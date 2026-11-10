"""Hand-written backward passes of the uniform-input store kernels (exact.py), one fused Triton reverse sweep per kernel.

Each kernel runs backwards in time. At step t it recomputes the step from the stored state at its start, carrying a tangent vector of
KT forward-mode derivatives with every intermediate (dual numbers: value [BLOCK], tangents [BLOCK, KT]); the tangents are seeded with the
state, the per-step inputs and the parameters of the step. With c = sum_m gout_m,t d out_m,t / d(.) + lam_t d x_t / d(.), the adjoint
of the state is lam_{t-1} = c[state] (+ the gradient of the stored state x_{t-1}), the gradients of the per-step inputs are the
matching entries of c, and the gradients of the parameters are the sums over t of their entries. No intermediate is stored: the step is
recomputed in registers from the stored state, so the backward reads the same arrays as the forward and the generic stored-state
adjoint (adjoint.backward_k, K+1 autograd passes over [N, T] tensors) is not needed. tests: tests/check_exact.py.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as ld

from .exact import _omtc, _psi, _tc

BLOCK = 32


# ----------------------------------------------------------------------------- dual-number primitives (value, tangents [BLOCK, KT])
@triton.jit
def dmul(a, da, b, db):
    return a * b, da * b[:, None] + a[:, None] * db


@triton.jit
def dmulc(a, da, c):
    """a * c, c without tangent ([BLOCK])"""
    return a * c, da * c[:, None]


@triton.jit
def ddiv(a, da, b, db):
    q = a / b
    return q, (da - q[:, None] * db) / b[:, None]


@triton.jit
def dexp(a, da):
    e = tl.exp(a)
    return e, da * e[:, None]


@triton.jit
def dlog(a, da):
    return tl.log(a), da / a[:, None]


@triton.jit
def dlog1p(a, da):
    return ld.log1p(a), da / (1.0 + a)[:, None]


@triton.jit
def dem(a, da):
    """1 - exp(-a)"""
    return -ld.expm1(-a), da * tl.exp(-a)[:, None]


@triton.jit
def dtanh(a, da):
    t = ld.tanh(a)
    return t, da * (1.0 - t * t)[:, None]


@triton.jit
def dmaxc(a, da, c):
    """max(a, c), c a constant"""
    k = a > c
    return tl.where(k, a, c), tl.where(k[:, None], da, 0.0)


@triton.jit
def dclampc(a, da, c):
    """max(a, c) with the derivative of torch.clamp(min=c) at a == c (1)"""
    k = a >= c
    return tl.where(k, a, c), tl.where(k[:, None], da, 0.0)


@triton.jit
def dminc(a, da, c):
    k = a < c
    return tl.where(k, a, c), tl.where(k[:, None], da, 0.0)


@triton.jit
def dpow35(r, dr):
    """r^3.5 for r > 0, else 0, as exp(3.5 log r) (the form of the quick_x kernel)"""
    pos = r > 0
    lr = tl.log(tl.maximum(r, 1e-30))
    return tl.where(pos, tl.exp(3.5 * lr), 0.0), dr * tl.where(pos, 3.5 * tl.exp(2.5 * lr), 0.0)[:, None]


@triton.jit
def dpow35s(r, dr):
    """r^3.5 = r^3 sqrt(r) for r >= 0 (the form of the lin_x kernel)"""
    s = tl.sqrt(r)
    return r * r * r * s, dr * (3.5 * r * r * s)[:, None]


@triton.jit
def _tcd(y):
    """tanh(sqrt(y)) / sqrt(y) and its derivative in y"""
    small = y < 0.04
    r = tl.sqrt(tl.where(small, 1.0, y))
    th = ld.tanh(r)
    dbig = ((1.0 - th * th) / r - th / (r * r)) / (2.0 * r)
    dsmall = -1.0 / 3.0 + y * (4.0 / 15.0 - y * (51.0 / 315.0 - y * 248.0 / 2835.0))
    return _tc(y), tl.where(small, dsmall, dbig)


@triton.jit
def _psid(x):
    """psi(x) = 1 - (1 - exp(-x)) / x and its derivative"""
    small = x < 0.05
    xs = tl.where(small, 1.0, x)
    dbig = (-ld.expm1(-xs) - xs * tl.exp(-xs)) / (xs * xs)
    dsmall = 0.5 - x * (1.0 / 3.0 - x * (1.0 / 8.0 - x / 30.0))
    return _psi(x), tl.where(small, dsmall, dbig)


@triton.jit
def _col(c, kk, j):
    return tl.sum(tl.where(kk == j, c, 0.0), axis=1)


@triton.jit
def _seed(Z, kk, j):
    return Z + tl.where(kk == j, 1.0, 0.0)


# ----------------------------------------------------------------------------- surface layer (tangents: U, p, e, cu, b)
@triton.jit
def _top_bwd(p_ptr, e_ptr, cu_ptr, b_ptr, x0_ptr, X_ptr, go_ptr, gd_ptr, ge_ptr, gX_ptr, gp_ptr, gee_ptr, gpar_ptr, N, T,
             HAS_GX: tl.constexpr, BLOCK: tl.constexpr, KT: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    kk = tl.arange(0, KT)[None, :]
    Z = tl.zeros([BLOCK, KT], dtype=tl.float32)
    cu = tl.load(cu_ptr + n, mask=m, other=1.0)
    b = tl.load(b_ptr + n, mask=m, other=1.0)
    x0 = tl.load(x0_ptr + n, mask=m, other=0.0)
    dcu = _seed(Z, kk, 3)
    db = _seed(Z, kk, 4)
    eb, deb = dem(b, db)
    lam = tl.zeros([BLOCK], dtype=tl.float32)
    acc = Z
    for i in range(T):
        t = T - 1 - i
        if HAS_GX:
            lam += tl.load(gX_ptr + n * T + t, mask=m, other=0.0)
        U = tl.where(t > 0, tl.load(X_ptr + n * T + t - 1, mask=m & (t > 0), other=0.0), x0)
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        dU = _seed(Z, kk, 0)
        dp = _seed(Z, kk, 1)
        de = _seed(Z, kk, 2)
        net = p - e
        dnet = dp - de
        a1, da1 = dmul(U, dU, 1.0 - eb, -deb)
        r_, dr_ = ddiv(eb, deb, b, db)
        a2, da2 = dmul(net, dnet, r_, dr_)
        Uf = a1 + a2
        dUf = da1 + da2
        over = (net >= 0.0) & (Uf > cu)
        bcu, dbcu = dmul(b, db, cu, dcu)
        sp, dsp = dmaxc(net - bcu, dnet - dbcu, 1e-12)
        cmU, dcmU = dmaxc(cu - U, dcu - dU, 0.0)
        q1, dq1 = dmul(b, db, cmU, dcmU)
        q1, dq1 = ddiv(q1, dq1, sp, dsp)
        l1, dl1 = dlog1p(q1, dq1)
        tcf, dtcf = ddiv(l1, dl1, b, db)
        tcf, dtcf = dminc(tcf, dtcf, 1.0)
        nn, dnn = dmaxc(-net, -dnet, 1e-12)
        bU, dbU = dmul(b, db, U, dU)
        q0, dq0 = ddiv(bU, dbU, nn, dnn)
        l0, dl0 = dlog1p(q0, dq0)
        t0, dt0 = ddiv(l0, dl0, b, db)
        empt = (net < 0.0) & (t0 < 1.0)
        ts = tl.where(over, tcf, tl.where(empt, t0, 1.0))
        dts = tl.where(over[:, None], dtcf, tl.where(empt[:, None], dt0, 0.0))
        x, dx = dmul(b, db, ts, dts)
        ex, dex = dem(x, dx)
        ps_, psp = _psid(x)
        dps_ = dx * psp[:, None]
        u1, du1 = dmul(U, dU, ex, dex)
        nt, dnt = dmul(net, dnet, ts, dts)
        u2, du2 = dmul(nt, dnt, ps_, dps_)
        u3, du3 = dmul(bcu, dbcu, 1.0 - tcf, -dtcf)
        ddu = du1 + du2 + tl.where(over[:, None], du3, 0.0)
        u4, du4 = dmul(sp, dsp, 1.0 - tcf, -dtcf)
        dou = tl.where(over[:, None], du4, 0.0)
        e1, de1 = dmul(e, de, t0, dt0)
        e2, de2 = dmul(p, dp, 1.0 - t0, -dt0)
        deu = tl.where(empt[:, None], de1 + de2, de)
        dUn = tl.where(over[:, None], dcu, tl.where(empt[:, None], 0.0, dUf))
        go = tl.load(go_ptr + n * T + t, mask=m, other=0.0)
        gd = tl.load(gd_ptr + n * T + t, mask=m, other=0.0)
        ge = tl.load(ge_ptr + n * T + t, mask=m, other=0.0)
        c = go[:, None] * dou + gd[:, None] * ddu + ge[:, None] * deu + lam[:, None] * dUn
        lam = _col(c, kk, 0)
        tl.store(gp_ptr + n * T + t, _col(c, kk, 1), mask=m)
        tl.store(gee_ptr + n * T + t, _col(c, kk, 2), mask=m)
        acc += c
    tl.store(gpar_ptr + n, _col(acc, kk, 3), mask=m)
    tl.store(gpar_ptr + N + n, _col(acc, kk, 4), mask=m)


# ----------------------------------------------------------------------------- quadratic groundwater store (tangents: W, g, a)
@triton.jit
def _quad_bwd(g_ptr, a_ptr, x0_ptr, X_ptr, gq_ptr, gX_ptr, gg_ptr, gpar_ptr, N, T, HAS_GX: tl.constexpr, BLOCK: tl.constexpr,
              KT: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    kk = tl.arange(0, KT)[None, :]
    Z = tl.zeros([BLOCK, KT], dtype=tl.float32)
    a = tl.load(a_ptr + n, mask=m, other=0.0)
    x0 = tl.load(x0_ptr + n, mask=m, other=0.0)
    da = _seed(Z, kk, 2)
    lam = tl.zeros([BLOCK], dtype=tl.float32)
    acc = Z
    for i in range(T):
        t = T - 1 - i
        if HAS_GX:
            lam += tl.load(gX_ptr + n * T + t, mask=m, other=0.0)
        W = tl.where(t > 0, tl.load(X_ptr + n * T + t - 1, mask=m & (t > 0), other=0.0), x0)
        g = tl.load(g_ptr + n * T + t, mask=m, other=0.0)
        dW = _seed(Z, kk, 0)
        g, dg = dclampc(g, _seed(Z, kk, 1), 0.0)
        y, dy = dmul(g, dg, a, da)
        Tc, Tcp = _tcd(y)
        dTc = dy * Tcp[:, None]
        om = _omtc(y)
        dom = -dTc
        aW, daW = dmul(a, da, W, dW)
        aWT, daWT = dmul(aW, daW, Tc, dTc)
        den = 1.0 + aWT
        n1, dn1 = dmul(aWT, daWT, W, dW)
        yW, dyW = dmul(y, dy, W, dW)
        n2, dn2 = dmul(yW, dyW, Tc, dTc)
        n3, dn3 = dmul(g, dg, om, dom)
        Q, dQ = ddiv(n1 + n2 + n3, dn1 + dn2 + dn3, den, daWT)
        gT, dgT = dmul(g, dg, Tc, dTc)
        Wn, dWn = ddiv(W + gT, dW + dgT, den, daWT)
        gq = tl.load(gq_ptr + n * T + t, mask=m, other=0.0)
        c = gq[:, None] * dQ + lam[:, None] * dWn
        lam = _col(c, kk, 0)
        tl.store(gg_ptr + n * T + t, _col(c, kk, 1), mask=m)
        acc += c
    tl.store(gpar_ptr + n, _col(acc, kk, 2), mask=m)


# ----------------------------------------------------------------------------- groundwater store with uptake (tangents: W, g, er, a, lam, w0)
@triton.jit
def _dupt(W, dW, le, dle, w0, dw0):
    """uptake over half a sub-step: W (1 - exp(-le / (W exp(-le / (2 (W + w0))) + w0))), le = lam er"""
    s, ds = ddiv(le, dle, W + w0, dW + dw0)
    h, dh = dexp(-0.5 * s, -0.5 * ds)
    Wh, dWh = dmul(W, dW, h, dh)
    x, dx = ddiv(le, dle, Wh + w0, dWh + dw0)
    ex, dex = dem(x, dx)
    return dmul(W, dW, ex, dex)


@triton.jit
def _gwet_bwd(g_ptr, er_ptr, a_ptr, lam_ptr, w0_ptr, x0_ptr, X_ptr, gq_ptr, ge_ptr, gX_ptr, gg_ptr, ger_ptr, gpar_ptr, N, T,
              NSUB: tl.constexpr, HAS_GX: tl.constexpr, BLOCK: tl.constexpr, KT: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    kk = tl.arange(0, KT)[None, :]
    Z = tl.zeros([BLOCK, KT], dtype=tl.float32)
    a = tl.load(a_ptr + n, mask=m, other=0.0) / NSUB
    da = _seed(Z, kk, 3) / NSUB
    la = tl.load(lam_ptr + n, mask=m, other=0.0) * (0.5 / NSUB)
    dla = _seed(Z, kk, 4) * (0.5 / NSUB)
    w0 = tl.load(w0_ptr + n, mask=m, other=1.0)
    dw0 = _seed(Z, kk, 5)
    x0 = tl.load(x0_ptr + n, mask=m, other=0.0)
    lmb = tl.zeros([BLOCK], dtype=tl.float32)
    acc = Z
    for i in range(T):
        t = T - 1 - i
        if HAS_GX:
            lmb += tl.load(gX_ptr + n * T + t, mask=m, other=0.0)
        W = tl.where(t > 0, tl.load(X_ptr + n * T + t - 1, mask=m & (t > 0), other=0.0), x0)
        dW = _seed(Z, kk, 0)
        g, dg = dclampc(tl.load(g_ptr + n * T + t, mask=m, other=0.0), _seed(Z, kk, 1), 0.0)
        g = g / NSUB
        dg = dg / NSUB
        er = tl.load(er_ptr + n * T + t, mask=m, other=0.0)
        le, dle = dmul(la, dla, er, _seed(Z, kk, 2))
        y, dy = dmul(g, dg, a, da)
        Tc, Tcp = _tcd(y)
        dTc = dy * Tcp[:, None]
        om = _omtc(y)
        dom = -dTc
        Q = tl.zeros([BLOCK], dtype=tl.float32)
        dQ = Z
        E = tl.zeros([BLOCK], dtype=tl.float32)
        dE = Z
        for j in tl.static_range(NSUB):
            e1, de1 = _dupt(W, dW, le, dle, w0, dw0)
            W = W - e1
            dW = dW - de1
            aW, daW = dmul(a, da, W, dW)
            aWT, daWT = dmul(aW, daW, Tc, dTc)
            n1, dn1 = dmul(aWT, daWT, W, dW)
            yW, dyW = dmul(y, dy, W, dW)
            n2, dn2 = dmul(yW, dyW, Tc, dTc)
            n3, dn3 = dmul(g, dg, om, dom)
            q, dq = ddiv(n1 + n2 + n3, dn1 + dn2 + dn3, 1.0 + aWT, daWT)
            Q += q
            dQ += dq
            gT, dgT = dmul(g, dg, Tc, dTc)
            W, dW = ddiv(W + gT, dW + dgT, 1.0 + aWT, daWT)
            e2, de2 = _dupt(W, dW, le, dle, w0, dw0)
            W = W - e2
            dW = dW - de2
            E += e1 + e2
            dE += de1 + de2
        gq = tl.load(gq_ptr + n * T + t, mask=m, other=0.0)
        ge = tl.load(ge_ptr + n * T + t, mask=m, other=0.0)
        c = gq[:, None] * dQ + ge[:, None] * dE + lmb[:, None] * dW
        lmb = _col(c, kk, 0)
        tl.store(gg_ptr + n * T + t, _col(c, kk, 1), mask=m)
        tl.store(ger_ptr + n * T + t, _col(c, kk, 2), mask=m)
        acc += c
    tl.store(gpar_ptr + n, _col(acc, kk, 3), mask=m)
    tl.store(gpar_ptr + N + n, _col(acc, kk, 4), mask=m)
    tl.store(gpar_ptr + 2 * N + n, _col(acc, kk, 5), mask=m)


# ----------------------------------------------------------------------------- GR4J routing store (tangents: R, x9, x1, X2, X3, X5)
@triton.jit
def _drel(R, dR, c, dc):
    R2, dR2 = dmul(R, dR, R, dR)
    R4, dR4 = dmul(R2, dR2, R2, dR2)
    z, dz = dmul(c, dc, R4, dR4)
    L, dL = dlog1p(z, dz)
    e, de = dem(0.25 * L, 0.25 * dL)
    return dmul(R, dR, e, de)


@triton.jit
def _dquick_chain(R, dR, x9s, dx9s, x1s, dx1s, X2s, dX2s, X3, dX3, X5, dX5, ch, dch, Z, CR: tl.constexpr, CD: tl.constexpr,
                  NS: tl.constexpr):
    Q = tl.zeros_like(R)
    dQ = Z
    xr = tl.zeros_like(R)
    dxr = Z
    for j in tl.static_range(NS):
        q1, dq1 = _drel(R, dR, ch, dch)
        R = R - q1
        dR = dR - dq1
        rr, drr = ddiv(R, dR, X3, dX3)
        g1, dg1 = dpow35(rr, drr)
        F1, dF1 = dmul(X2s, dX2s, g1 - X5, dg1 - dX5)
        h, dh = dmaxc(R + 0.5 * (x9s + CR * F1), dR + 0.5 * (dx9s + CR * dF1), 0.0)
        rm, drm = ddiv(h, dh, X3, dX3)
        gm, dgm = dpow35(rm, drm)
        F, dF = dmul(X2s, dX2s, gm - X5, dgm - dX5)
        Rn, dRn = dmaxc(R + x9s + CR * F, dR + dx9s + CR * dF, 0.0)
        Qd, dQd = dmaxc(x1s + CD * F, dx1s + CD * dF, 0.0)
        xr += (Rn - R - x9s) + (Qd - x1s)
        dxr += (dRn - dR - dx9s) + (dQd - dx1s)
        q2, dq2 = _drel(Rn, dRn, ch, dch)
        R = Rn - q2
        dR = dRn - dq2
        Q += q1 + q2 + Qd
        dQ += dq1 + dq2 + dQd
    return R, dR, Q, dQ, xr, dxr


@triton.jit
def _quick_bwd(x9_ptr, x1_ptr, x2_ptr, x3_ptr, x5_ptr, x0_ptr, X_ptr, gq_ptr, gxr_ptr, gX_ptr, g9_ptr, g1_ptr, gpar_ptr, N, T,
               DT: tl.constexpr, NSUB: tl.constexpr, CR: tl.constexpr, CD: tl.constexpr, HAS_GX: tl.constexpr, BLOCK: tl.constexpr,
               KT: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    kk = tl.arange(0, KT)[None, :]
    Z = tl.zeros([BLOCK, KT], dtype=tl.float32)
    X2 = tl.load(x2_ptr + n, mask=m, other=0.0)
    X3 = tl.load(x3_ptr + n, mask=m, other=1.0)
    X5 = tl.load(x5_ptr + n, mask=m, other=0.0)
    x0 = tl.load(x0_ptr + n, mask=m, other=0.0)
    dX2 = _seed(Z, kk, 3)
    dX3 = _seed(Z, kk, 4)
    dX5 = _seed(Z, kk, 5)
    X32 = X3 * X3
    ch = 0.5 * DT / (X32 * X32)
    dch = dX3 * (-4.0 * ch / X3)[:, None]
    fa = 1.0 / NSUB
    fb = 2.0 / NSUB
    lam = tl.zeros([BLOCK], dtype=tl.float32)
    acc = Z
    for i in range(T):
        t = T - 1 - i
        if HAS_GX:
            lam += tl.load(gX_ptr + n * T + t, mask=m, other=0.0)
        R = tl.where(t > 0, tl.load(X_ptr + n * T + t - 1, mask=m & (t > 0), other=0.0), x0)
        x9 = tl.load(x9_ptr + n * T + t, mask=m, other=0.0)
        x1 = tl.load(x1_ptr + n * T + t, mask=m, other=0.0)
        dR = _seed(Z, kk, 0)
        dx9 = _seed(Z, kk, 1)
        dx1 = _seed(Z, kk, 2)
        Ra, dRa, Qa, dQa, xa, dxa = _dquick_chain(R, dR, x9 * fa, dx9 * fa, x1 * fa, dx1 * fa, X2 * fa, dX2 * fa, X3, dX3, X5, dX5,
                                                  ch * fa, dch * fa, Z, CR, CD, NSUB)
        Rb, dRb, Qb, dQb, xb, dxb = _dquick_chain(R, dR, x9 * fb, dx9 * fb, x1 * fb, dx1 * fb, X2 * fb, dX2 * fb, X3, dX3, X5, dX5,
                                                  ch * fb, dch * fb, Z, CR, CD, NSUB // 2)
        Rr = (4.0 * Ra - Rb) / 3.0
        dRr = (4.0 * dRa - dRb) / 3.0
        dQ = (4.0 * dQa - dQb) / 3.0
        dxr = (4.0 * dxa - dxb) / 3.0
        ng = Rr < 0.0                                                                     # R = max(Rr, 0), xr += max(-Rr, 0)
        dRn = tl.where(ng[:, None], 0.0, dRr)
        dxr = dxr + tl.where(ng[:, None], -dRr, 0.0)
        gq = tl.load(gq_ptr + n * T + t, mask=m, other=0.0)
        gx = tl.load(gxr_ptr + n * T + t, mask=m, other=0.0)
        c = gq[:, None] * dQ + gx[:, None] * dxr + lam[:, None] * dRn
        lam = _col(c, kk, 0)
        tl.store(g9_ptr + n * T + t, _col(c, kk, 1), mask=m)
        tl.store(g1_ptr + n * T + t, _col(c, kk, 2), mask=m)
        acc += c
    tl.store(gpar_ptr + n, _col(acc, kk, 3), mask=m)
    tl.store(gpar_ptr + N + n, _col(acc, kk, 4), mask=m)
    tl.store(gpar_ptr + 2 * N + n, _col(acc, kk, 5), mask=m)


# ----------------------------------------------------------------------------- derived-release quick store (tangents: R, u1, u2, c0, c1, p1, p2, sc)
@triton.jit
def _dlrel(R, dR, p1h, dp1h, sc, dsc, c0, dc0, c1, dc1):
    Rc, dRc = dmaxc(R, dR, 1e-3)
    rs, drs = ddiv(Rc, dRc, sc, dsc)
    f, df = dlog(rs, drs)
    cf, dcf = dmul(c1, dc1, f, df)
    th, dth = dtanh((c0 + cf) / 6.0, (dc0 + dcf) / 6.0)
    ex, dex = dexp(6.0 * th, 6.0 * dth)
    A, dA = dmul(p1h, dp1h, ex, dex)
    t2, dt2 = dmul(th, dth, th, dth)
    co, dco = dmul(c1, dc1, 1.0 - t2, -dt2)
    nu = 4.0 + co
    R2, dR2 = dmul(R, dR, R, dR)
    R4, dR4 = dmul(R2, dR2, R2, dR2)
    z, dz = dmul(nu, dco, A, dA)
    z, dz = dmul(z, dz, R4, dR4)
    L, dL = dlog1p(z, dz)
    y, dy = ddiv(L, dL, nu, dco)
    e, de = dem(y, dy)
    return dmul(R, dR, e, de)


@triton.jit
def _dlin_chain(R, dR, u1s, du1s, u2s, du2s, p1h, dp1h, p2s, dp2s, sc, dsc, c0, dc0, c1, dc1, Z, NS: tl.constexpr):
    Q = tl.zeros_like(R)
    dQ = Z
    xr = tl.zeros_like(R)
    dxr = Z
    for j in tl.static_range(NS):
        q1, dq1 = _dlrel(R, dR, p1h, dp1h, sc, dsc, c0, dc0, c1, dc1)
        R = R - q1
        dR = dR - dq1
        Sc, dSc = dmaxc(R, dR, 0.0)
        s35, ds35 = dpow35s(Sc / 30.0, dSc / 30.0)
        e1, de1 = dmul(p2s, dp2s, s35, ds35)
        h, dh = dmaxc(R + 0.5 * (u1s + 0.9 * e1), dR + 0.5 * (du1s + 0.9 * de1), 0.0)
        m35, dm35 = dpow35s(h / 30.0, dh / 30.0)
        ex, dex = dmul(p2s, dp2s, m35, dm35)
        Rm, dRm = dmaxc(R + u1s + 0.9 * ex, dR + du1s + 0.9 * dex, 0.0)
        Qd, dQd = dmaxc(u2s + 0.1 * ex, du2s + 0.1 * dex, 0.0)
        xr += (Rm - R - u1s) + (Qd - u2s)
        dxr += (dRm - dR - du1s) + (dQd - du2s)
        q2, dq2 = _dlrel(Rm, dRm, p1h, dp1h, sc, dsc, c0, dc0, c1, dc1)
        R = Rm - q2
        dR = dRm - dq2
        Q += q1 + q2 + Qd
        dQ += dq1 + dq2 + dQd
    return R, dR, Q, dQ, xr, dxr


@triton.jit
def _lin_bwd(u1_ptr, u2_ptr, c0_ptr, c1_ptr, p1_ptr, p2_ptr, sc_ptr, x0_ptr, X_ptr, gq_ptr, gxr_ptr, gX_ptr, gu1_ptr, gu2_ptr, gc0_ptr,
             gc1_ptr, gpar_ptr, N, T, NSUB: tl.constexpr, HAS_GX: tl.constexpr, BLOCK: tl.constexpr, KT: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    kk = tl.arange(0, KT)[None, :]
    Z = tl.zeros([BLOCK, KT], dtype=tl.float32)
    p1h = tl.load(p1_ptr + n, mask=m, other=0.0) * 0.5
    dp1h = _seed(Z, kk, 5) * 0.5
    p2 = tl.load(p2_ptr + n, mask=m, other=0.0)
    dp2 = _seed(Z, kk, 6)
    sc = tl.load(sc_ptr + n, mask=m, other=1.0)
    dsc = _seed(Z, kk, 7)
    x0 = tl.load(x0_ptr + n, mask=m, other=0.0)
    fa = 1.0 / NSUB
    fb = 2.0 / NSUB
    lam = tl.zeros([BLOCK], dtype=tl.float32)
    acc = Z
    for i in range(T):
        t = T - 1 - i
        if HAS_GX:
            lam += tl.load(gX_ptr + n * T + t, mask=m, other=0.0)
        R = tl.where(t > 0, tl.load(X_ptr + n * T + t - 1, mask=m & (t > 0), other=0.0), x0)
        u1 = tl.load(u1_ptr + n * T + t, mask=m, other=0.0)
        u2 = tl.load(u2_ptr + n * T + t, mask=m, other=0.0)
        c0 = tl.load(c0_ptr + n * T + t, mask=m, other=0.0)
        c1 = tl.load(c1_ptr + n * T + t, mask=m, other=0.0)
        dR = _seed(Z, kk, 0)
        du1 = _seed(Z, kk, 1)
        du2 = _seed(Z, kk, 2)
        dc0 = _seed(Z, kk, 3)
        dc1 = _seed(Z, kk, 4)
        Ra, dRa, Qa, dQa, xa, dxa = _dlin_chain(R, dR, u1 * fa, du1 * fa, u2 * fa, du2 * fa, p1h * fa, dp1h * fa, p2 * fa, dp2 * fa,
                                                sc, dsc, c0, dc0, c1, dc1, Z, NSUB)
        Rb, dRb, Qb, dQb, xb, dxb = _dlin_chain(R, dR, u1 * fb, du1 * fb, u2 * fb, du2 * fb, p1h * fb, dp1h * fb, p2 * fb, dp2 * fb,
                                                sc, dsc, c0, dc0, c1, dc1, Z, NSUB // 2)
        Rr = (4.0 * Ra - Rb) / 3.0
        dRr = (4.0 * dRa - dRb) / 3.0
        dQ = (4.0 * dQa - dQb) / 3.0
        dxr = (4.0 * dxa - dxb) / 3.0
        ng = Rr < 0.0
        dRn = tl.where(ng[:, None], 0.0, dRr)
        dxr = dxr + tl.where(ng[:, None], -dRr, 0.0)
        gq = tl.load(gq_ptr + n * T + t, mask=m, other=0.0)
        gx = tl.load(gxr_ptr + n * T + t, mask=m, other=0.0)
        c = gq[:, None] * dQ + gx[:, None] * dxr + lam[:, None] * dRn
        lam = _col(c, kk, 0)
        tl.store(gu1_ptr + n * T + t, _col(c, kk, 1), mask=m)
        tl.store(gu2_ptr + n * T + t, _col(c, kk, 2), mask=m)
        tl.store(gc0_ptr + n * T + t, _col(c, kk, 3), mask=m)
        tl.store(gc1_ptr + n * T + t, _col(c, kk, 4), mask=m)
        acc += c
    tl.store(gpar_ptr + n, _col(acc, kk, 5), mask=m)
    tl.store(gpar_ptr + N + n, _col(acc, kk, 6), mask=m)
    tl.store(gpar_ptr + 2 * N + n, _col(acc, kk, 7), mask=m)


# ----------------------------------------------------------------------------- launchers
def _g(x, like):
    return torch.zeros_like(like) if x is None else x.detach().float().contiguous()


def top_bwd(p, e, cu, b, x0, X, gO, gD, gE, gX):
    N, T = p.shape
    gp, ge, gpar = torch.empty_like(p), torch.empty_like(p), torch.empty(2, N, device=p.device)
    _top_bwd[(triton.cdiv(N, BLOCK),)](p, e, cu, b, x0, X, _g(gO, p), _g(gD, p), _g(gE, p), _g(gX, p), gp, ge, gpar, N, T,
                                       HAS_GX=gX is not None, BLOCK=BLOCK, KT=8)
    return gp, ge, gpar[0], gpar[1]


def quad_bwd(g, a, x0, X, gq, gX):
    N, T = g.shape
    gg, gpar = torch.empty_like(g), torch.empty(1, N, device=g.device)
    _quad_bwd[(triton.cdiv(N, BLOCK),)](g, a, x0, X, _g(gq, g), _g(gX, g), gg, gpar, N, T, HAS_GX=gX is not None, BLOCK=BLOCK, KT=4)
    return gg, gpar[0]


def gwet_bwd(g, er, a, lam, w0, x0, X, gq, ge, gX, nsub):
    N, T = g.shape
    gg, ger, gpar = torch.empty_like(g), torch.empty_like(g), torch.empty(3, N, device=g.device)
    _gwet_bwd[(triton.cdiv(N, BLOCK),)](g, er, a, lam, w0, x0, X, _g(gq, g), _g(ge, g), _g(gX, g), gg, ger, gpar, N, T, NSUB=nsub,
                                        HAS_GX=gX is not None, BLOCK=BLOCK, KT=8)
    return gg, ger, gpar[0], gpar[1], gpar[2]


def quick_bwd(x9, x1, X2, X3, X5, x0, X, gq, gxr, gX, dt, nsub, cr, cd):
    N, T = x9.shape
    g9, g1, gpar = torch.empty_like(x9), torch.empty_like(x9), torch.empty(3, N, device=x9.device)
    _quick_bwd[(triton.cdiv(N, BLOCK),)](x9, x1, X2, X3, X5, x0, X, _g(gq, x9), _g(gxr, x9), _g(gX, x9), g9, g1, gpar, N, T, float(dt),
                                         NSUB=nsub, CR=float(cr), CD=float(cd), HAS_GX=gX is not None, BLOCK=BLOCK, KT=8)
    return g9, g1, gpar[0], gpar[1], gpar[2]


def lin_bwd(u1, u2, c0, c1, p1, p2, sc, x0, X, gq, gxr, gX, nsub):
    N, T = u1.shape
    gu1, gu2, gc0, gc1 = (torch.empty_like(u1) for _ in range(4))
    gpar = torch.empty(3, N, device=u1.device)
    _lin_bwd[(triton.cdiv(N, BLOCK),)](u1, u2, c0, c1, p1, p2, sc, x0, X, _g(gq, u1), _g(gxr, u1), _g(gX, u1), gu1, gu2, gc0, gc1, gpar,
                                       N, T, NSUB=nsub, HAS_GX=gX is not None, BLOCK=BLOCK, KT=8)
    return gu1, gu2, gc0, gc1, gpar[0], gpar[1], gpar[2]
