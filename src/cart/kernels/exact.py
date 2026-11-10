"""Stores advanced over one step for an input that arrives at a CONSTANT RATE within the step (uniform input), outputs = step totals.

Every operator below integrates its own differential equation over the step, so that a daily and an hourly run approximate the same
continuous model and the step length carries no convention about the time of the input within the step.
  closed form (exact):
    top_x     surface layer        dU/dt = p - e_act - U/tau_u, evaporation at the potential rate while wet, overflow at U = Cu
                                   (piecewise-linear equation: the times at which the layer fills or dries are solved in closed form)
    quad_x    groundwater store    dW/dt = g - a W^2  (Riccati equation, tanh solution)
  no closed form for the joint equation; solved by NSUB symmetric (Strang) sub-steps of the exact solutions of its parts, the input of
  the step spread evenly over the sub-steps (error of second order in dt / NSUB, the same NSUB at every step length); quick_x and lin_x
  combine the solutions with NSUB and NSUB / 2 sub-steps by Richardson extrapolation, (4 R_n - R_{n/2}) / 3, to fourth order:
    prod_x    GR4J production store   dS/dt = pn (1 - s^2) - en s (2 - s) - Kp s^5        parts: wetting / drying (tanh), percolation
    pareto_x  Pareto production store (and depression storage)                             parts: wetting / drying, percolation
    quick_x   GR4J routing store      dR/dt = x9 + cr F(R) - R^5 / (4 X3^4)                parts: release, inflow + exchange
    lin_x     quick store of the derived release law (CaRT+)                               parts: release, inflow + exchange
    gwet_x    groundwater store with uptake  dW/dt = g - a W^2 - lam Er W / (W + W0)       parts: uptake, inflow + release (Riccati)
All per-step quantities (inputs, rate coefficients) are amounts per step, as in the other kernels. Forward: fused Triton recursion that
stores the state after every step; backward: the hand-written fused reverse sweeps of exact_bwd.py. The same step is also written as
an elementwise torch function (`*_step`), which performs the same operations as the kernel; with CART_GENERIC_ADJ=1 the backward is the
generic stored-state adjoint (adjoint.backward_k) of that function (reference, slower). tests/check_exact.py checks both.
"""
import functools
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as ld

from .adjoint import backward_k

GENERIC = os.environ.get("CART_GENERIC_ADJ", "0") == "1"                              # 1: generic stored-state adjoint (reference)

BLOCK = 32


# ----------------------------------------------------------------------------- shared pieces (Triton / torch twins)
@triton.jit
def _em(x):
    """1 - exp(-x)"""
    return -ld.expm1(-x)


@triton.jit
def _rel(R, c):
    """release of the power-law store over (part of) a step: exact for dR/dt = -(c/4) R^5, R - R (1 + c R^4)^(-1/4)"""
    R2 = R * R
    return R * _em(0.25 * ld.log1p(c * R2 * R2))


@triton.jit
def _psi(x):
    """1 - (1 - exp(-x)) / x, x >= 0"""
    xs = tl.where(x < 0.05, 1.0, x)
    return tl.where(x < 0.05, x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))), 1.0 - _em(xs) / xs)


@triton.jit
def _tc(y):
    """tanh(sqrt(y)) / sqrt(y), y >= 0"""
    r = tl.sqrt(tl.where(y < 0.04, 1.0, y))
    return tl.where(y < 0.04, 1.0 - y * (1.0 / 3.0 - y * (2.0 / 15.0 - y * (17.0 / 315.0 - y * 62.0 / 2835.0))), ld.tanh(r) / r)


@triton.jit
def _omtc(y):
    """1 - tanh(sqrt(y)) / sqrt(y), y >= 0"""
    r = tl.sqrt(tl.where(y < 0.04, 1.0, y))
    return tl.where(y < 0.04, y * (1.0 / 3.0 - y * (2.0 / 15.0 - y * (17.0 / 315.0 - y * 62.0 / 2835.0))), 1.0 - ld.tanh(r) / r)


def em(x):
    return -torch.expm1(-x)


def rel(R, c):
    R2 = R * R
    return R * em(0.25 * torch.log1p(c * R2 * R2))


def psi(x):
    sm = x < 0.05
    xs = torch.where(sm, torch.ones_like(x), x)
    return torch.where(sm, x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))), 1.0 - em(xs) / xs)


def tc(y):
    sm = y < 0.04
    r = torch.sqrt(torch.where(sm, torch.ones_like(y), y))
    return torch.where(sm, 1.0 - y * (1.0 / 3.0 - y * (2.0 / 15.0 - y * (17.0 / 315.0 - y * 62.0 / 2835.0))), torch.tanh(r) / r)


def omtc(y):
    sm = y < 0.04
    r = torch.sqrt(torch.where(sm, torch.ones_like(y), y))
    return torch.where(sm, y * (1.0 / 3.0 - y * (2.0 / 15.0 - y * (17.0 / 315.0 - y * 62.0 / 2835.0))), 1.0 - torch.tanh(r) / r)


def _prep(*ts):
    return [t_.detach().float().contiguous() for t_ in ts]


# ----------------------------------------------------------------------------- surface layer, closed form
# dU/dt = p - e - b U while 0 < U < Cu (b = dt / tau_u, time in steps). Free solution U(t) = U0 e^(-bt) + (p - e)(1 - e^(-bt)) / b.
#   p >= e: the layer fills; if it reaches Cu at tc < 1 it stays full, drains b Cu and spills p - e - b Cu.
#   p <  e: the layer dries; if it empties at t0 < 1 the rest of the input evaporates as it falls (evaporation p, no drainage).
# Outputs: overflow, drainage into the production store, evaporation.

@triton.jit
def _top_fwd(p_ptr, e_ptr, cu_ptr, b_ptr, x0_ptr, X_ptr, o_ptr, d_ptr, ev_ptr, N, T, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    cu = tl.load(cu_ptr + n, mask=m, other=1.0)
    b = tl.load(b_ptr + n, mask=m, other=1.0)
    eb = _em(b)
    U = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        net = p - e
        Uf = U * (1.0 - eb) + net * (eb / b)                                              # free evolution over the whole step
        over = (net >= 0.0) & (Uf > cu)
        sp = tl.maximum(net - b * cu, 1e-12)                                              # spill rate of the full layer
        tcf = tl.minimum(ld.log1p(b * tl.maximum(cu - U, 0.0) / sp) / b, 1.0)             # time at which the layer is full
        t0 = ld.log1p(b * U / tl.maximum(-net, 1e-12)) / b                                # time at which the layer is empty
        empt = (net < 0.0) & (t0 < 1.0)
        ts = tl.where(over, tcf, tl.where(empt, t0, 1.0))                                 # length of the free phase
        x = b * ts
        du = U * _em(x) + net * ts * _psi(x) + tl.where(over, b * cu * (1.0 - tcf), 0.0)
        ou = tl.where(over, sp * (1.0 - tcf), 0.0)
        eu = tl.where(empt, e * t0 + p * (1.0 - t0), e)
        U = tl.where(over, cu, tl.where(empt, 0.0, Uf))
        tl.store(X_ptr + n * T + t, U, mask=m)
        tl.store(o_ptr + n * T + t, ou, mask=m)
        tl.store(d_ptr + n * T + t, du, mask=m)
        tl.store(ev_ptr + n * T + t, eu, mask=m)


def top_step(xp, p, e, cu, b):
    (U,) = xp
    net = p - e
    eb = em(b)
    Uf = U * (1.0 - eb) + net * (eb / b)
    over = (net >= 0.0) & (Uf > cu)
    sp = (net - b * cu).clamp(min=1e-12)
    tcf = (torch.log1p(b * (cu - U).clamp(min=0.0) / sp) / b).clamp(max=1.0)
    t0 = torch.log1p(b * U / (-net).clamp(min=1e-12)) / b
    empt = (net < 0.0) & (t0 < 1.0)
    one = torch.ones_like(p)
    ts = torch.where(over, tcf, torch.where(empt, t0, one))
    x = b * ts
    du = U * em(x) + net * ts * psi(x) + torch.where(over, b * cu * (1.0 - tcf), torch.zeros_like(p))
    ou = torch.where(over, sp * (1.0 - tcf), torch.zeros_like(p))
    eu = torch.where(empt, e * t0 + p * (1.0 - t0), e)
    Un = torch.where(over, cu, torch.where(empt, torch.zeros_like(p), Uf))
    return [Un], [ou, du, eu]


class TopX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, e, cu, b, x0):
        N, T = p.shape
        v = _prep(p, e, cu, b, x0)
        X, O, D, Ev = (torch.empty(N, T, device=p.device) for _ in range(4))
        _top_fwd[(triton.cdiv(N, BLOCK),)](*v, X, O, D, Ev, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(*v, X)
        return O, D, Ev, X

    @staticmethod
    def backward(ctx, gO, gD, gE, gX):
        p, e, cu, b, x0, X = ctx.saved_tensors
        if not GENERIC:
            from .exact_bwd import top_bwd
            gp_, ge_, gcu, gb = top_bwd(p, e, cu, b, x0, X, gO, gD, gE, gX)
            return gp_, ge_, gcu, gb, None
        gi, gp = backward_k(top_step, X[..., None], x0[:, None], gX[..., None] if gX is not None else None, [gO, gD, gE], [p, e], [cu, b])
        return gi[0], gi[1], gp[0], gp[1], None


def top_x(p, e, cu, b, U0):
    """p, e [N, T]; cu, b = dt / tau_u, U0 [N] -> overflow, drainage, evaporation, U after each step [N, T]"""
    return TopX.apply(p, e, cu, b, U0)


# ----------------------------------------------------------------------------- quadratic groundwater store, closed form
# dW/dt = g - a W^2 over one step:  W1 = (W + g Tc) / (1 + a W Tc),  Tc = tanh(sqrt(g a)) / sqrt(g a)  (g = 0: W / (1 + a W));
# Q = W + g - W1 = [a W^2 Tc + g a W Tc + g (1 - Tc)] / (1 + a W Tc)  (no cancellation).

@triton.jit
def _quad_fwd(g_ptr, a_ptr, x0_ptr, X_ptr, q_ptr, N, T, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    a = tl.load(a_ptr + n, mask=m, other=0.0)
    W = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        g = tl.maximum(tl.load(g_ptr + n * T + t, mask=m, other=0.0), 0.0)
        y = g * a
        Tc = _tc(y)
        den = 1.0 + a * W * Tc
        Q = (a * W * W * Tc + y * W * Tc + g * _omtc(y)) / den
        W = (W + g * Tc) / den
        tl.store(X_ptr + n * T + t, W, mask=m)
        tl.store(q_ptr + n * T + t, Q, mask=m)


def quad_step(xp, g, a):
    (W,) = xp
    g = g.clamp(min=0.0)
    y = g * a
    Tc = tc(y)
    den = 1.0 + a * W * Tc
    return [(W + g * Tc) / den], [(a * W * W * Tc + y * W * Tc + g * omtc(y)) / den]


class QuadX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, a, x0):
        N, T = g.shape
        v = _prep(g, a, x0)
        X, Q = torch.empty(N, T, device=g.device), torch.empty(N, T, device=g.device)
        _quad_fwd[(triton.cdiv(N, BLOCK),)](*v, X, Q, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(*v, X)
        return Q, X

    @staticmethod
    def backward(ctx, gq, gX):
        g, a, x0, X = ctx.saved_tensors
        if not GENERIC:
            from .exact_bwd import quad_bwd
            gg, ga = quad_bwd(g, a, x0, X, gq, gX)
            return gg, ga, None
        gi, gp = backward_k(quad_step, X[..., None], x0[:, None], gX[..., None] if gX is not None else None, [gq], [g], [a])
        return gi[0], gp[0], None


def quad_x(g, a, W0):
    """g [N, T]; a, W0 [N] -> Q, W after each step [N, T]"""
    return QuadX.apply(g, a, W0)


# ----------------------------------------------------------------------------- groundwater store with uptake, Strang sub-steps
# sub-step: half of the uptake (e_W = W (1 - exp(-lam Er / (Wm + W0))), Wm the storage half-way through it: second order), inflow +
# quadratic release (Riccati, exact), half of the uptake. lam = 0 is quad_x.

@triton.jit
def _gwet_fwd(g_ptr, er_ptr, a_ptr, lam_ptr, w0_ptr, x0_ptr, X_ptr, q_ptr, e_ptr, N, T, NSUB: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    a = tl.load(a_ptr + n, mask=m, other=0.0) / NSUB
    lam = tl.load(lam_ptr + n, mask=m, other=0.0) * (0.5 / NSUB)
    w0 = tl.load(w0_ptr + n, mask=m, other=1.0)
    W = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        g = tl.maximum(tl.load(g_ptr + n * T + t, mask=m, other=0.0), 0.0) / NSUB
        er = tl.load(er_ptr + n * T + t, mask=m, other=0.0)
        Q = tl.zeros_like(g)
        E = tl.zeros_like(g)
        y = g * a
        Tc = _tc(y)
        om = _omtc(y)
        for j in tl.static_range(NSUB):
            e1 = W * _em(lam * er / (W * tl.exp(-0.5 * lam * er / (W + w0)) + w0))        # rate at the midpoint of the half sub-step
            W = W - e1
            den = 1.0 + a * W * Tc
            Q += (a * W * W * Tc + y * W * Tc + g * om) / den
            W = (W + g * Tc) / den
            e2 = W * _em(lam * er / (W * tl.exp(-0.5 * lam * er / (W + w0)) + w0))
            W = W - e2
            E += e1 + e2
        tl.store(X_ptr + n * T + t, W, mask=m)
        tl.store(q_ptr + n * T + t, Q, mask=m)
        tl.store(e_ptr + n * T + t, E, mask=m)


def gwet_step(xp, g, er, a, lam, w0, nsub=4):
    (W,) = xp
    g = g.clamp(min=0.0) / nsub
    a = a / nsub
    lam = lam * (0.5 / nsub)
    y = g * a
    Tc, om = tc(y), omtc(y)
    Q, E = torch.zeros_like(g), torch.zeros_like(g)
    for _ in range(nsub):
        e1 = W * em(lam * er / (W * torch.exp(-0.5 * lam * er / (W + w0)) + w0)); W = W - e1
        den = 1.0 + a * W * Tc
        Q = Q + (a * W * W * Tc + y * W * Tc + g * om) / den
        W = (W + g * Tc) / den
        e2 = W * em(lam * er / (W * torch.exp(-0.5 * lam * er / (W + w0)) + w0)); W = W - e2
        E = E + e1 + e2
    return [W], [Q, E]


class GwEtX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, er, a, lam, w0, x0, nsub):
        N, T = g.shape
        v = _prep(g, er, a, lam, w0, x0)
        X, Q, Ew = (torch.empty(N, T, device=g.device) for _ in range(3))
        _gwet_fwd[(triton.cdiv(N, BLOCK),)](*v, X, Q, Ew, N, T, NSUB=nsub, BLOCK=BLOCK)
        ctx.nsub = nsub
        ctx.save_for_backward(*v, X)
        return Q, Ew, X

    @staticmethod
    def backward(ctx, gq, ge, gX):
        g, er, a, lam, w0, x0, X = ctx.saved_tensors
        if not GENERIC:
            from .exact_bwd import gwet_bwd
            gg, ger, ga, gl, gw = gwet_bwd(g, er, a, lam, w0, x0, X, gq, ge, gX, ctx.nsub)
            return gg, ger, ga, gl, gw, None, None
        gi, gp = backward_k(functools.partial(gwet_step, nsub=ctx.nsub), X[..., None], x0[:, None], gX[..., None] if gX is not None else None,
                            [gq, ge], [g, er], [a, lam, w0])
        return gi[0], gi[1], gp[0], gp[1], gp[2], None, None


def gwet_x(G, Er, a, lam, W0, x0, nsub=4):
    """G, Er [N, T]; a, lam, W0, x0 [N] -> Qs, e_W, W after each step [N, T]"""
    return GwEtX.apply(G, Er, a, lam, W0, x0, nsub)


# ----------------------------------------------------------------------------- production stores, Strang sub-steps
# sub-step: half of the percolation (exact for dS/dt = -Kp s^5: S (1 + kq s^4)^(-1/4), kq = 4 Kp dt / C for the whole step), wetting or
# drying by the net input of the sub-step (the exact GR4J / Pareto step), half of the percolation.
# PARETO = 0: GR4J partition (b, be unused); PARETO = 1: Pareto distribution of point capacities (pareto.py), NET = 0: p and e are
# used as given (depression storage).

@triton.jit
def _prod_fwd(p_ptr, e_ptr, c_ptr, k_ptr, b_ptr, be_ptr, x0_ptr, X_ptr, d_ptr, g_ptr, a_ptr, N, T, NSUB: tl.constexpr, PARETO: tl.constexpr,
              NET: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    Cc = tl.load(c_ptr + n, mask=m, other=1.0)
    kh = tl.load(k_ptr + n, mask=m, other=0.0) * (0.5 / NSUB)
    C4 = Cc * Cc * Cc * Cc
    b = tl.load(b_ptr + n, mask=m, other=1.0)
    be = tl.load(be_ptr + n, mask=m, other=2.0)
    cm = (b + 1.0) * Cc
    S = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        if NET:
            pn = tl.maximum(p - e, 0.0) / NSUB
            en = tl.maximum(e - p, 0.0) / NSUB
            ic = tl.minimum(p, e)
        else:
            pn = p / NSUB
            en = e / NSUB
            ic = p * 0.0
        tp = ld.tanh(pn / Cc)
        if NET:
            te = ld.tanh(en / Cc)
        else:
            te = ld.tanh(0.5 * en / Cc)                                                   # p and e act together: half of the drying on
                                                                                          # either side of the wetting
        D = tl.zeros_like(p)
        G = tl.zeros_like(p)
        A = tl.zeros_like(p)
        for j in tl.static_range(NSUB):
            g1 = _rel(S, kh / C4)
            S = S - g1
            if PARETO:
                s = tl.minimum(tl.maximum(S / Cc, 0.0), 1.0)
                d = tl.maximum(1.0 - s, 1e-7)
                z = tl.exp(tl.log(d) / (b + 1.0))
                rt = pn / (cm * z)                                                        # rise of the critical capacity relative to z
                fill = rt >= 1.0
                ps = Cc * d * tl.where(fill, 1.0, _em(-(b + 1.0) * ld.log1p(-tl.where(fill, 0.0, rt))))
                ps = tl.minimum(ps, pn)
                dbe = tl.exp(be * tl.log(d))
                es = tl.minimum(Cc * (1.0 - dbe) * te / (1.0 + dbe / d * te), S)
                if NET == 0:
                    S = S - es
                    s = tl.minimum(tl.maximum(S / Cc, 0.0), 1.0)
                    d = tl.maximum(1.0 - s, 1e-7)
                    z = tl.exp(tl.log(d) / (b + 1.0))
                    rt = pn / (cm * z)
                    fill = rt >= 1.0
                    ps = tl.minimum(Cc * d * tl.where(fill, 1.0, _em(-(b + 1.0) * ld.log1p(-tl.where(fill, 0.0, rt)))), pn)
                    S = S + ps
                    s = tl.minimum(tl.maximum(S / Cc, 0.0), 1.0)
                    d = tl.maximum(1.0 - s, 1e-7)
                    dbe = tl.exp(be * tl.log(d))
                    e2 = tl.minimum(Cc * (1.0 - dbe) * te / (1.0 + dbe / d * te), S)
                    S = S - e2
                    es = es + e2
                else:
                    S = S - es + ps
            else:
                s = S / Cc
                ps = Cc * (1.0 - s * s) * tp / (1.0 + s * tp)
                es = S * (2.0 - s) * te / (1.0 + (1.0 - s) * te)
                S = S - es + ps
            g2 = _rel(S, kh / C4)
            S = S - g2
            D += pn - ps
            G += g1 + g2
            A += es
        tl.store(X_ptr + n * T + t, S, mask=m)
        tl.store(d_ptr + n * T + t, D, mask=m)
        tl.store(g_ptr + n * T + t, G, mask=m)
        tl.store(a_ptr + n * T + t, A + ic, mask=m)


def prod_step(xp, p, e, Cc, kq, b, be, nsub=4, pareto=False, net=True):
    (S,) = xp
    if net:
        pn, en, ic = torch.relu(p - e) / nsub, torch.relu(e - p) / nsub, torch.minimum(p, e)
    else:
        pn, en, ic = p / nsub, e / nsub, torch.zeros_like(p)
    kh = kq * (0.5 / nsub) / Cc ** 4
    cm = (b + 1.0) * Cc
    tp, te = torch.tanh(pn / Cc), torch.tanh((en if net else 0.5 * en) / Cc)

    def wet(S_):
        s_ = (S_ / Cc).clamp(0.0, 1.0)
        d_ = (1.0 - s_).clamp(min=1e-7)
        rt = pn / (cm * torch.exp(torch.log(d_) / (b + 1.0)))
        fill = rt >= 1.0
        return torch.minimum(Cc * d_ * torch.where(fill, torch.ones_like(rt), em(-(b + 1.0) * torch.log1p(-torch.where(fill, torch.zeros_like(rt), rt)))), pn)

    def dry(S_):
        d_ = (1.0 - (S_ / Cc).clamp(0.0, 1.0)).clamp(min=1e-7)
        dbe = torch.exp(be * torch.log(d_))
        return torch.minimum(Cc * (1.0 - dbe) * te / (1.0 + dbe / d_ * te), S_)
    D, G, A = torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)
    for _ in range(nsub):
        g1 = rel(S, kh); S = S - g1
        if pareto and net:
            ps, es = wet(S), dry(S)
            S = S - es + ps
        elif pareto:
            es = dry(S); S = S - es
            ps = wet(S); S = S + ps
            e2 = dry(S); S = S - e2
            es = es + e2
        else:
            s = S / Cc
            ps = Cc * (1.0 - s * s) * tp / (1.0 + s * tp)
            es = S * (2.0 - s) * te / (1.0 + (1.0 - s) * te)
            S = S - es + ps
        g2 = rel(S, kh); S = S - g2
        D = D + pn - ps; G = G + g1 + g2; A = A + es
    return [S], [D, G, A + ic]


class ProdX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, e, Cc, kq, b, be, x0, nsub, pareto, net):
        N, T = p.shape
        v = _prep(p, e, Cc, kq, b, be, x0)
        X, D, G, A = (torch.empty(N, T, device=p.device) for _ in range(4))
        _prod_fwd[(triton.cdiv(N, BLOCK),)](*v, X, D, G, A, N, T, NSUB=nsub, PARETO=int(pareto), NET=int(net), BLOCK=BLOCK)
        ctx.opt = (nsub, pareto, net)
        ctx.save_for_backward(*v, X)
        return D, G, A, X

    @staticmethod
    def backward(ctx, gD, gG, gA, gX):
        p, e, Cc, kq, b, be, x0, X = ctx.saved_tensors
        nsub, pareto, net = ctx.opt
        gi, gp = backward_k(functools.partial(prod_step, nsub=nsub, pareto=pareto, net=net), X[..., None], x0[:, None],
                            gX[..., None] if gX is not None else None, [gD, gG, gA], [p, e], [Cc, kq, b, be])
        return gi[0], gi[1], gp[0], gp[1], gp[2], gp[3], None, None, None, None


def prod_x(p, e, Cc, kq, S0, nsub=4):
    """GR4J production store. p, e [N, T]; Cc, kq, S0 [N] -> D, G, AET, S after each step [N, T]"""
    one = torch.ones_like(Cc)
    return ProdX.apply(p, e, Cc, kq, one, 2.0 * one, S0, nsub, False, True)


def pareto_x(p, e, Cc, kq, b, S0, be=None, net=True, nsub=4):
    """Pareto production store (pareto.py). p, e [N, T]; Cc, kq, b, S0 [N]; be [N] or None (= 2) -> D, G, AET, S after each step [N, T]"""
    be = torch.full_like(b, 2.0) if be is None else be
    return ProdX.apply(p, e, Cc, kq, b, be, S0, nsub, True, net)


# ----------------------------------------------------------------------------- GR4J routing store with exchange, Strang sub-steps
# sub-step: half of the release (exact for dR/dt = -R^5 / (4 X3^4)), inflow x9 / NSUB and exchange F = X2 (g(R/X3) - X5) / NSUB with
# g(r) = r^3.5 at the state of the sub-step (CR F into the store, CD F into the direct branch), half of the release.
# Q = release + direct branch max(x1 + CD F, 0); xr = realised exchange.

@triton.jit
def _quick_chain(R, x9s, x1s, X2s, X3, X5, ch, CR: tl.constexpr, CD: tl.constexpr, NS: tl.constexpr):
    """NS sub-steps (x9s, x1s, X2s, ch already per sub-step) -> R, Q, realised exchange"""
    Q = tl.zeros_like(x9s)
    xr = tl.zeros_like(x9s)
    for j in tl.static_range(NS):
        q1 = _rel(R, ch)
        R = R - q1
        rr = R / X3
        F1 = X2s * (tl.where(rr > 0, tl.exp(3.5 * tl.log(tl.maximum(rr, 1e-30))), 0.0) - X5)
        rm = tl.maximum(R + 0.5 * (x9s + CR * F1), 0.0) / X3                              # state half-way through the inflow and exchange
        F = X2s * (tl.where(rm > 0, tl.exp(3.5 * tl.log(tl.maximum(rm, 1e-30))), 0.0) - X5)
        Rn = tl.maximum(R + x9s + CR * F, 0.0)
        Qd = tl.maximum(x1s + CD * F, 0.0)
        xr += (Rn - R - x9s) + (Qd - x1s)
        q2 = _rel(Rn, ch)
        R = Rn - q2
        Q += q1 + q2 + Qd
    return R, Q, xr


@triton.jit
def _quick_fwd(x9_ptr, x1_ptr, x2_ptr, x3_ptr, x5_ptr, x0_ptr, X_ptr, q_ptr, xr_ptr, N, T, DT: tl.constexpr, NSUB: tl.constexpr,
               CR: tl.constexpr, CD: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    X2 = tl.load(x2_ptr + n, mask=m, other=0.0)
    X3 = tl.load(x3_ptr + n, mask=m, other=1.0)
    X5 = tl.load(x5_ptr + n, mask=m, other=0.0)
    X32 = X3 * X3
    ch = 0.5 * DT / (X32 * X32)
    R = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        x9 = tl.load(x9_ptr + n * T + t, mask=m, other=0.0)
        x1 = tl.load(x1_ptr + n * T + t, mask=m, other=0.0)
        Ra, Qa, xa = _quick_chain(R, x9 / NSUB, x1 / NSUB, X2 / NSUB, X3, X5, ch / NSUB, CR, CD, NSUB)
        Rb, Qb, xb = _quick_chain(R, x9 * (2.0 / NSUB), x1 * (2.0 / NSUB), X2 * (2.0 / NSUB), X3, X5, ch * (2.0 / NSUB), CR, CD, NSUB // 2)
        R = (4.0 * Ra - Rb) / 3.0                                                         # Richardson extrapolation of the two solutions
        Q = (4.0 * Qa - Qb) / 3.0
        xr = (4.0 * xa - xb) / 3.0
        neg = tl.maximum(-R, 0.0)                                                         # (an extrapolated storage below zero is a loss
        R = R + neg                                                                       # the store could not supply)
        xr = xr + neg
        tl.store(X_ptr + n * T + t, R, mask=m)
        tl.store(q_ptr + n * T + t, Q, mask=m)
        tl.store(xr_ptr + n * T + t, xr, mask=m)


def quick_chain(R, x9, x1, X2, X3, X5, ch, cr, cd, ns):
    Q, xr = torch.zeros_like(x9), torch.zeros_like(x9)
    for _ in range(ns):
        q1 = rel(R, ch); R = R - q1
        F1 = X2 * ((R / X3).clamp(min=0.0) ** 3.5 - X5)
        F = X2 * ((torch.relu(R + 0.5 * (x9 + cr * F1)) / X3) ** 3.5 - X5)
        Rn = torch.relu(R + x9 + cr * F)
        Qd = torch.relu(x1 + cd * F)
        xr = xr + (Rn - R - x9) + (Qd - x1)
        q2 = rel(Rn, ch); R = Rn - q2
        Q = Q + q1 + q2 + Qd
    return R, Q, xr


def quick_step(xp, x9, x1, X2, X3, X5, dt=1.0, nsub=4, cr=0.9, cd=0.1, rich=True):
    (R,) = xp
    ch = 0.5 * dt / X3 ** 4
    Ra, Qa, xa = quick_chain(R, x9 / nsub, x1 / nsub, X2 / nsub, X3, X5, ch / nsub, cr, cd, nsub)
    if not rich:
        return [Ra], [Qa, xa]
    Rb, Qb, xb = quick_chain(R, x9 * (2.0 / nsub), x1 * (2.0 / nsub), X2 * (2.0 / nsub), X3, X5, ch * (2.0 / nsub), cr, cd, nsub // 2)
    Rn, Q, xr = (4.0 * Ra - Rb) / 3.0, (4.0 * Qa - Qb) / 3.0, (4.0 * xa - xb) / 3.0
    neg = torch.relu(-Rn)
    return [Rn + neg], [Q, xr + neg]


class QuickX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x9, x1, X2, X3, X5, x0, dt, nsub, cr, cd):
        N, T = x9.shape
        v = _prep(x9, x1, X2, X3, X5, x0)
        X, Q, Xr = (torch.empty(N, T, device=x9.device) for _ in range(3))
        _quick_fwd[(triton.cdiv(N, BLOCK),)](*v, X, Q, Xr, N, T, float(dt), NSUB=nsub, CR=float(cr), CD=float(cd), BLOCK=BLOCK)
        ctx.opt = (float(dt), nsub, float(cr), float(cd))
        ctx.save_for_backward(*v, X)
        return Q, Xr, X

    @staticmethod
    def backward(ctx, gq, gxr, gX):
        x9, x1, X2, X3, X5, x0, X = ctx.saved_tensors
        dt, nsub, cr, cd = ctx.opt
        if not GENERIC:
            from .exact_bwd import quick_bwd
            g9, g1, g2, g3, g5 = quick_bwd(x9, x1, X2, X3, X5, x0, X, gq, gxr, gX, dt, nsub, cr, cd)
            return g9, g1, g2, g3, g5, None, None, None, None, None
        gi, gp = backward_k(functools.partial(quick_step, dt=dt, nsub=nsub, cr=cr, cd=cd), X[..., None], x0[:, None],
                            gX[..., None] if gX is not None else None, [gq, gxr], [x9, x1], [X2, X3, X5])
        return gi[0], gi[1], gp[0], gp[1], gp[2], None, None, None, None, None


def quick_x(x9, x1, X2, X3, R0, X5=None, split=(1.0, 1.0), dt=1.0, nsub=4):
    """x9, x1 [N, T]; X2 (per step), X3, R0 [N]; X5 [N] exchange threshold (None = 0) -> Q, realised exchange, R after each step [N, T]"""
    X5 = torch.zeros_like(X3) if X5 is None else X5
    return QuickX.apply(x9, x1, X2, X3, X5, R0, dt, nsub, split[0], split[1])


# ----------------------------------------------------------------------------- quick store of the derived release law (lin_store, mode 0)
# release over part of a step with the coefficient frozen at the state: A = p1 exp(6 tanh((c0 + c1 ln(R / X3)) / 6)), R -> R (1 + 4 A R^4)^(-1/4);
# sub-step: half of the release, inflow u1 / NSUB and exchange j = p2 (R / 30)^3.5 / NSUB (0.9 j into the store, 0.1 j into the direct
# branch), half of the release. c0, c1 [N, T] are given series.

@triton.jit
def _lrel(R, p1h, sc, c0, c1):
    f = tl.log(tl.maximum(R, 1e-3) / sc)
    th = ld.tanh((c0 + c1 * f) / 6.0)
    A = p1h * tl.exp(6.0 * th)
    nu = 4.0 + c1 * (1.0 - th * th)                                                       # local exponent of the release law minus one
    R2 = R * R
    return R * _em(ld.log1p(nu * A * R2 * R2) / nu)


@triton.jit
def _lin_chain(R, u1s, u2s, p1h, p2s, sc, c0, c1, NS: tl.constexpr):
    Q = tl.zeros_like(u1s)
    xr = tl.zeros_like(u1s)
    for j in tl.static_range(NS):
        q1 = _lrel(R, p1h, sc, c0, c1)
        R = R - q1
        Sc = tl.maximum(R, 0.0) / 30.0
        e1 = p2s * Sc * Sc * Sc * tl.sqrt(Sc)
        Sm = tl.maximum(R + 0.5 * (u1s + 0.9 * e1), 0.0) / 30.0                          # state half-way through the inflow and exchange
        ex = p2s * Sm * Sm * Sm * tl.sqrt(Sm)
        Rm = tl.maximum(R + u1s + 0.9 * ex, 0.0)
        Qd = tl.maximum(u2s + 0.1 * ex, 0.0)
        xr += (Rm - R - u1s) + (Qd - u2s)
        q2 = _lrel(Rm, p1h, sc, c0, c1)
        R = Rm - q2
        Q += q1 + q2 + Qd
    return R, Q, xr


@triton.jit
def _lin_fwd(u1_ptr, u2_ptr, c0_ptr, c1_ptr, p1_ptr, p2_ptr, sc_ptr, x0_ptr, X_ptr, q_ptr, xr_ptr, N, T, NSUB: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    p1h = tl.load(p1_ptr + n, mask=m, other=0.0) * 0.5
    p2 = tl.load(p2_ptr + n, mask=m, other=0.0)
    sc = tl.load(sc_ptr + n, mask=m, other=1.0)
    R = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        u1 = tl.load(u1_ptr + n * T + t, mask=m, other=0.0)
        u2 = tl.load(u2_ptr + n * T + t, mask=m, other=0.0)
        c0 = tl.load(c0_ptr + n * T + t, mask=m, other=0.0)
        c1 = tl.load(c1_ptr + n * T + t, mask=m, other=0.0)
        Ra, Qa, xa = _lin_chain(R, u1 / NSUB, u2 / NSUB, p1h / NSUB, p2 / NSUB, sc, c0, c1, NSUB)
        Rb, Qb, xb = _lin_chain(R, u1 * (2.0 / NSUB), u2 * (2.0 / NSUB), p1h * (2.0 / NSUB), p2 * (2.0 / NSUB), sc, c0, c1, NSUB // 2)
        R = (4.0 * Ra - Rb) / 3.0
        Q = (4.0 * Qa - Qb) / 3.0
        xr = (4.0 * xa - xb) / 3.0
        neg = tl.maximum(-R, 0.0)
        R = R + neg
        xr = xr + neg
        tl.store(X_ptr + n * T + t, R, mask=m)
        tl.store(q_ptr + n * T + t, Q, mask=m)
        tl.store(xr_ptr + n * T + t, xr, mask=m)


def lrel(R, p1h, sc, c0, c1):
    f = torch.log(R.clamp(min=1e-3) / sc)
    th = torch.tanh((c0 + c1 * f) / 6.0)
    A = p1h * torch.exp(6.0 * th)
    nu = 4.0 + c1 * (1.0 - th * th)
    R2 = R * R
    return R * em(torch.log1p(nu * A * R2 * R2) / nu)


def lin_chain(R, u1, u2, p1h, p2, sc, c0, c1, ns):
    Q, xr = torch.zeros_like(u1), torch.zeros_like(u1)
    for _ in range(ns):
        q1 = lrel(R, p1h, sc, c0, c1); R = R - q1
        e1 = p2 * (R.clamp(min=0.0) / 30.0) ** 3.5
        ex = p2 * (torch.relu(R + 0.5 * (u1 + 0.9 * e1)) / 30.0) ** 3.5
        Rm = torch.relu(R + u1 + 0.9 * ex)
        Qd = torch.relu(u2 + 0.1 * ex)
        xr = xr + (Rm - R - u1) + (Qd - u2)
        q2 = lrel(Rm, p1h, sc, c0, c1); R = Rm - q2
        Q = Q + q1 + q2 + Qd
    return R, Q, xr


def lin_step(xp, u1, u2, c0, c1, p1, p2, sc, nsub=4, rich=True):
    (R,) = xp
    p1h = 0.5 * p1
    Ra, Qa, xa = lin_chain(R, u1 / nsub, u2 / nsub, p1h / nsub, p2 / nsub, sc, c0, c1, nsub)
    if not rich:
        return [Ra], [Qa, xa]
    Rb, Qb, xb = lin_chain(R, u1 * (2.0 / nsub), u2 * (2.0 / nsub), p1h * (2.0 / nsub), p2 * (2.0 / nsub), sc, c0, c1, nsub // 2)
    Rn, Q, xr = (4.0 * Ra - Rb) / 3.0, (4.0 * Qa - Qb) / 3.0, (4.0 * xa - xb) / 3.0
    neg = torch.relu(-Rn)
    return [Rn + neg], [Q, xr + neg]


class LinX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u1, u2, c0, c1, p1, p2, sc, x0, nsub):
        N, T = u1.shape
        v = _prep(u1, u2, c0.expand(N, T), c1.expand(N, T), p1, p2, sc, x0)
        X, Q, Xr = (torch.empty(N, T, device=u1.device) for _ in range(3))
        _lin_fwd[(triton.cdiv(N, BLOCK),)](*v, X, Q, Xr, N, T, NSUB=nsub, BLOCK=BLOCK)
        ctx.nsub = nsub
        ctx.save_for_backward(*v, X)
        return Q, Xr, X

    @staticmethod
    def backward(ctx, gq, gxr, gX):
        u1, u2, c0, c1, p1, p2, sc, x0, X = ctx.saved_tensors
        if not GENERIC:
            from .exact_bwd import lin_bwd
            gu1, gu2, gc0, gc1, gp1, gp2, gsc = lin_bwd(u1, u2, c0, c1, p1, p2, sc, x0, X, gq, gxr, gX, ctx.nsub)
            return gu1, gu2, gc0, gc1, gp1, gp2, gsc, None, None
        gi, gp = backward_k(functools.partial(lin_step, nsub=ctx.nsub), X[..., None], x0[:, None], gX[..., None] if gX is not None else None,
                            [gq, gxr], [u1, u2, c0, c1], [p1, p2, sc])
        return gi[0], gi[1], gi[2], gi[3], gp[0], gp[1], gp[2], None, None


def lin_x(u1, u2, p1, p2, sc, R0, c0, c1, nsub=4):
    """u1 = x9, u2 = x1, c0, c1 [N, T]; p1 = a_f (per step), p2 = Xj (per step), sc = X3, R0 [N] -> Q, realised exchange, R after each step"""
    return LinX.apply(u1, u2, c0, c1, p1, p2, sc, R0, nsub)
