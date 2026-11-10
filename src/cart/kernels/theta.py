"""Connectivity wetness state theta as a fused Triton recursion with a HAND-WRITTEN adjoint (reverse sweep of the per-day closed form,
intermediates recomputed from the stored state), in place of the generic stored-state adjoint of zone.py (adjoint.backward_k) when only
the state is needed.

Per unit area, capacity C, z = Z/C, drainage time tz (b = 1/tz), Pn = max(P - E, 0), En = max(E - P, 0):
    dZ/dt = Pn (1 - z^2) - En z (2 - z) - Z/tz
exact daily solution in the cancellation-free form of zone.py (rain or no-forcing day: Riccati; evaporation day: Bernoulli).
Output: Z after each day [N, T] (the caller uses theta = Z / C). Gradients: inputs P, E [N, T], parameters C, tz [N].
The forward performs the same operations as zone.zone()[2].
"""

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 32


@triton.jit
def _em(x):
    """1 - exp(-x), accurate for small x"""
    return tl.where(x < 1e-3, x - 0.5 * x * x + x * x * x / 6.0, 1.0 - tl.exp(-x))


@triton.jit
def _day(Z0, p, ev, cz, b):
    """one day: returns Z1 (the operations of zone._zone_fwd for the state)"""
    pn = tl.maximum(p - ev, 0.0)
    en = tl.maximum(ev - p, 0.0)
    z0 = Z0 / cz
    a = pn / cz
    D = tl.sqrt(b * b + 4.0 * a * a)
    zp = 2.0 * a / (b + D)
    den = 2.0 * a * z0 + b + D
    r0 = 2.0 * a * (z0 - zp) / den
    eD = 1.0 - _em(D)
    r1 = r0 * eD
    z1r = zp + (z0 - zp) * (1.0 - r0) * eD / (1.0 - r1)
    e = en / cz
    al = 2.0 * e + b
    ma = _em(al)
    x = z0 * (e / al) * ma
    z1e = z0 * (1.0 - ma) / (1.0 - x)
    return tl.where(en > 0.0, z1e, z1r) * cz


@triton.jit
def _theta_fwd(p_ptr, e_ptr, cz_ptr, tz_ptr, x0_ptr, X_ptr, N, T, DT, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    cz = tl.load(cz_ptr + n, mask=m, other=1.0)
    b = DT / tl.load(tz_ptr + n, mask=m, other=1.0)
    Z = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        ev = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        Z = _day(Z, p, ev, cz, b)
        tl.store(X_ptr + n * T + t, Z, mask=m)


@triton.jit
def _theta_bwd(p_ptr, e_ptr, cz_ptr, tz_ptr, x0_ptr, X_ptr, gX_ptr, gp_ptr, ge_ptr, gcz_ptr, gtz_ptr, N, T, DT, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    cz = tl.load(cz_ptr + n, mask=m, other=1.0)
    tz = tl.load(tz_ptr + n, mask=m, other=1.0)
    b = DT / tz
    lam = tl.zeros((BLOCK,), dtype=tl.float32)                   # adjoint of the state after day t (carried backwards)
    gcz = tl.zeros((BLOCK,), dtype=tl.float32)
    gb = tl.zeros((BLOCK,), dtype=tl.float32)
    for tt in range(T):
        t = T - 1 - tt
        gZ1 = lam + tl.load(gX_ptr + n * T + t, mask=m, other=0.0)
        Z0 = tl.where(t > 0, tl.load(X_ptr + n * T + t - 1, mask=m & (t > 0), other=0.0), tl.load(x0_ptr + n, mask=m, other=0.0))
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        ev = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        # ---- recompute the forward of day t
        pn = tl.maximum(p - ev, 0.0)
        en = tl.maximum(ev - p, 0.0)
        z0 = Z0 / cz
        a = pn / cz
        D = tl.sqrt(b * b + 4.0 * a * a)
        zp = 2.0 * a / (b + D)
        den = 2.0 * a * z0 + b + D
        u = z0 - zp
        r0 = 2.0 * a * u / den
        eD = 1.0 - _em(D)
        r1 = r0 * eD
        s = 1.0 / (1.0 - r1)
        v = 1.0 - r0
        z1r = zp + u * v * eD * s
        e = en / cz
        al = 2.0 * e + b
        ma = _em(al)
        rr = e / al
        x = z0 * rr * ma
        z1e = z0 * (1.0 - ma) / (1.0 - x)
        dry = en > 0.0
        z1 = tl.where(dry, z1e, z1r)
        # ---- reverse sweep
        gcz += gZ1 * z1
        gz1 = gZ1 * cz
        gz1e = tl.where(dry, gz1, 0.0)
        gz1r = tl.where(dry, 0.0, gz1)
        # evaporation branch
        gz0 = gz1e * (1.0 - ma) / (1.0 - x)
        gma = -gz1e * z0 / (1.0 - x)
        gx = gz1e * z1e / (1.0 - x)
        gz0 += gx * rr * ma
        grr = gx * z0 * ma
        gma += gx * z0 * rr
        ge_ = grr / al
        gal = -grr * e / (al * al) + gma * tl.exp(-al)
        ge_ += 2.0 * gal
        gb += gal
        gen = ge_ / cz
        gcz += -ge_ * en / (cz * cz)
        # rain branch
        gzp = gz1r
        gu = gz1r * v * eD * s
        gv = gz1r * u * eD * s
        gw = gz1r * u * v * s
        gr1 = gz1r * u * v * eD * s * s
        gr0 = -gv + gr1 * eD
        gw += gr1 * r0
        gD = -gw * tl.exp(-D)
        gnum = gr0 / den
        gden = -gr0 * r0 / den
        ga = gnum * 2.0 * u
        gu += gnum * 2.0 * a
        gz0 += gu
        gzp += -gu
        ga += gden * 2.0 * z0
        gz0 += gden * 2.0 * a
        gb += gden
        gD += gden
        bD = b + D
        ga += gzp * 2.0 / bD
        gbD = -gzp * 2.0 * a / (bD * bD)
        gb += gbD
        gD += gbD
        Dsafe = tl.maximum(D, 1e-30)
        gb += gD * b / Dsafe
        ga += gD * 4.0 * a / Dsafe
        gpn = ga / cz
        gcz += -ga * pn / (cz * cz)
        # state and inputs
        lam = gz0 / cz
        gcz += -gz0 * Z0 / (cz * cz)
        mp = (p - ev) >= 0.0                                       # torch.clamp(min=0) passes the gradient at the kink
        me = (ev - p) >= 0.0
        gp = tl.where(mp, gpn, 0.0) - tl.where(me, gen, 0.0)
        gev = -tl.where(mp, gpn, 0.0) + tl.where(me, gen, 0.0)
        tl.store(gp_ptr + n * T + t, gp, mask=m)
        tl.store(ge_ptr + n * T + t, gev, mask=m)
    tl.store(gcz_ptr + n, gcz, mask=m)
    tl.store(gtz_ptr + n, -gb * DT / (tz * tz), mask=m)


class ThetaStore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, ev, x0, cz, tz, dt=1.0):
        N, T = p.shape
        a = [v.detach().float().contiguous() for v in (p, ev, cz, tz, x0)]
        X = torch.empty(N, T, device=p.device)
        _theta_fwd[(triton.cdiv(N, BLOCK),)](*a, X, N, T, float(dt), BLOCK=BLOCK)
        ctx.dt = float(dt)
        ctx.save_for_backward(*a, X)
        return X

    @staticmethod
    def backward(ctx, gX):
        p, ev, cz, tz, x0, X = ctx.saved_tensors
        N, T = p.shape
        gp, ge = torch.empty_like(p), torch.empty_like(p)
        gcz, gtz = torch.empty_like(cz), torch.empty_like(tz)
        _theta_bwd[(triton.cdiv(N, BLOCK),)](p, ev, cz, tz, x0, X, gX.contiguous().float(), gp, ge, gcz, gtz, N, T, ctx.dt, BLOCK=BLOCK)
        return gp, ge, None, gcz, gtz, None


def theta_state(p, ev, cz, tz, x0=None, dt=1.0):
    """p, ev [N, T] liquid input and demand (mm per step); cz [N], tz [N] (days), dt (days) -> Z [N, T] after each step"""
    x0 = 0.3 * cz.detach() if x0 is None else x0
    return ThetaStore.apply(p, ev, x0, cz, tz, dt)
