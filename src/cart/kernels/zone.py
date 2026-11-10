"""Near-stream saturated zone Z as a fused Triton recursion with stored-state adjoint (adjoint.backward_k).

Per unit zone area, capacity Cz, z = Z / Cz, drainage time tz, GR4J net input Pn = max(P - E, 0), En = max(E - P, 0):
    dZ/dt = Pn (1 - z^2) - En z (2 - z) - Z / tz
    saturation runoff d_z = Pn z^2,  evaporation e_z = En z (2 - z),  drainage q_z = Z / tz
One day is integrated EXACTLY (input and drainage jointly), in a cancellation-free form (a = Pn/Cz, b = 1/tz, e = En/Cz):
  rain or no-forcing day (En = 0):  Riccati.  D = sqrt(b^2 + 4 a^2), z+ = 2a / (b + D), rho0 = 2a (z0 - z+) / (2a z0 + b + D),
        z1 = z+ + (z0 - z+)(1 - rho0) e^-D / (1 - rho0 e^-D)
        int_0^1 z dt = z+ + 2 (z0 - z+)(1 - e^-D) / [(2a z0 + b + D)(1 - rho0)] * psi(y),  y = rho0 (1 - e^-D) / (1 - rho0),  psi(y) = log(1+y)/y
  evaporation day (En > 0):  Bernoulli.  al = 2e + b, m = 1 - e^-al, x = z0 (e/al) m,
        z1 = z0 e^-al / (1 - x),   int_0^1 z dt = z0 (m / al) phi(x),  phi(x) = -log(1-x)/x
  q_z = Cz b int z dt;  rain day: d_z = Pn - Cz (z1 - z0) - q_z;  evaporation day: d_z = 0, e_z = Cz (z0 - z1) - q_z.
(a = 0 in the Riccati form reduces to pure drainage z0 e^-b, as does e = 0 in the Bernoulli form.)
Outputs per day: d_z, q_z (mm/d per unit zone area); the caller weights them by the zone fraction w.
"""

import functools

import numpy as np
import torch
import triton
import triton.language as tl

from .adjoint import backward_k

BLOCK = 32


@triton.jit
def _expm1n(x):
    """1 - exp(-x) for x >= 0, accurate for small x"""
    return tl.where(x < 1e-3, x - 0.5 * x * x + x * x * x / 6.0, 1.0 - tl.exp(-x))


@triton.jit
def _psi(y):
    """log(1 + y) / y, y > -1"""
    ys = tl.where(tl.abs(y) < 1e-3, 1.0, y)
    return tl.where(tl.abs(y) < 1e-3, 1.0 - 0.5 * y + y * y / 3.0, tl.log(1.0 + ys) / ys)


@triton.jit
def _phi(x):
    """-log(1 - x) / x, x < 1"""
    xs = tl.where(tl.abs(x) < 1e-3, 0.5, x)
    return tl.where(tl.abs(x) < 1e-3, 1.0 + 0.5 * x + x * x / 3.0, -tl.log(1.0 - xs) / xs)


@triton.jit
def _zone_fwd(p_ptr, e_ptr, cz_ptr, tz_ptr, x0_ptr, X_ptr, dz_ptr, qz_ptr, N, T, DT, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    cz = tl.load(cz_ptr + n, mask=m, other=1.0)
    b = DT / tl.load(tz_ptr + n, mask=m, other=1.0)                                     # drainage over a step of DT days
    Z = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        ev = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        pn = tl.maximum(p - ev, 0.0)
        en = tl.maximum(ev - p, 0.0)
        z0 = Z / cz
        a = pn / cz
        # rain / no-forcing day
        D = tl.sqrt(b * b + 4.0 * a * a)
        zp = 2.0 * a / (b + D)
        den = 2.0 * a * z0 + b + D
        r0 = 2.0 * a * (z0 - zp) / den
        mD = _expm1n(D)
        eD = 1.0 - mD
        r1 = r0 * eD
        z1r = zp + (z0 - zp) * (1.0 - r0) * eD / (1.0 - r1)
        y = r0 * mD / (1.0 - r0)
        Izr = zp + 2.0 * (z0 - zp) * mD / (den * (1.0 - r0)) * _psi(y)
        # evaporation day
        e = en / cz
        al = 2.0 * e + b
        ma = _expm1n(al)
        x = z0 * (e / al) * ma
        z1e = z0 * (1.0 - ma) / (1.0 - x)
        Ize = z0 * (ma / al) * _phi(x)
        dry = en > 0.0
        z1 = tl.where(dry, z1e, z1r)
        qz = cz * b * tl.where(dry, Ize, Izr)
        dz = tl.where(dry, 0.0, pn - cz * (z1 - z0) - qz)
        Z = z1 * cz
        tl.store(X_ptr + n * T + t, Z, mask=m)
        tl.store(dz_ptr + n * T + t, dz, mask=m)
        tl.store(qz_ptr + n * T + t, qz, mask=m)


def _t_expm1n(x):
    return torch.where(x < 1e-3, x - 0.5 * x * x + x * x * x / 6.0, 1.0 - torch.exp(-x))


def _t_psi(y):
    small = y.abs() < 1e-3
    ys = torch.where(small, torch.ones_like(y), y)
    return torch.where(small, 1.0 - 0.5 * y + y * y / 3.0, torch.log(1.0 + ys) / ys)


def _t_phi(x):
    small = x.abs() < 1e-3
    xs = torch.where(small, torch.full_like(x, 0.5), x)
    return torch.where(small, 1.0 + 0.5 * x + x * x / 3.0, -torch.log(1.0 - xs) / xs)


def zone_step(xp, p, ev, cz, tz, dt=1.0):
    """the kernel's one-step update, elementwise torch (for adjoint.backward_k and as a float64 reference)"""
    (Z,) = xp
    b = dt / tz
    pn = torch.clamp(p - ev, min=0.0)
    en = torch.clamp(ev - p, min=0.0)
    z0 = Z / cz
    a = pn / cz
    D = torch.sqrt(b * b + 4.0 * a * a)
    zp = 2.0 * a / (b + D)
    den = 2.0 * a * z0 + b + D
    r0 = 2.0 * a * (z0 - zp) / den
    mD = _t_expm1n(D)
    eD = 1.0 - mD
    r1 = r0 * eD
    z1r = zp + (z0 - zp) * (1.0 - r0) * eD / (1.0 - r1)
    y = r0 * mD / (1.0 - r0)
    Izr = zp + 2.0 * (z0 - zp) * mD / (den * (1.0 - r0)) * _t_psi(y)
    e = en / cz
    al = 2.0 * e + b
    ma = _t_expm1n(al)
    x = z0 * (e / al) * ma
    z1e = z0 * (1.0 - ma) / (1.0 - x)
    Ize = z0 * (ma / al) * _t_phi(x)
    dry = en > 0.0
    z1 = torch.where(dry, z1e, z1r)
    qz = cz * b * torch.where(dry, Ize, Izr)
    dz = torch.where(dry, torch.zeros_like(pn), pn - cz * (z1 - z0) - qz)
    return [z1 * cz], [dz, qz]


class ZoneStore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, ev, x0, cz, tz, dt=1.0):
        N, T = p.shape
        a = [v.detach().float().contiguous() for v in (p, ev, cz, tz, x0)]
        X = torch.empty(N, T, device=p.device); dz = torch.empty(N, T, device=p.device); qz = torch.empty(N, T, device=p.device)
        _zone_fwd[(triton.cdiv(N, BLOCK),)](*a, X, dz, qz, N, T, float(dt), BLOCK=BLOCK)
        ctx.dt = float(dt)
        ctx.save_for_backward(a[0], a[1], a[4], X, a[2], a[3])
        return dz, qz, X

    @staticmethod
    def backward(ctx, gdz, gqz, gX):
        p, ev, x0, X, cz, tz = ctx.saved_tensors
        step = functools.partial(zone_step, dt=ctx.dt)
        gi, gp = backward_k(step, X[..., None], x0[:, None], gX[..., None] if gX is not None else None, [gdz, gqz], [p, ev], [cz, tz])
        return (gi[0], gi[1], None, *gp, None)


def zone(p, ev, cz, tz, x0=None, dt=1.0):
    """p, ev [N, T] precipitation and PET (mm per step); cz [N], tz [N] (days), dt step (days) -> d_z, q_z [N, T] per step, Z [N, T]"""
    x0 = 0.3 * cz.detach() if x0 is None else x0
    return ZoneStore.apply(p, ev, x0, cz, tz, dt)


def zone_np(P, E, cz, tz, z0frac=0.3):
    """independent float64 reference: the closed forms written directly (agrees with an RK4 integration to 1e-14). P, E [N, T]"""
    N, T = P.shape
    Pn, En = np.maximum(P - E, 0), np.maximum(E - P, 0)
    Z = z0frac * cz.astype(np.float64); b = 1.0 / tz; dz = np.zeros((N, T)); qz = np.zeros((N, T)); ez = np.zeros((N, T)); Zs = np.zeros((N, T))
    for t in range(T):
        z0 = Z / cz; pn, en = Pn[:, t], En[:, t]; rain = en <= 0
        a = np.where(pn > 0, pn / cz, 1e-300); D = np.sqrt(b * b + 4 * a * a); zp = (-b + D) / (2 * a); zm = (-b - D) / (2 * a)
        with np.errstate(all="ignore"):
            r0 = (z0 - zp) / (z0 - zm); r1 = r0 * np.exp(-D); z1r = (zp - zm * r1) / (1 - r1)
            Izr = zm + (zp - zm) * (1 + np.log((1 - r1) / (1 - r0)) / D)
        Izr = np.where(pn > 0, Izr, z0 * (1 - np.exp(-b)) / b); z1r = np.where(pn > 0, z1r, z0 * np.exp(-b))
        e = en / cz; al = 2 * e + b; A = e / al
        with np.errstate(all="ignore"):
            B = 1 / np.maximum(z0, 1e-300) - A; z1e = np.where(z0 > 0, 1 / (A + B * np.exp(al)), 0.0)
            Ize = np.where(A > 0, (al - np.log(A + B * np.exp(al)) + np.log(A + B)) / (al * A), z0 * (1 - np.exp(-al)) / al)
        Ize = np.where(z0 > 0, Ize, 0.0)
        z1 = np.where(rain, z1r, z1e); q = cz * b * np.where(rain, Izr, Ize)
        dz[:, t] = np.where(rain, pn - cz * (z1 - z0) - q, 0.0); ez[:, t] = np.where(rain, 0.0, cz * (z0 - z1) - q); qz[:, t] = q
        Z = z1 * cz; Zs[:, t] = Z
    return dz, qz, ez, Zs
