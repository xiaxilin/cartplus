"""GR4J unit hydrographs (Perrin et al. 2003) and their causal convolution. The model uses ordinates_rate (input at a uniform rate within
the step); ordinates (input at the start of the step, the GR4J and main-branch convention) is kept for reference and the tests."""
import torch

import numpy as np

X4_MAX = 6.0                                                                                  # upper bound of X4 (d): buffer lengths


def _sc1(t, x4):
    """S-curve of UH1: 0 for t <= 0, (t/X4)^2.5 for 0 < t < X4, 1 afterwards."""
    tt = t.clamp(min=0)
    return torch.where(t <= 0, torch.zeros_like(tt / x4), torch.where(t < x4, (tt / x4) ** 2.5, torch.ones_like(tt / x4)))


def _sc2(t, x4):
    """S-curve of UH2: 0.5 (t/X4)^2.5 for t <= X4, 1 - 0.5 (2 - t/X4)^2.5 for X4 < t < 2 X4, 1 afterwards."""
    tt = t.clamp(min=0)
    a = 0.5 * (tt / x4) ** 2.5
    b = 1 - 0.5 * (2 - tt / x4).clamp(min=0) ** 2.5
    return torch.where(t <= 0, torch.zeros_like(a), torch.where(t <= x4, a, torch.where(t < 2 * x4, b, torch.ones_like(a))))


def ordinates(x4, dt=1.0):
    """x4 [N] (days) -> UH1 ordinates [N, L1], UH2 ordinates [N, L2] for steps of dt days, o_j = S(j dt) - S((j-1) dt);
    L1 = ceil(X4_MAX/dt) + 1 (7 for the daily model, whose seventh ordinate is zero), L2 = 2 L1."""
    L1 = int(np.ceil(X4_MAX / dt)) + 1; L2 = 2 * L1
    x4 = x4[..., None]
    j1 = torch.arange(1, L1 + 1, dtype=x4.dtype, device=x4.device) * dt; j2 = torch.arange(1, L2 + 1, dtype=x4.dtype, device=x4.device) * dt
    return _sc1(j1, x4) - _sc1(j1 - dt, x4), _sc2(j2, x4) - _sc2(j2 - dt, x4)


def uh_conv(x, o):
    """Q(t) = sum_i o_i x(t - i) as one depthwise causal convolution: x [..., T], o [..., L]."""
    shp, L = x.shape, o.shape[-1]
    xi = x.reshape(1, -1, shp[-1])
    y = torch.nn.functional.conv1d(torch.nn.functional.pad(xi, (L - 1, 0)), o.reshape(-1, 1, L).flip(-1).to(x.dtype), groups=xi.shape[1])
    return y.reshape(shp)


def _T1(x, x4):
    """integral of the UH1 S-curve from 0 to x (x >= 0): X4/3.5 (x/X4)^3.5 below X4, X4/3.5 + x - X4 above"""
    xc = x.clamp(min=0)
    return torch.where(xc <= x4, x4 / 3.5 * (xc / x4).clamp(max=1.0) ** 3.5, x4 / 3.5 + xc - x4)


def _T2(x, x4):
    """integral of the UH2 S-curve from 0 to x (x >= 0); equals x - X4 beyond 2 X4"""
    xc = x.clamp(min=0); r = xc / x4
    a = 0.5 * x4 / 3.5 * r.clamp(max=1.0) ** 3.5
    b = 0.5 * x4 / 3.5 + (xc - x4) + 0.5 * x4 / 3.5 * ((2 - r).clamp(min=0.0, max=1.0) ** 3.5 - 1)
    return torch.where(r <= 1, a, torch.where(r < 2, b, xc - x4))


def ordinates_rate(x4, dt=1.0):
    """ordinates for input at a uniform rate within each step and outflow as the step mean:
    o_m = [T((m+1) dt) - 2 T(m dt) + T((m-1) dt)] / dt, m = 0, 1, ..., with T the integral of the S-curve and T(<0) = 0.
    o_0 = (1/dt) int_0^dt S is the same-step share; the ordinates sum to 1. L1 = ceil(X4_MAX/dt) + 2, L2 = ceil(2 X4_MAX/dt) + 2."""
    L1 = int(np.ceil(X4_MAX / dt)) + 2; L2 = int(np.ceil(2 * X4_MAX / dt)) + 2
    x4 = x4[..., None]
    def o(Tf, L):
        j = torch.arange(-1, L + 1, dtype=x4.dtype, device=x4.device) * dt                  # (m - 1) dt for m = 0 .. L
        Tv = torch.where(j < 0, torch.zeros_like(j + x4), Tf(j, x4))                           # [N, L + 2]
        return (Tv[..., 2:] - 2 * Tv[..., 1:-1] + Tv[..., :-2]) / dt
    return o(_T1, L1), o(_T2, L2)
