"""Five-band snowpack with sublimation, an ADDITIVE radiation-index melt term and an orographic precipitation gradient. Per band k and day: precipitation P_k = f_k P with f_k = exp(gP dz_k) / sum_j a_j exp(gP dz_j) (area-weighted mean exactly P;
gP per catchment, dz_k band elevation minus the mean, km); snowfall SFCF f_k P below TT; sublimation u = min(ks E, N);
melt on days with T_k > TT:  M = min(CFMAX (T_k - TT) + rf I, N)  with I the potential (extraterrestrial) radiation index of the day and
latitude. Radiation, not air temperature, dominates spring melt energy, so the additive term lowers the temperature sensitivity of the
degree-day store. rf = 0 and f_k = 1 reproduce snow_sub.snow_bands_sub exactly. Fused Triton recursion with stored-state adjoint.
CaRT+ calls it with f_k = 1 and the PET series as the index I (PET-index melt rf PET)."""
import numpy as np
import torch
import triton
import triton.language as tl
from .adjoint import backward_k

BLOCK = 32


@triton.jit
def _bra_fwd(p_ptr, t_ptr, e_ptr, r_ptr, dT_ptr, a_ptr, fk_ptr, tt_ptr, cf_ptr, sf_ptr, ks_ptr, rf_ptr, x0_ptr, X_ptr, q_ptr, u_ptr,
             N, T, K: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    tt = tl.load(tt_ptr + n, mask=m, other=0.0); cf = tl.load(cf_ptr + n, mask=m, other=0.0); sf = tl.load(sf_ptr + n, mask=m, other=1.0)
    ks = tl.load(ks_ptr + n, mask=m, other=0.0); rf = tl.load(rf_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        ta = tl.load(t_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        r = tl.load(r_ptr + n * T + t, mask=m, other=0.0)
        liq = tl.zeros_like(p); sub = tl.zeros_like(p)
        for k in tl.static_range(K):
            dT = tl.load(dT_ptr + n * K + k, mask=m, other=0.0); a = tl.load(a_ptr + n * K + k, mask=m, other=0.0)
            fk = tl.load(fk_ptr + n * K + k, mask=m, other=1.0)
            if t == 0:
                s = tl.load(x0_ptr + n * K + k, mask=m, other=0.0)
            else:
                s = tl.load(X_ptr + (n * T + t - 1) * K + k, mask=m, other=0.0)
            tk = ta + dT
            cold = tk < tt
            pk = fk * p
            s = s + tl.where(cold, sf * pk, 0.0)
            uu = ks * e
            mm = tl.where(cold, 0.0, cf * (tk - tt) + rf * r)
            fr = tl.where(uu + mm > s, s / tl.maximum(uu + mm, 1e-30), 1.0)             # constant rates until the pack is gone
            u = uu * fr
            mlt = mm * fr
            s = tl.maximum(s - u - mlt, 0.0)
            liq += a * (tl.where(cold, 0.0, pk) + mlt)
            sub += a * u
            tl.store(X_ptr + (n * T + t) * K + k, s, mask=m)
        tl.store(q_ptr + n * T + t, liq, mask=m)
        tl.store(u_ptr + n * T + t, sub, mask=m)


def bra_step(xp, p, ta, e, r, *prm):
    K = len(xp); tt, cf, sf, ks, rf = prm[:5]; dT = prm[5:5 + K]; a = prm[5 + K:5 + 2 * K]; fk = prm[5 + 2 * K:5 + 3 * K]
    xs, liq, sub = [], torch.zeros_like(p), torch.zeros_like(p)
    for k in range(K):
        tk = ta + dT[k]; cold = tk < tt; pk = fk[k] * p
        s = xp[k] + torch.where(cold, sf * pk, torch.zeros_like(p))
        uu, mm = ks * e, torch.where(cold, torch.zeros_like(p), cf * (tk - tt) + rf * r)
        fr = torch.where(uu + mm > s, s / (uu + mm).clamp(min=1e-30), torch.ones_like(s))
        u, mlt = uu * fr, mm * fr
        s = torch.relu(s - u - mlt)
        liq = liq + a[k] * (torch.where(cold, torch.zeros_like(p), pk) + mlt); sub = sub + a[k] * u
        xs.append(s)
    return xs, [liq, sub]


class BandsRadA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, ta, e, r, dT, a, fk, tt, cf, sf, ks, rf, x0):
        N, T = p.shape; K = dT.shape[1]
        v = [t_.detach().float().contiguous() for t_ in (p, ta, e, r, dT, a, fk, tt, cf, sf, ks, rf, x0)]
        X = torch.empty(N, T, K, device=p.device); q = torch.empty(N, T, device=p.device); u = torch.empty(N, T, device=p.device)
        _bra_fwd[(triton.cdiv(N, BLOCK),)](*v, X, q, u, N, T, K=K, BLOCK=BLOCK)
        ctx.save_for_backward(*v, X)
        return q, u, X

    @staticmethod
    def backward(ctx, gq, gu, gX):
        p, ta, e, r, dT, a, fk, tt, cf, sf, ks, rf, x0, X = ctx.saved_tensors
        K = dT.shape[1]
        prm = [tt, cf, sf, ks, rf] + [dT[:, k] for k in range(K)] + [a[:, k] for k in range(K)] + [fk[:, k] for k in range(K)]
        gi, gp = backward_k(bra_step, X, x0, gX if gX is not None else None, [gq, gu], [p, ta, e, r], prm)
        z = lambda g: g if g is not None else torch.zeros_like(tt)
        gdT = torch.stack([z(g) for g in gp[5:5 + K]], 1)
        gfk = torch.stack([z(g) for g in gp[5 + 2 * K:5 + 3 * K]], 1)
        return None, None, gi[2], None, gdT, None, gfk, gp[0], gp[1], gp[2], gp[3], gp[4], None


def band_factors(dz, a, gP):
    """f_k = exp(gP dz_k) / sum_j a_j exp(gP dz_j): area-weighted mean exactly 1 [N, K]"""
    w = torch.exp(gP[:, None] * dz)
    return w / (a * w).sum(1, keepdim=True)


def snow_bands_rada(p, ta, e, r, dT, a, fk, tt, cf, sf, ks, rf, x0=None):
    """p, ta, e, r [N, T]; dT, a, fk [N, K]; tt, cf, sf, ks, rf [N] -> liquid [N, T], sublimation [N, T], SWE per band [N, T, K]"""
    x0 = torch.zeros_like(dT) if x0 is None else x0
    return BandsRadA.apply(p, ta, e, r, dT, a, fk, tt, cf, sf, ks, rf, x0)
