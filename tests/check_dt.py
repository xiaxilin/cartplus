"""Checks of the step length (ModelConfig.dt) and of the uniform-input scheme of the hourly branch (rate-form unit hydrographs, stores
solved for input at a constant rate within the step, X4 down to one hour). The store kernels of kernels/exact.py are checked by
tests/check_exact.py.

1 Unit hydrographs (float64, CPU): the ordinates of uh.ordinates and uh.ordinates_rate sum to 1 for any step; the same-step share of
  the rate ordinates is (1/dt) int_0^dt S, which for UH1 is X4/3.5 + dt - X4 (in units of dt) below one step and S(dt)/3.5 above; the
  mean delay of the rate ordinates (in steps) is exactly that of the continuous unit hydrograph, (2.5/3.5) X4 / dt for UH1 and
  X4 / dt for UH2; rate and point ordinates converge as dt -> 0; the gradient of the same-step share with respect to X4 is the
  analytic one.
2 Kernels outside exact.py that take the step length (zone, theta_state): directional derivative of the adjoint against central
  finite differences of the Triton forward at dt = 1 and dt = 1/24, perturbing every input and parameter.
3 Hourly water balance of CaRT+ (full system, the bookkeeping of tests/check_balance.py) at dt = 1/24, on the daily forcing of the
  first K catchments spread evenly over the 24 hours of each day (2003-2008).
4 Step invariance: with the same parameters (fresh network, fixed seed), the hourly run of CaRT and CaRT+ on that evenly spread forcing,
  summed over days, against the daily run on the daily forcing (2005-2008, after the spin-up): median NSE over the K catchments.
  The remaining difference is the coupling between operators, each of which passes the total of its step to the next as a uniform
  input. Under the step-start scheme of the main branch the same comparison gives a median NSE of 0.81 (CaRT) and 0.85 (CaRT+) on
  the first 20 GB catchments; the check guards against a return of the input-timing lead.

Pass: (1) errors < 1e-9; (2) relative error < 1e-2 for the best of three finite-difference steps (float32); (3) max |residual| /
accumulated P < 1e-5; (4) median NSE > 0.98.

    python tests/check_dt.py [--gpu 0] [--dataset gb] [--K 20]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]


def check_uh(results):
    from cart.uh import ordinates, ordinates_rate
    x4 = torch.tensor([0.05, 0.3, 0.7, 1.0, 1.2, 2.0, 3.7, 5.99], dtype=torch.float64)
    err_sum, err_share, err_delay = 0.0, 0.0, 0.0
    for dt in (1.0, 0.5, 1 / 24):
        p1, p2 = ordinates(x4, dt); r1, r2 = ordinates_rate(x4, dt)
        err_sum = max(err_sum, *(float((o.sum(1) - 1).abs().max()) for o in (p1, p2, r1, r2)))
        u = x4 / dt                                                                            # X4 in steps
        share = torch.where(u < 1, u / 3.5 + 1 - u, u.clamp(min=1) ** -2.5 / 3.5)
        err_share = max(err_share, float((r1[:, 0] - share).abs().max()))
        mean = lambda o: (o * torch.arange(o.shape[1], dtype=o.dtype)).sum(1)               # mean delay in steps
        err_delay = max(err_delay, float((mean(r1) - 2.5 / 3.5 * x4 / dt).abs().max()), float((mean(r2) - x4 / dt).abs().max()))
    results.append(("UH ordinates sum to 1 (point and rate, UH1 and UH2, dt 1, 1/2, 1/24)", err_sum < 1e-9, f"max error {err_sum:.1e}"))
    results.append(("rate UH1 same-step share = (1/dt) int_0^dt S", err_share < 1e-9, f"max error {err_share:.1e}"))
    results.append(("mean delay of the rate ordinates = that of the continuous UH (UH1 and UH2)", err_delay < 1e-9, f"max error {err_delay:.1e} steps"))
    conv = []
    for dt in (1 / 6, 1 / 24, 1 / 96):
        p1, _ = ordinates(x4[4:5], dt); r1, _ = ordinates_rate(x4[4:5], dt)
        n = min(p1.shape[1], r1.shape[1])
        conv.append(float(((r1[:, :n] - p1[:, :n]).abs().sum()) / 2))
    ok = conv[0] > conv[1] > conv[2]
    results.append(("rate and point ordinates converge as dt -> 0 (X4 = 1.2 d)", ok, "half L1 distance " + ", ".join(f"{c:.3f}" for c in conv)))
    x = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    ordinates_rate(x, 1.0)[0][0, 0].backward()
    g_err = abs(float(x.grad) - (1 / 3.5 - 1))
    results.append(("d(same-day share)/dX4 = 1/3.5 - 1 below one day (autograd)", g_err < 1e-9, f"error {g_err:.1e}"))


def check_kernels(dev, results):
    from cart.kernels.theta import theta_state
    from cart.kernels.zone import zone
    N, T = 32, 2000

    def check(name, f, args, pos):
        a = [x.clone().requires_grad_(True) for x in args]
        outs = f(*a); outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        gen = torch.Generator(device=dev).manual_seed(1)
        ws = [torch.rand(o.shape, device=dev, generator=gen) for o in outs]
        L = lambda os_: sum((w * o).sum() for w, o in zip(ws, os_)) / T
        L(outs).backward()
        v = [torch.randn(x.shape, device=dev, generator=gen) for x in args]
        dirs = [(vi * x if p else vi * x.abs().mean()) if ai.grad is not None else torch.zeros_like(x) for vi, x, p, ai in zip(v, args, pos, a)]
        g = sum((ai.grad * d).sum() for ai, d in zip(a, dirs) if ai.grad is not None).item()
        tup = lambda r: r if isinstance(r, (tuple, list)) else (r,)
        fd = []
        for eps in (1e-2, 1e-3, 1e-4):
            with torch.no_grad():
                lp = L(tup(f(*[x + eps * d for x, d in zip(args, dirs)]))).item(); lm = L(tup(f(*[x - eps * d for x, d in zip(args, dirs)]))).item()
            fd.append((lp - lm) / (2 * eps))
        best = min(abs(x - g) / abs(g) for x in fd)
        results.append((f"adjoint vs finite differences: {name}", best < 1e-2, f"best relative error {best:.1e}"))

    for dt in (1.0, 1.0 / 24.0):
        torch.manual_seed(0)
        p = torch.rand(N, T, device=dev) ** 3 * 30 * dt; ev = torch.rand(N, T, device=dev) * 3 * dt
        cz = 50 + 150 * torch.rand(N, device=dev); tz = 2 + 18 * torch.rand(N, device=dev); x0 = 0.3 * cz
        check(f"zone, dt = {dt:.4g}", lambda p, ev, cz, tz, x0: zone(p, ev, cz, tz, x0, dt=dt), [p, ev, cz, tz, x0], [True] * 5)
        check(f"theta_state, dt = {dt:.4g}", lambda p, ev, cz, tz, x0: theta_state(p, ev, cz, tz, x0, dt=dt), [p, ev, cz, tz, x0], [True] * 5)


def check_step_invariance(a, dev, results):
    from cart import data as D
    from cart.config import config_for
    from cart.model import CaRT
    d0 = D.load(a.dataset).subset(a.K)
    k = np.array([date(2003, 1, 1) <= d <= date(2008, 12, 31) for d in d0.days])
    from cart.data import Dataset
    dd = Dataset(d0.name, d0.ids, d0.gauged, d0.ungauged, d0.static, [d for d, kk in zip(d0.days, k) if kk], d0.P[k], d0.E[k], d0.q[k], d0.Tm[k],
                 {"train": ("2003-01-01", "2006-12-31"), "val": ("2007-01-01", "2008-12-31")}, d0.band_H, d0.band_A, d0.static_names, d0.p_mean)
    dh = hourly_dataset(d0)
    ev = np.array([d >= date(2005, 1, 1) for d in dd.days])
    for preset in ("cart", "cartplus"):
        sims = []
        for ds_, dt in ((dd, 1.0), (dh, 1.0 / 24.0)):
            torch.manual_seed(0); np.random.seed(0)
            m = CaRT(ds_, config_for(preset, a.dataset, dt=dt), dev)
            with torch.no_grad():
                sims.append(m.simulate(TT=m.T)[0].double().cpu().numpy())
        qd, qh = sims[0], sims[1].reshape(sims[1].shape[0], -1, 24).sum(-1)
        nse = [1 - ((qh[c, ev] - qd[c, ev]) ** 2).sum() / ((qd[c, ev] - qd[c, ev].mean()) ** 2).sum() for c in range(qd.shape[0])]
        vol = np.median(qh[:, ev].sum(1) / qd[:, ev].sum(1))
        results.append((f"step invariance, {preset}: hourly run on evenly spread forcing, summed over days, vs the daily run",
                        float(np.median(nse)) > 0.98, f"median NSE {np.median(nse):.4f} (min {np.min(nse):.4f}), volume ratio {vol:.4f}"))


def hourly_dataset(ds, a0="2003-01-01", a1="2008-12-31"):
    """the daily forcing of ds spread evenly over the 24 hours of each day; every step carries its day"""
    from cart.data import Dataset
    k = np.array([date.fromisoformat(a0) <= d <= date.fromisoformat(a1) for d in ds.days])
    days = [d for d, kk in zip(ds.days, k) for _ in range(24) if kk]
    rep = lambda x, s: np.repeat(x[k], 24, axis=0) * s
    return Dataset(ds.name, ds.ids, ds.gauged, ds.ungauged, ds.static, days, rep(ds.P, 1 / 24), rep(ds.E, 1 / 24), rep(ds.q, np.nan),
                   rep(ds.Tm, 1.0), {"train": (a0, "2006-12-31"), "val": ("2007-01-01", a1)}, ds.band_H, ds.band_A, ds.static_names, ds.p_mean)


def check_hourly_balance(a, dev, results):
    sys.path.insert(0, str(PKG / "tests"))
    from check_balance import book
    from cart import data as D
    from cart.config import config_for
    from cart.model import CaRT
    ds = hourly_dataset(D.load(a.dataset).subset(a.K))
    torch.manual_seed(0); np.random.seed(0)
    model = CaRT(ds, config_for("cartplus", a.dataset, dt=1.0 / 24.0), dev)
    keep = {}
    with torch.no_grad():
        Q, ps, _ = model.simulate(TT=model.T, keep=keep)
    res, Pr, _, _ = book(keep, ps, model, model.T)
    r = float((res / Pr).abs().max())
    results.append((f"hourly full-system water balance of CaRT+ (dt = 1/24, {a.K} catchments, {model.T} steps)",
                    r < 1e-5, f"max |residual| / P {r:.1e}; finite Q: {bool(torch.isfinite(Q).all())}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", default="gb", choices=["gb", "us", "aus"])
    ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, str(PKG / "src"))
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev)
    results = []
    check_uh(results)
    check_kernels(dev, results)
    check_hourly_balance(a, dev, results)
    check_step_invariance(a, dev, results)
    fail = 0
    for name, ok, msg in results:
        tag = "SKIPPED" if ok is None else ("PASSED" if ok else "FAILED")
        fail += ok is False
        print(f"{tag:8s} {name}: {msg}")
    print(f"SUMMARY check_dt: {'all passed' if not fail else f'{fail} FAILED'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
