"""Checks of the uniform-input store kernels of the hourly branch (src/cart/kernels/exact.py, exact_bwd.py), at one step length:
  1 the Triton forward against the float64 recursion of its torch step function;
  2 the water balance of every kernel;
  3 step invariance: one step of length dt with a constant input against the same input over 96 steps of dt / 96 (closed-form
    kernels: equal to rounding; sub-stepped kernels: the error at nsub 1, 2, 4, 8 is printed, the configured nsub is checked);
  4 gradients of the hand-written adjoint (exact_bwd.py; with CART_GENERIC_ADJ=1 the generic stored-state adjoint) against float64
    autograd of the plain recursion.
Inputs: 48 synthetic catchments, 300 steps, gamma-distributed rain on 35 % of the steps, amounts per step.
Pass: kernel vs the float32 step function < 1e-4 at dt = 1 and < 5e-4 at dt < 1 (relative to the largest value; the same arithmetic in
another order, so a larger difference is a kernel error); kernel vs the float64 step < 2e-4 at dt = 1 and < 2e-3 at dt < 1; water balance < 2e-5 of the input at dt = 1 and < 3e-4
at dt < 1; one step vs 96 fine steps as listed; adjoint < 2e-3 (relative). The looser bounds at dt < 1 are float32 precision: each
step carries 1/24 of the daily amount against a store of tens of mm, and the float32 step function departs from the float64 one by
the same amount as the kernel does (about 6e-4 of the largest hourly exchange flux of the routing store).

    python tests/check_exact.py [--gpu 0] [--dt 1.0] [--only top_x,quad_x]
"""
import argparse, functools, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cart.kernels import exact as X

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--dt", type=float, default=1.0)
ap.add_argument("--only", default=""); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); torch.manual_seed(0); rng = np.random.default_rng(0)
N, T, dt = 48, 300, a.dt
f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
wet = rng.random((N, T)) < 0.35
P = f32(np.where(wet, rng.gamma(0.7, 9.0, (N, T)), 0.0) * dt)                       # per step
E = f32((1.5 + 1.4 * np.sin(np.arange(T) / 58.0))[None] * rng.uniform(0.5, 1.5, (N, 1)) * dt)
U = lambda lo, hi: f32(rng.uniform(lo, hi, N))
fails = []
nsub_chk = [4]                                                                             # sub-step count whose invariance is checked


CUR = [""]


def report(name, val, tol):
    ok = val <= tol; print(f"{'PASSED' if ok else 'FAILED':8s} {CUR[0]}, {name}: {val:.2e} (tol {tol:.0e})")
    if not ok: fails.append(f"{CUR[0]}: {name}")


def recur(step, x0, ins, prm, dtype=torch.float64):
    """plain recursion of a step function: returns states [N, T] and outputs"""
    x = [x0.to(dtype)]; ins = [v.to(dtype) for v in ins]; prm = [v.to(dtype) for v in prm]
    xs, outs = [], None
    for t in range(ins[0].shape[1]):
        x, o = step(x, *[v[:, t] for v in ins], *prm)
        xs.append(x[0]); outs = [[q] for q in o] if outs is None else [l + [q] for l, q in zip(outs, o)]
    return torch.stack(xs, 1), [torch.stack(l, 1) for l in outs]


def relerr(a_, b_):
    return float((a_.double() - b_.double()).abs().max() / b_.double().abs().max().clamp(min=1e-12))


def check(name, kern, step, x0, ins, prm, storage_in, storage_out, fine=None, tol_inv=2e-5, grad_scale=1.0):
    """kern(ins, prm, x0) -> (outs..., X); storage_in / storage_out: functions of (ins, outs) -> total in / out of the store"""
    if a.only and not any(k in name for k in a.only.split(",")): return
    CUR[0] = f"{name}, dt = {dt:.4g}"
    outs = kern(ins, prm, x0); Xs = outs[-1]; outs = outs[:-1]
    Xr, outr = recur(step, x0, ins, prm)
    X32, out32 = recur(step, x0, ins, prm, torch.float32)
    report("kernel vs float32 step, state and outputs", max([relerr(Xs, X32)] + [relerr(o, r) for o, r in zip(outs, out32)]),
           1e-4 if dt >= 1.0 else 5e-4)
    tol64 = 2e-4 if dt >= 1.0 else 2e-3
    report("kernel vs float64 step, state", relerr(Xs, Xr), tol64)
    report("kernel vs float64 step, outputs", max(relerr(o, r) for o, r in zip(outs, outr)), tol64)
    bal = (x0 + storage_in(ins, outs).sum(1) - storage_out(ins, outs).sum(1) - Xs[:, -1]).abs().max() / storage_in(ins, outs).sum(1).max()
    report("water balance / input", float(bal), 2e-5 if dt >= 1.0 else 3e-4)
    if fine is not None:                                                                   # one step against 96 fine steps, float64 step function
        errs = fine()
        for k, v in errs.items():
            if k in ("closed form", f"nsub {nsub_chk[0]}"):
                report(f"one step vs 96 fine steps, {k}", v, tol_inv)
            else:
                print(f"{'INFO':8s} {CUR[0]}, one step vs 96 fine steps, {k}: {v:.2e}")
    # gradients: loss = weighted sum of outputs and last state
    wts = [torch.randn_like(o) for o in outs] + [torch.randn_like(Xs[:, -1])]
    ins_g = [v.clone().requires_grad_(True) for v in ins]; prm_g = [v.clone().requires_grad_(True) for v in prm]
    o = kern(ins_g, prm_g, x0); L = sum((w * q).sum() for w, q in zip(wts[:-1], o[:-1])) + (wts[-1] * o[-1][:, -1]).sum()
    g = torch.autograd.grad(L, ins_g + prm_g, allow_unused=True)
    ins_d = [v.double().clone().requires_grad_(True) for v in ins]; prm_d = [v.double().clone().requires_grad_(True) for v in prm]
    Xd, od = recur(step, x0, ins_d, prm_d); Ld = sum((w.double() * q).sum() for w, q in zip(wts[:-1], od)) + (wts[-1].double() * Xd[:, -1]).sum()
    gd = torch.autograd.grad(Ld, ins_d + prm_d, allow_unused=True)
    worst = 0.0
    for ga, gb in zip(g, gd):
        if gb is None: continue
        ga = torch.zeros_like(gb) if ga is None else ga
        worst = max(worst, float((ga.double() - gb).abs().max() / gb.abs().max().clamp(min=1e-12)))
    report("adjoint vs float64 autograd (max over inputs and parameters)", worst, 2e-3 * grad_scale)


def fine_test(step_of_nsub, x0, ins1, prm, scale_prm, nsubs=(1, 2, 4, 8), nf=96):
    """ins1: inputs of ONE step [N, 1]; scale_prm(prm, k) -> parameters for a step k times shorter. Reference: nf fine steps at nsub 4."""
    d = torch.float64
    ref_x, ref_o = recur(step_of_nsub(4), x0, [v.repeat(1, nf) / nf if s else v.repeat(1, nf) for v, s in ins1], scale_prm(prm, nf), d)
    ref = [ref_x[:, -1]] + [o.sum(1) for o in ref_o]
    out = {}
    for ns in nsubs:
        x, o = recur(step_of_nsub(ns), x0, [v for v, s in ins1], prm, d)
        got = [x[:, -1]] + [q.sum(1) for q in o]
        out[f"nsub {ns}"] = max(float((g_ - r_).abs().max() / r_.abs().max().clamp(min=1e-9)) for g_, r_ in zip(got, ref))
    return out


one = lambda v: [(v[:, 150:151] * 0 + v.mean(1, keepdim=True) * 4.0, True)]              # a wet step: 4 x the mean input

# ---- surface layer
cu, b = U(2.0, 12.0), U(0.05, 3.0) * dt
k_top = lambda ins, prm, x0: X.top_x(ins[0], ins[1], prm[0], prm[1], x0)
ft = lambda: fine_test(lambda ns: X.top_step, 0.3 * cu, one(P) + [(E[:, 150:151], True)], [cu, b], lambda prm, k: [prm[0], prm[1] / k], nsubs=(1,))
check("top_x (surface layer, closed form)", k_top, X.top_step, 0.3 * cu, [P, E], [cu, b],
      lambda i, o: i[0], lambda i, o: o[0] + o[1] + o[2], fine=lambda: {k.replace("nsub 1", "closed form"): v for k, v in ft().items()})
# ---- quadratic groundwater store
ag = U(2e-4, 5e-3) * dt; W0 = U(20.0, 80.0); Gin = P * 0.3
k_quad = lambda ins, prm, x0: X.quad_x(ins[0], prm[0], x0)
fq = lambda: fine_test(lambda ns: X.quad_step, W0, one(Gin), [ag], lambda prm, k: [prm[0] / k], nsubs=(1,))
check("quad_x (groundwater store, closed form)", k_quad, X.quad_step, W0, [Gin], [ag], lambda i, o: i[0], lambda i, o: o[0],
      fine=lambda: {k.replace("nsub 1", "closed form"): v for k, v in fq().items()})
# ---- groundwater store with uptake
lam = U(0.05, 0.6)
for ns in (1, 4):                                                                          # 1 = the model's nsub_slow
    k_gw = lambda ins, prm, x0: X.gwet_x(ins[0], ins[1], prm[0], prm[1], prm[2], x0, nsub=ns)
    fg = lambda: fine_test(lambda n_: functools.partial(X.gwet_step, nsub=n_), W0, one(Gin) + [(E[:, 150:151], True)], [ag, lam, W0],
                           lambda prm, k: [prm[0] / k, prm[1], prm[2]])
    nsub_chk[0] = ns
    check(f"gwet_x (groundwater store with uptake, nsub {ns})", k_gw, functools.partial(X.gwet_step, nsub=ns), W0, [Gin, E], [ag, lam, W0],
          lambda i, o: i[0], lambda i, o: o[0] + o[1], fine=fg, tol_inv=1e-3 if ns >= 4 else 2e-2)
nsub_chk[0] = 4
# ---- production stores
Cc = U(150.0, 600.0); kq = 4.0 * ((4.0 / 9.0) ** 4 / 4.0) * U(0.3, 30.0) * dt; bP = U(0.2, 3.0); beP = U(1.0, 3.0)
for nm, par, net in (("prod_x (GR4J production store)", False, True), ("pareto_x (Pareto production store)", True, True), ("pareto_x (depression storage, net = False)", True, False)):
    beP = U(1.0, 3.0) if net else 2.0 + 0.0 * bP                                           # the model uses be = 2 (exact drying step)
    stp = lambda n_: functools.partial(X.prod_step, nsub=n_, pareto=par, net=net)
    k_pr = lambda ins, prm, x0: X.ProdX.apply(ins[0], ins[1], prm[0], prm[1], prm[2], prm[3], x0, 4, par, net)
    fp = lambda: fine_test(stp, 0.6 * Cc, one(P) + [(E[:, 150:151], True)], [Cc, kq, bP, beP], lambda prm, k: [prm[0], prm[1] / k, prm[2], prm[3]])
    check(nm + ", nsub 4", k_pr, stp(4), 0.6 * Cc, [P, E], [Cc, kq, bP, beP],
          (lambda i, o: i[0]) if net else (lambda i, o: i[0]), lambda i, o: o[0] + o[1] + o[2], fine=fp, tol_inv=1e-3)
# ---- GR4J routing store
X3 = U(15.0, 120.0); X2 = -U(0.0, 3.0) * dt; X5 = U(0.0, 0.4); x9, x1 = 0.9 * P * 0.5, 0.1 * P * 0.5
stq = lambda n_: functools.partial(X.quick_step, dt=dt, nsub=n_, cr=0.9, cd=0.1)
k_q = lambda ins, prm, x0: X.quick_x(ins[0], ins[1], prm[0], prm[1], x0, X5=prm[2], split=(0.9, 0.1), dt=dt, nsub=4)
fqk = lambda: {k: v for n96 in (96,) for k, v in fine_test_dt(n96).items()}


def fine_test_dt(nf):
    """the routing store takes dt itself: the fine reference runs at dt / nf"""
    d = torch.float64; ins1 = [x9.mean(1, keepdim=True) * 6.0, x1.mean(1, keepdim=True) * 6.0]; x0 = 0.5 * X3
    rx, ro = recur(functools.partial(X.quick_step, dt=dt / nf, nsub=4, cr=0.9, cd=0.1), x0, [v.repeat(1, nf) / nf for v in ins1], [X2 / nf, X3, X5], d)
    ref = [rx[:, -1]] + [o.sum(1) for o in ro]; out = {}
    for ns in (1, 2, 4, 8):
        x, o = recur(stq(ns), x0, ins1, [X2, X3, X5], d); got = [x[:, -1]] + [q.sum(1) for q in o]
        out[f"nsub {ns}"] = max(float((g_ - r_).abs().max() / r_.abs().max().clamp(min=1e-9)) for g_, r_ in zip(got, ref))
    return out


check("quick_x (GR4J routing store with exchange), nsub 4", k_q, stq(4), 0.5 * X3, [x9, x1], [X2, X3, X5],
      lambda i, o: i[0] + i[1] + o[1], lambda i, o: o[0], fine=fqk, tol_inv=2e-2)
# ---- quick store of the derived release law
af = 0.25 * X3 ** -4 * dt; Xj = -U(0.0, 0.5) * dt
c0 = f32(rng.normal(-0.5, 0.8, (N, T))); c1 = f32(np.repeat(rng.uniform(0.0, 1.5, (N, 1)), T, 1))
stl = lambda n_: functools.partial(X.lin_step, nsub=n_)
k_l = lambda ins, prm, x0: X.lin_x(ins[0], ins[1], prm[0], prm[1], prm[2], x0, ins[2], ins[3], nsub=4)
fl = lambda: fine_test(stl, 0.5 * X3, [(x9.mean(1, keepdim=True) * 6.0, True), (x1.mean(1, keepdim=True) * 6.0, True), (c0[:, 150:151], False), (c1[:, 150:151], False)],
                       [af, Xj, X3], lambda prm, k: [prm[0] / k, prm[1] / k, prm[2]])
check("lin_x (quick store of the derived release), nsub 4", k_l, stl(4), 0.5 * X3, [x9, x1, c0, c1], [af, Xj, X3],
      lambda i, o: i[0] + i[1] + o[1], lambda i, o: o[0], fine=fl, tol_inv=2e-2)
print(f"SUMMARY check_exact (dt = {dt:.4g}): " + (f"{len(fails)} FAILED: " + "; ".join(fails) if fails else "all passed"))
sys.exit(1 if fails else 0)
