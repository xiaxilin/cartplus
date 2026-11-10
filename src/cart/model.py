"""CaRT / CaRT+ forward model (all catchments in parallel, any time step, GPU).

Order of the daily water path (each item names the refinement switch of config.ModelConfig that enables it):
  1  phenology (pheno):           PET demand  E_eff = PET [1 - kv (1 - v_t)],  v_t = EMA_tau_v sigmoid((T - T_v)/s_v)
  2  snowpack (always):           5 elevation bands (GB: equal-area bands from the elevation quantiles; US, AUS: up to 5 groups of the
                                  CAMELS 100 m bands), lapse rate, degree-day melt, sublimation ks*PET;
                                  snow_rp adds the melt term rp*PET on melt days                       -> liquid input P_in
  3  surface layer (top):         U, capacity C_u, drainage time tau_u; its overflow joins the production-store input
  4  infiltration excess (ie):    i_e = P_n - F (1 - exp(-P_n/F)),  F = P* exp(-g w_t / Pbar) (>= fc_min if fc_min > 0)  -> direct runoff
  5  production store:            GR4J partition (or Pareto capacities, prod_pareto), evaporation, percolation G (kq)
  6  near-stream zone (zone):     share w of the area: saturation runoff, evaporation, drainage q_z to the outlet
  7  depression storage (depress): fill and spill of direct runoff; held water evaporates and seeps to groundwater
  8  connectivity (fast_law_der / route_conn / conn_gw): wetness state theta, connected fraction
                                  A = A_min + (1 - A_min) sigmoid((theta - theta_c + lam_s s + lam_W (W/W_ss - 1))/s_theta)
  9  unit hydrographs:            UH1 (quick path, share 0.9, or 1 - gamma A with route_conn) and UH2 (direct path)
 10  quick store:                 GR4J routing store with exchange X2 (r/X3)^3.5 (CaRT; X2 <= 0 with exch_loss), or the derived release (fast_law_der) with
                                  coefficient a_f A exp(viscosity(T) - aF frost(T)) (R/X3)^k inside the exact daily integral;
                                  exch_diss: dissipative exchange, exch_in: exterior inflow J_in (without fast_law_der, as in
                                  CaRT+ without the quick-release group P2, the dissipative exchange X2 (g - X5) acts inside the
                                  GR4J routing store);
                                  phys_bounds: k, aF, lam_s, lam_W >= 0
 11  groundwater:                 quadratic store, steady storage at the training-mean recharge, residence time tau;
                                  gw_et: uptake lam E_res W/(W + W0) (lam = 0 with no_uptake: the quadratic store)
 12  channel loss (chan_loss):    loss = L (1 - exp(-Q/L))
Spin-up: 4 years (1460 d) for the production store, surface layer, zone, wetness theta and depressions; 2 years (730 d) for the quick
store; the training-mean steady state for groundwater. Initial states carry no gradient.

Time step (hourly branch): ModelConfig.dt is the step length in days, and every parameter keeps its daily unit. The input of a step
arrives at a constant rate within the step (uniform-input scheme), and every store integrates its own equation over the step:
  unit hydrographs   rate-form ordinates (uh.ordinates_rate), exact for input constant within each step; X4 down to one hour
  surface layer      closed form, piecewise linear (kernels/exact.top_x)
  groundwater        Riccati solution (exact.quad_x); with uptake one symmetric sub-step (exact.gwet_x, nsub_slow)
  quick store        nsub symmetric (Strang) sub-steps with Richardson extrapolation, fourth order (exact.quick_x, exact.lin_x)
  production store   wetting / drying exact in the net input, percolation in sequence (prod_lie; percolation and depression storage
                     have relaxation rates of about 0.01 to 0.02 per day, where a first-order error is negligible)
  zone, theta        Riccati / Bernoulli solutions (kernels/zone, kernels/theta)
  interception, snow, infiltration excess, channel loss: constant rates on step totals
States that modulate another operator (theta, the production-store storage, the groundwater head) enter it as their mean over the
step. A network trained on daily flows with this scheme runs at an hourly step without change (tests/check_dt.py checks that a daily
and an hourly run of the same network agree on evenly spread forcing). The step-start scheme of the published daily models is on the
main branch.

The order of operations is fixed, so that training is reproducible to the bit for a given seed and device.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .data import Dataset
from .kernels.exact import gwet_x, lin_x, pareto_x, prod_x, quad_x, quick_x, top_x
from .kernels.grstores import gprod_k
from .kernels.pareto import pareto_prod
from .kernels.snow_rp import snow_bands_rada
from .kernels.snow_sub import snow_bands_sub
from .kernels.theta import theta_state
from .kernels.zone import zone
from .parameters import ABL_FILL, MappingNetwork
from .uh import ordinates_rate, uh_conv

def _mid(x, x0):
    """mean of a state over each step from its values at the start and the end of the step (x after each step [C, T], x0 [C]): a state
    that modulates another operator acts on it with its mean over the step, as every flux does"""
    return 0.5 * (x + torch.cat([x0[:, None], x[:, :-1]], 1))


R_REF = 30.0                                                                                  # reference storage of the exchange law (mm)
KP_GR4J = (4.0 / 9.0) ** 4 / 4.0                                                               # GR4J percolation: Kp = KP_GR4J * C
SPIN_STORE_D, SPIN_QUICK_D = 1460.0, 730.0                                                     # spin-up lengths (days)
VEG_KERNEL_D, VEG_INIT_D = 2048.0, 365.0                                                       # phenology EMA kernel and initial window (days)
IE_TAU = 3.0                                                                                   # time constant of the near-surface wetness (d)


class CaRT:
    """Holds the forcing on the device and the mapping network; simulate() runs the model for all catchments."""

    def __init__(self, ds: Dataset, cfg: ModelConfig, dev):
        self.cfg, self.dev, self.ds = cfg, dev, ds
        T, C = ds.P.shape
        self.T = T
        tr = ds.mask("train")
        self.tr = tr
        self.Ttr = int(np.where(ds.mask("val"))[0].max()) + 1                                 # training simulations end with the selection period
        self.dt = float(cfg.dt)                                                              # step length (days)
        self.spin_store, self.spin_quick = int(round(SPIN_STORE_D / self.dt)), int(round(SPIN_QUICK_D / self.dt))
        self.Lv, self.n_init = int(round(VEG_KERNEL_D / self.dt)), int(round(VEG_INIT_D / self.dt))
        self.Hdev = torch.from_numpy((ds.band_H - (ds.band_A * ds.band_H).sum(1, keepdims=True)).astype(np.float32)).to(dev) / 1000.0  # km
        self.Aband = torch.from_numpy(ds.band_A.astype(np.float32)).to(dev)
        if cfg.ie:
            self.Pbar_ie = torch.from_numpy(np.maximum(ds.P[tr].mean(0), 0.1 * self.dt).astype(np.float32)).to(dev)[:, None]
        self.xs = torch.from_numpy(ds.static).to(dev)
        self.Pg = torch.from_numpy(ds.P.T.astype(np.float32)).to(dev)
        self.Eg = torch.from_numpy(ds.E.T.astype(np.float32)).to(dev)
        self.trm = torch.from_numpy(tr).to(dev)
        self.Tcl = torch.from_numpy(np.ascontiguousarray((ds.Tm - 273.15).T).astype(np.float32)).to(dev)
        self.nfftV = 1 << int(np.ceil(np.log2(self.Tcl.shape[1] + self.Lv)))
        self.net = MappingNetwork(self.xs.shape[1], cfg, dev)

    # ------------------------------------------------------------------ auxiliary states
    def veg_state(self):
        """v = causal EMA (time constant tau_v) of sigmoid((T - T_v) / s_v), [C, T] (by FFT, 2048-day kernel)"""
        phg, Tcl, nfftV, dev = self.net.phg, self.Tcl, self.nfftV, self.dev
        n0 = self.n_init
        a_ = torch.exp(-self.dt / torch.exp(phg["ltv"]))
        k_ = (1 - a_) * a_ ** torch.arange(self.Lv, device=dev, dtype=torch.float32)
        ema = lambda x_: (torch.fft.irfft(torch.fft.rfft(x_ - x_[:, :n0].mean(1, keepdim=True), n=nfftV) * torch.fft.rfft(k_, n=nfftV)[None],
                                          n=nfftV)[:, :Tcl.shape[1]] + x_[:, :n0].mean(1, keepdim=True))
        return ema(torch.sigmoid((Tcl - phg["Tv"]) / torch.exp(phg["lsv"]))).clamp(0.0, 1.0)

    def _iex(self, ps, Ps_):
        """infiltration excess with an exponential distribution of daily capacities of mean F_t = P* exp(-g w_t / Pbar)"""
        dev = self.dev
        a_ie = float(np.exp(-self.dt / IE_TAU)); L_ie = int(np.ceil(12 * IE_TAU / self.dt))
        m_ie = (1.0 - a_ie) / (self.dt / IE_TAU)                                               # mean of exp(-t / tau) over the step
        k_ie = ((1 - a_ie) * a_ie ** torch.arange(L_ie, device=dev, dtype=torch.float32)).flip(0)[None, None]
        Pbar_ie = self.Pbar_ie
        fc_min = self.cfg.fc_min * self.dt

        def iex(pp, ee):
            pp_ = pp.clamp(min=0.0)                                                            # w_t = EMA of the input up to day t-1
            w = F.conv1d(F.pad(pp_[:, None], (L_ie, 0), value=0.0), k_ie)[:, 0, :pp.shape[1]]
            w = w + (a_ie ** torch.arange(0, pp.shape[1], device=dev, dtype=torch.float32))[None] * Pbar_ie   # start at the mean
            w = pp_ + (w - pp_) * m_ie                                                         # mean wetness over the step (uniform input)
            Fc = Ps_ * torch.exp(-ps["gie"][:, None] * w / Pbar_ie)
            if fc_min > 0:                                                                     # lower bound of F_c (float32 underflow of
                Fc = Fc.clamp(min=fc_min)                                                      # the exponential in arid catchments)
            pn_ = (pp - ee).clamp(min=0.0)
            x_ = pn_ / Fc
            return pn_ - Fc * torch.where(x_ < 1e-3, x_ - 0.5 * x_ * x_, -torch.expm1(-x_))
        return iex

    def _connected_fraction(self, h, pk, ek, Pspin, Espin, srel, Wn_pre):
        """A(theta): connected fraction of the quick-release closure, [C, TT] (theta from its own 4-year spin-up)"""
        net = self.net
        lc_ = net.law(h)
        Ct_, tt_ = torch.exp(lc_["lCt"][:, 0]), torch.exp(lc_["ltt"][:, 0])
        with torch.no_grad():
            th0_ = theta_state(Pspin.detach().contiguous(), Espin.contiguous(), Ct_.detach(), tt_.detach(), dt=self.dt)[:, -1].contiguous()
        th_ = _mid(theta_state(pk.contiguous(), ek.contiguous(), Ct_, tt_, th0_, dt=self.dt), th0_) / Ct_[:, None]
        sth_ = torch.exp(net.lawg["lsth"])
        return lc_, lc_["Amin"] + (1 - lc_["Amin"]) * torch.sigmoid((th_ - net.lawg["thc"] + lc_["lams"] * srel
                                                                    + (lc_["lamW"] * Wn_pre if Wn_pre is not None else 0.0)) / sth_)

    # ------------------------------------------------------------------ forward model
    def simulate(self, noise=0.0, TT=None, keep=None, xs_in=None):
        """-> Q [C, TT] (mm/d), per-catchment parameters, water-balance terms. noise: SD of the Gaussian attribute noise (training).
        keep: dict that receives every tensor of the forward pass (for diagnostics and tests/check_balance.py; no effect on the result).
        xs_in: standardised attributes [C, F] used instead of the catchments' own (donor ensembles of cart.predict); noise is added to them.
        bal: the balance of the stores below the snow bands and the surface layer (training.py prints its residual); the full-system
        balance from the raw precipitation is booked by tests/check_balance.py from keep."""
        cfg, net, dev = self.cfg, self.net, self.dev
        dt, SPIN_STORE, SPIN_QUICK = self.dt, self.spin_store, self.spin_quick
        ns = cfg.nsub                                                                          # sub-steps of the quick store (fast)
        nl = cfg.nsub_slow or cfg.nsub                                                         # sub-steps of the slow stores
        TT = self.Ttr if TT is None else TT
        Pg, Eg0, Tcl, trm = self.Pg, self.Eg, self.Tcl, self.trm
        xs = self.xs if xs_in is None else xs_in
        xsn = xs + noise * torch.randn_like(xs) if noise > 0 else xs
        h, ps = net(xsn)
        if cfg.ablate != "none":                                                               # removed components: fill-in constants, unused
            drop = {"no_gw": ("ltau",), "no_quick": ("lX3", "X2"), "no_uh": ("X4",), "no_exch": ("X2",),
                    "single": ("X4", "lX3", "X2", "ltau")}[cfg.ablate]
            ps.update({k_: torch.full_like(ps["lC"], ABL_FILL[k_]) for k_ in drop})

        def Eup(ek_, A_):                                                                      # demand on the groundwater uptake
            return (ek_ - A_).clamp(min=0.0)
        Cc, X3s = torch.exp(ps["lC"]), torch.exp(ps["lX3"])
        kq = 4.0 * KP_GR4J * torch.exp(ps["rKp"]) * dt                                         # 4 Kp dt / C
        af = 0.25 * X3s ** -4 * dt
        Xj = ps["X2"] * (R_REF / X3s) ** 3.5 * dt                                               # exchange per step
        if cfg.ablate in ("no_exch", "no_quick", "single"):
            Xj = torch.zeros_like(Xj)
        X3eq = X3s
        Eg = Eg0
        Eeff = Eg
        # ---- 1 phenology
        if cfg.pheno:
            vveg = self.veg_state()
            Eeff = Eg * (1.0 - ps["kv"][:, None] * (1.0 - vveg))
        pk, ek = Pg[:, :TT], Eeff[:, :TT]
        # ---- 2 snowpack (5 elevation bands; liquid input = rain + melt)
        # (band temperature offsets dT_k = -Gamma (z_k - z_mean) are formed inside each call)
        if cfg.snow_rp:
            fk_ = torch.ones_like(self.Aband)
            rfa_ = torch.exp(ps["lrfa"])
            Ir_ = Eg[:, :TT].contiguous()                                                      # melt index = PET (not the phenology-scaled demand)
            Pin, subl, Xsn = snow_bands_rada(Pg[:, :TT].contiguous(), Tcl[:, :TT].contiguous(), Eg[:, :TT].contiguous(), Ir_,
                                           -torch.exp(ps["lGam"])[:, None] * self.Hdev, self.Aband, fk_.contiguous(), ps["TTs"],
                                           torch.exp(ps["lCF"]) * dt, ps["SFCF"], torch.exp(ps["lks"]), rfa_)
        else:
            Pin, subl, Xsn = snow_bands_sub(Pg[:, :TT].contiguous(), Tcl[:, :TT].contiguous(), Eg[:, :TT].contiguous(),
                                          -torch.exp(ps["lGam"])[:, None] * self.Hdev, self.Aband, ps["TTs"], torch.exp(ps["lCF"]) * dt,
                                          ps["SFCF"], torch.exp(ps["lks"]))
        pk = Pin
        Pspin = Pin[:, :SPIN_STORE]
        Espin = Eeff[:, :SPIN_STORE]
        ie = None
        topq = None
        # ---- 3 surface layer (overflow into the production store)
        if cfg.top:
            cu_, ku_ = torch.exp(ps["lCu"]), dt * torch.exp(-ps["ltu"])                        # b_u = dt / tau_u
            with torch.no_grad():
                _, dsp, esp, Usp = top_x(Pspin.detach().contiguous(), Espin.detach().contiguous(), cu_.detach(), ku_.detach(), 0.5 * cu_.detach())
            topq, du_, eu_, User = top_x(pk.contiguous(), ek.contiguous(), cu_, ku_, Usp[:, -1].contiguous())
            osp = top_x(Pspin.detach().contiguous(), Espin.detach().contiguous(), cu_.detach(), ku_.detach(), 0.5 * cu_.detach())[0]
            Pspin, du_, topq = dsp + osp, du_ + topq, None
            Espin = (Espin - esp).clamp(min=0.0)
            pk, ek = du_, (ek - eu_).clamp(min=0.0)
        # ---- 4 infiltration excess
        if cfg.ie:
            Ps_ = torch.exp(ps["lPs"])[:, None] * dt                                           # capacity per step
            iex = self._iex(ps, Ps_)
            ie = iex(pk, ek)
            pk = pk - ie
            Pspin = Pspin - iex(Pspin, Espin)
        # ---- 5 production store
        with torch.no_grad():
            if cfg.prod_pareto:
                s0 = pareto_x(Pspin.detach().contiguous(), Espin.contiguous(), Cc.detach(), kq.detach(), torch.exp(ps["lbP"]).detach(),
                              0.7 * Cc.detach(), be=None, nsub=nl)[3][:, -1].contiguous()
            else:
                s0 = prod_x(Pspin.detach().contiguous(), Espin.contiguous(), Cc.detach(), kq.detach(), 0.7 * Cc.detach(), nsub=nl)[3][:, -1].contiguous()
        if cfg.prod_pareto:
            D, G, A, Sk = (None,) * 4 if cfg.prod_lie else pareto_x(pk.contiguous(), ek.contiguous(), Cc, kq, torch.exp(ps["lbP"]), s0, be=None, nsub=nl)
        else:
            D, G, A, Sk = (None,) * 4 if cfg.prod_lie else prod_x(pk.contiguous(), ek.contiguous(), Cc, kq, s0, nsub=nl)
        if cfg.prod_lie:                                                                       # wetting / drying (exact in the net input), then percolation
            if cfg.prod_pareto:
                D, G, A, Sk = pareto_prod(pk.contiguous(), ek.contiguous(), Cc, kq, torch.exp(ps["lbP"]), s0, be=None)
            else:
                D, G, A, _ = gprod_k(pk, ek, Cc, kq, s0)
        Dp, Gp, Ap = D, G, A                                                                   # (bookkeeping only: keep)
        Sser = s0[:, None] + torch.cumsum(pk - D - G - A, 1)
        srel = (_mid(Sser, s0) / Cc[:, None]).clamp(0.0, 1.0)                                    # mean over the step
        QZ, Sbal, s0bal = None, Sser, s0
        # ---- 6 near-stream saturated zone, in parallel with the production store
        if cfg.zone:
            wz, cz_, tz_ = ps["wz"][:, None], torch.exp(ps["lCz"]), torch.exp(ps["ltz"])
            with torch.no_grad():
                Zsp = zone(Pspin.detach().contiguous(), Espin.contiguous(), cz_.detach(), tz_.detach(), dt=dt)[2]
            z0_ = Zsp[:, -1].contiguous()
            dzz, qzz, Zser = zone(pk.contiguous(), ek.contiguous(), cz_, tz_, z0_, dt=dt)
            Zprev = torch.cat([z0_[:, None], Zser[:, :-1]], 1)
            Az = torch.minimum(pk, ek) + ((pk - ek).clamp(min=0.0) - dzz - qzz - (Zser - Zprev))   # interception + zone evaporation
            Dsrc_Z = wz * dzz
            D = (1 - wz) * D + wz * dzz
            G = (1 - wz) * G
            A = (1 - wz) * A + wz * Az
            QZ = wz * qzz
            Sbal, s0bal = (1 - wz) * Sser + wz * Zser, (1 - wz[:, 0]) * s0 + wz[:, 0] * z0_
        if ie is not None:                                                                     # infiltration excess over the whole area
            D = D + ie
        Sd_change = None
        # ---- 7 depression storage (fill and spill), Pareto-distributed depression capacities
        if cfg.depress:
            Cd_, bd_, kd_ = torch.exp(ps["lCd"]), torch.exp(ps["lbd"]), torch.exp(ps["lkd"]) * dt
            Din_dep = D                                                                        # (bookkeeping only: keep)
            Erd = (ek - A).clamp(min=0.0)
            with torch.no_grad():
                s0d = pareto_x(D[:, :SPIN_STORE].detach().contiguous(), Erd[:, :SPIN_STORE].detach().contiguous(), Cd_.detach(), kd_.detach(),
                               bd_.detach(), 0.5 * Cd_.detach(), net=False, nsub=nl)[3][:, -1].contiguous()
            if not cfg.prod_lie:
                Dd, Gd, Ad, Sdser = pareto_x(D.contiguous(), Erd.contiguous(), Cd_, kd_, bd_, s0d, net=False, nsub=nl)
            else:
                Dd, Gd, Ad, Sdser = pareto_prod(D.contiguous(), Erd.contiguous(), Cd_, kd_, bd_, s0d, net=False)
            D, G, A = Dd, G + Gd, A + Ad
            Sd_change = Sdser[:, -1] - s0d
        # ---- 8a groundwater head before routing (conn_gw; the head depends on G and the residual demand only)
        Wn_pre = None
        if cfg.conn_gw:
            gbar_ = G[:, trm[:TT]].mean(-1) / dt                                              # training-mean recharge (mm/d)
            ag_ = 1.0 / (gbar_.clamp(min=1e-6) * torch.exp(ps["ltau"]) ** 2)
            W0_ = torch.sqrt(gbar_.detach().clamp(min=1e-6) / ag_.detach())
            ag_ = ag_ * dt                                                                      # per step
            _, _, Wpre_ = gwet_x(G.contiguous(), Eup(ek, A).contiguous(), ag_, ps["lamE"], W0_, W0_, nsub=nl)
            Wn_pre = _mid(Wpre_, W0_) / W0_[:, None] - 1.0                                     # mean over the step
        # ---- 8b connected fraction before routing (route_conn)
        Ath_pre = None
        if cfg.route_conn:
            _, Ath_pre = self._connected_fraction(h, pk, ek, Pspin, Espin, srel, Wn_pre)
        if cfg.ablate == "no_gw":                                                              # no groundwater store: percolation joins the
            D, G = D + G, torch.zeros_like(G)                                                  # direct-runoff split (GR4J connection)
        # ---- 9 unit hydrographs
        Duh = D                                                                                # (bookkeeping only: keep)
        o1, o2 = ordinates_rate(ps["X4"], dt)                                                  # input at a uniform rate within the step
        if cfg.route_conn:                                                                     # gamma A(theta): connected units bypass the quick store
            gR = ps["gR"][:, None]
            if cfg.zone and not cfg.depress:
                Dby = gR * (Ath_pre * (D - Dsrc_Z) + Dsrc_Z)
            else:
                Dby = gR * Ath_pre * D
            x9, x1 = uh_conv(D - Dby, o1), uh_conv(Dby, o2)
        elif cfg.ablate == "no_uh":                                                            # X4 -> 0: both kernels are the same-day delta
            x9, x1 = 0.9 * D, 0.1 * D
        elif cfg.ablate == "single":                                                           # all direct runoff to the outlet the same day
            x9, x1 = torch.zeros_like(D), D
        else:
            x9, x1 = uh_conv(0.9 * D, o1), uh_conv(0.1 * D, o2)
        # ---- 10 quick store
        cX = None
        if cfg.exch_diss:                                                                      # exterior head: constant inflow part of the exchange
            cX = torch.exp(ps["lJin"]) * dt if cfg.exch_in else (-Xj).clamp(min=0.0) * ps["X5e"]   # bounded inflow >= 0, per step
            x9e, x1e = x9 + 0.9 * cX[:, None], x1 + 0.1 * cX[:, None]
        else:
            x9e, x1e = x9, x1
        if cfg.fast_law_der:                                                                   # derived release law inside the recursion
            lc = net.law(h)
            Tl = Tcl[:, :TT]
            frost = torch.sigmoid((net.lawg["TF"] - Tl) / torch.exp(net.lawg["lsF"]))           # frozen-ground share
            Ct, tt = torch.exp(lc["lCt"][:, 0]), torch.exp(lc["ltt"][:, 0])                     # wetness state: capacity, drainage time
            with torch.no_grad():
                th0 = theta_state(Pspin.detach().contiguous(), Espin.contiguous(), Ct.detach(), tt.detach(), dt=dt)[:, -1].contiguous()
            theta = _mid(theta_state(pk.contiguous(), ek.contiguous(), Ct, tt, th0, dt=dt), th0) / Ct[:, None]   # mean over the step
            Am = lc["Amin"]
            sth = torch.exp(net.lawg["lsth"])
            Ath = Am + (1 - Am) * torch.sigmoid((theta - net.lawg["thc"] + lc["lams"] * srel + (lc["lamW"] * Wn_pre if Wn_pre is not None else 0.0)) / sth)
            m_conn = torch.log(Ath)                                                            # connected fraction
            Tk = Tl.clamp(min=0.0) + 273.15
            m_visc = -np.log(10.0) * 247.8 * (1.0 / (Tk - 140.0) - 1.0 / (293.15 - 140.0))   # water viscosity (Vogel equation)
            c0 = m_conn + m_visc - lc["aF"] * frost - 0.0
            c1 = lc["k"] * torch.ones_like(c0)                                                 # storage exponent of the release
            with torch.no_grad():
                r0 = lin_x(x9e[:, :SPIN_QUICK].contiguous(), x1e[:, :SPIN_QUICK].contiguous(), af, Xj, X3eq, 0.5 * X3eq, c0[:, :SPIN_QUICK].contiguous(),
                          c1[:, :SPIN_QUICK].contiguous(), nsub=ns)[2][:, -1].contiguous()
            Qg, xr, Rser = lin_x(x9e.contiguous(), x1e.contiguous(), af, Xj, X3eq, r0, c0, c1, nsub=ns)
            if keep is not None:
                keep.update(theta_A=Ath, Rq=Rser, Qg=Qg, xr=xr)
        elif cfg.ablate in ("no_quick", "single"):                                             # no quick store: UH1 output straight to the outlet
            Qg, xr, Rser = x9 + x1, torch.zeros_like(x9), torch.zeros_like(x9)
            r0 = torch.zeros_like(Cc)
        else:                                                                                  # GR4J routing store with exchange
            X2s = Xj * (X3s / R_REF) ** 3.5
            X5s = ps["X5e"] if cfg.exch_diss else None                                         # exch_diss without the derived release (CaRT+
            with torch.no_grad():                                                              # without P2): dissipative exchange X2 (g - X5)
                r0 = quick_x(x9e[:, :SPIN_QUICK].detach().contiguous(), x1e[:, :SPIN_QUICK].detach().contiguous(), X2s.detach(), X3s.detach(),
                             0.5 * X3s.detach(), X5=X5s.detach() if X5s is not None else None, split=(0.9, 0.1), dt=dt, nsub=ns)[2][:, -1].contiguous()
            Qg, xr, Rser = quick_x(x9e.contiguous(), x1e.contiguous(), X2s, X3s, r0, X5=X5s, split=(0.9, 0.1), dt=dt, nsub=ns)
        # ---- 11 groundwater
        gbar = G[:, trm[:TT]].mean(-1) / dt                                                     # training-mean recharge (mm/d)
        ag = 1.0 / (gbar.clamp(min=1e-6) * torch.exp(ps["ltau"]) ** 2)
        W0 = torch.sqrt(gbar.detach().clamp(min=1e-6) / ag.detach())                            # steady storage at the training-mean recharge
        ag = ag * dt                                                                            # per step
        if cfg.gw_et:                                                                          # uptake from groundwater
            w_start = W0
            Er = Eup(ek, A)
            Qs, ewg, Wser = gwet_x(G.contiguous(), Er.contiguous(), ag, ps["lamE"], W0, W0, nsub=nl)
            wend = Wser[:, -1]
            A = A + ewg
        elif cfg.ablate in ("no_gw", "single"):                                                # no groundwater store
            w_start = torch.zeros_like(Cc)
            Qs, wend = G, w_start
        else:
            w_start = W0
            Qs, Wser = quad_x(G.contiguous(), ag, W0)
            wend = Wser[:, -1]
        Q = Qg + Qs
        if QZ is not None:
            Q = Q + QZ
        # ---- 12 channel transmission loss
        closs = None
        Qpre_cl = Q                                                                            # (bookkeeping only: keep)
        if cfg.chan_loss:
            Lc = torch.exp(ps["lLc"])[:, None] * dt                                              # loss capacity per step
            xq = Q.clamp(min=0.0) / Lc
            closs = Lc * torch.where(xq < 1e-3, xq - 0.5 * xq * xq, -torch.expm1(-xq))
            Q = Q - closs
        if keep is not None:                                                                   # every tensor of the forward pass (final bindings;
            keep.update({k_: v_ for k_, v_ in locals().items() if torch.is_tensor(v_)})       # tests/check_balance.py books the full water balance)
            keep.update(S=Sser)
        bal = dict(P=(pk + (ie if ie is not None else 0.0)).sum(1), A=A.sum(1), Q=Q.sum(1),
                   X=xr.sum(1) + (cX * TT if cX is not None else 0.0) - (closs.sum(1) if closs is not None else 0.0),
                   dS=Sbal[:, -1] - s0bal + (Sd_change if Sd_change is not None else 0.0), dR=Rser[:, -1] - r0, dW=wend - w_start,
                   transit=D.sum(1) - x9.sum(1) - x1.sum(1))
        return Q, ps, bal
