"""Per-catchment parameter tables and the mapping network (static attributes -> parameters).

Every per-catchment parameter p has bounds (lo, hi) and an initial value p0: p = lo + (hi - lo) sigmoid(head(h)), h = MLP(attributes),
with the head's weights initialised to zero and its bias to logit((p0 - lo)/(hi - lo)), so that training starts at p0 everywhere.
Names starting with 'l' are logarithms (the model uses exp(.)).

Inactive output of the static head. The exterior-head factor X5e of the product form of the exterior inflow, (-X2)^+ X5e, acts only
with the dissipative exchange and either without the derived release (inside the GR4J routing store: cartplus_no_p2) or without the
exterior inflow J_in as its own parameter. With the derived release and J_in (cartplus, cartplus_no_p1, _p3, _p5) it has no effect and
is not a parameter of the model. For these configurations the static head is built and evaluated at the full table size, X5e
included: the head is created with the full number of outputs (so that the initial weights drawn for a seed do not depend on whether
X5e is used) and the X5e row is dropped afterwards (`head_rows`); its matrix product is taken at the full size with a zero row in place
of X5e (`MappingNetwork.static_head`), and train.clip_grad_norm takes the gradient norm with that zero row. cuBLAS chooses its
algorithm, and hence the float32 rounding of every output, by the shape of the product, so a product of the smaller size would give
results that differ in the last bits (up to 8e-3 mm/d in the flows for the 561 CAMELS-AUS catchments at once).
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def x5e_used(cfg: ModelConfig) -> bool:
    """X5e acts with the dissipative exchange, either in the GR4J routing store (no derived release) or as the product-form inflow
    (no J_in); with the derived release and J_in it has no effect."""
    return cfg.exch_diss and not (cfg.fast_law_der and cfg.exch_in)


def static_table(cfg: ModelConfig, full: bool = False) -> "OrderedDict[str, tuple]":
    """'STATIC' parameters (head 1) in head order: name -> (lo, hi, init). full=True: the full table, which also has X5e where it has
    no effect (see the module docstring)."""
    S = OrderedDict([
        ("lC", (np.log(25.0), np.log(1600.0), np.log(225.0))),        # production-store capacity C (mm)
        ("X4", (1.0 / 24.0, 6.0, 1.26)),                               # unit-hydrograph time base (d), down to one hour
        ("lX3", (np.log(5.0), np.log(1000.0), np.log(33.0))),          # routing-store capacity X3 (mm)
        ("ltau", (np.log(3.0), np.log(1460.0), np.log(78.0))),         # groundwater residence time tau (d)
        ("X2", (-20.0, 10.0, -0.35)),                                  # exchange coefficient X2 (mm/d); [-20, 0] with exch_loss / exch_diss
        ("rKp", (np.log(1.0 / 30.0), np.log(30.0), 0.0)),              # percolation coefficient relative to GR4J
    ])
    drop = {"none": (), "no_gw": ("ltau",), "no_quick": ("lX3", "X2"), "no_uh": ("X4",), "no_exch": ("X2",),
            "single": ("X4", "lX3", "X2", "ltau")}[cfg.ablate]
    for k in drop:                                                                              # removed component: parameter leaves the set
        del S[k]
    # snowpack, 5 elevation bands
    S.update({"TTs": (-2.5, 2.5, 0.0), "lCF": (np.log(0.5), np.log(10.0), np.log(3.0)), "SFCF": (0.5, 1.5, 1.0),
              "lGam": (np.log(3.0), np.log(9.0), np.log(6.5))})
    if cfg.zone:                                                                                # zone_small initialisation
        S.update({"wz": (0.0, 0.5, 0.05), "lCz": (np.log(5.0), np.log(300.0), np.log(15.0)), "ltz": (np.log(2.0), np.log(120.0), np.log(4.0))})
    if cfg.pheno:
        S["kv"] = (0.0, cfg.kv_max, 0.1)
    if cfg.gw_et:
        S["lamE"] = (0.0, 1.0, 0.05)                                                           # uptake coefficient (0 with no_uptake)
    if cfg.prod_pareto:
        S["lbP"] = (np.log(0.03), np.log(8.0), np.log(0.7))
    if cfg.top:                                                                                 # top_init 20,10
        S.update({"lCu": (np.log(1.0), np.log(300.0), np.log(20.0)), "ltu": (np.log(0.05), np.log(60.0), np.log(10.0))})
    if cfg.chan_loss:
        S["lLc"] = (np.log(1e-3), np.log(5.0), np.log(3e-3))                                  # mean loss capacity L (mm/d)
    if cfg.depress:
        S.update({"lCd": (np.log(0.02), np.log(300.0), np.log(0.2)), "lbd": (np.log(0.05), np.log(8.0), 0.0),
                  "lkd": (np.log(1e-4), np.log(10.0), np.log(0.01))})
    if cfg.route_conn:
        S["gR"] = (0.0, 1.0, 0.13)
    if cfg.exch_loss and "X2" in S:                                                             # GR4J exchange as a loss only (Law 2)
        S["X2"] = (-20.0, 0.0, -0.35)
    if cfg.exch_diss:
        S["X2"] = (-20.0, 0.0, -0.35)                                                          # loss branch only (keeps its position)
        if full or x5e_used(cfg):
            S["X5e"] = (0.0, 1.0, 0.001)                                                       # exterior head (product form), see x5e_used
        if cfg.exch_in:
            S["lJin"] = (np.log(1e-4), np.log(20.0), np.log(1e-3))
    S["lks"] = (np.log(1e-4), np.log(1.0), np.log(1e-3))                                       # sublimation factor
    if cfg.snow_rp:
        S["lrfa"] = (np.log(1e-3), np.log(10.0), np.log(1e-2))                                 # PET-index melt factor
    if cfg.ie:                                                                                  # ie_form exp, wetness-dependent
        S["lPs"] = (np.log(2.0), np.log(3000.0), np.log(1000.0))
        S["gie"] = (0.0, 4.0, 0.3)
    return S


def law_table(cfg: ModelConfig) -> "OrderedDict[str, tuple]":
    """'LAWC' parameters (head 2) of the derived quick release (only with cfg.fast_law_der); with cfg.phys_bounds a_F, k, lambda_s
    (and lambda_W) are restricted to >= 0 and k, lambda_s, lambda_W start at 0.05."""
    L = OrderedDict([("aF", (-1.0, 6.0, 1.0)), ("k", (-3.0, 3.0, 0.0)), ("Amin", (cfg.amin_lo, 0.999, 0.75)),
                     ("lCt", (np.log(5.0), np.log(500.0), np.log(50.0))), ("ltt", (np.log(2.0), np.log(200.0), np.log(20.0))),
                     ("lams", (-3.0, 3.0, 0.0))])
    if cfg.phys_bounds:                                                                         # physical ranges: frost can only suppress the
        L["aF"] = (0.0, 6.0, 1.0); L["k"] = (0.0, 3.0, 0.05); L["lams"] = (0.0, 3.0, 0.05)     # release, k >= 0, wetter soil only connects more
    if cfg.conn_gw:                                                                             # a higher groundwater head only connects more
        L["lamW"] = (0.0, 3.0, 0.05) if cfg.phys_bounds else (-3.0, 3.0, 0.0)
    return L


ABL_FILL = {"X4": 1.26, "lX3": np.log(33.0), "X2": 0.0, "ltau": np.log(78.0)}                # fill-in values of removed parameters (unused)


def head_rows(cfg: ModelConfig):
    """rows of the full static head that are model parameters (None: all)"""
    S, Sr = static_table(cfg), static_table(cfg, full=True)
    return None if list(S) == list(Sr) else [list(Sr).index(k) for k in S]


def _head(n_out, table, dev, rows=None):
    """zero-weight head with bias logit((p0 - lo)/(hi - lo)) (the default initialisation of nn.Linear draws random numbers in
    proportion to its size before the weights are zeroed); rows: keep only these outputs afterwards (same random-number stream)"""
    lin = nn.Linear(128, n_out).to(dev)
    nn.init.zeros_(lin.weight)
    with torch.no_grad():
        lin.bias.copy_(torch.tensor([np.log((v[2] - v[0]) / (v[1] - v[2])) for v in table.values()]))
    if rows is not None:
        lin.weight = nn.Parameter(lin.weight.detach()[rows].clone())
        lin.bias = nn.Parameter(lin.bias.detach()[rows].clone())
        lin.out_features = len(rows)
    return lin


class MappingNetwork(nn.Module):
    """h = 3-layer GELU MLP (128 units) of the standardised attributes; head 'static' -> STATIC parameters, head 'law' -> LAWC
    parameters (derived quick release), global constants of the derived release and of the phenology state.

    Random-number order: three further linear layers (128->16, 128->5 and 128->4) are instantiated at fixed positions and discarded;
    they advance the random-number stream, so that a given --seed gives the initial weights of the paper's runs. The static head is
    built at the full size (with X5e) and the inactive X5e row dropped afterwards (head_rows); the head is evaluated at the full size
    with a zero X5e row (static_head), and train.clip_grad_norm takes the gradient norm with that zero row (see the module docstring)."""

    def __init__(self, n_attr, cfg: ModelConfig, dev):
        super().__init__()
        self.cfg = cfg
        self.static_par = static_table(cfg)
        self.law_par = law_table(cfg) if cfg.fast_law_der else OrderedDict()
        self.senc = nn.Sequential(nn.Linear(n_attr, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU()).to(dev)
        self.rows = head_rows(cfg)                                                              # (inactive X5e row dropped)
        Sr = static_table(cfg, full=True)
        self.shead = _head(len(Sr), Sr, dev, self.rows)
        self.n_full = len(Sr)
        self._rows_t = None if self.rows is None else torch.tensor(self.rows, device=dev)
        nn.Linear(128, 16)                                                                      # discarded (RNG order only)
        nn.Linear(128, 5)                                                                       # discarded (RNG order only)
        pv = lambda v: nn.Parameter(torch.tensor(float(v), device=dev))
        if cfg.fast_law_der:
            # with conn_gw a head without lamW is drawn first and discarded (RNG order only)
            if cfg.conn_gw:
                nn.Linear(128, len(self.law_par) - 1)                                           # discarded (RNG order only)
            self.lawhead = _head(len(self.law_par), self.law_par, dev)
            # globals of the derived release: frost threshold T_F and width s_F, connectivity centre theta_c and width s_theta
            self.lawg = nn.ParameterDict({"TF": pv(0.0), "lsF": pv(np.log(2.4)), "thc": pv(0.4), "lsth": pv(np.log(0.15))})
        # globals of the phenology state: threshold T_v, width s_v, time constant tau_v
        self.phg = nn.ParameterDict({"Tv": pv(8.0), "lsv": pv(np.log(3.0)), "ltv": pv(np.log(20.0))} if cfg.pheno else {})
        nn.Linear(128, 4)                                                                       # discarded (RNG order only)

    def static_head(self, h):
        """the static head's pre-activations [N, n_static]; where the full head has the inactive X5e row, the product is taken at the
        full size with a zero row in its place and that column dropped (fixes the cuBLAS algorithm and hence the float32 rounding;
        the other columns do not depend on the values of that row)"""
        if self.rows is None:
            return self.shead(h)
        W = self.shead.weight.new_zeros((self.n_full, self.shead.in_features)).index_copy(0, self._rows_t, self.shead.weight)
        b = self.shead.bias.new_zeros(self.n_full).index_copy(0, self._rows_t, self.shead.bias)
        return F.linear(h, W, b).index_select(1, self._rows_t)

    def forward(self, xs):
        """xs [N, n_attr] -> h [N, 128], STATIC parameters {name: [N]}"""
        h = self.senc(xs)
        u = torch.sigmoid(self.static_head(h))
        if self.cfg.no_uptake:                                                                  # lambda_E fixed at its lower bound 0: its head
            cols = list(u.unbind(1))                                                            # column is replaced by 0 (the column stays
            cols[list(self.static_par).index("lamE")] = torch.zeros_like(cols[0])              # in the head, so checkpoints keep their
            u = torch.stack(cols, 1)                                                            # shape)
        ps = {k: lo + (hi - lo) * u[:, i] for i, (k, (lo, hi, _)) in enumerate(self.static_par.items())}
        return h, ps

    def law(self, h):
        """LAWC parameters {name: [N, 1]} of the derived quick release"""
        ul = torch.sigmoid(self.lawhead(h))
        return {k: (lo + (hi - lo) * ul[:, i])[:, None] for i, (k, (lo, hi, _)) in enumerate(self.law_par.items())}

    def train_parameters(self):
        """parameters in the order of the optimiser's parameter list"""
        p = list(self.senc.parameters()) + list(self.shead.parameters())
        if self.cfg.fast_law_der:
            p += list(self.lawhead.parameters()) + list(self.lawg.values())
        return p + list(self.phg.values())
