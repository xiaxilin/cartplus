"""Model configurations of the paper as named presets.

Every configuration is the CaRT core (GR4J-type production store, unit hydrographs, routing store with exchange, quadratic groundwater
store, 5-band snowpack with sublimation) plus a set of refinements, each switched on by a field of ModelConfig.

The refinements of CaRT+ form five groups by the physical process they act on (paper Section 5.3, Figure fig:lofo, Table tab:lofo;
a refinement that needs another is in the same group):
  P1  runoff generation: surface layer, Pareto partition of the root zone, infiltration excess, depression storage, near-stream zone
  P2  quick release: connected release (connected fraction A(theta), release exponent k, viscosity, frost; derived release law,
      A_min >= 0.5), connected bypass, groundwater-controlled connectivity
  P3  evaporation: vegetation demand (phenology, kv <= 1), groundwater uptake
  P4  exchange with the exterior: exterior head (dissipative exchange), exterior inflow J_in, channel transmission loss
  P5  snow: melt energy term (PET-index melt)
CaRT+ = CaRT + P1..P5 with the physical bounds k >= 0, a_F >= 0, lambda >= 0, lambda_W >= 0 (`phys_bounds`; k, lambda, lambda_W start
at 0.05). The leave-one-group-out presets cartplus_no_p<i> are CaRT+ with the switches of group i reset, with two exceptions that keep
the other groups intact:
  - without P3 the groundwater store keeps its uptake kernel (gw_et), because the groundwater-controlled connectivity (P2) computes the
    head with it, but the uptake coefficient lambda_E is fixed at its lower bound 0 (`no_uptake`); with lambda_E = 0 the store is the
    quadratic store of CaRT;
  - without P4 the routing store keeps the GR4J exchange, restricted to losses (`exch_loss`).

Exchange (Law 2): the exchange coefficient X2 of the GR4J routing store is restricted to losses, X2 in [-20, 0] mm/d (`exch_loss`;
the GR4J range is [-20, 10]). This applies to cart and cartplus_no_p4, the presets without the dissipative exchange; with the
dissipative exchange (cartplus and the other leave-one-out presets) X2 is in [-20, 0] already.

  preset            model
  cart              CaRT
  cartplus          CaRT+ = CaRT + P1..P5
  cartplus_no_p<i>  CaRT+ without group P<i> (i = 1..5)
On CAMELS-AUS, config_for(preset, "aus") adds the lower bound fc_min = 1e-4 mm/d of the infiltration capacity to the presets with
infiltration excess (the paper's CAMELS-AUS runs are cart and cartplus).

Other combinations of the switches, including the component-removal variants of CaRT (`ablate`), can be built with ModelConfig(...)
directly.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    name: str = "cart"
    # quick release: derived law (theta connectivity, viscosity, frost) inside the quick-store recursion
    fast_law_der: bool = False
    amin_lo: float = 0.02          # lower bound of the permanently connected fraction A_min (paper: 0.5 whenever fast_law_der)
    phys_bounds: bool = False      # physical ranges k, a_F, lambda (and lambda_W with conn_gw) >= 0 of the derived release
    zone: bool = False             # near-stream saturated zone (always with the small, event-scale initialisation: Cz 15 mm, tz 4 d)
    pheno: bool = False            # vegetation activity state scaling PET
    kv_max: float = 0.9            # upper bound of the phenology amplitude kv (paper: 1.0 whenever pheno)
    gw_et: bool = False            # evapotranspiration from groundwater
    no_uptake: bool = False        # (with gw_et) uptake coefficient lambda_E fixed at its lower bound 0: the gw_et kernel stays
                                   # (conn_gw computes the head with it) and with lambda_E = 0 is the quadratic store
    top: bool = False              # surface layer (C_u = 20 mm, tau_u = 10 d initial; overflow into the production store)
    prod_pareto: bool = False      # Pareto distribution of point capacities in the production store
    ie: bool = False               # infiltration excess, exponential capacity with near-surface wetness dependence
    route_conn: bool = False       # direct-runoff bypass fraction gamma * A(theta)
    depress: bool = False          # depression storage (fill and spill)
    exch_diss: bool = False        # dissipative exchange with the exterior
    exch_loss: bool = False        # GR4J exchange as a loss only, X2 in [-20, 0] instead of [-20, 10] (no effect with exch_diss, which
                                   # has X2 <= 0 already, or when the ablation removes X2)
    exch_in: bool = False          # exterior inflow J_in as its own parameter (with exch_diss)
    conn_gw: bool = False          # groundwater-head shift of the connectivity threshold
    chan_loss: bool = False        # channel transmission loss
    snow_rp: bool = False          # additive melt term rp * PET in the snowpack
    fc_min: float = 0.0            # (with ie) lower bound (mm/d) of the infiltration capacity F_c; 0 = no bound.
                                   # exp(-g w / Pbar) underflows in float32 in arid catchments under large storms and P_n / F_c then
                                   # gives a NaN gradient; the CAMELS-AUS runs of CaRT+ used 1e-4 (DATASET_DEFAULTS), GB / US none
    ablate: str = "none"           # none | no_gw | no_quick | no_uh | no_exch | single: CaRT with one component removed (no preset)
    dt: float = 1.0                # step length in days (1 = daily, 1/24 = hourly). Every parameter keeps its daily unit; the step length
                                   # enters the discrete operators only. Uniform-input scheme: the input of every step arrives at a constant
                                   # rate within the step and every store integrates its own equation over the step (kernels/exact.py), so a
                                   # daily and an hourly run approximate the same continuous model. dt = 1 is therefore not the daily model of
                                   # the paper (main branch), whose input arrives at the start of the step, and the networks of the main branch
                                   # are not meant for this scheme: train with it (cart.train) and run the trained network at any step.
    nsub: int = 4                  # symmetric (Strang) sub-steps of the quick store per step, combined with nsub / 2 sub-steps by Richardson
                                   # extrapolation (fourth order); its relaxation rate times the daily step reaches 0.3 to 0.7 in the median
                                   # catchment and 1.8 to 3.7 at the 95th percentile
    nsub_slow: int = 1             # symmetric sub-steps of the slow stores with no closed form (groundwater uptake, and the production and
                                   # depression stores with prod_lie = False); relaxation rate times the daily step about 0.01 to 0.1
    prod_lie: bool = True          # production and depression stores: wetting / drying solved exactly in the net input of the step, then the
                                   # percolation in sequence (first order); False = the symmetric splitting of kernels/exact.prod_x / pareto_x
                                   # with nsub_slow sub-steps (the daily flow differs by a median NSE of 1e-5)

    def __post_init__(self):
        assert self.dt > 0.0, self.dt
        assert self.ablate in ("none", "no_gw", "no_quick", "no_uh", "no_exch", "single"), self.ablate
        if self.route_conn:
            assert self.fast_law_der
        if self.exch_in:
            assert self.exch_diss
        if self.conn_gw:
            assert self.fast_law_der and self.gw_et
        if self.no_uptake:
            assert self.gw_et
        assert self.fc_min >= 0.0 and (self.fc_min == 0.0 or self.ie), "fc_min bounds the infiltration capacity of ie"
        if self.ablate != "none":
            refine = [k for k in ("fast_law_der", "zone", "pheno", "gw_et", "no_uptake", "top", "prod_pareto", "ie", "route_conn", "depress", "exch_diss",
                                  "exch_in", "conn_gw", "chan_loss", "snow_rp", "phys_bounds") if getattr(self, k)]
            assert not refine, f"ablations apply to the CaRT baseline only (refinements set: {refine})"


# refinement groups by physical process (switch -> value in CaRT+; absent = the ModelConfig default)
P1 = dict(top=True, prod_pareto=True, ie=True, depress=True, zone=True)
P2 = dict(fast_law_der=True, amin_lo=0.5, route_conn=True, conn_gw=True)
P3 = dict(pheno=True, kv_max=1.0, gw_et=True)
P4 = dict(exch_diss=True, exch_in=True, chan_loss=True)
P5 = dict(snow_rp=True)
GROUPS = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5}
# what replaces a removed group, besides the ModelConfig defaults of its switches
REPLACEMENT = {"P3": dict(gw_et=True, no_uptake=True),                                       # uptake kernel kept, lambda_E = 0
               "P4": dict(exch_loss=True)}                                                    # GR4J exchange restricted to losses

_PLUS = dict(phys_bounds=True, **P1, **P2, **P3, **P4, **P5)
PRESETS = {
    "cart": ModelConfig("cart", exch_loss=True),                                              # loss-only GR4J exchange
    "cartplus": ModelConfig("cartplus", **_PLUS),
}
for _i in range(1, 6):
    _g = f"P{_i}"
    _off = {k: getattr(ModelConfig(), k) for k in GROUPS[_g]}
    _off.update(REPLACEMENT.get(_g, {}))
    PRESETS[f"cartplus_no_p{_i}"] = replace(PRESETS["cartplus"], name=f"cartplus_no_p{_i}", **_off)

# settings that depend on the dataset (applied by config_for): on CAMELS-AUS the infiltration capacity of the presets with infiltration
# excess is bounded below at 1e-4 mm/d (CaRT has no infiltration excess)
DATASET_DEFAULTS = {"aus": lambda cfg: dict(fc_min=1e-4) if cfg.ie else {}}


def config_for(preset, dataset, **overrides):
    """the configuration of a preset as run on a dataset (the preset plus DATASET_DEFAULTS of the dataset, then overrides)"""
    cfg = PRESETS[preset]
    if dataset in DATASET_DEFAULTS:
        cfg = replace(cfg, **DATASET_DEFAULTS[dataset](cfg))
    return replace(cfg, **overrides) if overrides else cfg
