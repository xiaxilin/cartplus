"""Fused GPU (Triton) recursions of the CaRT stores, each with an exact adjoint (hand-written or the generic stored-state adjoint).

They need a CUDA GPU. On the hourly branch the model uses the uniform-input kernels of exact.py for the surface layer, the quick
store and the groundwater stores; top_layer, gr_route_tv, lin_store, gw_et and quad_store are the step-start kernels of the main branch.
  exact      top_x, quad_x, gwet_x, prod_x, pareto_x, quick_x, lin_x: stores solved for input at a constant rate within the step
             (closed forms, or symmetric sub-steps; quick_x and lin_x with Richardson extrapolation); step length in the per-step amounts
  exact_bwd  hand-written fused reverse sweeps of the exact.py kernels (forward-mode recomputation of each step from the stored state)
  grstores   gprod_k (GR4J production store, percolation coefficient kq), gr_route_tv (GR4J routing store with exchange), quad_store
  pareto     pareto_prod (production store with Pareto-distributed capacities; also the depression store with net=False)
  top        top_layer (surface layer in series above the production store)
  zone       zone (near-stream saturated zone in parallel with the production store)
  theta      theta_state (connectivity wetness state)
  lin_store  lin_store (quick store with time-varying release coefficient)
  gw_et      gw_et (quadratic groundwater store with groundwater evapotranspiration)
  snow_sub   snow_bands_sub (elevation-band degree-day snowpack with sublimation)            -- CaRT, CaRT+ without P5
  snow_rp    snow_bands_rada, band_factors (snow_sub + additive PET-index melt term)      -- CaRT+
  adjoint    backward_k (generic stored-state adjoint used by several of the above)
"""
