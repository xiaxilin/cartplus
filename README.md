# CaRT / CaRT+

Code for the paper *From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three
continents* (Xia, 2026, [arXiv:2611.06382](https://arxiv.org/abs/2611.06382)). The repository contains the regional conceptual models CaRT and CaRT+ (lumped daily
catchment models whose per-catchment parameters come from a network of static catchment attributes), CaRT+ without each of its five
process groups of refinements, the four benchmarks (regional HBV, dHBV, a stateful LSTM and the EA-LSTM), the scripts that build the
CAMELS-US and CAMELS-AUS input files from public sources, the evaluation scripts that write the paper's tables and figures, and the
lists of catchments and runs that define the experiments on CAMELS-GB, CAMELS-US and CAMELS-AUS.

```
configs/        paper_runs.csv (the 84 training runs), paper_forward.csv (the 42 forward runs of the parameter-space ensemble),
                train_all.sh (all of them), hbv9_init_medians.json (initial values of the regional HBV)
data/           model input files (downloaded with scripts/data/fetch_data.py; see DATA.md)
data_lists/     catchment order and roles, held-out catchments of each dataset
environment/    pinned package versions
scripts/data/   fetch_data.py; us/ and aus/: build the input files from the public sources
scripts/eval/   tables and figures of the paper
src/cart/       CaRT / CaRT+: config.py (presets), data.py (datasets, periods), parameters.py (mapping network), model.py,
                train.py, predict.py (forward runs, parameter-space ensemble), uh.py, kernels/ (Triton GPU kernels)
src/baselines/  hbv_regional.py, dhbv.py, lstm_stateful.py, ealstm.py, kernels/
tests/          checks of the presets, input files, kernels and water balance (run_all.sh)
```

## Installation

A CUDA GPU is required: the stores of CaRT, HBV and dHBV are Triton GPU kernels. The paper's runs used Python 3.11, PyTorch 2.7.1
(CUDA 12.6) and Triton 3.3.1 on NVIDIA L40S GPUs.

```
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r environment/requirements-model.txt
```

`environment/requirements-dataext.txt` lists the extra packages needed only to rebuild the CAMELS-US input files from the raw sources
and to prepare the map geometry (`scripts/eval/prep_map_geometry.py`). Full freezes of both environments are in `environment/`.
Simulations are reproducible to the bit for a given seed on the same GPU type and library versions.

## Data

The CAMELS-US, CAMELS-AUS and CAMELS-GB input files are published as assets of the GitHub release. Download them (with a SHA256 check of each file) with

```
python scripts/data/fetch_data.py
```

or download them by hand from the release page into `data/`:

| file | content |
|---|---|
| `camels_us_daymet671_v4full.npz` | CAMELS-US, 671 basins, 1980-2019: Daymet v4 forcing, Priestley-Taylor PET, discharge, 27 attributes |
| `elev_bands671.npz` | CAMELS-US elevation bands |
| `camels_aus_v2.npz` | CAMELS-AUS v2, 561 catchments, 1980-2019: AGCD precipitation and temperature, SILO Morton PET, discharge, 27 attributes |
| `elev_bands_aus.npz` | CAMELS-AUS elevation bands |
| `camels_gb_cart.npz` | CAMELS-GB, 671 catchments, 1988-2019: HadUK-Grid precipitation, CAMELS-GB temperature, PET and discharge, 21 attributes |

`DATA.md` gives the sources, licences and variables of the released files; `scripts/data/README.md` describes how to
rebuild them.

The loaders read `data/` and `data_lists/` relative to the repository. Other locations can be set with environment variables
(`CART_DATA` for the folder; `CART_GB_NPZ`, `CAMELS_US_NPZ671`, `CART_US_BANDS`, `CAMELS_US_UNGAUGED`, `CART_AUS_NPZ`,
`CART_AUS_BANDS`, `CART_AUS_UNGAUGED` for single files; see `src/cart/data.py`).

## Training

Run from the repository root with `export PYTHONPATH=$PWD/src` (or from `src/` directly). `--dataset` is `gb`, `us` or `aus`;
`--gpu` selects the CUDA device.

```
export PYTHONPATH=$PWD/src
python -m cart.train --model cartplus --dataset us --seed 0                  # CaRT+
python -m cart.train --model cart --dataset us --seed 0                      # CaRT
python -m cart.train --model cartplus_no_p3 --dataset us --seed 0            # CaRT+ without process group P3
python -m baselines.hbv_regional --dataset us --seed 0                       # regional HBV
python -m baselines.dhbv --dataset us --seed 0                               # dHBV
python -m baselines.lstm_stateful --dataset us --seed 0                      # stateful LSTM
python -m baselines.ealstm --dataset us --seed 0                             # EA-LSTM
```

| preset | model |
|---|---|
| `cart` | CaRT: 5-band snowpack with sublimation, GR4J-type production store, GR4J unit hydrographs, routing store with loss-only exchange, quadratic groundwater store |
| `cartplus` | CaRT+ = CaRT + P1..P5, with the physical bounds of the derived release |
| `cartplus_no_p1` | CaRT+ without P1, runoff generation (surface layer, Pareto partition, infiltration excess, depression storage, near-stream zone) |
| `cartplus_no_p2` | CaRT+ without P2, quick release (connected release, connected bypass, groundwater-controlled connectivity) |
| `cartplus_no_p3` | CaRT+ without P3, evaporation (vegetation demand, groundwater uptake) |
| `cartplus_no_p4` | CaRT+ without P4, exchange with the exterior (exterior head, exterior inflow, channel loss) |
| `cartplus_no_p5` | CaRT+ without P5, snow (melt energy term) |

Each run writes `runs/<dataset>_<model>_s<seed>/` (`--out` sets another folder) with `sim.npy`, the simulated discharge as float32
[days, catchments] in mm/d over the whole record for all catchments including the held-out ones, the trained weights and `run.json`.

### Forward runs and the parameter-space ensemble

```
python -m cart.predict --run runs/us_cartplus_s0                         # forward run -> sim_forward.npy
python -m cart.predict --run runs/us_cartplus_s0 --noise 0.3 --draws 32  # parameter-space ensemble -> sim_noise0.3.npy
```

With `--noise SIGMA` the standardised attributes of every catchment are perturbed with Gaussian noise of standard deviation SIGMA,
the model is run once per draw and the flows are averaged. The paper uses SIGMA 0.1, 0.2, 0.3, 0.5, 0.7 and 1.0 on GB and US and 0.3
and 0.5 on AUS (`configs/paper_forward.csv`).

## Hourly branch: a time-step invariant scheme

This branch adds a step length to every operator (`ModelConfig.dt`, in days; every parameter keeps its daily unit) and replaces the
step-start scheme of the published daily models by a uniform-input scheme. The input of a step arrives at a constant rate within
the step, and every store integrates its own equation over the step:

| operator | solution over a step | order |
|---|---|---|
| unit hydrographs | rate-form ordinates (`uh.ordinates_rate`), X4 down to one hour | exact |
| surface layer (CaRT+) | closed form, piecewise linear (`kernels/exact.top_x`) | exact |
| groundwater store (CaRT) | Riccati solution (`exact.quad_x`) | exact |
| groundwater store with uptake (CaRT+) | one symmetric sub-step (`exact.gwet_x`) | 2 |
| quick store | 4 symmetric sub-steps with Richardson extrapolation (`exact.quick_x`, `exact.lin_x`) | 4 |
| production store | wetting and drying exact in the net input, then percolation | 1 in the percolation |
| near-stream zone, hillslope wetness | Riccati and Bernoulli solutions | exact |

The order of each store follows its relaxation rate at the daily step: high for the quick store, first order for percolation and
depression storage, whose rates are about 0.01 to 0.02 per day. The new kernels have hand-written adjoints (`kernels/exact_bwd.py`).
A daily and an hourly run of the same network then agree, so a network trained on daily flows runs at an hourly step as trained.
The networks of the main branch belong to the step-start scheme; train new ones on this branch with `cart.train` as above
(daily data, `dt = 1`).

To run a trained network hourly, build a `cart.data.Dataset` whose forcing and discharge are amounts per hour (mm/h) on an hourly
`days` list (each step carries its date). Its `train` and `val` periods set the training-mean recharge of the groundwater store and
the end of the default simulation. Then pass `dt = 1/24` to the configuration:

```python
import torch
from cart.config import config_for
from cart.model import CaRT
cfg = config_for("cartplus", "gb", dt=1 / 24)
model = CaRT(ds_hourly, cfg, "cuda:0"); model.net.load_state_dict(torch.load("runs/gb_cartplus_s0/net.pt"))
Q = model.simulate(TT=model.T)[0]                   # [catchments, hours], mm/h
```

`tests/check_exact.py` checks the new kernels and their adjoints at `dt = 1` and `1/24`. `tests/check_dt.py` checks the unit
hydrographs, the hourly water balance and the agreement of an hourly and a daily run on evenly spread forcing. The first run
compiles the Triton kernels, the backward of the quick store of CaRT+ in about 13 minutes; Triton caches them afterwards.

## Citation

Xia, X. (2026). From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three
continents. arXiv:2611.06382. https://doi.org/10.48550/arXiv.2611.06382

```bibtex
@misc{xia2026catchmentlaws,
  author        = {Xia, Xilin},
  title         = {From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three continents},
  year          = {2026},
  eprint        = {2611.06382},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2611.06382},
  url           = {https://arxiv.org/abs/2611.06382}
}
```

Please also cite the datasets used (`DATA.md`).

## Licence

The code is released under the MIT licence (`LICENSE`). The released input files are derived from CAMELS-US, Daymet, USGS NWIS,
CAMELS-AUS, CAMELS-GB and HadUK-Grid. The US and AUS files are distributed under CC BY 4.0 and the GB file under the Open Government
Licence v3.0, with the attributions given in `DATA.md`.
