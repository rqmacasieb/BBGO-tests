# BBGO for PRAMS

Scripts and configuration for running the **Batch Bayesian Global Optimization (BBGO)** algorithm on the **Perth Regional Aquifer Modelling System (PRAMS)**, a regional groundwater model of the Perth metropolitan area developed by the Department of Water and Environmental Regulation (DWER) of Western Australia.

---

## Model (PRAMS) Licence Notice

PRAMS is developed and maintained by DWER and distributed under a **project-specific, non-exclusive and non-transferable licence agreement**. Key conditions include:

- **Non-transferable use** — the model may not be copied, assigned, or provided to any other person without prior written consent from DWER.
- **Publication consent required** — written approval from DWER is required before using the model, in part or in full, for publication or promotion. All outputs must acknowledge DWER as the owner of all intellectual property in the model.
- **No warranty; indemnification** — the model is used at the user's own risk. DWER does not warrant accuracy and accepts no liability for any loss or damage arising from use of the model. The user agrees to indemnify DWER against any claims arising from such use.

This repository does **not** include the PRAMS model files. Users must obtain the model independently from DWER under their own licence agreement.

- **PRAMS Model information:** https://www.wa.gov.au/government/publications/perth-regional-aquifer-modelling-system-prams
- **Contact:** groundwater.info@dwer.wa.gov.au

---

## Overview

BBGO is a surrogate-assisted single-objective optimisation framework that operates with two main workflows:

1. **Outer sweep** — runs the full PRAMS groundwater model (MODFLOW-based) to evaluate candidate pumping strategies. Results are used to train Gaussian Process (GP) emulators.
2. **Inner sweep** — optimises against the cheap GP emulators using NSGA-II (via PEST++ `pestpp-mou`), rapidly exploring the decision space at low cost.

After each pair of sweeps, intermediate processing resamples infill points from the inner Pareto front and updates the GP training data for the next iteration. This loop repeats for a configurable number of BBGO iterations (default: 20).

```
for iteration in 0..20:
    outer_sweep   →  run full PRAMS model  →  train GP emulators
    inner_sweep   →  optimise on GP emulators  →  select infill points
```

---

## Repository Structure

```
PRAMS/
├── bbgo_run.sbatch            # Main SLURM wrapper — orchestrates the full BBGO loop
├── cleanup.sbatch             # Removes intermediate worker directories
├── interproc.py               # Intermediate processing: GP training, infill resampling
├── LHS_sampler.py             # Generates LHS initial populations for outer and GP training
│
├── outer_raisemaster.sbatch   # Launches pestpp-mou master for outer sweep
├── outer_iter.sbatch          # SLURM array job for outer worker nodes
├── outer_runworker            # Shell script each outer worker node executes
│
├── inner_raisemaster.sbatch   # Launches pestpp-mou master for inner sweep
├── inner_iter.sbatch          # SLURM array job for inner worker nodes
├── inner_runworker            # Shell script each inner worker node executes
│
├── template/                  # Base PEST++ configuration (full-model run)
│   ├── forward_pbrun.py       # Forward run script: runs PRAMS model and post-processing
│   ├── model/                 # *** EMPTY — PRAMS model files go here (see below) ***
│   ├── pest/                  # PEST++ control files, template files, instruction files
│   └── pestpp-mou             # PEST++ executable
│
├── template_outer/            # Outer sweep working template (generated from template/)
│   ├── forward_pbrun.py
│   ├── model/                 # *** EMPTY — PRAMS model files go here (see below) ***
│   ├── pest/
│   └── pestpp-mou
│
├── template_inner/            # Inner sweep working template (GP emulator, no model/)
│   ├── forward_gprun.py       # Forward run script: evaluates GP emulators
│   ├── gp_model/              # GP emulator data and trained model files
│   ├── pest/
│   └── pestpp-mou
│
└── template_repo_update/      # Outer Pareto repository management template
    ├── forward_pbrun.py
    ├── model/                 # *** EMPTY — PRAMS model files go here (see below) ***
    ├── pest/
    └── pestpp-mou
```

---

## Setting Up the PRAMS Model

The `model/` directories in `template/`, `template_outer/`, and `template_repo_update/` are intentionally empty and are not included in this repository due to the PRAMS licence restrictions described above.

To use these scripts, obtain the PRAMS model from DWER and populate each `model/` directory with the required files. The expected internal structure is:

```
model/
├── preproc/
│   └── wel-decvar/
│       └── wel/
│           ├── pumpdist          # Executable: distributes pumping rates to MODFLOW WEL package
│           ├── pump.dat          # Input: per-bore pumping rates (written by forward_pbrun.py)
│           ├── pumping-bore-data.csv
│           └── prams36_opt.wel   # Output: generated WEL file for MODFLOW
├── input/
│   └── prams36_opt.wel           # Copied from preproc output
├── vfm-mf2k                      # MODFLOW executable
├── prams36.nam                   # MODFLOW name file
└── postproc/
    ├── zonebudget                 # ZoneBudget executable
    ├── zonebudget_inputs.txt      # ZoneBudget configuration
    ├── swi_postproc.py            # SWI post-processing module
    └── obj_func.py                # Objective function calculation
```

The same model directory structure is required in all three template locations. After `bbgo_run.sbatch` is launched, the framework copies these templates to worker directories automatically — you do not need to replicate them manually beyond the three `template*/model/` locations.

---

## Prerequisites

### Python packages

```
numpy
scipy
pandas
scikit-learn
pyemu
laGPy
```

Install with:
```bash
pip install numpy scipy pandas scikit-learn pyemu laGPy
```

### Other software

| Software | Purpose |
|---|---|
| `pestpp-mou` | Multi-objective optimisation master/worker (included as executable in each template/) |
| MODFLOW (`vfm-mf2k`) | Groundwater flow simulation (part of PRAMS model) |
| ZoneBudget | Zone-based water budget post-processing (part of PRAMS model) |
| SLURM | Job scheduling on HPC cluster |

---

## Workflow

### 1. Generate initial populations

Before the first run, generate the LHS training set (500 samples for GP training) and starter population (100 samples for the first outer sweep):

```bash
python LHS_sampler.py
```

This writes:
- `template/pest/gp.lhs.dv_pop.csv` — initial GP training ensemble
- `template/pest/starter.dv_pop.csv` — initial outer sweep starting population

### 2. Submit the main BBGO job

```bash
sbatch bbgo_run.sbatch
```

This script manages the full BBGO loop. For each iteration it:
1. Copies `template_outer/` to a `master/` directory and submits `outer_raisemaster.sbatch` (pestpp-mou master).
2. Submits `outer_iter.sbatch` (100 SLURM array workers) to run the full PRAMS model.
3. Waits for the outer sweep to finish, then renames `master/` to `outer_<N>/`.
4. Calls `interproc.intermediate_processing_1()` to train GP emulators on the outer sweep results.
5. Copies `template_inner/` to `master/` and submits the inner master and workers similarly.
6. After the inner sweep completes, calls `interproc.intermediate_processing_2()` to resample infill points for the next outer sweep.

### 3. Restarting

To restart from a partially completed run, set `restart=True` in `bbgo_run.sbatch` (line 11) and `interproc.py` (line 19). The framework will skip the outer sweep setup for the current iteration and resume from the inner sweep.

---

## Key Configuration Parameters

All of the following are set at the top of [interproc.py](interproc.py):

| Parameter | Default | Description |
|---|---|---|
| `nmax_outer` | 20 | Max generations per outer sweep |
| `nmax_inner` | 100 | Max generations per inner sweep |
| `num_workers` | 100 | Number of parallel workers |
| `max_infill` | 100 | Max infill points per iteration |
| `pop_size` | 100 | NSGA-II population size |
| `port` | 4020 | TCP port for pestpp-mou master/worker communication |

SLURM resource settings (memory, time limits, array size) are configured directly in the respective `.sbatch` files.

---

## Outputs

After each BBGO iteration the following directories are created in the working directory:

- `outer_<N>/` — pestpp-mou master output for outer sweep iteration N, including Pareto archive CSV files.
- `inner_<N>/` — pestpp-mou master output for inner sweep iteration N, including all-generation population CSVs and GP training data.

The outer Pareto archive (`outer_<N>/pest/*.archive.{dv,obs}_pop.csv`) contains the non-dominated pumping strategies found by that iteration's full-model evaluation.
