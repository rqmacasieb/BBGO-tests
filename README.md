# BBGO-tests

This repository contains scripts and template files for testing the Batch Bayesian Global Optimization (BBGO) algorithm on benchmark functions.

## Repository Structure

```
BBGO-tests/
├── ROS20D/          # 20-dimensional Rosenbrock benchmark
├── ROS50D/          # 50-dimensional Rosenbrock benchmark
├── LEV10D/          # 10-dimensional Levy benchmark
├── G07/             # G07 engineering benchmark
└── PRAMS/           # Perth groundwater resource allocation problem
```

Each benchmark directory (`ROS20D`, `ROS50D`, `LEV10D`, `G07`) contains:

- `bbgo.py` — main script for running the BBGO algorithm
- `LHS_sampler.py` — Latin Hypercube Sampling script for generating initial populations
- `template/` — PEST++ model template files (`.pst`, forward run scripts, instruction/template files)
- `CBGO/` — Classic BGO implementation for the same benchmark, containing `cbgo.py` and supporting scripts


For the PRAMS application problem, refer to the PRAMS/README for more information.

## Configuration

### BBGO (`bbgo.py`)

User-configurable settings are at the top of each benchmark's `bbgo.py`:

| Setting | Description | Options |
|---------|-------------|---------|
| `CASE` | Optimization objective for the inner loop | `"MinVar"` or `"MaxVar"` |
| `pop_size` | Population size for the inner optimization | Integer (e.g., `100`) |
| `nmax_outer` | Maximum number of outer iterations | Integer |
| `nmax_inner` | Maximum number of inner iterations | Integer |

### Classic BGO (`CBGO/cbgo.py`)

| Setting | Description |
|---------|-------------|
| `n_initial_points` | Number of initial random samples (initial training dataset) before BGO iterations |
| `bgo_iter` | Number of BGO iterations |

## Usage

Run BBGO from a benchmark directory:

```bash
python bbgo.py --output-dir <path/to/output> --seed <integer>
```

Run Classic BGO from the `CBGO/` subdirectory:

```bash
python cbgo.py --output-dir <path/to/output> --seed <integer>
```
