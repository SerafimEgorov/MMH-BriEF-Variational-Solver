# Random fluctuation fields and pre-integration

This folder contains the disordered fracture-energy fields used by the propagation solvers. A ready-to-use field GaussianUniform_f.npz is archived on Zenodo (DOI: https://doi.org/10.5281/zenodo.20056873), so the following steps are **not** required for the default demo. Use them only to generate a new realization or change the discretization.

## Scripts

- **`generate_disorder.py`** — draws a random fluctuation field $f(z, x)$ with zero mean, unit standard deviation, Gaussian two-point correlations of characteristic length $d$, and a uniform marginal distribution (non-linear mapping of a Gaussian field through the error function; based on `FyeldGenerator`).
- **`compute_integrated_fluctuations.py`** — pre-computes, on a polar grid $(r, \theta)$, the radial integral

$$F(r, \theta) = \int_0^r r'\, f(r', \theta)\, \mathrm{d}r'$$

  and its first two derivatives $\mathrm{d}F$, $\mathrm{d}^2 F$. These are read by the propagation solvers and interpolated by a quintic polynomial along $r$.

## Usage

Run the two scripts in this order, from the repository root:

```bash
# Step 1 - generate the field f (saved as fields/{NAME}_f.npz)
python fields/generate_disorder.py -f GaussianUniform -L 256 -n 16

# Step 2 - precompute F, dF, d2F for a given front discretization N
#          (saved as fields/{NAME}_F_N{N}pts.npz). Must use the same
#          N as the propagation run that will consume it.
python fields/compute_integrated_fluctuations.py -f GaussianUniform \
    -N 2048 -path ./ -spath ./
```

## Parameters — `generate_disorder.py`

- **`-f`, `--field_name`** — Basename of the field. Outputs `fields/{f}_f.npz`.
- **`-L`, `--domain_size`** — Width of the (square) heterogeneous interface, in units of the heterogeneity length $d$. Domain spans $[-L/2, L/2]$ in $z$ and $x$. Pick `L > 2 * a_max` of the intended propagation run.
- **`-n`, `--density`** — Number of grid points per heterogeneity width $d$. Total grid is $(n \cdot L) \times (n \cdot L)$. Higher $n$ means better-resolved heterogeneities but more memory.
- **`-nopopup`** — Suppress the matplotlib pop-up.

## Parameters — `compute_integrated_fluctuations.py`

- **`-f`, `--field`** — Basename of an already-generated field (`{f}_f.npz` must exist in `fields/`).
- **`-N`, `--number_of_points`** — Number of azimuthal points used by the propagation solver (must match the run-time `-N` argument; power of two recommended).
- **`-path`, `--path`** — Global path (backward compatibility).
- **`-spath`, `--save_path`** — Root path; the script reads `{spath}fields/{f}_f.npz` and writes `{spath}fields/{f}_F_N{N}pts.npz`. End with a trailing `/`.

## Output files

| File | Content |
|---|---|
| `{f}_f.npz` | `domain_size`, `number_of_points`, position (`z`, `x`), fluctuations `f`. |
| `{f}_F_N{N}pts.npz` | radius `r`, angle `theta`, integrated field `F` and its derivatives `dF`, `d2F` (`float32`, sub-sampled). |
