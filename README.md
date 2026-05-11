# Variational perturbative framework for 3D coplanar brittle fracture (MMH-BriEF-Variational-Solver)

Python implementation of a reduced-order model for the propagation of sharp crack fronts in heterogeneous, perfectly brittle media under mixed-mode I+II+III loading. Equilibrium positions of a quasi-circular crack front are obtained by minimizing the total energy

$$\Pi_{\mathrm{tot}}([a]) = \Pi_{\mathrm{pot}}([a]) + \Pi_{\mathrm{dis}}([a]),$$

where the potential energy $\Pi_{\mathrm{pot}}$ is evaluated asymptotically from front deformations using the perturbation theory of Rice (Bueckner–Rice weight-function approach), and the dissipated energy $\Pi_{\mathrm{dis}}$ is set by the heterogeneous fracture energy field $G_c(r, \theta) = \langle G_c \rangle \, (1 + \mathrm{sign} \cdot \sigma \cdot f)$. The non-convex bound-constrained minimization is performed with the PETSc TAO Bounded Newton Trust Region (BNTR) method; gradient and Hessian-vector products are evaluated matrix-free through JAX automatic differentiation and FFT.

> **Reference:** S. Egorov, A. Sanner, J. Sulem, L. Pastewka, M. Lebihain, *"Bridging perturbation and variational approaches in brittle fracture"*.

## Dependencies

Python ≥ 3.9 with: `numpy`, `scipy`, `matplotlib`, `jax` (double precision enabled), `petsc4py` (PETSc with TAO), and a working LaTeX installation (`matplotlib` is configured with `usetex=True`).

## Files

- **`propagate_tensile_rupture.py`** — Mode I crack propagation under a pair of normal point forces $P$ applied at the centre of the reference circular front. First-order expansion of the potential energy in $G$.
- **`propagate_shear_rupture.py`** — Mixed mode II+III crack propagation under a pair of antisymmetric shear forces $Q$ applied at the centre. First-order expansion in $G$.
- **`fields/`** — should contain pre-generated fluctuation field and its integrated quantities accesible through https://doi.org/10.5281/zenodo.20122852; see `fields/readme.md` for regeneration.
- **`results/`** — created at runtime; contains the `.npz`.

## Quick start

The pre-generated field `{NAME}_f.npz` and its integrated counterpart `{NAME}_F_N{N}pts.npz` must be present in `./fields/` before launching a solver (here, `NAME = GaussianUniform` and `N = 2048`). A demo fluctuation field GaussianUniform_f.npz and its integrated quantities in GaussianUniform_F_2048.npz is archived on Zenodo (DOI: https://doi.org/10.5281/zenodo.20122852). Download it into fields/ before running the solver, or regenerate it locally with generate_disorder.py (see fields/readme.md).

**Mode I (tensile, normal point force $P$):**

```bash
python propagate_tensile_rupture.py -sim test_results_tensile \
    -f GaussianUniform -a_ini 0.01 -a_max 121 -da 0.1 -tr 0.1 \
    -sigma 0.5 -sign 1 -N 2048 -path ./ -spath ./ -user_pc
```

**Mode II+III (shear, pair of antisymmetric shear forces $Q$):**

```bash
python propagate_shear_rupture.py -sim test_results_shear \
    -f GaussianUniform -a_ini 0.01 -a_max 121 -da 0.1 -tr 0.1 \
    -sigma 0.5 -nu 0.2 -sign 1 -N 2048 -path ./ -spath ./ -user_pc
```

## Command-line parameters

Common to both solvers.

### Required

- **`-sim`, `--simulation_name`** — Identifier of the run; used as the basename of all output files.
- **`-f`, `--field`** — Basename of the random field (without `_f.npz`). The solver reads `{spath}fields/{f}_f.npz` and the precomputed integrated field `{spath}fields/{f}_F_N{N}pts.npz`.
- **`-a_max`, `--max_radius`** — Maximum mean crack radius, expressed in units of the heterogeneity length $d$. Must satisfy `a_max < L/2`, where $L$ is the field domain size; otherwise the run aborts with a warning.
- **`-da`, `--step_radius`** — Increment of the prescribed mean radius $a_0$ between consecutive loading steps. The applied load is rebuilt as $P$ (or $Q$) $= \sqrt{E \langle G_c \rangle}\,(\pi a_0)^{3/2}$ so that, in the absence of disorder, the crack remains in steady-state Griffith propagation.
- **`-sigma`, `--sigma`** — Normalized standard deviation of the toughness field $\sigma = \mathrm{std}(G_c)/\langle G_c \rangle$; controls the disorder intensity. The fracture-energy field is built as $G_c = \langle G_c \rangle\,(1 + \mathrm{sign} \cdot \sigma \cdot f)$.
- **`-nu`, `--poisson_ratio`** — Poisson's ratio of the elastic medium. Effective only for the shear solver (mixed II/III split); accepted by the tensile solver for argument-list uniformity.
- **`-N`, `--number_of_points`** — Number of discretization points on the front. Must match the $N$ used to precompute the integrated field `{f}_F_N{N}pts.npz`. A power of two is strongly recommended (FFT-based operators).
- **`-path`, `--path`** — Global path; kept for backward compatibility.
- **`-spath`, `--save_path`** — Root path: `./fields/` is read from it and `./results/` is created in it. End with a trailing `/`.

### Optional

- **`-a_ini`, `--initial_radius`** — Initial circular front radius. Default `0`; the first loading step then starts at `a0[0] ~ 1e-4`. A value `> 0` imposes a lower bound on $a(\theta)$ at step 0 (active irreversibility from the start).
- **`-tr`, `--tr_radius`** — Maximum trust-region radius for BNTR, in units of $d$. Should be a fraction of the heterogeneity length scale (typically `0.05`–`0.5`) so that the optimizer cannot cross energy barriers of $\Pi_{\mathrm{tot}}$ during a single Newton step and ends up on the next physical metastable equilibrium.
- **`-sign`, `--sign`** — Sign of the fluctuation prefactor (`+1` / `-1`). With `+1` the field $f$ is used as generated; with `-1` it is mirrored, swapping tough and weak regions on the same realization. Default `+1`.
- **`-nopopup`, `--display_options`** — Suppress the matplotlib pop-up; only save the figure on disk.
- **`-user_pc`, `--preconditioner`** — Replace the default PETSc preconditioner by the physics-based matrix-free preconditioner built from the inverse Hessian of the homogeneous penny-shaped crack. Recommended for low / moderate $\sigma$ and small $\nu$; the speed-up degrades at high contrast or large $\nu$.

## Output

- Tensile → `{spath}results/{sim}.npz`
- Shear → `{spath}results/{sim}_results.npz`

**Common fields:**

| Key | Content |
|---|---|
| `completed` | `True` if `a_max` was reached |
| `number_of_points` | $N$ |
| `field_name`, `sign` | used for post-processing |
| `Young_modulus` | $E$ |
| `fracture_energy` | $\langle G_c \rangle$ (set internally to 1) |
| `disorder_intensity` | $\sigma$ |
| `force` | $P$ (tensile) or $Q$ (shear) at each step |
| `reference_position` | prescribed $a_0$ at each step |
| `front_position` | $a(\theta)$ at each step, shape `(n_steps, N)` |
| `dissipated_energy` | $\Pi_{\mathrm{dis}}$ at each step |
| `potential_energy` | $\Pi_{\mathrm{pot}}$ at each step |
| `initial_radius` | `a_ini` |
| `maximum_radius` | `a_max` |
| `radius_increment` | `da` |
| `tr_radius` | `Delta` |
| `maximum_error` | max Griffith residual per step |
| `average_error` | RMS Griffith residual per step |
| `step_duration` | wall-clock time per step |
| `iteration_counts` | TAO iteration count per step |
| `total_time` | `np.sum(step_duration)` |

**Shear only:**

| Key | Content |
|---|---|
| `Poisson_ratio` | $\nu$ |

A `{sim}.pdf` showing the field and successive front positions is also produced unless `-nopopup` is set, and saved in the same directory as the `.py` file.

## Notes

- The integrated fluctuations file `{f}_F_N{N}pts.npz` must be precomputed for the exact `N` requested at runtime (the solver cross-checks the angular grid and aborts on mismatch).
- For very large simulations consider disabling the LaTeX matplotlib backend by editing `rc('text', usetex=True)` at the top of the scripts (allowed only as a local workaround).
- The code is RAM-heavy: for the current examples, ≈ 8 GB of free RAM is needed.
- The notation is different from one used in assosiated article. In particular: i) $K_1 = K_\mathrm{I}$ ; $K_2 = K_\mathrm{II}$ ; $K_3 = - K_\mathrm{III}$; ii) Hilbert transform operator $\mathcal{S} = - \mathcal{H}$
