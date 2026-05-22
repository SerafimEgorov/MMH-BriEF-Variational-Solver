# Installation Guide

Total time estimated at **roughly 20-30 minutes**.

**Supported systems:** Linux, tested on Ubuntu 24.04.

> **Why Conda?**
> Some of the libraries this code relies on (PETSc, OpenMPI) are written in C/C++/Fortran,
> and are notoriously hard to install by hand. *Conda* is a tool that downloads
> pre-compiled, mutually-compatible versions of these libraries and isolates them in a named
> "environment" so they never clash with the rest of your system. *conda-forge* is simply
> the community package source we pull from. 

## Step 0 — Get the code

If you do not already have the repository on your machine, clone it with git clone or download a ZIP instead. 
Then open a terminal from your local repository, every command in this guide assumes your terminal is **inside the repository folder**.

---

## Step 1 — Install the LaTeX system packages

The plotting scripts render their text and equations with LaTeX. For now, without it, 
any script that produces a figure will fail immediately. Install LaTeX with:

```bash
sudo apt-get update
sudo apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    cm-super \
    dvipng
```

This downloads a few hundred megabytes and takes a couple of minutes. Later, we will add an option to turn it off.

---

## Step 2 — Install Conda (only if you don't have it)

First check whether Conda is already installed:

```bash
conda --version
```

- If this prints a version number (e.g. `conda 24.x.x`), **skip to Step 3.**
- If it says `command not found`, install Miniforge (a minimal, conda-forge-based Conda):

```bash
# Download the installer
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

# Run it in unattended mode, installing into your home folder
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"

# Activate Conda for your shell, permanently
"$HOME/miniforge3/bin/conda" init bash
```

Now **close your terminal and open a new one** (this is required for `conda init` to take
effect), then `cd` back into the repository folder. Verify it worked:

```bash
conda --version
```

> If you use a shell other than bash (e.g. zsh), replace `bash` with your shell name in the
> `conda init` command, e.g. `conda init zsh`.

---

## Step 3 — Create and activate the environment

This creates an isolated environment named `PyPETSc` with a fixed Python, NumPy, and SciPy:

```bash
conda create -n PyPETSc python=3.12 numpy=2.0 scipy=1.14
conda activate PyPETSc
```

> **Important — you must activate the environment every time you open a new terminal**
> before running the code. The command is always:
> ```bash
> conda activate PyPETSc
> ```
> When the environment is active, your prompt will start with `(PyPETSc)`. If you ever see
> a "module not found" error, the first thing to check is whether `(PyPETSc)` is showing.

---

## Step 4 — Install MPI and PETSc (the hard libraries)

With the environment **active**, run these three commands in order. Each may take a few
minutes; Conda will print "Solving environment..." while it works — this is normal, let it
finish.

```bash
conda install -c conda-forge openmpi openmpi-mpicc openmpi-mpicxx openmpi-mpifort
conda install -c conda-forge petsc
conda install -c conda-forge petsc4py
```

> **Do NOT install petsc4py or mpi4py with `apt` or with the system Python.** The Ubuntu
> system versions are built against an older NumPy and will crash with
> `ValueError: numpy.dtype size changed` when used with NumPy 2.0. Installing them inside
> the Conda environment, as above, avoids this entirely.


---

## Step 5 — Install mpi4py

`mpi4py` is the Python bridge to OpenMPI. It must be compiled against the OpenMPI you just
installed in Step 4, so install it with `pip` **only after** the environment is active and
Step 4 is done:

```bash
python -m pip install mpi4py
```

---

## Step 6 — Install the remaining Python packages

```bash
python -m pip install matplotlib jax FyeldGenerator
```

That is the complete set of dependencies.

---

## Step 7 — Run the demo

The solver reads a pre-generated random field from a `fields/` subfolder. A ready-made field
is archived on Zenodo:

```
DOI: https://doi.org/10.5281/zenodo.20122852
```

1. Create the folder and download `GaussianUniform_f.npz` into it:
   ```bash
   mkdir -p fields
   # then place the downloaded GaussianUniform_f.npz inside the fields/ folder
   ```
   You only need the `_f.npz` field file. The companion "integrated fluctuations" file
   (`GaussianUniform_F_N2048pts.npz`) is generated automatically by the solver on first run
   if it is absent — so downloading it is optional.

2. Run a Mode-I (tensile) propagation:
   ```bash
   python propagate_tensile_rupture.py \
       -sim demo -f GaussianUniform \
       -a_max 50 -da 0.5 -sigma 0.1 -N 2048
   ```
   Results are written to a `results/` folder created automatically and figure of propagation saved at root.

> **About the demo parameters:** `-sigma` (disorder strength) and the radii (`-a_max`, `-da`)
> are illustrative values for a quick test. `-a_max`
> **must be smaller than the field size** used to generate the field, or the script will
> stop with a warning. Read the accompanying paper and `fields/readme.md` for meaningful
> parameter choices.

To generate your own field instead of downloading one, see `fields/readme.md`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` | Conda not installed, or terminal not reopened after `conda init` | Redo Step 2; open a fresh terminal |
| `ModuleNotFoundError` for any package | Environment not active | Run `conda activate PyPETSc`; check prompt shows `(PyPETSc)` |
| `ValueError: numpy.dtype size changed` | A system/apt `petsc4py` or `mpi4py` is being picked up | Make sure `(PyPETSc)` is active; never `apt install python3-petsc4py` |
| Matplotlib error mentioning `latex` or `dvipng` | LaTeX missing or incomplete | Redo Step 1 |
| Script stops: "Maximum radius is larger than field size" | `-a_max` exceeds the field's domain size | Lower `-a_max`, or generate a larger field |
| Solver re-computes integrated field every run (slow start) | `_F_N{N}pts.npz` absent for your chosen `-N` | Let it run once; the file is cached in `fields/` afterwards |

