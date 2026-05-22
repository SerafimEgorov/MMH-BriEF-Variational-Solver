# Installation Guide

Total time estimated at **roughly 20-30 minutes**.

**Supported systems:** Linux (tested on Ubuntu 24.04) and macOS on Apple Silicon
(M1/M2/M3/M4). The two paths differ only in **Step 1** (LaTeX) and the installer name in
**Step 2**; everything from Step 3 onward is identical, because Conda provides the same
packages on both systems.

> macOS support relies on the same conda-forge packages as Linux but has had less testing
> than the Ubuntu path. If you hit a problem specific to macOS, see the Troubleshooting
> section.

> **Why Conda?**
> Some of the libraries this code relies on (PETSc, OpenMPI) are written in C/C++/Fortran,
> and are notoriously hard to install by hand. *Conda* is a tool that downloads
> pre-compiled, mutually-compatible versions of these libraries and isolates them in a named
> "environment" so they never clash with the rest of your system. *conda-forge* is simply
> the community package source we pull from.

## Step 0 — Get the code

If you do not already have the repository on your machine, clone it with git clone or
download a ZIP instead. Then open a terminal from your local repository; every command in
this guide assumes your terminal is **inside the repository folder**.

---

## Step 1 — Install the LaTeX system packages

The plotting scripts render their text and equations with LaTeX. Without it, any script that
produces a figure will fail immediately.

### On Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    cm-super \
    dvipng
```

### On macOS

macOS has no built-in package manager, so first install [Homebrew](https://brew.sh) if you
do not already have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install MacTeX, which bundles LaTeX, the required fonts, and `dvipng`:

```bash
brew install --cask mactex-no-gui
```

After it finishes, open a new terminal (so the LaTeX binaries are on your `PATH`).

> Either way this downloads a few hundred megabytes (MacTeX is larger, around 2 GB for the
> `no-gui` variant) and takes a few minutes.

---

## Step 2 — Install Conda (only if you don't have it)

First check whether Conda is already installed:

```bash
conda --version
```

- If this prints a version number (e.g. `conda 24.x.x`), **skip to Step 3.**
- If it says `command not found`, install Miniforge (a minimal, conda-forge-based Conda).
  **Pick the command matching your system:**

### On Linux

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
```

### On macOS (Apple Silicon)

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init zsh
```

> macOS uses **zsh** by default, which is why the macOS command ends in `conda init zsh`.
> The Linux command assumes **bash**. If your shell differs, substitute its name (e.g.
> `conda init bash` on a Mac configured for bash).

Now **close your terminal and open a new one** (this is required for `conda init` to take
effect), then return to the repository folder. Verify it worked:

```bash
conda --version
```

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

> **Do NOT install petsc4py or mpi4py with your system package manager (`apt`, Homebrew) or
> with the system Python.** Those builds are compiled against a different NumPy and will
> crash with `ValueError: numpy.dtype size changed` when used with the NumPy 2.0 here.
> Installing them inside the Conda environment, as above, avoids this entirely.

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

That is the complete set of dependencies. On Apple Silicon, `pip` installs the CPU build of
JAX, which is what these scripts use.

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
   Results are written to a `results/` folder created automatically, and a figure of the
   propagation is saved at the repository root (`demo.pdf`).

> **About the demo parameters:** `-sigma` (disorder strength) and the radii (`-a_max`, `-da`)
> are illustrative values for a quick test. `-a_max` **must be smaller than the field size**
> used to generate the field, or the script will stop with a warning. Read the accompanying
> paper and `fields/readme.md` for meaningful parameter choices.

To generate your own field instead of downloading one, see `fields/readme.md`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` | Conda not installed, or terminal not reopened after `conda init` | Redo Step 2; open a fresh terminal |
| `ModuleNotFoundError` for any package | Environment not active | Run `conda activate PyPETSc`; check prompt shows `(PyPETSc)` |
| `ValueError: numpy.dtype size changed` | A system `petsc4py` or `mpi4py` (from `apt` or Homebrew) is being picked up | Make sure `(PyPETSc)` is active; never install these outside Conda |
| Matplotlib error mentioning `latex` or `dvipng` | LaTeX missing or incomplete; on macOS, terminal opened before MacTeX finished | Redo Step 1; open a fresh terminal so the LaTeX binaries are on `PATH` |
| `brew: command not found` (macOS) | Homebrew not installed or not on `PATH` | Install Homebrew (Step 1) and follow its post-install `PATH` instructions |
| Script stops: "Maximum radius is larger than field size" | `-a_max` exceeds the field's domain size | Lower `-a_max`, or generate a larger field |
| Solver re-computes integrated field every run (slow start) | `_F_N{N}pts.npz` absent for your chosen `-N` | Let it run once; the file is cached in `fields/` afterwards |
