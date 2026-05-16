"""
 @file   propagate_rupture.py
 @authors Mathias Lebihain <mathias.lebihain@enpc.fr>
 @date   Mon 15 Jul 2024
 @brief  python script to model crack propagation along strongly heterogeneous interfaces at first order in G
         using JAX for just-in-time compilation and automatic differentiation
         and PETSc/TAO for optimization with bounded trust-region Newton conjugate gradient method
"""
# Parser
import sys
import argparse
# Numpy/Scipy
import numpy as np
# JAX
import jax as jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import jit
from functools import partial
# PETSc
import petsc4py
from petsc4py import PETSc
# Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Timer
import time
# File management
import os
import subprocess

# LateX font
from matplotlib import rc
rc('text', usetex=True)

jax.config.update("jax_enable_x64", True)  # use double-precision

#############################################################
# Simulation parameters

# Get simulation parameters from args
parser = argparse.ArgumentParser(description='Simulate crack propagation along a given heterogeneous interface')
parser.add_argument('-sim', '--simulation_name', type=str,
                    help=('Name of the simulation'), required=True)
parser.add_argument('-f', '--field', type=str,
                    help=('Name of the random field'), required=True)
parser.add_argument('-a_ini', '--initial_radius', type=float,
                    help=('Initial radius'), default=0, nargs='?')
parser.add_argument('-a_max', '--max_radius', type=float,
                    help=('Maximum computed radius'), required=True)
parser.add_argument('-da', '--step_radius', type=float,
                    help=('Step for crack advance'), required=True)
parser.add_argument('-tr', '--tr_radius', type=float,
                    help=('Maximum radius of the trust region'), default=0.5, nargs='?')
parser.add_argument('-sigma', '--sigma', type=float,
                    help=('Normalized field standard deviation'), required=True)
parser.add_argument('-nu', '--poisson_ratio', type=float,
                    help=('Poisson ratio for the simulation'), required=False)
parser.add_argument('-sign', '--sign', type=float,
                    help=('Sign factor'), default=+1, nargs='?')
parser.add_argument('-N', '--number_of_points', type=int,
                    help=('Number of points along the crack front'), required=True)
parser.add_argument('-fpath', '--fields_path', type=str,
                    help=('Path to the folder containing fields'), default='./fields/', nargs='?')  
parser.add_argument('-spath', '--save_path', type=str,
                    help=('Path to the results'), required=True)
parser.add_argument('-nopopup', '--display_options', action='store_true',
                    help=('Display options'), required=False)
parser.add_argument('-user_pc', '--preconditioner', action='store_true',
                    help=('Use a user-defined preconditioner'), required=False)
args = parser.parse_args()

# Create save folder
if not(os.path.isdir(args.save_path+'/results')) :
    os.mkdir(args.save_path+'/results')

# Get path
save_path = args.save_path
# Get field name
field_name = args.field
# Get the number of discretization points along the crack front
N = args.number_of_points
# Set average toughness Gc_0
Gc_0 = 1.
# Set Young's modulus 
E = 1.
# Get the normalized standard deviation of the toughness field std(Gc)/E(Gc)
sigma = args.sigma
# Get the sign of the fluctuation prefactor
sign = np.sign(args.sign)
# Get initial radius
a_ini = args.initial_radius
# Get maximum radius
a_max = args.max_radius
# Get radius increment
da = args.step_radius
# Get trust region radius
Delta = args.tr_radius

# Set simulation name
simulation_name = args.simulation_name
print("Launch {:s}.".format(simulation_name))

########################################################################
# Reference axisymmetric problem

# Potential energy
Pi_pot_0 = lambda a, P : 2*np.pi * (P**2 /(E*jnp.pi**3))/ a 
# G and its derivatives
G0 = lambda a, P : (P**2 /(E*jnp.pi**3)) / a**3
G0_prime = lambda a, P : -3 * (P**2 /(E*jnp.pi**3)) / a**4
# aG and its derivatives
aG0 = lambda a, P : a * G0(a, P)
aG0_prime = lambda a, P : G0(a, P) + a * G0_prime(a, P)

########################################################################
# Pre-computed quantities for the front position

# FrontUtils class
class FrontUtils():
    def __init__(self, n):
        # Number of points
        self.N = n
        # Angle
        self.theta = jnp.linspace(0, 2*jnp.pi, num=n, endpoint=False, dtype='float64')
        # Wavenumber
        self.k = jnp.arange(n//2+1, dtype='float64')
        # Mask for not-zero frequency
        self.not_zeroFreq = (self.k != 0) * 1.
        # Mask for Nyquist frequency
        self.is_Nyquist = (self.k == n//2) * 1.
        # Vector to compute DFT at the Nyquist frequency
        self.v_Nyquist = (1/n) * jnp.exp(-1j*(n//2)*self.theta).real
        
        # Padded quantities
        self.N_pad = 3*n//2
        self.k_pad = jnp.arange(self.N_pad//2+1, dtype='float64')
        self.is_Nyquist_pad = (self.k_pad == n//2) * 1.
        # Super padded quantities
        self.N_superpad = 2*n
        self.k_superpad = jnp.arange(self.N_superpad//2+1, dtype='float64')
        self.is_Nyquist_superpad = (self.k_superpad == n//2) * 1.

########################################################################
# Pre-computed quantities for the fracture energy field

# Field class
class FieldData:
    def __init__(self, fname, average_frac_ener, sign_prefactor, amplitude):
        # Set field name
        self.name = fname
        # Get field
        field_data = np.load(args.fields_path + fname + '_f.npz')
        # Get width of the heterogeneous interface
        self.L = field_data['domain_size']
        # Set average fracture energy
        self.Gc_0 = average_frac_ener
        # Set disorder intensiy
        self.sigma = amplitude
        # Set sign
        self.sign = sign_prefactor
        # Define position
        self.z, self.x = None, None
        # Define fluctuations field f
        self.f = None
        # Define fracture energy field
        self.Gc = None
    
    def load_field(self):
        # Get field
        field_data = np.load(args.fields_path + self.name + '_f.npz')
        # Set position
        self.z, self.x = field_data['position']
        # Set fluctuations field f
        f = field_data['fluctuations']
        # Set fracture energy field
        self.Gc = self.Gc_0 * (1 + self.sign *  self.sigma * f)

# Interpolator class
class FieldInterpolator():
    def __init__(self, fdata, futils):
        # Generate data if not existing
        if not(os.path.isfile(args.fields_path + '/{:s}_F_N{:d}pts.npz'.format(fdata.name, futils.N))):
            print("Precomputing the integrated field F and its derivatives for N={:d}.".format(futils.N))
            subprocess.call('python {:s}/compute_integrated_fluctuations.py -f {:s} -N {:d} -fpath {:s} -spath {:s}'.format(args.fields_path,
    fdata.name, futils.N, args.fields_path , args.fields_path ), shell=True)
        # Load data
        data = np.load(args.fields_path + '/{:s}_F_N{:d}pts.npz'.format(fdata.name, futils.N))
        # Radius
        r = data['radius']
        # Angle
        theta = data['angle']
        # Integrated fluctuations F
        F = fdata.sign * data['F'].astype('float64')
        # Its first and second derivatives
        dF = fdata.sign * data['dF'].astype('float64')
        d2F = fdata.sign * data['d2F'].astype('float64')
        # Step size
        dr = np.mean(np.diff(r))
        # Maximum radius
        r_max = r.max()
        # Linear interpolation of the field after r_max
        r = np.append(r, [r_max+dr, 2*r_max])
        F = np.vstack((F, F[-1,:] + dF[-1,:]*dr, F[-1,:] + dF[-1,:]*r_max))
        dF = np.vstack((dF, dF[-1,:], dF[-1,:]))
        d2F = np.vstack((d2F, np.zeros(theta.size), np.zeros(theta.size)))
        # Save to JAX array
        self.r = jnp.asarray(r)
        self.theta = jnp.asarray(theta)
        self.F = jnp.asarray(F)
        self.dF = jnp.asarray(dF)
        self.d2F = jnp.asarray(d2F)
    
    @partial(jit, static_argnums=0)
    def F_ev(self, a_i):
        # Get interval
        indices = jnp.digitize(a_i, self.r, right=False) - 1
        i = jnp.arange(a_i.size)
        # Interval limits
        lb_i = self.r[indices]
        ub_i = self.r[indices+1]
        # Step size
        Dx_i = ub_i - lb_i
        # Reduced position
        x_i = a_i - lb_i
        # Field and its derivatives at the limits
        Fl_i = self.F[indices, i]
        Fu_i = self.F[indices+1, i]
        dFl_i = self.dF[indices, i]
        dFu_i = self.dF[indices+1, i]
        d2Fl_i = self.d2F[indices, i]
        d2Fu_i = self.d2F[indices+1, i]
        # Residuals
        res1 = Fu_i - Fl_i - dFl_i * Dx_i - d2Fl_i/2 * Dx_i**2
        res2 = dFu_i - dFl_i - d2Fl_i * Dx_i
        res3 = d2Fu_i - d2Fl_i
        # Coefficients of the polynomials interpolation
        C0 = Fl_i
        C1 = dFl_i
        C2 = d2Fl_i/2
        C3 =  10/Dx_i**3 * res1 - 4/Dx_i**2 * res2 + 1/2/Dx_i    * res3
        C4 = -15/Dx_i**4 * res1 + 7/Dx_i**3 * res2 -   1/Dx_i**2 * res3
        C5 =   6/Dx_i**5 * res1 - 3/Dx_i**4 * res2 + 1/2/Dx_i**3 * res3
        # Set quintic interpolation
        F_i = C5*x_i**5 + C4*x_i**4 + C3*x_i**3 \
            + C2*x_i**2 + C1*x_i    + C0
        
        return F_i

########################################################################
# Functions related to the potential energy

# Compute objective
@partial(jit, static_argnums=(2,))
def obj_potential_energy(a_i, P, futils):
    #Number of points
    N = futils.N
    # Wavenumber
    k = futils.k
    # Average radius
    hat_a_0 = np.mean(a_i)
    # Perturbation
    da_i = a_i - hat_a_0
    dft_N_da = jfft.rfft(da_i, norm="forward")
    # Fractional laplacian
    dft_N_Lda = - jnp.abs(k) * dft_N_da # Note: (-|k|) Fourier convention for L; the sign is reabsorbed in the Pi_pot expression below (cf. l. 289). Shear solver uses the standard (+|k|) convention.
    Lda_i = jfft.irfft(dft_N_Lda, norm="forward")
    
    # Dealiased averaged quantities
    hat_da2_0 = jnp.mean(da_i**2) - 0.5 * dft_N_da[N//2].real**2
    hat_daLda_0 = jnp.mean(da_i * Lda_i) + 0.5 *  N//2 * dft_N_da[N//2].real**2
    
    # Objective
    obj_Pi_pot =  Pi_pot_0(hat_a_0, P) \
                - jnp.pi   * aG0_prime(hat_a_0, P) * hat_da2_0 \
                - jnp.pi   * G0(hat_a_0, P) * hat_daLda_0 # Note: The minus sign here compensates the (-|k|) convention used for L_code on l. 279, so that the result matches +π  G0  <Δa·L[Δa]> as in Eq. (17) of the article.
    
    return obj_Pi_pot

# Compute gradient
grad_potential_energy = jax.jit(jax.grad(obj_potential_energy, argnums=0), static_argnums=(2,))

# Compute Hessian vector product
@partial(jit, static_argnums=(3,))
def hessp_potential_energy(a_i, v_i, P, futils):
    return jax.jvp(lambda x : grad_potential_energy(x, P, futils), (a_i,), (v_i,))[1]

########################################################################
# Functions related to the dissipated energy

# Compute objective
@partial(jit, static_argnums=(1,2))
def obj_dissipated_energy(a_i, futils, finterp):
    # Vector to compute DFT at the Nyquist frequency
    v_Nyquist_i = futils.v_Nyquist
    # F function and its derivatives
    F_i = finterp.F_ev(a_i)
    
    # Fast Fourier transform
    dft_N_a_Nyquist = jnp.sum(v_Nyquist_i * a_i)
    # Dealiased averaged quantities
    hat_a2_0 = jnp.mean(a_i**2) - 0.5*dft_N_a_Nyquist**2
    
    # Objective
    obj_Pi_dis =   jnp.pi * Gc_0 * hat_a2_0 \
               + 2*jnp.pi * Gc_0 * sigma * jnp.mean(F_i)
    
    return obj_Pi_dis

# Compute gradient
grad_dissipated_energy = jax.jit(jax.grad(obj_dissipated_energy, argnums=0), static_argnums=(1,2))
# Compute Hessian vector product
@partial(jit, static_argnums=(2,3))
def hessp_dissipated_energy(a_i, v_i, futils, finterp):
    return jax.jvp(lambda x : grad_dissipated_energy(x, futils, finterp), (a_i,), (v_i,))[1]

########################################################################
# Functions related to the total energy

# Compute objective
@partial(jit, static_argnums=(2,3))
def obj_total_energy(a_i, P, futils, finterp):
    # Potential energy
    obj_Pi_pot = obj_potential_energy(a_i, P, futils)
    # Dissipated energy
    obj_Pi_dis = obj_dissipated_energy(a_i, futils, finterp)
    
    return obj_Pi_pot + obj_Pi_dis

# Compute gradient
@partial(jit, static_argnums=(2,3))
def grad_total_energy(a_i, P, futils, finterp):
    # Gradient of the potential energy
    grad_Pi_pot = grad_potential_energy(a_i, P, futils)
    # Gradient of the dissipated energy
    grad_Pi_dis = grad_dissipated_energy(a_i, futils, finterp)
    
    return grad_Pi_pot + grad_Pi_dis

# Compute hessian matrix product
@partial(jit, static_argnums=(3,4))
def hessp_total_energy(a_i, v_i, P, futils, finterp):
    # Hessian product of the potential energy
    hessp_Pi_pot = hessp_potential_energy(a_i, v_i, P, futils)
    # Hessian product of the dissipated energy
    hessp_Pi_dis = hessp_dissipated_energy(a_i, v_i, futils, finterp)
        
    return hessp_Pi_pot + hessp_Pi_dis

########################################################################
# Preconditionner
# Note: This is based on the matrix-vector product
#       between the inverse of the Hessian of the penny-shaped crack
#       and a random vector

@partial(jit, static_argnums=(3,))
def invH0_product(a_i, v_i, P, futils):
    #Number of points
    N = futils.N
    # Wavenumber
    k = futils.k
    # Average radius
    hat_a_0 = jnp.mean(a_i)
    # Fast Fourier transform
    dft_N_v = jfft.rfft(v_i, norm="forward")
    # Diagonal of the zero-order Hessian inverse in Fourier space
    kernel = hat_a_0 * G0_prime(hat_a_0, P) - G0(hat_a_0, P) * jnp.abs(k)  
    # Compute matrix-free vector product in the Fourier space
    dft_N_Pv = - N/(2*jnp.pi) / kernel * dft_N_v
    # Back in real space
    Pv_i = jfft.irfft(dft_N_Pv, norm="forward")
    
    return Pv_i

########################################################################
# Define class for PETSc

class FractureProblem :
    def __init__(self, P, futils, finterp):
        self.load = P
        self.step = 0
        self.futils = futils
        self.finterp = finterp
        self.solver = PETSc.TAO().create()
        self.is_inactive = np.zeros(futils.N, dtype='bool')
    
    def update_load(self, P):
        self.load  = P
    
    def incr_step(self):
        self.step += 1
    
    def formObjective(self, tao, x):
        # Get loading
        P = self.load
        # Get front utilities
        futils = self.futils
        # Get field interpolator
        finterp = self.finterp
        # Get front position
        a_i = x.getArray(readonly=True)
        # Compute objective
        obj_Pi_tot = obj_total_energy(a_i, P, futils, finterp)
        
        return obj_Pi_tot
    
    def formGradient(self, tao, x, g):
        # Get loading
        P = self.load
        # Get front utilities
        futils = self.futils
        # Get array size
        N = futils.N
        # Get field interpolator
        finterp = self.finterp
        # Get front position
        a_i = x.getArray(readonly=True)
        # Compute gradient
        grad_Pi_tot = np.asarray(grad_total_energy(a_i, P, futils, finterp))
        # Set gradient
        g.setValues(range(N), grad_Pi_tot)
        # Set inactive set
        lb, _ = self.solver.getVariableBounds()
        lb_i = lb.getArray(readonly=True)
        self.is_inactive = (lb_i < a_i) | ((a_i == lb_i) & (grad_Pi_tot < 0))
    
    def formObjGrad(self, tao, x, g):
        # Get loading
        P = self.load
        # Get front utilities
        futils = self.futils
        # Get array size
        N = futils.N
        # Get field interpolator
        finterp = self.finterp
        # Get front position
        a_i = x.getArray(readonly=True)
        # Compute objective
        obj_Pi_tot = obj_total_energy(a_i, P, futils, finterp)
        # Compute gradient
        grad_Pi_tot = np.asarray(grad_total_energy(a_i, P, futils, finterp))
        # Set gradient
        g.setValues(range(N), grad_Pi_tot)
        # Set inactive set
        lb, _ = self.solver.getVariableBounds()
        lb_i = lb.getArray(readonly=True)
        self.is_inactive = (lb_i < a_i) | ((a_i == lb_i) & (grad_Pi_tot < 0))
        
        return obj_Pi_tot
    
    def formHessian(self, tao, x, Hess, HessP):
        pass
    
    def formHessianProduct(self, v, Hv):
        # Get loading
        P = self.load
        # Get front utilities
        futils = self.futils
        # Get array size
        N = futils.N
        # Get field interpolator
        finterp = self.finterp
        # Get solver
        tao = self.solver
        # Get front position
        x = tao.getSolution()
        a_i = x.getArray(readonly=True)
        # Compute gradient
        v_i = v.getArray(readonly=True)
        hessp_Pi_tot = np.asarray(hessp_total_energy(a_i, v_i, P, futils, finterp))
        # Set Hessian vector product
        Hv.setValues(range(N), hessp_Pi_tot)
    
    def applyPreconditioner(self, v, Pv):
        # could be implemented separately and jitted        
        v_i = v.getArray(readonly=True)
        # Get solver
        tao = self.solver
        # Get loading
        P = self.load
        # Get front utilities
        futils = self.futils
        # Get array size
        N = futils.N
        # Get front position
        a_i = tao.getSolution().getArray(readonly=True)
        # Check if the vector space has been reduced
        if (v_i.size != N):
            # Get indices of inactive points
            indices = np.nonzero(self.is_inactive)
            # New vector, with v_i = 0 at active points
            v_new = np.zeros(N)
            v_new[indices] = v_i
            # Apply preconditioner
            Pv_new = invH0_product(a_i, v_new, P, futils)
            # Reduce space to inactive points
            Pv_i = Pv_new[self.is_inactive]
            # Set values
            Pv.setValues(range(v_i.size), Pv_i)
        else:
            # Apply preconditioner
            Pv_i = invH0_product(a_i, v_i, P, futils)
            # Set values
            Pv.setValues(range(N), Pv_i)
    
    def mult(self, A, x, y):
        "y <- A * x"
        self.formHessianProduct(x, y)
    
    def apply(self, P, x, y):
        "y <- P * x"
        self.applyPreconditioner(x, y)

########################################################################
# Initiate FractureProblem and set PETSc solvers

# Set front
front = FrontUtils(N)
# Set field
field = FieldData(field_name, Gc_0, sign, sigma)
if (a_max > field.L):
    print("Warning: Maximum radius is larger than field size. Please reduce it or generate a larger field.")
    exit()
# Set interpolator for integrated fluctuations F
interpolator = FieldInterpolator(field, front)
if (not np.all(np.isclose(interpolator.theta, front.theta))):
    print("The integrated fluctuations must be run again with the right number of points.")
    print(interpolator.theta, front.theta)
    exit()
# Create fracture problem
problem = FractureProblem(1, front, interpolator)
# Create solution vector
x = PETSc.Vec().createSeq(N)
a_computed = x.getArray(readonly=True)
# Create gradient vector
g = PETSc.Vec().createSeq(N)
grad_Pi_tot = g.getArray(readonly=True)
# Create Hessian matrix
H = PETSc.Mat().create()
H.setSizes([N,N])
H.setType('python')
H.setPythonContext(problem)
H.setUp()

# Create TAO Solver
tao = problem.solver
tao.setPythonContext(problem)
# Set the solver to Bounded Newton Trust Region
tao.setType('bntr')
# Set maximum number of iterations
tao.setMaximumIterations(5000)
# Set functions
tao.setObjectiveGradient(problem.formObjGrad, g)
tao.setObjective(problem.formObjective)
tao.setGradient(problem.formGradient, g)
tao.setHessian(problem.formHessian, H=H, P=H)

# Get linear KSP solver
ksp = tao.getKSP()
# Get associated preconditioner
pc = ksp.getPC()
# User-defined preconditioner
if '-user_pc' in sys.argv:
    print("Linear solver will use a user-defined preconditioner.")
    pc.setType('python')
    pc.setPythonContext(problem)
else:
    print("Linear solver will use a predefined preconditioner.")

# Set additional TAO options to control the trust region behavior
OptDB = PETSc.Options()
OptDB.setValue('tao_bnk_init_type','constant')
OptDB.setValue('tao_bnk_update_type','reduction')
OptDB.setValue('tao_trust0', Delta * np.sqrt(N))
OptDB.setValue('tao_bnk_max_radius', Delta * np.sqrt(N))
OptDB.setValue('tao_bnk_as_type', 'none')
tao.setFromOptions()
# Create entities for lower and upper bounds
lb = PETSc.Vec().createSeq(N)
ub = PETSc.Vec().createSeq(N)
tao.setVariableBounds(lb, ub)
# Set upper bound to infinity
ub.set(np.inf)
# Create entity for first guess
x_guess = PETSc.Vec().createSeq(N)
tao.setInitial(x_guess)

########################################################################
# Propagate

## Loading steps
# Radius
a0_low = np.logspace(np.log10(1E-4), np.log10(0.1), base = 10, num = 34, endpoint = True)
a0_high = np.linspace(0.1, a_max, num=int((a_max-1)/da)+1, endpoint=True)[1:]
a0 = np.concatenate((a0_low, a0_high)) 
# Loading
P = np.sqrt(E*Gc_0) * (np.pi*a0)**(3/2)
# Array for data save
print(simulation_name)
file_for_a = open(save_path+'results/'+simulation_name+'_position.bin', "wb")
Pi_dis = np.zeros(a0.size)
Pi_pot = np.zeros(a0.size)
max_error = np.zeros(a0.size)
average_error = np.zeros(a0.size)
step_duration = np.zeros(a0.size)
iteration_counts = np.zeros(a0.size)
# Set tolerance on gradient gtol
my_tol_on_Griffith = min(1E-5, 1E-3*sigma) * Gc_0

# Initialization
# Note: Also modifies a_computed (same pointer)
x.set(max(a_ini, a0[0]))

# Propagate
i = problem.step
success = True
print("Computing crack propagation...", end='\n')
while(success & (i < P.size)):
    print("Computing step n.{:d}/{:d}".format(i, P.size-1), end='\n')
    # Reset ksp & pc
    # Note: This was added to avoid the following error:
    #       "Cannot change local size of Amat after use old sizes [...] new sizes"
    ksp.reset()
    pc.reset()
    if '-user_pc' in sys.argv:
        pc.setType('python')
        pc.setPythonContext(problem)
    
    # Update loading
    problem.update_load(P[i])
    # Current gtol based on Griffith's criterion
    a_mean = a_computed.mean()
    my_gatol = 2*np.pi*a_mean/N * my_tol_on_Griffith
    my_grtol = np.finfo(float).eps
    my_gttol = np.finfo(float).eps
    tao.setTolerances(my_gatol, my_grtol, my_gttol)
    # Update bounds
    if (i==0):
        lb.set(max(a_ini, 1E-6*a0[0]))
    else:
        lb.setValues(range(N), a_computed)
    # Update initial value
    x_guess.setValues(range(N), a_computed)
    tao.setInitial(x_guess)
    
    # Solve
    start_time = time.time()
    tao.solve(x)
    stop_time = time.time()
    # Check success
    Griffith_criterion = grad_Pi_tot / (-2*np.pi/N * a_computed)
    average_error_on_Griffith = np.sqrt(np.mean(Griffith_criterion**2))
    max_error_on_Griffith = np.max(np.abs(Griffith_criterion))
    success = (average_error_on_Griffith <= my_tol_on_Griffith) | (max_error_on_Griffith <= 10 * my_tol_on_Griffith)
    # Save step
    a_computed.astype('float64').tofile(file_for_a)
    Pi_dis[i] = obj_dissipated_energy(a_computed, front, interpolator)
    Pi_pot[i] = obj_potential_energy(a_computed, P[-1], front)
    max_error[i] = max_error_on_Griffith
    average_error[i] = average_error_on_Griffith
    step_duration[i] = stop_time - start_time
    iteration_counts[i] = tao.getIterationNumber()
    # Write step infos
    print("Mean radius: {:.2e}/{:.0f} - Maximum deviation: {:.2e} - Error on Griffith's criterion: {:.2e}/{:.2e} (avg/max)".format(a_computed.mean(), a_max, np.abs(a_computed-a_computed.mean()).max(), average_error_on_Griffith, max_error_on_Griffith))
    print("Converged reason: {:d} - Number of iterations: {:d} - Step duration: {:.3E}".format(tao.getConvergedReason(), tao.getIterationNumber(), stop_time - start_time), end='\n\n')
    # Update step count
    problem.incr_step()
    i = problem.step

print("Simulation has ended at step n.{:d}/{:d}".format(i-1, P.size-1), end='\n')
# Check if simulation is completed
completed = (i == P.size) & success
print("Is completed? {:s}".format('Yes! :)' if completed else 'No! :()'))
# Resize arrays
P = P[:i] if success else P[:i-1]
a0 = a0[:i] if success else a0[:i-1]
Pi_dis = Pi_dis[:i] if success else Pi_dis[:i-1]
Pi_pot = Pi_pot[:i] if success else Pi_pot[:i-1]
max_error = max_error[:i] if success else max_error[:i-1]
average_error = average_error[:i] if success else average_error[:i-1]
step_duration = step_duration[:i] if success else step_duration[:i-1]
iteration_counts = iteration_counts[:i] if success else iteration_counts[:i-1]

# Delete large class elements
del interpolator
# Destroy PETSc entities
x.destroy()
g.destroy()
H.destroy()
lb.destroy()
ub.destroy()
x_guess.destroy()
tao.destroy()

# Get the front position as a numpy array
file_for_a = open(save_path+'results/'+simulation_name+'_position.bin', "rb")
if success :
    a = np.fromfile(file_for_a, dtype=np.float64).reshape(a0.size, N)
if (not success):
    a = np.fromfile(file_for_a, dtype=np.float64).reshape(a0.size+1, N)
    a = a[:-1]
#Save results
with open(save_path+'results/'+simulation_name+'.npz', 'wb') as outfile:
    np.savez(outfile, completed = completed, number_of_points = N, \
                      field_name = field_name, sign = sign, \
                      Young_modulus = E, \
                      fracture_energy = Gc_0, disorder_intensity = sigma,  \
                      force = P, reference_position = a0, \
                      front_position = a, dissipated_energy = Pi_dis, potential_energy = Pi_pot,\
                      initial_radius = a_ini, maximum_radius = a_max, \
                      radius_increment = da, tr_radius = Delta, \
                      maximum_error = max_error, average_error = average_error, \
                      step_duration = step_duration, iteration_counts = iteration_counts, total_time = np.sum(step_duration))
# Remove bin file
cmd_rmv = 'rm '+save_path+'results/'+simulation_name+'_position.bin'
subprocess.call(cmd_rmv, shell = True)

########################################################################
# Plot propagation

if '-nopopup' not in sys.argv:
    # Load field
    field.load_field()
    #Plot parameters
    my_fts = 18
    my_lbs = 14
    my_lw = 1.5
    my_cw = 1.5
    # Colormap
    Gc_cmap = 'pink_r'
    #Limit values
    r_max = min(field.L/2, 1.25*a[-1].max())
    Gc_min, Gc_max = field.Gc.min(), field.Gc.max()
    # Step
    n = max(int(0.1/da), 1)
    
    #Plot options
    fig = plt.figure(figsize=(6,5))
    #Fracture energy field and crack front position
    ax_01 = fig.add_subplot(111)
    ax_01.set_aspect(1)
    #Plot
    fracture_energy = ax_01.pcolormesh(field.z, field.x, field.Gc.T, cmap=Gc_cmap, vmin=Gc_min, vmax=Gc_max, shading='auto', rasterized=True)
    theta_plot = np.append(front.theta, 2*np.pi)
    for i in range(a.shape[0]-1)[::n]:
        a_plot = np.append(a[i], a[i,0])
        ax_01.plot(a_plot*np.cos(theta_plot), a_plot*np.sin(theta_plot), color='black', linewidth=1, linestyle='-')
    
    ax_01.plot(a_computed*np.cos(front.theta), a_computed*np.sin(front.theta), color='firebrick', linewidth=1, linestyle='-')
    #Axes
    for axis in ['top','bottom','left','right']:
        ax_01.spines[axis].set_linewidth(my_cw)
    
    ax_01.tick_params(axis='x', which='major', direction='out', length=2.5*my_cw, width=my_cw, labelsize=my_lbs)
    ax_01.tick_params(axis='y', which='major', direction='out', length=2.5*my_cw, width=my_cw, labelsize=my_lbs)
    ax_01.set_xlim(-r_max, r_max)
    ax_01.set_ylim(-r_max, r_max)
    #Legend
    ax_01.set_xlabel(r'Position $z/d$', size=my_fts)
    ax_01.set_ylabel(r'Position $x/d$', size=my_fts)
    #Colorbar
    divider_01 = make_axes_locatable(ax_01)
    ax_01_cbar = divider_01.append_axes("right", size="5%", pad=0.33)
    cbar_01 = fig.colorbar(fracture_energy, cax=ax_01_cbar)
    cbar_01.ax.set_title(r'Fracture energy $G_\mathrm{c}/G^0_\mathrm{c}$', fontsize=my_fts, ha='right', va='bottom', x=1.4)
    cbar_01.ax.tick_params(labelsize=my_lbs, width=my_cw)
    cbar_01.outline.set_linewidth(my_cw)
    cbar_01.set_alpha(1)
    #Arrange
    fig.tight_layout(pad=1)
    #Save
    plt.savefig('{:s}.pdf'.format(simulation_name))
