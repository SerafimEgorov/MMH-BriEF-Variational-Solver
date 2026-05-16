"""
 @file   propagate_shear_rupture.py
 @authors Serafim Egorov <serafim.egorov@enpc.fr>
          Mathias Lebihain <mathias.lebihain@enpc.fr>
 @date   Thud 1 Aug 2024
 @brief  python script to model crack propagation in shear modes II+III 
         along weakly heterogeneous interfaces at first order in G
         using JAX for just-in-time compilation and automatic differentiation
         and PETSc/TAO for optimization with a preconditioned BNTR method
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
                    help=('Poisson ratio for the simulation'), required=True)
parser.add_argument('-sign', '--sign', type=float,
                    help=('Sign factor'), default=+1, nargs='?')
parser.add_argument('-N', '--number_of_points', type=int,
                    help=('Number of points along the crack front'), required=True)
parser.add_argument('-fpath', '--field_path', type=str,
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

# Get field name
field_name = args.field
# Get the number of discretization points along the crack front
N = args.number_of_points
# Set average toughness Gc_0
Gc_0 = 1.
# Set Young's modulus 
E = 1.
# Set Poisson's ratio 
nu = args.poisson_ratio
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
# Reference problem of the penny crack 
# propagating under the action of a pair of antisymmetric punctual 
# shear forces applied at its center

## Stress intensity factors
# Note: For axisymmetric loading tau(r, theta) = f(r),
#       SIFS can be written as:
#       K20(a, Q, theta) = k2(a,Q) * np.cos(theta), and
#       K30(a, Q, theta) = k3(a,Q) * np.sin(theta),
#       where k2 = k0 + 2*nu/(2-nu) * k1
#       and   k3 = k0 - 2*nu/(2-nu) * k1.

# Loading functions
k0 = lambda a, Q : Q / (jnp.pi * a)**1.5
k1 = lambda a, Q : 3/2 * Q / (jnp.pi * a)**1.5
k2 = lambda a, Q, nu : k0(a, Q) + 2*nu/(2-nu) * k1(a, Q)
k3 = lambda a, Q, nu : k0(a, Q) - 2*nu/(2-nu) * k1(a, Q)
# Their first derivative
k0_prime = lambda a, Q : - 3/2 * k0(a, Q)/a
k1_prime = lambda a, Q : - 3/2 * k1(a, Q)/a
k2_prime = lambda a, Q, nu : k0_prime(a, Q) + 2*nu/(1-nu) * k1_prime(a, Q)
k3_prime = lambda a, Q, nu : k0_prime(a, Q) - 2*nu/(1-nu) * k1_prime(a, Q)

## Energy release rate
# G0(a, Q, theta) = (1-nu**2)/E * K20(a, Q, theta)**2 + (1+nu)/E * K30(a, Q, theta)**2
# Loading functions
g0 = lambda a, Q, E, nu : ((1-nu**2) * k2(a, Q, nu)**2 + (1+nu) * k3(a, Q, nu)**2)/2/E
g2 = lambda a, Q, E, nu : ((1-nu**2) * k2(a, Q, nu)**2 - (1+nu) * k3(a, Q, nu)**2)/2/E
# Their first derivative
g0_prime = lambda a, Q, E, nu : ((1-nu**2) * k2(a, Q, nu) * k2_prime(a, Q, nu) + (1+nu) * k3(a, Q, nu) * k3_prime(a, Q, nu))/E
g2_prime = lambda a, Q, E, nu : ((1-nu**2) * k2(a, Q, nu) * k2_prime(a, Q, nu) - (1+nu) * k3(a, Q, nu) * k3_prime(a, Q, nu))/E

## Total potential energy of the circular crack of radius a
Pi_pot_0 = lambda a, Q, E, nu : 2 * (1+nu) * (1-nu+nu**2) / (2-nu) * 2 * Q**2 / jnp.pi**2 / E / a

########################################################################
# Pre-computed quantities for the front position

# FrontUtils class
class FrontUtils():
    def __init__(self, N, E, nu):
        # Number of points
        self.N = N
        # Young's modulus
        self.E = E
        # Poisson's ratio
        self.nu = nu
        # Prefactors associated with Poisson's ratio
        self.A1_nu = (2-3*nu) * (1-nu**2) / (2-nu)
        self.A2_nu = (2+nu) * (1+nu) / (2-nu)
        self.A3_nu = 2 * (1-nu**2) / (2-nu)
        # Angle
        self.theta = jnp.linspace(0, 2*jnp.pi, num=N, endpoint=False, dtype='float64')
        # Angular functions
        self.sin = jnp.sin(self.theta)
        self.cos = jnp.cos(self.theta)
        self.cos2 = jnp.cos(2*self.theta)
        # Wavenumber
        self.k = jnp.arange(N//2+1, dtype='float64')
        # Mask for not-zero frequency
        self.not_zeroFreq = (self.k != 0) * 1.
        # Mask for Nyquist frequency
        self.is_Nyquist = (self.k == N//2) * 1.
        self.is_Nyquist_minus_1 = (self.k == N//2-1) * 1.
        # Vector to compute DFT at the Nyquist frequency
        self.v_Nyquist = (1/N) * jnp.exp(-1j*(N//2)*self.theta).real

########################################################################
# Pre-computed quantities for the fracture energy field

# Field class
class FieldData:
    def __init__(self, fname, average_frac_ener, sign_prefactor, amplitude):
        # Set field name
        self.name = fname
        # Get field
        field_data = np.load(args.field_path + fname +'_f.npz')
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
        field_data = np.load(args.field_path + self.name + '_f.npz')
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
        if not(os.path.isfile(args.field_path + '/{:s}_F_N{:d}pts.npz'.format(fdata.name, futils.N))):
            print("Precomputing the integrated field F and its derivatives for N={:d}.".format(futils.N))
            subprocess.call('python {:s}/compute_integrated_fluctuations.py -f {:s} -N {:d} -fpath {:s} -spath {:s}'.format(args.field_path,
    fdata.name, futils.N, args.field_path , args.field_path ), shell=True)
        # Load data
        data = np.load(args.field_path + '{:s}_F_N{:d}pts.npz'.format(fdata.name, futils.N))
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
        r_indices = jnp.digitize(a_i, self.r, right=False) - 1
        theta_indices = jnp.arange(a_i.size)
        # Interval limits
        lb_i = self.r[r_indices]
        ub_i = self.r[r_indices+1]
        # Step size
        Dx_i = ub_i - lb_i
        # Reduced position
        x_i = a_i - lb_i
        # Field and its derivatives at the limits
        Fl_i = self.F[r_indices, theta_indices]
        Fu_i = self.F[r_indices+1, theta_indices]
        dFl_i = self.dF[r_indices, theta_indices]
        dFu_i = self.dF[r_indices+1, theta_indices]
        d2Fl_i = self.d2F[r_indices, theta_indices]
        d2Fu_i = self.d2F[r_indices+1, theta_indices]
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
def obj_potential_energy(a_i, Q, futils):
    #Number of points
    N = futils.N
    # Young's modulus
    E = futils.E
    # Poisson's ratio
    nu = futils.nu
    # Angular functions
    sin_i = futils.sin
    cos_i = futils.cos 
    cos2_i = futils.cos2
    # Wavenumber
    k = futils.k
    # Filter for near-Nyquist frequencies
    is_Nyquist_minus_1 = futils.is_Nyquist_minus_1
    
    # Fast Fourier transform of the front position
    dft_N_a = jfft.rfft(a_i, norm="forward")
    # Average radius
    hat_a_0 = dft_N_a[0].real
    # Perturbations
    da_i = a_i - hat_a_0
    
    # Side quantities
    acos_i = a_i * cos_i
    asin_i = a_i * sin_i
    # Fast Fourier transform of side quantities
    dft_N_acos = jfft.rfft(acos_i, norm="forward")
    dft_N_asin = jfft.rfft(asin_i, norm="forward")
    # Dealiasing
    dft_N_acos -=   dft_N_a[N//2]/4  * is_Nyquist_minus_1
    dft_N_asin -= - dft_N_a[N//2]/4j * is_Nyquist_minus_1
    # Fractional Laplacian in Fourier space
    dft_N_Lacos = jnp.abs(k) * dft_N_acos
    dft_N_Lasin = jnp.abs(k) * dft_N_asin
    # Fractional Laplacian in real space
    Lacos_i = jfft.irfft(dft_N_Lacos, norm="forward")
    Lasin_i = jfft.irfft(dft_N_Lasin, norm="forward")
    # Signe operator in Fourier space
    dft_N_Sacos = 1j * jnp.sign(k) * dft_N_acos
    dft_N_Sasin = 1j * jnp.sign(k) * dft_N_asin
    # Signe operator in real space 
    Sacos_i = jfft.irfft(dft_N_Sacos, norm="forward")
    Sasin_i = jfft.irfft(dft_N_Sasin, norm="forward")
    
    # Magnitude of SIFs
    k2_0 = k2(hat_a_0, Q, nu)
    k3_0 = k3(hat_a_0, Q, nu)
    # Energy release rate
    G0_i = g0(hat_a_0, Q, E, nu) + g2(hat_a_0, Q, E, nu) * cos2_i
    dG0_i = g0_prime(hat_a_0, Q, E, nu) + g2_prime(hat_a_0, Q, E, nu) * cos2_i
    
    # Average of side quantities
    hat_G_0 = jnp.mean(G0_i)
    hat_Ga2_0 = jnp.mean(G0_i * a_i**2)
    hat_dGda2_0 = jnp.mean(dG0_i * da_i**2)
    hat_a2_0 = jnp.mean(a_i**2)
    hat_acos_0 = dft_N_acos[0].real
    hat_a2cos2_0 = jnp.mean(acos_i**2) - 0.5 * np.real(dft_N_acos[N//2]**2)
    hat_acosLacos_0 = jnp.mean(acos_i * Lacos_i) - 0.5 * np.real(dft_N_acos[N//2]*dft_N_Lacos[N//2])
    hat_acosSasin_0 = jnp.mean(acos_i * Sasin_i) - 0.5 * np.real(dft_N_acos[N//2]*dft_N_Sasin[N//2])
    hat_asin_0 = dft_N_asin[0].real
    hat_a2sin2_0 = jnp.mean(asin_i**2) - 0.5 * np.real(dft_N_asin[N//2]**2)
    hat_asinLasin_0 = jnp.mean(asin_i * Lasin_i) - 0.5 * np.real(dft_N_asin[N//2]*dft_N_Lasin[N//2])
    hat_asinSacos_0 = jnp.mean(asin_i * Sacos_i) - 0.5 * np.real(dft_N_asin[N//2]*dft_N_Sacos[N//2])
    # Dealiasing
    hat_Ga2_0 -= g0(hat_a_0, Q, E, nu) * 0.5 * np.real(dft_N_a[N//2]**2) \
               + g2(hat_a_0, Q, E, nu) * (np.real(dft_N_a[N//2]*dft_N_a[N//2-2]) + np.real(dft_N_a[N//2-1]**2))
    hat_dGda2_0 -= g0_prime(hat_a_0, Q, E, nu) * 0.5 * np.real(dft_N_a[N//2]**2) \
                 + g2_prime(hat_a_0, Q, E, nu) * (np.real(dft_N_a[N//2]*dft_N_a[N//2-2]) + np.real(dft_N_a[N//2-1]**2))
    hat_a2_0 -= 0.5 * np.real(dft_N_a[N//2]**2)
    hat_a2cos2_0 -= 0.5 * np.real(dft_N_acos[N//2]**2)
    hat_acosLacos_0 -= 0.5 * np.real(dft_N_acos[N//2]*dft_N_Lacos[N//2])
    hat_acosSasin_0 -= 0.5 * np.real(dft_N_acos[N//2]*dft_N_Sasin[N//2])
    hat_a2sin2_0 -= 0.5 * np.real(dft_N_asin[N//2]**2)
    hat_asinLasin_0 -= 0.5 * np.real(dft_N_asin[N//2]*dft_N_Lasin[N//2])
    hat_asinSacos_0 -= 0.5 * np.real(dft_N_asin[N//2]*dft_N_Sacos[N//2])
    
    # Prefactors associated with Poisson's ratio
    A1_nu = futils.A1_nu
    A2_nu = futils.A2_nu
    A3_nu = futils.A3_nu
    
    # Objective
    obj_Pi_pot = + Pi_pot_0(hat_a_0, Q, E, nu) \
                 - jnp.pi * hat_Ga2_0 + np.pi * hat_a_0**2 * hat_G_0 \
                 - jnp.pi * hat_a_0 * hat_dGda2_0 \
                 + jnp.pi/E * A1_nu * k2_0**2 * (hat_acosLacos_0 - hat_a2cos2_0) \
                 + jnp.pi/E * A2_nu * k3_0**2 * (hat_asinLasin_0 - hat_a2sin2_0) \
                 + jnp.pi/E * A3_nu * (k2_0**2 * hat_acos_0**2 + k3_0**2 * hat_asin_0**2) \
                 + jnp.pi/E * A3_nu * k2_0 * k3_0 * (hat_asinSacos_0 - hat_acosSasin_0 + hat_a2_0)
    
    return obj_Pi_pot #checked and coherent with an article, but there is simplification here due to axis symetric K2 and K3, 
                        #also for last two terms order is inversed and simplification cos**2 + sin**2 was performed 

# Compute gradient
grad_potential_energy = jax.jit(jax.grad(obj_potential_energy, argnums=0), static_argnums=(2,))

# Compute Hessian vector product
@partial(jit, static_argnums=(3,))
def hessp_potential_energy(a_i, v_i, Q, futils):
    return jax.jvp(lambda x : grad_potential_energy(x, Q, futils), (a_i,), (v_i,))[1]

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
def obj_total_energy(a_i, Q, futils, finterp):
    # Potential energy
    obj_Pi_pot = obj_potential_energy(a_i, Q, futils)
    # Dissipated energy
    obj_Pi_dis = obj_dissipated_energy(a_i, futils, finterp)
    
    return obj_Pi_pot + obj_Pi_dis

# Compute gradient
@partial(jit, static_argnums=(2,3))
def grad_total_energy(a_i, Q, futils, finterp):
    # Gradient of the potential energy
    grad_Pi_pot = grad_potential_energy(a_i, Q, futils)
    # Gradient of the dissipated energy
    grad_Pi_dis = grad_dissipated_energy(a_i, futils, finterp)
    
    return grad_Pi_pot + grad_Pi_dis

# Compute hessian matrix product
@partial(jit, static_argnums=(3,4))
def hessp_total_energy(a_i, v_i, Q, futils, finterp):
    # Hessian product of the potential energy
    hessp_Pi_pot = hessp_potential_energy(a_i, v_i, Q, futils)
    # Hessian product of the dissipated energy
    hessp_Pi_dis = hessp_dissipated_energy(a_i, v_i, futils, finterp)
        
    return hessp_Pi_pot + hessp_Pi_dis

########################################################################
# Preconditionner
# Note: This is based on the matrix-vector product
#       between the inverse of the Hessian of the quasi-circular crack
#       and a random vector

# Zero-order approximation: Penny-shaped crack
@partial(jit, static_argnums=(3,))
def invH0_product(a_i, v_i, Q, futils):
    #Number of points
    N = futils.N
    # Wavenumber
    k = futils.k
    # Average radius
    hat_a_0 = jnp.mean(a_i)
    # Fast Fourier transform
    dft_N_v = jfft.rfft(v_i, norm="forward")
    # Diagonal of the zero-order Hessian inverse in Fourier space
    R0 = k0(hat_a_0,Q)**2 / E
    R1 = 2 * hat_a_0 * k0(hat_a_0,Q) * k0_prime(hat_a_0,Q) / E
    kernel = R1 - R0 * np.abs(k)
    # Compute matrix-free vector product in the Fourier space
    dft_N_Pv = - N/(2*np.pi) / kernel * dft_N_v
    # Back in real space
    Pv_i = jfft.irfft(dft_N_Pv, norm="forward")
    
    return Pv_i

########################################################################
# Define class for PETSc

class FractureProblem :
    def __init__(self, Q, futils, finterp):
        self.load = Q
        self.step = 0
        self.futils = futils
        self.finterp = finterp
        self.solver = PETSc.TAO().create()
        self.is_inactive = np.zeros(futils.N, dtype='bool')
    
    def update_load(self, Q):
        self.load  = Q
    
    def incr_step(self):
        self.step += 1
        
    def formObjective(self, tao, x):
        # Get loading
        Q = self.load
        # Get front utilities
        futils = self.futils
        # Get field interpolator
        finterp = self.finterp
        # Get front position
        a_i = x.getArray(readonly=True)
        # Compute objective
        obj_Pi_tot = obj_total_energy(a_i, Q, futils, finterp)
        
        return obj_Pi_tot
    
    def formGradient(self, tao, x, g):
        # Get loading
        Q = self.load
        # Get front utilities
        futils = self.futils
        # Get array size
        N = futils.N
        # Get field interpolator
        finterp = self.finterp
        # Get front position
        a_i = x.getArray(readonly=True)
        # Compute gradient
        grad_Pi_tot = np.asarray(grad_total_energy(a_i, Q, futils, finterp))
        # Set gradient
        g.setValues(range(N), grad_Pi_tot)
        # Set inactive set
        lb, _ = self.solver.getVariableBounds()
        lb_i = lb.getArray(readonly=True)
        self.is_inactive = (lb_i < a_i) | ((a_i == lb_i) & (grad_Pi_tot < 0))
    
    def formObjGrad(self, tao, x, g):
        # Get loading
        Q = self.load
        # Get front utilities
        futils = self.futils
        # Get array size
        N = futils.N
        # Get field interpolator
        finterp = self.finterp
        # Get front position
        a_i = x.getArray(readonly=True)
        # Compute objective
        obj_Pi_tot = obj_total_energy(a_i, Q, futils, finterp)
        # Compute gradient
        grad_Pi_tot = np.asarray(grad_total_energy(a_i, Q, futils, finterp))
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
        Q = self.load
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
        hessp_Pi_tot = np.asarray(hessp_total_energy(a_i, v_i, Q, futils, finterp))
        # Set Hessian vector product
        Hv.setValues(range(N), hessp_Pi_tot)
    
    def applyPreconditioner(self, v, Pv):
        # could be implemented separately and jitted        
        v_i = v.getArray(readonly=True)
        # Get solver
        tao = self.solver
        # Get loading
        Q = self.load
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
            Pv_new = invH0_product(a_i, v_new, Q, futils)
            # Reduce space to inactive points
            Pv_i = Pv_new[self.is_inactive]
            # Set values
            Pv.setValues(range(v_i.size), Pv_i)
        else:
            # Apply preconditioner
            Pv_i = invH0_product(a_i, v_i, Q, futils)
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
front = FrontUtils(N, E, nu)
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
tao.setMaximumIterations(1000 * max(int(da / Delta),1))
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
Q = np.sqrt(E*Gc_0) * (np.pi*a0)**(3/2)
# Array for data save
file_for_a = open(args.save_path+'results/'+simulation_name+'_position.bin', "wb")
Pi_dis = np.zeros(a0.size)
Pi_pot = np.zeros(a0.size)
max_error = np.zeros(a0.size)
average_error = np.zeros(a0.size)
step_duration = np.zeros(a0.size)
iteration_counts = np.zeros(a0.size)
# Set tolerance on gradient gtol
my_tol_on_Griffith = min(1E-6, 1E-4*sigma) * Gc_0

# Initialization
# Note: Also modifies a_computed (same pointer)
x.set(max(a_ini, a0[0]))

# Propagate
i = problem.step
success = True
print("Computing crack propagation...", end='\n')
while(success & (i < Q.size)):
    print("Computing step n.{:d}/{:d}".format(i, Q.size-1), end='\n')
    # Reset ksp & pc
    # Note: This was added to avoid the following error:
    #       "Cannot change local size of Amat after use old sizes [...] new sizes"
    ksp.reset()
    pc.reset()
    if '-user_pc' in sys.argv:
        pc.setType('python')
        pc.setPythonContext(problem)
    
    # Update loading
    problem.update_load(Q[i])
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
    Pi_pot[i] = obj_potential_energy(a_computed, Q[-1], front)
    max_error[i] = max_error_on_Griffith
    average_error[i] = average_error_on_Griffith
    step_duration[i] = stop_time - start_time
    iteration_counts[i] = tao.getIterationNumber()
    # Write step infos
    print("Mean radius: {:.2e}/{:.0f} - Maximum deviation: {:.2e} – Error on Griffith's criterion: {:.2e}/{:.2e} (avg/max)".format(a_computed.mean(), a_max, np.abs(a_computed-a_computed.mean()).max(), average_error_on_Griffith, max_error_on_Griffith))
    print("Converged reason: {:d} – Number of iterations: {:d} – Step duration: {:.3E}".format(tao.getConvergedReason(), tao.getIterationNumber(), stop_time - start_time), end='\n\n')
    # Update step count
    problem.incr_step()
    i = problem.step

print("Simulation has ended at step n.{:d}/{:d}".format(i-1, Q.size-1), end='\n')
# Check if simulation is completed
completed = (i == Q.size) & success
print("Is completed? {:s}".format('Yes! :)' if completed else 'No! :()'))
# Resize arrays
Q = Q[:i] if success else Q[:i-1]
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

# Get a
file_for_a = open(args.save_path+'results/'+simulation_name+'_position.bin', "rb")
a = np.fromfile(file_for_a, dtype=np.float64).reshape(a0.size, N)
if (not success):
    a = a[:-1]
#Save results
with open(args.save_path+'results/'+simulation_name+'_results.npz', 'wb') as outfile:
    np.savez(outfile, completed = completed, number_of_points = N, \
                      field_name = field_name, sign = sign, \
                      Young_modulus = E, Poisson_ratio = nu, \
                      fracture_energy = Gc_0, disorder_intensity = sigma, \
                      force = Q, reference_position = a0, \
                      front_position = a, dissipated_energy = Pi_dis, potential_energy = Pi_pot,\
                      initial_radius = a_ini, maximum_radius = a_max, \
                      radius_increment = da, tr_radius = Delta, \
                      maximum_error = max_error, average_error = average_error, \
                      step_duration = step_duration, iteration_counts = iteration_counts, total_time = np.sum(step_duration))
# Remove bin file
cmd_rmv = 'rm '+args.save_path+'results/'+simulation_name+'_position.bin'
subprocess.call(cmd_rmv, shell = True)

########################################################################
# Plot propagation

if '-nopopup' not in sys.argv:
    # Load field
    field.load_field()
    # Plot parameters
    my_fts = 18
    my_lbs = 14
    my_lw = 1.5
    my_cw = 1.5
    # Colormap
    Gc_cmap = 'pink_r'
    # Limit values
    r_max = min(field.L/2, 1.25*a[-1].max())
    Gc_min, Gc_max = field.Gc.min(), field.Gc.max()
    # Step
    n = 1 # max(int(0.2/da), 1)
    
    # Plot options
    fig = plt.figure(figsize=(6,5))
    # Fracture energy field and crack front position
    ax_01 = fig.add_subplot(111)
    ax_01.set_aspect(1)
    # Plot
    fracture_energy = ax_01.pcolormesh(field.z, field.x, field.Gc.T, cmap=Gc_cmap, vmin=Gc_min, vmax=Gc_max, shading='auto', rasterized=True)
    theta_plot = np.append(front.theta, 2*np.pi)
    for i in range(a.shape[0]-1):#[::n]:
        a_plot = np.append(a[i], a[i,0])
        ax_01.plot(a_plot*np.cos(theta_plot), a_plot*np.sin(theta_plot), color='black', linewidth=1, linestyle='-')
    # Axes
    for axis in ['top','bottom','left','right']:
        ax_01.spines[axis].set_linewidth(my_cw)
    
    ax_01.tick_params(axis='x', which='major', direction='out', length=2.5*my_cw, width=my_cw, labelsize=my_lbs)
    ax_01.tick_params(axis='y', which='major', direction='out', length=2.5*my_cw, width=my_cw, labelsize=my_lbs)
    ax_01.set_xlim(-r_max, r_max)
    ax_01.set_ylim(-r_max, r_max)
    # Legend
    ax_01.set_xlabel(r'Position $z/d$', size=my_fts)
    ax_01.set_ylabel(r'Position $x/d$', size=my_fts)
    # Colorbar
    divider_01 = make_axes_locatable(ax_01)
    ax_01_cbar = divider_01.append_axes("right", size="5%", pad=0.33)
    cbar_01 = fig.colorbar(fracture_energy, cax=ax_01_cbar)
    cbar_01.ax.set_title(r'Fracture energy $G_\mathrm{c}/G^0_\mathrm{c}$', fontsize=my_fts, ha='right', va='bottom', x=1.4)
    cbar_01.ax.tick_params(labelsize=my_lbs, width=my_cw)
    cbar_01.outline.set_linewidth(my_cw)
    cbar_01.set_alpha(1)
    # Arrange
    fig.tight_layout(pad=1)
    # Save
    plt.savefig('{:s}.pdf'.format(simulation_name))
