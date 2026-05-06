"""
 @file   compute_integrated_fluctuations.py
 @authors Mathias Lebihain <mathias.lebihain@enpc.fr>
 @date   Tue 30 Apr 2024
 @brief  python script to pre-compute the integral of a fluctuation field
"""
# Parser
import argparse
# Scientific packages
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
# File management packages
import os

#############################################################
# Simulation parameters

# Get simulation parameters from args
parser = argparse.ArgumentParser(description=r'Compute the integral $\int_{0}^{a} r*f(r,theta) dr$ of a fluctuation field f.')
parser.add_argument('-f', '--field', type=str,
                    help=('Name of the random field'), required=True)
parser.add_argument('-N', '--number_of_points', type=int,
                    help=('Number of points along the crack front'), required=True)
parser.add_argument('-path', '--path', type=str,
                    help=('Global path'), required=True)
parser.add_argument('-spath', '--save_path', type=str,
                    help=('Path to the results'), required=True)
args = parser.parse_args()

# Save and working directories 


#Global path 
global_path = args.path
# Save path
save_path = args.save_path

# Get field
field_data = np.load(save_path + 'fields/'+args.field+'_f.npz')
# Get width of the heterogeneous interface
L_field = field_data['domain_size']
# Position
z, x = field_data['position']
# Fluctuations field f
f = field_data['fluctuations']
# Interpolate field with a cubic spline
interp_f_zx = RectBivariateSpline(z, x, f, s=0)

# Get the number of discretization points along the crack front
N = args.number_of_points
# Discretized angle
theta = np.linspace(0, 2*np.pi, num=N, endpoint=False)

## Evaluate f on concentric circles
# Hyper-discretization factor
# Note: This should help improving the local quadratic/cubic interpolation of F
#       during propagation
n = 4
# Radius
r = np.linspace(0, L_field//2, num=n*z.size, endpoint=True)
# Radius with negative values (to properly compute derivatives near r=0)
r_negpos = np.concatenate((-r[::-1][:-1], r))

# Compute F(a, theta) = \int_{0}^{a} r*f(r,theta) dr
#         and its derivatives
F = np.zeros((r.size, theta.size))
dF = np.zeros((r.size, theta.size))
d2F = np.zeros((r.size, theta.size))
print("Completion 0/100%", end='')
for i in range(N):
    print("\rCompletion {:.0f}/100%".format(i/N*100), end='')
    # Compute f
    f_negpos = interp_f_zx.ev(r_negpos*np.cos(theta[i]), r_negpos*np.sin(theta[i]))
    # Spline interpolation for r*f
    rf_spl = UnivariateSpline(r_negpos, r_negpos*f_negpos, s=0, k=3)
    # F(r, theta) = \int_{0}^{r} r f(r, theta) dr
    F[:,i] = rf_spl.antiderivative(1)(r) - rf_spl.antiderivative(1)(0)
    # dF/dr(r, theta) = r f(r, theta)
    dF[:,i] = rf_spl(r)
    # d^2F/dr^2(r, theta) = (rf)'(r, theta)
    d2F[:,i] = rf_spl.derivative(1)(r)

# Save results
with open(save_path + 'fields/{:s}_F_N{:d}pts.npz'.format(args.field, N), 'wb') as outfile:
    np.savez(outfile, radius = r[::n], angle = theta, \
                      F = F[::n,:].astype('float32'), dF = dF[::n,:].astype('float32'), d2F = d2F[::n,:].astype('float32'))