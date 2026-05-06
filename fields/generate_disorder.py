"""
 @file   generate_disorder.py
 @authors Mathias Lebihain <mathias.lebihain@enpc.fr>
 @date   Wed 09 Aug 2023
 @brief  python script to generate random fluctuations fields
         with Gaussian correlations and uniform distribution
"""
# Parser
import argparse
# Scientific packages
import numpy as np
from FyeldGenerator import generate_field
from scipy.special import erf
# Graphical packages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
# File management packages
import os
import sys

#LateX font
from matplotlib import rc
rc('text', usetex=True)

if not(os.path.isdir('fields')) :
    os.mkdir('fields')

#############################################################
# Parser
 
# Get simulation parameters from args
parser = argparse.ArgumentParser(description='Create disordered Gc field')
parser.add_argument('-f', '--field_name', type=str,
                    help=('Name of the field'), required=True)
parser.add_argument('-L', '--domain_size', type=int,
                    help=('Set width of the heterogeneous interface'), required=True)
parser.add_argument('-n', '--density', type=int,
                    help=('Number of points per heterogeneity width'), required=True)
parser.add_argument('-nopopup', '--display_options', action='store_true',
                    help=('Display options'), required=False)
args = parser.parse_args()

#############################################################
# Simulation parameters

# Set simulation name
field_name = args.field_name
# Set width of the heterogeneous interface
L_field = args.domain_size
# Number of points per heterogeneity width
n_field = args.density

########################################################################
# Generation of the random field using FyeldGenerator

# Draw samples from a normal distribution
def generate_gaussian_distribution(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

# Generate field with zero average, unit standard deviation, 
#                     uniform probability distribution, and Gaussian correlations
def generate_random_field(total_number_of_points, points_per_heterogeneity):
    # Generate Gaussian field with Gaussian correlations
    shape = (total_number_of_points, total_number_of_points)
    psd_func = lambda k : np.exp(-(np.pi*points_per_heterogeneity*k)**2)
    gaussian_field = generate_field(generate_gaussian_distribution, psd_func, shape)
    gaussian_field -= np.mean(gaussian_field)
    gaussian_field /= np.std(gaussian_field)
    cdf_field = 0.5*(1.+erf(gaussian_field/np.sqrt(2)))
    # Min/max values of the distribution of the uniform distribution
    Delta = 1. / np.sqrt(0.25 + cdf_field.std()**2)
    # Non-linear mapping to the toughness-fluctuations field f
    f = (- Delta + 2 * Delta * cdf_field)
    f -= f.mean()
    f /= f.std()
    
    return f

## Interface properties
# Number of points per domain width
N_field = n_field*int(L_field)
# Cartesian coordinates
z = x = np.linspace(-L_field/2, L_field/2, num=N_field, endpoint=False)
# Fracture energy-fluctuations field f
f = generate_random_field(N_field, n_field)

#Save results
with open('fields/'+field_name+'_f.npz', 'wb') as outfile:
    np.savez(outfile, domain_size = L_field, number_of_points = N_field, \
                      position = (z,x), fluctuations = f.astype('float32'))

########################################################################
# Plot field

if '-nopopup' not in sys.argv :
    #Plot parameters
    my_fts = 18
    my_lbs = 14
    my_lw = 1.5
    my_cw = 1.5
    
    #Plot options
    fig = plt.figure(figsize=(6,5))
    #Fracture energy field and dissipation
    ax_01 = fig.add_subplot(111)
    ax_01.set_aspect(1)
    #Plot
    fluctuations = ax_01.pcolormesh(z, x, f.T, cmap='pink_r', shading='auto', rasterized=True)
    #Axes
    for axis in ['top','bottom','left','right']:
        ax_01.spines[axis].set_linewidth(my_cw)
    
    ax_01.tick_params(axis='x', which='major', direction='out', length=2.5*my_cw, width=my_cw, labelsize=my_lbs)
    ax_01.tick_params(axis='y', which='major', direction='out', length=2.5*my_cw, width=my_cw, labelsize=my_lbs)
    ax_01.set_xlim(-L_field/2, L_field/2)
    ax_01.set_ylim(-L_field/2, L_field/2)
    #Legend
    ax_01.set_xlabel(r'Position $z/d$', size=my_fts)
    ax_01.set_ylabel(r'Position $x/d$', size=my_fts)
    #Colorbar
    divider_01 = make_axes_locatable(ax_01)
    ax_01_cbar = divider_01.append_axes("right", size="5%", pad=0.1)
    cbar_01 = fig.colorbar(fluctuations, cax=ax_01_cbar)
    cbar_01.ax.set_ylabel(r'Fluctuations $f$ (a.u.) ', size=my_fts, rotation=270, labelpad=28)
    cbar_01.ax.tick_params(labelsize=my_lbs, width=my_cw)
    cbar_01.outline.set_linewidth(my_cw)
    cbar_01.set_alpha(1)
    #Arrange
    fig.tight_layout(pad=2)
    #Save
    fig.savefig('../fields/'+field_name+".pdf", dpi=320)
    plt.show()