# Python code to
# generate the list of halo mass, central redshift and frequency for halo intersection in the range 7-6
# Min halo mass - 10^8 Msun/h

import numpy as np
from scipy.stats import rv_continuous
from scipy.integrate import romberg
from scipy.integrate import quad

from colossus.cosmology import cosmology
from colossus.utils import constants
from colossus.lss import mass_function
from colossus.halo import mass_so


# mass unit = Msun/h
# mass func unit = (h/Mpc)**3 in comoving coord
# dm halo radius = kpc/h in physical coord

cosmo = cosmology.setCosmology('planck18')

# Defining the constants
c = constants.C
mpc = constants.MPC
kpc = constants.KPC
H0 = cosmo.H0
hp = constants.H
kb = constants.KB
G = constants.G
mp = constants.M_PROTON
msun = constants.MSUN
h = H0/100

mass_expo = #m_f#
zmax = #z_f#
seed = #s_d#
delz = #delz_f#

halomass_min = np.power(10.0,mass_expo) # in Msolar/h
halomass_max = 1.e20 # in Msolar/h

zmin = zmax-delz
seed_new = seed + int(zmin*100) + mass_expo
np.random.seed(seed_new)

num1=str(mass_expo)
num2=str(zmax)
num3=str(seed)

# Parameters for dark matter
f_0 = (c/0.192065)*10**-6 # MHz

# DM temperature
def Tdm(z):
    zdec = 466400
    return 2.725*(1+zdec)*((1+z)/(1+zdec))**2

# Max impact parameter for intersection of light ray with a halo at redshift z
def max_alpha(m, z):
    #r = mass_defs.changeMassDefinition(m, concentration.concentration(m, '200c', z=z_halo, model = 'diemer19'), z, '200c', 'vir')[0]
    return 4.5*mass_so.M_to_R(m, z, 'vir')*(0.001) #kpc/h to mpc/h phys   #np.power(cosmo.rho_m(z)*(10**9)*h/mmin, -1/3)/2

# integrand for finding the number of intersections with halos at redshift z
def p(m, z):
    area = (np.pi)*max_alpha(np.exp(m), z)**2
    return mass_function.massFunction(np.exp(m), z, mdef = '200c', model = 'tinker08', q_out = 'dndlnM')*area #com mass func

# probability of intersection/ total number of intersections at redshift z
def prob(z, mmin, mmax, integrand):
    return (1+z)**2*(c/(cosmo.Hz(z)*(10**5/mpc)))*(h/mpc)*romberg(integrand, np.log(mmin), np.log(mmax), args=(z,))

# Here, for a given redshift and a given prob. we find the halo mass by
# sampling a point from pdf decided by normalised mass func at z

# Define function to normalise the PDF

def normalisation(z):
    val = romberg(p,  np.log(halomass_min), np.log(halomass_max), args=(z,))
    return val

# Define the distribution using rv_continuous
class halomass_prob(rv_continuous):
    def _pdf(self, x, rs, const):
        return (1.0/const) * p(x, rs)

halomass_p = halomass_prob(name="halomass_p", a=np.log(halomass_min), b=np.log(halomass_max))

def halomass(z):
    norm_constant = normalisation(z)
    sample = halomass_p.rvs(const = norm_constant, rs=z, size = 1)
    return np.exp(sample)

# randomly choosing the impact parameter in 0 to 5*rvir in mpc/h
def impact_param(z, mh):
    radius = max_alpha(mh, z)
    area = np.pi * radius**2
    return np.sqrt(np.random.uniform(0, area)/np.pi)


f_min = f_0/(1+zmax)
f_max = f_0/(1+zmin)

m_halo = []
z_array = []
f_array = []
alpha_array=[]
fi = f_min
delz = 0.1/(max(prob(zmin, halomass_min, halomass_max, p), prob(zmax, halomass_min, halomass_max, p)))
delnu = delz*f_0/((1+zmax)**2)


file = open("intersection_M_"+num1+"_zmax_"+num2+"_seed_"+num3+".txt", "w")
L = ["redshift \t Frequency (MHz) \t Halo mass (Msolar/h) Impact parameter (Mpc/h)\t IP(rvir) \t"]
file.writelines(L)
file.write("%.20E\t"%delnu)
file.writelines(["seed\t"])
file.write("%.20E\n"%seed_new)

while(fi < f_max):
    zi = f_0/fi - 1
    #probtot = prob(zi, halomass_min, halomass_max, p)*((1+zi)**2/f_0)*delnu
    rand = np.random.rand(1)[0]
    if rand < 0.1 :
        z_array.append(zi)
        f_array.append(fi)
        mh = halomass(zi)
        m_halo.append(mh)
        alpha_array.append(impact_param(zi, mh))
        ipbyr = impact_param(zi, mh)/(mass_so.M_to_R(mh, zi, 'vir')*(0.001))
        file.write("%.20E\t"%zi)
        file.write("%.20E\t"%fi)
        file.write("%.20E\t"%mh)
        file.write("%.20E\t"%impact_param(zi, mh))
        file.write("%.20E\n"%ipbyr)
    fi = fi + delnu

file.close()
