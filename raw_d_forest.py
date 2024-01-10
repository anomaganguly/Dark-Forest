# Python code to :
# plot optical depth with frequency
# optical depth contribution from different regions

# generate points for optical depth for the intersections with different halos


import numpy as np

from scipy.integrate import quad
from scipy.integrate import romberg
from scipy import optimize

from colossus.cosmology import cosmology
from colossus.utils import constants
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration
from colossus.halo import profile_nfw

coll = #c_f#
mass_expo = #m_f#
zmax = #z_f#
seed = #s_d#

num0 = str(coll)
num1 = str(mass_expo)
num2 = str(zmax)
num3 = str(seed)

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
mp = constants.M_PROTON
msun = constants.MSUN
h = H0/100
mev = 1.783e-33*(1.e6)
mdm = 1.e0*mev

# Parameters for dark matter
f_0 = (c/0.192065)*10**-6 # MHz
A10 = 10**-15
Tstar = hp*f_0*10**6/(kb)

# Defining functions

# Halo density profile (h2Msun/kpc3 physical units)
def density_profile(z_halo, m):
    return profile_nfw.NFWProfile(M = m, c = concentration.concentration(m, '200c', z=z_halo, model = 'bullock01'), z = z_halo, mdef = '200c')

# Virial radius in Mpc/h
def rvir(z, m):
    rt = mass_so.M_to_R(m, z, 'vir')
    return rt*0.001

# Radius at which DM number density = mean DM number density
def func(r, z, m):
    val = density_profile(z, m).density(r*1000)-(cosmo.rho_m(z)-cosmo.rho_b(z))
    return val

# CMB temperature (K)
def Tcmb(z):
    return 2.726*(1+z)

# Kinetic temp in the halo (K)
def Tk(r, z, m):
    if r <= 10*rvir(z, m):
        # conc = concentration.concentration(m, '200c', z, model = 'diemer19')
        # dm_density = density_profile(z, m).density(r*1000)*(msun)*(1.e12)*(h**2/(kpc**3))/(2.93e-12*(mev*1.e3))
        # Tk_halo_old = (1/3.)*(mdm/kb)*np.power(dm_density/(np.power(r*conc/rvir(z, m), -1.875)), 2./3.)
        Tk_halo = (1/3.)*(mdm/kb)*np.power((density_profile(z, m).density(r*1000)/cosmo.rho_c(z))*np.power(r/rvir(z, m), 1.9)\
        *(np.power(10, -1.46))*(density_profile(z, m).circularVelocity(rvir(z, m)*1000)*1.e5)**3, 2./3.)
        return Tk_halo
    else:
        return 1.e-12

# DM number density in cm-3
def ndm(r, z, m):
    if r > 10*rvir(z, m):
        return ((cosmo.rho_m(z)-cosmo.rho_b(z)))*(msun/mdm)*(h**2/(kpc**3)) #cm-3
    else :
        return density_profile(z, m).density(r*1000)*(msun/mdm)*(h**2/(kpc**3))

# Spin temperature (if collsions are strong inside virial radius the level population
# is set by DM kinetic temperature otherwise by CMB temperature)

def Ts(r, z, m, collision, root):
    if r <= root:
        if collision == 0:
            return Tcmb(z)
        else:
            return Tk(r, z, m)
    else:
        return Tcmb(z)

# Line of sight velocity of DM particles: random motion within 10 r_vir outside it is the Hubble flow
def vlos(R, r, z, m):
   if r <= 10*rvir(z, m):
       val = 0
       return val
   else:
       return cosmo.Hz(z)*R*1.e5/(c*h)


#Optical depth!

# integrand to calculate the optical depth
def integrand_od(R, z, delf, m, alpha, root, collision):
    r = np.sqrt(alpha**2 + R**2)
    b2_r = (2*kb*Tk(r, z, m)/mdm)/(c**2)
    v_f = delf/(f_0)  #m s-1
    value = (1/(np.sqrt(np.pi*b2_r)))*np.exp(-(v_f-vlos(R, r, z, m))**2/(b2_r))*ndm(r, z, m)*3*(1-np.exp(-Tstar/Ts(r, z, m, collision, root)))/(1+3*np.exp(-Tstar/Ts(r, z, m, collision, root)))
    return value

# optical depth depends on frequency, impact parameter, mass of halo, collision
def opticaldepth(z, delf, m, alpha, root, collision, integrand_od):
    R_h = delf*(c*h)/(f_0*cosmo.Hz(z)*1.e5)
    r_h = np.sqrt(R_h**2 + alpha**2)
    # odsat1 = (3*c**3*A10)/(8*(np.pi)*(f_0*10**6)**3)*(ndm(r_h, z, m, root)/(cosmo.Hz(z)*(10**5/mpc)))\
    #     *(1-np.exp(-Tstar/Ts(r_h, z, m, collision, root)))/(1+3*np.exp(-Tstar/Ts(r_h, z, m, collision, root)))
    rc = 10*rvir(z, m)
    valt = quad(integrand_od, -rc, +rc, args=(z, delf, m, alpha, root, collision), limit = 100)
    return valt[0]*(c**2*A10)/(8*(np.pi)*(f_0*10**6)**3)*(mpc/h)

file1 = open("intersection_M_"+num1+"_zmax_"+num2+"_seed_"+num3+".txt","r")

lines = file1.readlines()
file1.close()

for line in lines[0:1]:
    p = line.split()
    delnu = float(p[10])


zh = []
fh = []
mh = []
ah = []
ar = []

insec = np.shape(lines)[0]-1

for line in lines[1:]:
    p = line.split()
    zh.append(float(p[0]))
    fh.append(float(p[1]))
    mh.append(float(p[2]))
    ah.append(float(p[3]))
    ar.append(float(p[4]))

file1.close()
del(lines)

file2 = open("raw_int_"+num0+"_M_"+num1+"_zmax_"+num2+"_seed_"+num3+".txt", "w")
L = ["frequency (MHz) \t Optical depth \t Mass \t imp param \t del nu"]
file2.writelines(L)
file2.write("%.20E\n"%delnu)
file2.flush()

for index in range(insec):
    delf = 0
    root = optimize.bisect(func, 2*rvir(zh[index], mh[index]), 6*rvir(zh[index], mh[index]), args = (zh[index], mh[index]))
    od = opticaldepth(zh[index], delf*(1+zh[index]), mh[index], ah[index], root, coll, integrand_od)
    ibyr = ah[index]/rvir(zh[index], mh[index])
    while(abs(od)>0):
        if delf == 0:
            fobs = fh[index]
            file2.write("%.20E\t"%fobs)
            file2.write("%.20E\t"%od)
            file2.write("%.20E\t"%mh[index])
            file2.write("%.20E\t"%ar[index])
            file2.write("%.20E\n"%ibyr)
            file2.flush()
        else:
            fobs = fh[index] - delf
            od = opticaldepth(zh[index], delf*(1+zh[index]), mh[index], ah[index], root, coll, integrand_od)
            file2.write("%.20E\t"%fobs)
            file2.write("%.20E\t"%od)
            file2.write("%.20E\t"%mh[index])
            file2.write("%.20E\t"%ar[index])
            file2.write("%.20E\n"%ibyr)
            fobs = fh[index] + delf
            file2.write("%.20E\t"%fobs)
            file2.write("%.20E\t"%od)
            file2.write("%.20E\t"%mh[index])
            file2.write("%.20E\t"%ar[index])
            file2.write("%.20E\n"%ibyr)
            file2.flush()
        delf = delf + delnu


file2.close()
