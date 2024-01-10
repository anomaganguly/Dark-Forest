#combining the forest data

import numpy as np
from scipy import interpolate
import itertools
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.cosmology import cosmology
c = constants.C

cosmo = cosmology.setCosmology('planck18')

coll = #c_f#
mass_expo = #m_f#
zmax = #z_f#
seed = #s_d#
delz = #del_z#
div = #div_z#

num0 = str(coll)
num1 = str(mass_expo)
num2 = str(zmax)
num3 = str(seed)


f_0 = (c/0.192065)*10**-9

path=[]

num=zmax

#div = total number of divisions in redshift 7-6 (7.0, 6.9, 6.8, 6.7....)
if delz == 0.05 :
    dec = 2
else:
    dec = 1

for i in range(0, div):
    num4 = str(float(num))
    path.append("raw_int_"+num0+"_M_"+num1+"_zmax_"+num4+"_seed_"+num3+".txt")
    num = np.round(num-delz, decimals = dec)

lines = []
count=0
for item in path:
    file1 = open(item, "r")
    lines.append(file1.readlines())
    file1.close()
    count += 1

fi = []
oi = []
mi = []
ai = []
fr = []
oraw = []
zr = []
f=[]
o=[]
z=[]
mh=[]
ah=[]

width = []
flag=0
j=0
k=0
# count=2
for i in range(0, count):
    fi=[]
    oi=[]
    mi=[]
    ai=[]
    zr = []
    fr = []
    oraw = []
    mr = []
    ar = []
    for l in lines[i][0:1]:
        p = l.split()
        width.append(float(p[8][2:])*1.e-3)
    for l in lines[i][1:]:
        p = l.split()
        fraw = float(p[0])*1.e-3
        fi.append(fraw)
        oi.append(float(p[1]))
        mi.append(float(p[2]))
        ai.append(float(p[3]))
    arr = np.asarray(np.transpose([fi, oi, mi, ai]))
    arr = arr[np.argsort(arr[:, 0])]
    for index in range(0, len(arr)-1):
        if (flag==0):
            fhf = arr[index][0]
            od = arr[index][1]
            zhf = f_0/fhf - 1
            j = index
            while (abs(arr[j][0]-arr[j+1][0]) < 0.10*width[i]):
                od = od+arr[j+1][1]
                j = j+1
                flag = flag + 1
            zr.append(zhf)
            fr.append(fhf)
            oraw.append(od)
            list_m = np.transpose(arr[index:j+1])[1]
            max = index+np.where(list_m==np.max(list_m))[0][0]
            # print(arr[max][2])
            mr.append(arr[max][2]) #mass and imp param for which we get max opdepth
            ar.append(arr[max][3])
        else:
            flag=flag-1
    last = len(arr)-1
    if (abs(arr[last-1][0]-arr[last][0])>=0.10*width[i]):
        od = arr[last][1]
        fhf = arr[last][0]
        zhf = f_0/fhf - 1
        od = arr[last][1]
        zr.append(zhf)
        fr.append(fhf)
        oraw.append(od)
        mr.append(arr[last][2])
        ar.append(arr[last][3])
    f.append(fr)
    z.append(zr)
    o.append(oraw)
    mh.append(mr)
    ah.append(ar)

del(lines)
del(fr)
del(oraw)
del(zr)
del(mr)
del(ar)
del(max)

op_d = []

for index in range(0, count):
    op_i = interpolate.interp1d(f[index], o[index])
    op_d.append(op_i)

def op_depth(fobs):
    val=[]
    for index in range(0, count):
        if fobs>=f[index][0] and fobs<=f[index][-1]:
            val.append(op_d[index](fobs))
        else:
            val.append(0)
    return [sum(val), val.index(max(val))]

pathw = "cdf_int_"+num0+"_M_"+num1+"_zmax_"+num2+"_"+num4+"_seed_"+num3+".txt" #'M_'+str(mass)+'_'+str(coll)+'/'+str(zmax)+'_'+str(zmax-1)+'.txt'

file2 = open(pathw,"w")
L = ["File_no\t Redshift\t Frequency(GHz)\t Optical_depth\t Rel_trans\t mass\t i_param\t bin_size\n"]
file2.writelines(L)

fp=[]
op=[]
tp=[]
wp=[]

for i in range(0, count):
    for j in range(0, len(f[i])):
        oph = op_depth(f[i][j])
        if oph[0] > 1.e-15 and i==np.round(oph[1]):
            file2.write("%E\t"%i)
            file2.write("%E\t"%z[i][j])
            fp.append(f[i][j])
            file2.write("%E\t"%f[i][j])
            op.append(oph[0])
            file2.write("%E\t"%oph[0])
            trans = np.exp(-oph[0])
            tp.append(trans)
            file2.write("%E\t"%trans)
            # file2.write("%E\t"%mh[i][j])
            # file2.write("%E\t"%ah[i][j])
            wp.append(width[i])
            file2.write("%E\n"%width[i])

file2.close()
