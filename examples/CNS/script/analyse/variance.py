# Need to import the plotting package:
import matplotlib.pyplot as plt
import numpy as np

vol    = 2*np.pi*np.pi*2

re_tau = 550
mu     = 0.0003
tau_w  = 1.080/vol;#(mu*re_tau)**2
u_tau  = np.sqrt(tau_w)

filename1     = 'u_mean_0186000' 
filename2     = 'dns_fluc_550' 

def getDNSdata(filename):
    # Read the file. 
    f = open(filename, 'r')
    f.readline()
    lines = f.readlines()
    f.close()
    
    # initialize some variable to be lists:
    yl  = []
    uul = []
    vvl = []
    wwl = []
    uvl = []
    
    # scan the rows of the file stored in lines, and put the values into some variables:
    for line in lines:
        p = line.split()
        yl.append(float(p[1]))
        uul.append(float(p[2]))
        vvl.append(float(p[3]))
        wwl.append(float(p[4]))
        uvl.append(float(p[5]))
    
    yv  = np.array(yl)
    uuv = np.array(uul)
    vvv = np.array(vvl)
    wwv = np.array(wwl)
    uvv = np.array(uvl)

    return yv, uuv, vvv, wwv, uvv
 
def getFileData(filename):
    # Read the file. 
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    
    # initialize some variable to be lists:
    yl  = []
    uul = []
    vvl = []
    wwl = []
    uvl = []

    # scan the rows of the file stored in lines, and put the values into some variables:
    coun = 0
    for line in lines:
        p = line.split()
        yl.append(float(p[0])*re_tau)
        uul.append(float(p[3]))
        vvl.append(float(p[4]))
        wwl.append(float(p[5]))
        uvl.append(float(p[6]))

        ubar = float(p[2])

        uul[coun] = (uul[coun] - ubar**2)/(u_tau**2)
        vvl[coun] = (vvl[coun]          )/(u_tau**2)
        wwl[coun] = (wwl[coun]          )/(u_tau**2)
        uvl[coun] = (uvl[coun]          )/(u_tau**2)

        coun = coun + 1
    
    yv  = np.array(yl)
    uuv = np.array(uul)
    vvv = np.array(vvl)
    wwv = np.array(wwl)
    uvv = np.array(uvl)

    return yv, uuv, vvv, wwv, uvv
    
yv1, uuv1, vvv1, wwv1, uvv1 = getFileData(filename1)
yv2, uuv2, vvv2, wwv2, uvv2 = getDNSdata(filename2)

fig, axis = plt.subplots(2, 2, figsize = (16, 6))

axis[0, 0].scatter(yv1[:22], uuv1[:22], color = "blue", linewidth =  2.5, label = 'MFEM')
axis[0, 0].semilogx(yv2, uuv2, color = "red", linewidth =  2.5, label = 'DNS')
axis[0, 0].set_ylabel('<uu+>')

axis[0, 1].scatter(yv1[:22], vvv1[:22], color = "blue", linewidth =  2.5, label = 'MFEM')
axis[0, 1].semilogx(yv2, vvv2, color = "red", linewidth =  2.5, label = 'DNS')
axis[0, 1].set_ylabel('<vv+>')

axis[1, 0].scatter(yv1[:22], wwv1[:22], color = "blue", linewidth =  2.5, label = 'MFEM')
axis[1, 0].semilogx(yv2, wwv2, color = "red", linewidth =  2.5, label = 'DNS')
axis[1, 0].set_xlabel('y+')
axis[1, 0].set_ylabel('<ww+>')

axis[1, 1].scatter(yv1[:22], uvv1[:22], color = "blue", linewidth =  2.5, label = 'MFEM')
axis[1, 1].semilogx(yv2, uvv2, color = "red", linewidth =  2.5, label = 'DNS')
axis[1, 1].set_xlabel('y+')
axis[1, 1].set_ylabel('<uv+>')

for i, ax in enumerate(fig.axes):
    ax.set_xlim([0.5 , 500])


#plt.ylim([1.65 , 2.0])

#plt.title('U at B')
##plt.title('U at point near GF')
#plt.xlabel('y+')
#plt.ylabel('<uu+>')
plt.legend(loc='lower right', fontsize = 10)
####plt.legend(loc='upper center', bbox_to_anchor = (0.95, 1.12), fontsize = 10)

#plt.show()
plt.savefig('re550_variance.png', format = 'png')
