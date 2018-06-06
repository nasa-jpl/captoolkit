import sys
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import matplotlib.animation as animation

np.warnings.filterwarnings('ignore')

vmin, vmax = -1, 1

def smooth_images(Z, kernel_size=3):
    for k in xrange(Z.shape[2]):
        Z[:,:,k] = ndi.median_filter(Z[:,:,k], kernel_size)
    return Z


def filter_images(Z):
    for k in xrange(Z.shape[2]):
        z = Z[:,:,k]
        z[np.abs(z)>np.nanstd(z)*3] = np.nan
        Z[:,:,k] = z
    return Z


def center_images(Z):
    for k in xrange(Z.shape[2]):
        Z[:,:,k] -= np.nanmean(Z[:,:,k])
    return Z


def init():
    """ Initialize with empty frame. """ 
    zz = np.full_like(Z[:,:,0], np.nan)
    plt.pcolormesh(xx, yy, Z[:,:,0], vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)
    plt.colorbar(shrink=0.8)


def animate(k):
    plt.title('%.2f' % t[k])
    plt.pcolormesh(xx, yy, Z[:,:,k], vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)


fname = sys.argv[1]

with h5py.File(fname, 'r') as f:
    t = f['t'][:]
    x = f['x'][:]
    y = f['y'][:]
    Z = f['z'][:]
    E = f['e'][:]
    N = f['n'][:]

# Plot time series
if 0:
    i = int(Z.shape[0]/2.)
    j = int(Z.shape[1]/2.)
    plt.plot(t, Z[i,j,:])
    plt.show()
    sys.exit()

xx, yy = np.meshgrid(x, y)

Z = smooth_images(Z) 
Z = filter_images(Z)
#Z = center_images(Z)

Z = np.ma.masked_invalid(Z)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()

print 'building animation ...'

init()

anim = animation.FuncAnimation(fig, animate, frames=Z.shape[2], blit=False)

anim.save('anim.mp4', writer=writer)

print 'animation saved.'

#plt.show()
