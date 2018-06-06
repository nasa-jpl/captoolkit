import numpy as np
import matplotlib.pyplot as plt

""" Covariance models. """

def gauss(r, s, R):
    return s**2 * np.exp(-r**2/R**2)

def markov(r, s, R):
    return s**2 * (1 + r/R) * np.exp(-r/R)

def generic(r, s, R):
    return s * (1 + (r/R) - 0.5 * (r/R)**2) * np.exp(-r/R)


def covxt(r, l, s, v, R, L):
    return markov(r, s, R) * markov(l, v, L)



# Modeled params: Spatial Cov
#s, R = [0.831433532, 1536.21967]  # gauss
s, R = [0.85417087, 655.42437697]  # markov
#s, R = [0.720520448, 1084.11029]  # generic


# Modeled params: Temporal Cov
#gauss [ 0.15033057, 3.9811011 ] # gauss
v, L = [ 0.15301943, 1.97284949]  # markov
#generic [ 0.02315648, 3.02922434]
#
#gauss [ 0.18734364  3.32761283]
#markov [ 0.19784374  1.35463026]
#generic [ 0.03775496  2.25583848]



rr = np.linspace(0, 10000, 100)
ll = np.linspace(0, 5, 100)

cc = covxt(rr, ll, s, v, R, L)

plt.figure()
plt.plot(rr, cc)

plt.figure()
plt.plot(ll, cc)

plt.show()
