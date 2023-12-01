# -*- coding: utf-8 -*-
########################################################################################################################

import sys
import math
import tqdm

import numpy as np

from kerr.initial import initial
from kerr.geodesic import integrate

import matplotlib.pyplot as plt

########################################################################################################################

# noinspection PyPep8Naming
def ray_tracer(a: float, r_cam: float, θ_cam: float, ϕ_cam: float, size_x: int, size_y: int, width: float):

    ####################################################################################################################

    x_obs0 = width * np.linspace(-0.5, +0.5, size_x)
    y_obs0 = width * np.linspace(-0.5, +0.5, size_y)

    #x_obs0 = width * (np.arange(0, size_x) - (size_x + 1.0) / 2.0) / (size_x - 1.0)
    #y_obs0 = width * (np.arange(0, size_y) - (size_y + 1.0) / 2.0) / (size_y - 1.0)

    x_obs, y_obs = np.meshgrid(x_obs0, y_obs0)

    x_obs = x_obs.ravel()
    y_obs = y_obs.ravel()

    ####################################################################################################################

    z_obs = np.zeros(size_x * size_y, dtype = np.float32)

    ####################################################################################################################

    r, θ, ϕ, pr, pθ, E, L, Q, κ = initial(
        a,
        r_cam, θ_cam, ϕ_cam,
        x_obs, y_obs, z_obs
    )

    ####################################################################################################################

    if False:

        plt.imshow(r.reshape(size_x, size_y))
        plt.title('r')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(θ.reshape(size_x, size_y))
        plt.title('θ')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(ϕ.reshape(size_x, size_y))
        plt.title('ϕ')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(pr.reshape(size_x, size_y))
        plt.title('pr')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(pθ.reshape(size_x, size_y))
        plt.title('pθ')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(E.reshape(size_x, size_y))
        plt.title('E')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(L.reshape(size_x, size_y))
        plt.title('L')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(Q.reshape(size_x, size_y))
        plt.title('Q')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(κ.reshape(size_x, size_y))
        plt.title('κ')
        plt.colorbar()
        plt.show()
        plt.close()

    #for i in range(size_x * size_y):

    #    print('x=%f  y=%f  r=%f  θ=%f  ϕ=%f  t=0.000000  pr=%f  pθ=%.9f  E=%f  L=%f  Q=%f  κ=%f' % (
    #        x_obs[i], y_obs[i],
    #        r[i], θ[i], ϕ[i],
    #        pr[i], pθ[i],
    #        E[i], L[i], Q[i], κ[i]
    #    ), flush = True)

    #sys.exit(0)

    ####################################################################################################################

    r_sky = np.full(size_x * size_y, np.nan, dtype = np.float64)
    θ_sky = np.full(size_x * size_y, np.nan, dtype = np.float64)
    ϕ_sky = np.full(size_x * size_y, np.nan, dtype = np.float64)

    steps = np.empty(size_x * size_y, dtype = np.int32)

    for i in tqdm.tqdm(range(r.shape[0])):

        var, step = integrate(r[i], θ[i], ϕ[i], pr[i], pθ[i], a, L[i], κ[i])

        r_sky[i] = var[0]
        θ_sky[i] = var[1]
        ϕ_sky[i] = var[2]

        steps[i] = step

    ####################################################################################################################

    if True:

        plt.imshow(r_sky.reshape(size_x, size_y))
        plt.title('r_sky')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(θ_sky.reshape(size_x, size_y))
        plt.title('θ_sky')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(ϕ_sky.reshape(size_x, size_y))
        plt.title('ϕ_sky')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(steps.reshape(size_x, size_y))
        plt.title('steps')
        plt.colorbar()
        plt.show()
        plt.close()

########################################################################################################################

if __name__ == '__main__':

    size = 250

    inclination = 85.0

    ray_tracer(
        0.999,
        #
        10.0,
        np.pi / 180.0 * inclination,
        0.0,
        #
        size,
        size,
        25.0
    )

########################################################################################################################
