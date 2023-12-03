# -*- coding: utf-8 -*-
########################################################################################################################

import sys
import math
import timeit

import tqdm

import numpy as np
import numba as nb

from PIL import Image

from kerr.initial import initial
from kerr.geodesic import integrate

import matplotlib.image as img
import matplotlib.pyplot as plt

########################################################################################################################

@nb.njit(parallel = True)
def ray_tracer_step2(
    r_sky, θ_sky, ϕ_sky,
    steps,
    r, θ, ϕ, pr, pθ,
    a, L, κ
):

    for i in nb.prange(r.shape[0]):

        var, steps[i] = integrate(r[i], θ[i], ϕ[i], pr[i], pθ[i], a, L[i], κ[i])

        r_sky[i] = var[0]
        θ_sky[i] = var[1]
        ϕ_sky[i] = var[2]

########################################################################################################################

# noinspection PyPep8Naming
def ray_tracer(a: float, r_cam: float, θ_cam: float, ϕ_cam: float, size_x: int, size_y: int, width: float):

    ####################################################################################################################

    x_obs0 = 1.000000000000000 * width * np.linspace(+0.5, -0.5, size_x)
    y_obs0 = (size_y / size_x) * width * np.linspace(-0.5, +0.5, size_y)

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

        plt.imshow(r.reshape(size_y, size_x))
        plt.title('r')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(θ.reshape(size_y, size_x))
        plt.title('θ')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(ϕ.reshape(size_y, size_x))
        plt.title('ϕ')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(pr.reshape(size_y, size_x))
        plt.title('pr')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(pθ.reshape(size_y, size_x))
        plt.title('pθ')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(E.reshape(size_y, size_x))
        plt.title('E')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(L.reshape(size_y, size_x))
        plt.title('L')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(Q.reshape(size_y, size_x))
        plt.title('Q')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(κ.reshape(size_y, size_x))
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

    ####################################################################################################################

    r_sky = np.full(size_x * size_y, np.nan, dtype = np.float64)
    θ_sky = np.full(size_x * size_y, np.nan, dtype = np.float64)
    ϕ_sky = np.full(size_x * size_y, np.nan, dtype = np.float64)

    steps = np.empty(size_x * size_y, dtype = np.int32)

    time0 = timeit.default_timer()

    ray_tracer_step2(
        r_sky, θ_sky, ϕ_sky,
        steps,
        r, θ, ϕ, pr, pθ,
        a, L, κ
    )

    time1 = timeit.default_timer()

    print('Ray tracing done within {}s'.format(time1 - time0))

    ####################################################################################################################

    if True:

        plt.imshow(r_sky.reshape(size_y, size_x))
        plt.title('r_sky')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(θ_sky.reshape(size_y, size_x))
        plt.title('θ_sky')
        plt.show()
        plt.close()

        plt.imshow(ϕ_sky.reshape(size_y, size_x))
        plt.title('ϕ_sky')
        plt.show()
        plt.close()

        plt.imshow(steps.reshape(size_y, size_x))
        plt.title('steps')
        plt.colorbar()
        plt.show()
        plt.close()

    ####################################################################################################################

    return (
        θ_sky.reshape(size_y, size_x),
        ϕ_sky.reshape(size_y, size_x),
    )

########################################################################################################################

if __name__ == '__main__':

    ####################################################################################################################

    size_x = 300
    size_y = 150

    inclination = 85.0

    θ_sky, ϕ_sky = ray_tracer(
        0.5,
        #
        10.0,
        np.pi / 180.0 * inclination,
        0.0,
        #
        size_x,
        size_y,
        50.0
    )

    sys.exit(0)

    ####################################################################################################################

    sky_orig = img.imread('/Users/jodier/PycharmProjects/kerr/rainbow.png')

    X = sky_orig.shape[1]
    Y = sky_orig.shape[0]

    ####################################################################################################################

    θ_sky_image = np.array(Image.fromarray(θ_sky).resize((Y, X), Image.Resampling.LANCZOS))
    ϕ_sky_image = np.array(Image.fromarray(ϕ_sky).resize((Y, X), Image.Resampling.LANCZOS))

    print(θ_sky_image.shape)

    ####################################################################################################################

    sky_bh = np.zeros((Y, X, 3), dtype = np.float32)

    for y in range(Y):
        for x in range(X):

            θ = θ_sky_image[y, x]
            ϕ = ϕ_sky_image[y, x]

            if not math.isnan(θ)\
               and              \
               not math.isnan(ϕ):

                i = int(θ / (1.0 * np.pi) * X)
                j = int(ϕ / (2.0 * np.pi) * Y)

                sky_bh[y, x] = sky_orig[j, i]

    ####################################################################################################################

    plt.imshow(sky_orig)
    plt.show()

    plt.imshow(sky_bh)
    plt.show()

    ####################################################################################################################

    sys.exit(0)

########################################################################################################################
