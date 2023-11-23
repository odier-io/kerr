# -*- coding: utf-8 -*-
########################################################################################################################

import tqdm

import numpy as np

from kerr.camera import Camera
from kerr.initial import initial
from kerr.geodesic import integrate

import matplotlib.pyplot as plt

########################################################################################################################

# noinspection PyPep8Naming
def ray_tracer(a: float, r_cam: float, θ_cam: float, ϕ_cam: float, br_cam: float, bθ_cam: float, bϕ_cam: float, x: int, y: int):

    ####################################################################################################################

    camera = Camera(a, r_cam, θ_cam, ϕ_cam, br_cam, bθ_cam, bϕ_cam)

    ####################################################################################################################

    θ, ϕ = np.meshgrid(
        np.linspace(0.0, 1.0 * np.pi, x),
        np.linspace(0.0, 2.0 * np.pi, y)
    )

    ####################################################################################################################

    p_r, p_θ, E, L, C = initial(camera, θ.reshape(x * y), ϕ.reshape(x * y))

    ####################################################################################################################

    if True:

        plt.imshow(p_r.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(p_θ.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(E.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(L.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(C.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

    ####################################################################################################################

    θ_sky = np.empty(x * y, dtype = np.float32)
    ϕ_sky = np.empty(x * y, dtype = np.float32)

    steps = np.empty(x * y, dtype = np.int32)

    for i in tqdm.tqdm(range(p_r.shape[0])):

        var, step = integrate(r_cam, θ_cam, ϕ_cam, p_r[i], p_θ[i], a, L[i], C[i])

        θ_sky[i] = var[1]
        ϕ_sky[i] = var[2]

        steps[i] = step

    ####################################################################################################################

    if True:

        plt.imshow(θ_sky.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(ϕ_sky.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(steps.reshape(x, y))
        plt.colorbar()
        plt.show()
        plt.close()

########################################################################################################################

if __name__ == '__main__':

    ray_tracer(
        0.5,
        5.0,
        1.5708,
        0.0,
        0.0,
        0.0,
        0.1,
        64,
        64
    )

########################################################################################################################
