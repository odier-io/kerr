# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import numba as nb

from kerr.camera import Camera

########################################################################################################################

# noinspection PyPep8Naming
def initial(camera: Camera, θ_obs: np.ndarray, ϕ_obs: np.ndarray):

    return _initial(
        camera.a,
        camera.Δ,
        camera.ρ,
        camera.α,
        camera.ϖ,
        camera.ω,
        #
        camera.θ,
        camera.br,
        camera.bθ,
        camera.bϕ,
        camera.speed,
        #
        θ_obs,
        ϕ_obs
    )

########################################################################################################################

# noinspection PyPep8Naming, PyRedundantParentheses
@nb.njit(fastmath = True)
def _initial(
    a: float,
    Δ: float,
    ρ: float,
    α: float,
    ϖ: float,
    ω: float,
    #
    θ_cam: float,
    br_cam: float,
    bθ_cam: float,
    bϕ_cam: float,
    speed_cam,
    #
    θ_obs: np.ndarray,
    ϕ_obs: np.ndarray
):

    ####################################################################################################################
    # FIDUCIAL OBSERVER                                                                                                #
    ####################################################################################################################

    x_obs = np.sin(θ_obs) \
            * np.cos(ϕ_obs)
    y_obs = np.sin(θ_obs) \
            * np.sin(ϕ_obs)
    z_obs = np.cos(θ_obs)

    ####################################################################################################################

    n = np.sqrt(1.0 - speed_cam * speed_cam)

    d = 1.0 - speed_cam * y_obs

    x_fido = -x_obs * n / d
    y_fido = -(y_obs - speed_cam) / d
    z_fido = -z_obs * n / d

    ####################################################################################################################

    k = np.sqrt(1.0 - bθ_cam * bθ_cam)

    r_fido = (+x_fido * bϕ_cam / k) + (y_fido * br_cam) + (z_fido * br_cam * bθ_cam / k)
    θ_fido = y_fido * bθ_cam - z_fido * k
    ϕ_fido = (-x_fido * br_cam / k) + (y_fido * bϕ_cam) + (z_fido * bθ_cam * bϕ_cam / k)

    ####################################################################################################################
    # INITIAL CONDITIONS                                                                                               #                                                                                                             #
    ####################################################################################################################

    E = 1.0 / (α + ϖ * ω * ϕ_fido)

    ####################################################################################################################

    p_r = r_fido * ρ / (Δ)         # !!! This is p_r renormalized to E to simplify geodesic calculations
    p_θ = θ_fido * ρ / 1.0         # !!! This is p_θ renormalized to E to simplify geodesic calculations
    p_ϕ = ϕ_fido * ϖ / 1.0         # !!! This is p_ϕ renormalized to E to simplify geodesic calculations

    ####################################################################################################################

    L = p_ϕ

    ####################################################################################################################

    C = p_ϕ * p_ϕ + np.cos(θ_cam) ** 2 * (L * L / np.sin(θ_cam) ** 2 - a * a)

    ####################################################################################################################

    return p_r, p_θ, E, L, C

########################################################################################################################
