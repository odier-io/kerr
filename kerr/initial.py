# -*- coding:utf-8 -*-
########################################################################################################################

"""Photon initial conditions."""

########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

from kerr.transformation import obs_to_bh, cartesian_to_boyer_lindquist

########################################################################################################################

# noinspection PyPep8Naming, PyRedundantParentheses, SpellCheckingInspection
@nb.njit
def initial(
    a: float,
    r_obs: float,
    θ_obs: float,
    ϕ_obs: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:

    """
    For photon (rest mass :math:`\\mu=0`), initial conditions are:

    .. math::
        p_{r}=\\dot{r}\\frac{\\rho^2}{\\Delta}

    .. math::
        p_{\\theta}=\\dot{\\theta}\\frac{\\rho^2}{1}

    .. math::
        E=\\sqrt{(\\rho^2 -2r)\\left(\\frac{\\dot{r}_{bh}^2}{\\Delta}+\\dot{\\theta}^2\\right)+\\Delta\\sin^2\\theta\\dot{\\phi}^2}

    .. math::
        L=\\frac{(\\rho^2\\Delta\\dot{\\phi}- 2arE)\\sin^2\\theta}{\\rho^2-2r}

    where:

    .. math::
        \\dot{r}=-\\frac{r\\mathcal{R}\\sin\\theta\\sin\\theta_{obs}\\cos\\Phi+\\mathcal{R}^2\\cos\\theta\\cos\\theta_{obs}}{\\rho^2}

    .. math::
        \\dot{\\theta}=+\\frac{r\\sin\\theta\\cos\\theta_{obs}-\\mathcal{R}\\cos\\theta\\sin\\theta_{obs}\\cos\\Phi}{\\rho^2}

    .. math::
        \\dot{\\phi}=\\frac{\\sin\\theta_{obs}\\sin\\Phi}{\\mathcal{R}\\sin\\theta}

    and :math:`\\mathcal{R}\\equiv\\sqrt{r^2+a^2}` and :math:`\\Phi\\equiv\\phi-\\phi_{obs}`.
    """

    ####################################################################################################################

    x_bh, y_bh, z_bh = obs_to_bh(a, r_obs, θ_obs, ϕ_obs, x, y, z)

    r_bh, θ_bh, ϕ_bh = cartesian_to_boyer_lindquist(a, x_bh, y_bh, z_bh)

    ####################################################################################################################

    Φ = ϕ_bh - ϕ_obs

    ####################################################################################################################

    sinθ_obs = math.sin(θ_obs)
    cosθ_obs = math.cos(θ_obs)

    sinθ_bh = np.sin(θ_bh)
    cosθ_bh = np.cos(θ_bh)

    sinΦ = np.sin(Φ)
    cosΦ = np.cos(Φ)

    ####################################################################################################################

    sin2θ_bh = sinθ_bh * sinθ_bh
    cos2θ_bh = cosθ_bh * cosθ_bh

    ####################################################################################################################

    a2 = a * a

    r2_bh = r_bh * r_bh

    ####################################################################################################################

    R2 = r2_bh + a2

    R1 = np.sqrt(r2_bh + a2)

    ####################################################################################################################
    # KERR PARAMETERS                                                                                                  #
    ####################################################################################################################

    Δ = r2_bh - 2.0 * r_bh + a2

    ρ2 = r2_bh + a2 * cos2θ_bh

    ####################################################################################################################
    # INITIAL CONDITIONS                                                                                               #
    ####################################################################################################################

    zdot = -1.0

    rdot_bh = zdot * (-r_bh * R1 * sinθ_bh * sinθ_obs * cosΦ - R2 * cosθ_bh * cosθ_obs) / ρ2

    θdot_bh = zdot * (+r_bh * sinθ_bh * cosθ_obs - R1 * cosθ_bh * sinθ_obs * cosΦ) / ρ2

    ϕdot_bh = zdot * (sinθ_obs * sinΦ) / (R1 * sinθ_bh)

    ####################################################################################################################

    pr_bh = rdot_bh * ρ2 / (Δ)
    pθ_bh = θdot_bh * ρ2 / 1.0

    ####################################################################################################################

    E = np.sqrt((ρ2 - 2.0 * r_bh) * (rdot_bh * rdot_bh / Δ + θdot_bh * θdot_bh) + Δ * sin2θ_bh * ϕdot_bh * ϕdot_bh)

    L = (ρ2 * Δ * ϕdot_bh - 2.0 * a * r_bh * E) * sin2θ_bh / (ρ2 - 2.0 * r_bh)

    ####################################################################################################################

    pr_bh /= E # For simplifying geodesic calculations
    pθ_bh /= E # For simplifying geodesic calculations
    L     /= E # For simplifying geodesic calculations

    ####################################################################################################################

    L2 = L * L

    pθ2_bh = pθ_bh * pθ_bh

    Q = pθ2_bh + (L2 / sin2θ_bh - a2) * cos2θ_bh

    κ = Q + L2 + a2

    ####################################################################################################################

    return (
        r_bh, θ_bh, ϕ_bh,
        pr_bh, pθ_bh,
        #
        E, L,
        Q, κ
    )

########################################################################################################################
