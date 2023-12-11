# -*- coding:utf-8 -*-
########################################################################################################################

"""Photon initial conditions."""

########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

from kerr.coord import obs_to_bh, cartesian_to_boyer_lindquist

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
    z: np.ndarray,
    µ: float = 0.0
) -> typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:

    """
    In the black hole coordinate system, initial conditions are:

    .. math::
        p_{r}=\\frac{\\rho^2}{\\Delta}\\dot{r}

    .. math::
        p_{\\theta}=\\frac{\\rho^2}{1}\\dot{\\theta}

    .. math::
        p_{\\phi}=L_z

    .. math::
        E=\\sqrt{\\frac{\\rho^2 -2r}{\\rho^2\\Delta}\\left(\\rho^2\\dot{r}^2+\\rho^2\\Delta\\dot{\\theta}^2-\\Delta\\mu\\right)+\\Delta\\sin^2\\theta\\,\\dot{\\phi}^2}

    .. math::
        L_z=\\left[\\frac{\\rho^2\\Delta\\dot{\\phi}-2arE}{\\rho^2-2r}\\right]\\sin^2\\theta

    .. math::
        Q=p_\\theta^2+\\left[\\frac{L_z^2}{\\sin^2\\theta}-a^2(E^2+\\mu)\\right]\\cos^2\\theta

    where:

    .. math::
        \\rho^2\\equiv r^2+a^2\\cos^2\\theta

    .. math::
        \\Delta\\equiv r^2-2r+a^2

    and:

    .. math::
        \\dot{r}=-\\frac{r\\mathcal{R}\\sin\\theta\\sin\\theta_{obs}\\cos\\Phi+\\mathcal{R}^2\\cos\\theta\\cos\\theta_{obs}}{\\rho^2}

    .. math::
        \\dot{\\theta}=+\\frac{r\\sin\\theta\\cos\\theta_{obs}-\\mathcal{R}\\cos\\theta\\sin\\theta_{obs}\\cos\\Phi}{\\rho^2}

    .. math::
        \\dot{\\phi}=\\frac{\\sin\\theta_{obs}\\sin\\Phi}{\\mathcal{R}\\sin\\theta}

    and :math:`\\mathcal{R}\\equiv\\sqrt{r^2+a^2}` and :math:`\\Phi\\equiv\\phi-\\phi_{obs}`.

    .. codeblock:: python

        x_bh, y_bh, z_bh = obs_to_bh(a, r_obs, θ_obs, ϕ_obs, x, y, z)
        r, θ, ϕ = cartesian_to_boyer_lindquist(a, x_bh, y_bh, z_bh)

    Parameters
    ----------
    a : float
        The black hole spin :math:`\\in ]-1,+1[`.
    r_obs : float
        The observer radial distance.
    θ_obs : float
        The observer polar angle :math:`\\in [0,\\pi]`.
    ϕ_obs : float
        The observer azimuthal angle :math:`\\in [0,2\\pi]`.
    x : np.ndarray
        The :math:`x` cartesian coordinate.
    y : np.ndarray
        The :math:`y` cartesian coordinate.
    z : np.ndarray
        The :math:`z` cartesian coordinate.
    µ : float
        The rest mass (0 for massless particles and -1 for particles with mass).
    """

    if µ != 0.0 and µ != -1.0:

        raise ValueError('Rest mass have to be either 0 or -1.')

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
    # INITIAL CONDITIONS - STEP 1                                                                                      #
    ####################################################################################################################

    zdot = -1.0

    rdot_bh = zdot * (-r_bh * R1 * sinθ_bh * sinθ_obs * cosΦ - R2 * cosθ_bh * cosθ_obs) / ρ2

    θdot_bh = zdot * (+r_bh * sinθ_bh * cosθ_obs - R1 * cosθ_bh * sinθ_obs * cosΦ) / ρ2

    ϕdot_bh = zdot * (sinθ_obs * sinΦ) / (R1 * sinθ_bh)

    ####################################################################################################################
    # INITIAL CONDITIONS - STEP 2                                                                                      #
    ####################################################################################################################

    pr_bh = rdot_bh * ρ2 / (Δ)
    pθ_bh = θdot_bh * ρ2 / 1.0

    ####################################################################################################################

    E = np.sqrt((ρ2 - 2.0 * r_bh) * (ρ2 * rdot_bh * rdot_bh + ρ2 * Δ * θdot_bh * θdot_bh - Δ * µ) / (ρ2 * Δ) + Δ * sin2θ_bh * ϕdot_bh * ϕdot_bh)

    L = (ρ2 * Δ * ϕdot_bh - 2.0 * a * r_bh * E) * sin2θ_bh / (ρ2 - 2.0 * r_bh)

    ####################################################################################################################

    pr_bh /= E # For simplifying geodesic calculations
    pθ_bh /= E # For simplifying geodesic calculations
    L     /= E # For simplifying geodesic calculations

    ####################################################################################################################

    L2 = L * L

    a21mu = a2 * (1.0 + µ)

    Q = pθ_bh * pθ_bh + (L2 / sin2θ_bh - a21mu) * cos2θ_bh

    κ = Q + L2 + a21mu

    ####################################################################################################################

    return (
        r_bh, θ_bh, ϕ_bh,
        pr_bh, pθ_bh,
        #
        E, L,
        Q, κ
    )

########################################################################################################################
