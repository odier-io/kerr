# -*- coding:utf-8 -*-
########################################################################################################################

"""Initial conditions of particles starting from the observer’s grid."""

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
    Initial conditions of particles starting from the observer’s grid.

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
        The rest mass (0 for massless particles, 1 otherwise).

    Returns
    -------
        The initial condition tuple :math:`(r,\\theta,\\phi,p_r,p_\\theta,E,L,Q,\\kappa)`.

    Notes
    -----
    From the Kerr lagrangian:

    .. math::
        \\begin{eqnarray}
            \\mathscr{L}(x^\\mu,\\dot{x}^\\mu)&=&\\frac{1}{2}g_{\\mu\\nu}\\dot{x}^\\mu\\dot{x}^\\nu\\\\
                                              &=&\\frac{1}{2}\\left[-\\left(1-\\frac{2r}{\\Sigma}\\right)\\dot{t}^2-\\frac{4ar\\sin^2\\theta}{\\Sigma}\\dot{t}\\dot{\\phi}\\right.\\\\
                                              & &\\left.+\\frac{\\Sigma}{\\Delta}\\dot{r}^2+\\frac{\\Sigma}{1}\\dot{\\theta}^2+\\left(r^2+a^2+\\frac{2a^2r\\sin^2\\theta}{\\Sigma}\\right)\\sin^2\\theta\,\\dot{\\phi}^2\\right]\\\\
        \\end{eqnarray}

    .. math::
        \\Sigma\\equiv r^2+a^2\\cos^2\\theta

    .. math::
        \\Delta\\equiv r^2-2r+a^2

    initial conditions are:

    .. math::
        \\left\\{
        \\begin{eqnarray}
            p_t\\equiv\\frac{\\partial\\mathscr{L}}{\\partial\\dot{t}}&=&-E\\\\
            p_r\\equiv\\frac{\\partial\\mathscr{L}}{\\partial\\dot{r}}&=&\\frac{\\Sigma}{\\Delta}\\dot{r}\\\\
            p_\\theta\\equiv\\frac{\\partial\\mathscr{L}}{\\partial\\dot{\\theta}}&=&\\frac{\\Sigma}{1}\\dot{\\theta}\\\\
            p_\\phi\\equiv\\frac{\\partial\\mathscr{L}}{\\partial\\dot{\\phi}}&=&+L_z\\\\
        \\end{eqnarray}
        \\right.

    where (see :ref:`[1] <reference_1>`):

    .. math::
        E=\\sqrt{\\frac{\\Sigma -2r}{\\Sigma\\Delta}\\left(\\Sigma\\dot{r}^2+\\Sigma\\Delta\\dot{\\theta}^2+\\Delta\\mu\\right)+\\Delta\\sin^2\\theta\\,\\dot{\\phi}^2}

    .. math::
        L_z=\\left[\\frac{\\Sigma\\Delta\\dot{\\phi}-2arE}{\\Sigma-2r}\\right]\\sin^2\\theta

    .. math::
        Q=p_\\theta^2+\\left[\\frac{L_z^2}{\\sin^2\\theta}+a^2(\\mu-E^2)\\right]\\cos^2\\theta

    .. math::
        \\kappa=Q+L_z^2-a^2(\\mu-E^2)

    :math:`a\\equiv\\frac{L}{M}` is the Kerr parameter (conventionally, :math:`M=1`), :math:`L_z` is the projection of the particle angular momentum along the black hole spin axis and :math:`Q` the Carter constant and :math:`\\kappa` another constant that is always non-negative.

    In the black hole system coordinate, initial cartesian velocities :math:`(\\dot{x},\\dot{y},\\dot{z})` are determined by differentiating :func:`kerr.coord.obs_to_bh` along the photon arrival direction (= z direction):

    .. math::
        \\left\\{
        \\begin{eqnarray}
            \\dot{x}&=&-\\sin\\theta_\\text{obs}\\cos\\phi_\\text{obs}\\\\
            \\dot{y}&=&-\\sin\\theta_\\text{obs}\\sin\\phi_\\text{obs}\\\\
            \\dot{z}&=&-\\cos\\theta_\\text{obs}\\\\
        \\end{eqnarray}
        \\right.

    Then, initial spherical velocities :math:`(\\dot{r},\\dot{\\theta},\\dot{\\phi})` are determined by differentiating :math:`(r,\\theta,\\phi)` = :func:`kerr.coord.cartesian_to_boyer_lindquist` and substituting :math:`(x,y,z)\\to` :func:`kerr.coord.boyer_lindquist_to_cartesian` and :math:`(\\dot{x},\\dot{y},\\dot{z})`:

    .. math::
        \\left\\{
        \\begin{eqnarray}
            \\dot{r}&=&\\frac{\\partial r}{\\partial x}\\dot{x}+\\frac{\\partial r}{\\partial y}\\dot{y}+\\frac{\\partial r}{\\partial z}\\dot{z}\\\\
            \\dot{\\theta}&=&\\frac{\\partial\\theta}{\\partial x}\\dot{x}+\\frac{\\partial\\theta}{\\partial y}\\dot{y}+\\frac{\\partial\\theta}{\\partial z}\\dot{z}\\\\
            \\dot{\\phi}&=&\\frac{\\partial\\phi}{\\partial x}\\dot{x}+\\frac{\\partial\\phi}{\\partial y}\\dot{y}+\\frac{\\partial\\phi}{\\partial z}\\dot{z}\\\\
        \\end{eqnarray}
        \\right|_{\\text{subs. }(x,y,z)\\text{ and }(\\dot{x},\\dot{y},\\dot{z})}

    This gives:

    .. math::
        \\left\\{
        \\begin{eqnarray}
            \\dot{r}&=&-\\frac{r\\mathcal{R}\\sin\\theta\\sin\\theta_\\text{obs}\\cos\\Phi+\\mathcal{R}^2\\cos\\theta\\cos\\theta_\\text{obs}}{\\Sigma}\\\\
            \\dot{\\theta}&=&+\\frac{r\\sin\\theta\\cos\\theta_\\text{obs}-\\mathcal{R}\\cos\\theta\\sin\\theta_\\text{obs}\\cos\\Phi}{\\Sigma}\\\\
            \\dot{\\phi}&=&\\frac{\\sin\\theta_\\text{obs}\\sin\\Phi}{\\mathcal{R}\\sin\\theta}\\\\
        \\end{eqnarray}
        \\right.

    where :math:`\\mathcal{R}\\equiv\\sqrt{r^2+a^2}` and :math:`\\Phi\\equiv\\phi-\\phi_\\text{obs}`.
    """

    if µ != 0.0 and µ != 1.0:

        raise ValueError('Rest mass have to be either 0 or 1.')

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

    Σ = r2_bh + a2 * cos2θ_bh

    ####################################################################################################################
    # INITIAL CONDITIONS - STEP 1                                                                                      #
    ####################################################################################################################

    zdot = -1.0

    rdot_bh = zdot * (-r_bh * R1 * sinθ_bh * sinθ_obs * cosΦ - R2 * cosθ_bh * cosθ_obs) / Σ

    θdot_bh = zdot * (+r_bh * sinθ_bh * cosθ_obs - R1 * cosθ_bh * sinθ_obs * cosΦ) / Σ

    ϕdot_bh = zdot * (sinθ_obs * sinΦ) / (R1 * sinθ_bh)

    ####################################################################################################################
    # INITIAL CONDITIONS - STEP 2                                                                                      #
    ####################################################################################################################

    pr_bh = rdot_bh * Σ / (Δ)
    pθ_bh = θdot_bh * Σ / 1.0

    ####################################################################################################################

    E = np.sqrt((Σ - 2.0 * r_bh) * (Σ * rdot_bh * rdot_bh + Σ * Δ * θdot_bh * θdot_bh + Δ * µ) / (Σ * Δ) + Δ * sin2θ_bh * ϕdot_bh * ϕdot_bh)

    L = (Σ * Δ * ϕdot_bh - 2.0 * a * r_bh * E) * sin2θ_bh / (Σ - 2.0 * r_bh)

    ####################################################################################################################

    E2 = E * E
    L2 = L * L

    a21mu = a2 * (µ - E2)

    Q = pθ_bh * pθ_bh + (L2 / sin2θ_bh + a21mu) * cos2θ_bh

    κ = Q + L2 - a21mu

    ####################################################################################################################

    return (
        r_bh, θ_bh, ϕ_bh,
        pr_bh, pθ_bh,
        #
        E, L,
        Q, κ
    )

########################################################################################################################
