# -*- coding: utf-8 -*-
########################################################################################################################

"""Coordinate system transformations."""

########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

########################################################################################################################

@nb.njit
def obs_to_bh(
    a: float,
    r_obs: float,
    θ_obs: float,
    ϕ_obs: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Transforms the observer coordinates to the black hole coordinates.

    Parameters
    ----------
    a : float
        The black hole spin :math:`\in ]-1,+1[`.
    r_obs : float
        The observer radial distance.
    θ_obs : float
        The observer polar angle :math:`\in [0,\\pi]`.
    ϕ_obs : float
        The observer azimuthal angle :math:`\in [0,2\\pi]`.
    x : np.ndarray
        The :math:`x` cartesian coordinate.
    y : np.ndarray
        The :math:`y` cartesian coordinate.
    z : np.ndarray
        The :math:`z` cartesian coordinate.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
        The resulting :math:`(x',y',z')` black hole coordinates.
    """

    ####################################################################################################################

    a2 = a * a

    r2_obs = r_obs * r_obs

    ####################################################################################################################

    sinθ_obs = math.sin(θ_obs)
    cosθ_obs = math.cos(θ_obs)

    sinϕ_obs = math.sin(ϕ_obs)
    cosϕ_obs = math.cos(ϕ_obs)

    ####################################################################################################################

    d = (np.sqrt(r2_obs + a2) - z) * sinθ_obs - y * cosθ_obs

    δ = r_obs - z

    ####################################################################################################################

    return (
        d * cosϕ_obs - x * sinϕ_obs,
        d * sinϕ_obs + x * cosϕ_obs,
        δ * cosθ_obs + y * sinθ_obs,
    )

########################################################################################################################

@nb.njit
def cartesian_to_boyer_lindquist(
    a: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Transforms cartesian coordinates into Boyer-Lindquist coordinates.

    With :math:`w\equiv x^2+y^2+z^2-a^2`:

    .. math::
        \\begin{cases}
            r=\\sqrt{\\frac{w+\\sqrt{w^2+4a^2z^2}}{2}}\\\\
            \\theta=\\mathrm{acos}(z/r)\\\\
            \\phi=\\mathrm{atan2}(y,x)\\\\
        \end{cases}

    Parameters
    ----------
    a : float
        The black hole spin :math:`\in ]-1,+1[`.
    x : np.ndarray
        The :math:`x` cartesian coordinate.
    y : np.ndarray
        The :math:`y` cartesian coordinate.
    z : np.ndarray
        The :math:`z` cartesian coordinate.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
        The resulting :math:`(r,θ,ϕ)` Boyer-Lindquist coordinates.
    """

    ####################################################################################################################

    a2 = a * a
    x2 = x * x
    y2 = y * y
    z2 = z * z

    ####################################################################################################################

    w = x2 + y2 + z2 - a2

    ####################################################################################################################

    r = np.sqrt((w + np.sqrt(w * w + 4.0 * a2 * z2)) / 2.0)

    return (
        r,
        np.arccos(z / r),
        np.arctan2(y, x),
    )

########################################################################################################################

@nb.njit
def boyer_lindquist_to_cartesian(
    a: float,
    r: np.ndarray,
    θ: np.ndarray,
    ϕ: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Transforms Boyer-Lindquist coordinates into cartesian coordinates.

    .. math::
        \\begin{cases}
            x = \\sqrt{r^2 + a^2}\\sin\\theta\\cos\\phi\\\\
            y = \\sqrt{r^2 + a^2}\\sin\\theta\\sin\\phi\\\\
            z = r\\cos\\theta\\\\
        \end{cases}

    Parameters
    ----------
    a : float
        The black hole spin :math:`\in ]-1,+1[`.
    r : np.ndarray
        The :math:`r` Boyer-Lindquist coordinate.
    θ : np.ndarray
        The :math:`\\theta` Boyer-Lindquist coordinate.
    ϕ : np.ndarray
        The :math:`\\phi` Boyer-Lindquist coordinate.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
        The resulting :math:`(x,y,z)` cartesian coordinates.
    """

    ####################################################################################################################

    r2 = r * r
    a2 = a * a

    R = np.sqrt(r2 + a2)

    ####################################################################################################################

    return (
        R * np.sin(θ) * np.cos(ϕ),
        R * np.sin(θ) * np.sin(ϕ),
        r * np.cos(θ),
    )

########################################################################################################################
