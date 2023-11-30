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
        The black hole spin ∈ ]-1,+1[.
    r_obs : float
        The observer radial distance.
    θ_obs : float
        The observer polar angle ∈ [0,π].
    ϕ_obs : float
        The observer azimuthal angle ∈ [0,2π].
    x : np.ndarray
        The 1st cartesian coordinate.
    y : np.ndarray
        The 2nd cartesian coordinate.
    z : np.ndarray
        The 3rd cartesian coordinate.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
        The resulting black hole coordinates.
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

    Parameters
    ----------
    a : float
        The black hole spin ∈ ]-1,+1[.
    x : np.ndarray
        The 1st cartesian coordinate.
    y : np.ndarray
        The 2nd cartesian coordinate.
    z : np.ndarray
        The 3rd cartesian coordinate.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
        The resulting r, θ, ϕ Boyer-Lindquist coordinates.
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
