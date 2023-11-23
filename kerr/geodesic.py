# -*- coding: utf-8 -*-
########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

########################################################################################################################

TINY = 1.0e-30

ONE_PI = 1.0 * np.pi
TWO_PI = 2.0 * np.pi

########################################################################################################################

# Cash-Karp Parameters for Embedded Runga-Kutta

B21 = +2.000000000000000e-01
B31 = +7.500000000000000e-02
B32 = +2.250000000000000e-01
B41 = +3.000000000000000e-01
B42 = -9.000000000000000e-01
B43 = +1.200000000000000e+00
B51 = -2.037037037037037e-01
B52 = +2.500000000000000e+00
B53 = -2.592592592592593e+00
B54 = +1.296296296296296e+00
B61 = +2.949580439814815e-02
B62 = +3.417968750000000e-01
B63 = +4.159432870370371e-02
B64 = +4.003454137731481e-01
B65 = +6.176757812500000e-02

C1 = +9.788359788359788e-02
C2 = +0.000000000000000e+00
C3 = +4.025764895330113e-01
C4 = +2.104377104377105e-01
C5 = +0.000000000000000e+00
C6 = +2.891022021456804e-01

D1 = -4.293774801587311e-03
D2 = +0.000000000000000e+00
D3 = +1.866858609385785e-02
D4 = -3.415502683080807e-02
D5 = -1.932198660714286e-02
D6 = +3.910220214568039e-02

########################################################################################################################

ERROR_THRESHOLD = 1.89e-4

SAFETY = 0.9
ADAPTIVE = 5.0

POSITIVE_GROWTH = -0.20
POSITIVE_SHRINK = -0.25

N_MAX_ITERS = 10_000

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit(fastmath = True)
def model(out_dydx, var, a, L, C) -> None:

    """
    System of differential equations, in Boyer–Lindquist coordinates, for computing photon geodesics:

    .. math::
        \\frac{dr}{dz}=p_r\\times\\frac{\\Delta}{\\rho^2}

    .. math::
        \\frac{d\\theta}{dz}=p_\\theta\\times\\frac{1}{\\rho^2}

    .. math::
        \\frac{d\\phi}{dz}=\\frac{2ar+(\\rho^2-2r)L/\\sin^2\\theta}{\\rho^2\\Delta}

    .. math::
        \\frac{dp_r}{dz}=\\frac{(-\\kappa)(r-1)+2r(r^2+a^2)-2aL}{\\rho^2\\Delta}-\\frac{2p_r^2(r-1)}{\\rho^2}

    .. math::
        \\frac{dp_\\theta}{dz}=\\frac{\\sin\\theta\\cos\\theta}{\\rho^2}\\left[(L/\\sin^2\\theta)^2-a^2\\right]

    Where :math:`a\\equiv\\frac{J}{M}` is the Kerr parameter (conventionally, :math:`M=1`), :math:`L` is the projection of the particle angular momentum along the black hole spin axis, :math:`C` the Carter constant and:

    .. math::
        \\rho^2\\equiv r^2+a^2\\cos^2\\theta

    .. math::
        \\Delta\\equiv r^2-2r+a^2

    .. math::
        \\kappa\\equiv C+L^2+a^2
    """

    ####################################################################################################################

    r = var[0]
    θ = var[1]
    # ϕ not needed
    pr = var[3]
    pθ = var[4]

    a2 = a * a
    r2 = r * r
    L2 = L * L

    twoa = 2.0 * a
    twor = 2.0 * r

    ####################################################################################################################

    sinθ = math.sin(θ)
    cosθ = math.cos(θ)

    sin2θ = sinθ * sinθ
    cos2θ = cosθ * cosθ

    sin4θ = sin2θ * sin2θ

    if sinθ < TINY:
        sinθ = TINY
        sin2θ = TINY
        sin4θ = TINY

    ####################################################################################################################

    Δ = r2 - twor + a2

    ρ2 = r2 + a2 * cos2θ

    κ = C + L2 + a2

    ####################################################################################################################

    # drdz
    out_dydx[0] = pr * (Δ / ρ2)
    # dθdz
    out_dydx[1] = pθ * (1.0 / ρ2)
    # dϕdz
    out_dydx[2] = (2.0 * a * r + (ρ2 - twor) * L / sin2θ) / (ρ2 * Δ)
    # dprdz
    out_dydx[3] = ((-κ) * (r - 1.0) + twor * (r2 + a2) - twoa * L) / (ρ2 * Δ) - (pr * pr) * (twor - 2.0) / ρ2
    # dpθdz
    out_dydx[4] = (sinθ * cosθ) * (L2 / sin4θ - a2) / ρ2

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit(fastmath = True)
def rkck(
    out_var: np.ndarray,
    out_err: np.ndarray,
    #
    var: np.ndarray,
    a: float,
    L: float,
    C: float,
    #
    dvdz: np.ndarray,
    h: float,
    aks: np.ndarray,
    tmp: np.ndarray
) -> None:

    ####################################################################################################################

    ak2 = aks[0]
    ak3 = aks[1]
    ak4 = aks[2]
    ak5 = aks[3]
    ak6 = aks[4]

    ####################################################################################################################

    for i in range(var.shape[0]):
        tmp[i] = var[i] + h * (B21 * dvdz[i])
    model(ak2, tmp, a, L, C)

    for i in range(var.shape[0]):
        tmp[i] = var[i] + h * (B31 * dvdz[i] + B32 * ak2[i])
    model(ak3, tmp, a, L, C)

    for i in range(var.shape[0]):
        tmp[i] = var[i] + h * (B41 * dvdz[i] + B42 * ak2[i] + B43 * ak3[i])
    model(ak4, tmp, a, L, C)

    for i in range(var.shape[0]):
        tmp[i] = var[i] + h * (B51 * dvdz[i] + B52 * ak2[i] + B53 * ak3[i] + B54 * ak4[i])
    model(ak5, tmp, a, L, C)

    for i in range(var.shape[0]):
        tmp[i] = var[i] + h * (B61 * dvdz[i] + B62 * ak2[i] + B63 * ak3[i] + B64 * ak4[i] + B65 * ak5[i])
    model(ak6, tmp, a, L, C)

    ####################################################################################################################

    for i in range(var.shape[0]):

        out_var[i] = var[i] + h * (C1 * dvdz[i] + C3 * ak3[i] + C4 * ak4[i] + C6 * ak6[i])

        out_err[i] = 0.0000 + h * (D1 * dvdz[i] + D3 * ak3[i] + D4 * ak4[i] + D5 * ak5[i] + D6 * ak6[i])

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit(fastmath = True)
def rkqs(
    inout_var: np.ndarray,
    a: float,
    L: float,
    C: float,
    #
    dvdz: np.ndarray,
    z: float,
    h: float,
    accuracy: float,
    scale: np.ndarray,
    aks: np.ndarray,
    tmp_int: np.ndarray,
    tmp_var: np.ndarray,
    tmp_err: np.ndarray
) -> typing.Tuple[float, float]:

    ####################################################################################################################

    rkck(tmp_var, tmp_err, inout_var, a, L, C, dvdz, h, aks, tmp_int)

    ####################################################################################################################

    err_max = np.max(np.abs(tmp_err / scale)) / accuracy

    ####################################################################################################################

    if err_max <= 1.0:

        z += h

        inout_var[:] = tmp_var

        if err_max < ERROR_THRESHOLD:

            h *= ADAPTIVE

        else:

            h *= SAFETY * pow(err_max, POSITIVE_GROWTH)

    else:

        h *= min(SAFETY * pow(err_max, POSITIVE_SHRINK), 0.1)

    ####################################################################################################################

    return z, h

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit(fastmath = True)
def odeint(
    inout_var: np.ndarray,
    a: float,
    L: float,
    C: float,
    #
    z_end: float,
    accuracy: float,
    default_step: float,
) -> typing.Tuple[np.ndarray, int]:

    ####################################################################################################################

    dim = inout_var.shape[0]

    ####################################################################################################################

    dvdz = np.zeros(dim)

    aks = np.zeros((5, dim))

    tmp_int = np.zeros(dim)
    tmp_var = np.zeros(dim)
    tmp_err = np.zeros(dim)

    ####################################################################################################################

    z = 0.000000000000 * 0x0000000000
    h = np.sign(z_end) * default_step

    for curr_iter in range(N_MAX_ITERS):

        model(dvdz, inout_var, a, L, C)

        scale = np.abs(inout_var) + np.abs(dvdz * h) + TINY

        z, h = rkqs(inout_var, a, L, C, dvdz, z, h, accuracy, scale, aks, tmp_int, tmp_var, tmp_err)

        if z <= z_end:

            return inout_var, curr_iter + 1

    ####################################################################################################################

    return inout_var, N_MAX_ITERS

########################################################################################################################

@nb.njit(fastmath = True)
def wrap(θ: float, ϕ: float) -> typing.Tuple[float, float]:

    θ = θ % TWO_PI
    while θ < 0: θ += TWO_PI

    if θ > ONE_PI:
        θ = TWO_PI - θ
        ϕ = ONE_PI + ϕ

    while ϕ < 0: ϕ += TWO_PI
    ϕ = ϕ % TWO_PI

    return θ, ϕ

########################################################################################################################

# noinspection PyPep8Naming
def integrate(
    r: float, θ: float, ϕ: float, pr: float, pθ: float,
    a: float, L: float, C: float,
    z_end: float = -1.0e+7,
    accuracy: float = +1.0e-5,
    default_step: float = +1.0e-2
) -> typing.Tuple[np.ndarray, int]:

    var, step = odeint(np.array([r, θ, ϕ, pr, pθ], dtype = np.float32), a, L, C, z_end, accuracy, default_step)

    var[1], var[2] = wrap(float(var[1]), float(var[2])) # θ and ϕ

    return var, step

########################################################################################################################
