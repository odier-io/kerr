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

########################################################################################################################

C1 = +9.788359788359788e-02
C2 = +0.000000000000000e+00
C3 = +4.025764895330113e-01
C4 = +2.104377104377105e-01
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

N_MAX_STEPS = 10_000

########################################################################################################################

# noinspection PyPep8Naming
def model(result_dydx, var, a, L, C) -> None:

    """
    System of differential equations in Boyer–Lindquist coordinates for computing photon geodesics:

    .. math::
        \\frac{dr}{dz}=p_r\\times\\frac{\\Delta}{\\Sigma}

    .. math::
        \\frac{d\\theta}{dz}=p_\\theta\\times\\frac{1}{\\Sigma}

    .. math::
        \\frac{d\\phi}{dz}=\\frac{2ar+(\\Sigma-2r)L/\\sin^2\\theta}{\\Sigma\\Delta}

    .. math::
        \\frac{dp_r}{dz}=\\frac{(-\\kappa)(r-1)+2r(r^2+a^2)-2aL}{\\Sigma\\Delta}-\\frac{2p_r^2(r-1)}{\\Sigma}

    .. math::
        \\frac{dp_\\theta}{dz}=\\frac{\\sin\\theta\\cos\\theta}{\\Sigma}(L^2/\\sin^4\\theta-a^2)

    Where :math:`a\\equiv\\frac{J}{M}` is the Kerr parameter (conventionally, :math:`M=1`), :math:`L` is the projection of the particle angular momentum along the black hole spin axis, :math:`C` the Carter constant and:

    .. math::
        \\Sigma\\equiv r^2+a^2\\cos^2\\theta

    .. math::
        \\Delta\\equiv r^2-2r+a^2

    .. math::
        \\kappa\\equiv C+L^2+a^2
    """

    ####################################################################################################################

    r = var[0]
    θ = var[1]
    # ϕ not needed
    # t not needed
    pr = var[4]
    pθ = var[5]

    r2 = r * r
    a2 = a * a
    L2 = L * L

    twor = 2.0 * r
    twoa = 2.0 * a

    sinθ = math.sin(θ)
    cosθ = math.cos(θ)

    sin2θ = sinθ * sinθ
    cos2θ = cosθ * cosθ

    if sinθ < TINY:
        sinθ = TINY
        sin2θ = TINY

    ####################################################################################################################

    Σ = r2 + a2 * cos2θ

    Δ = r2 - twor + a2

    κ = C + L2 + a2

    ####################################################################################################################

    ΣΔ = Σ * Δ

    ####################################################################################################################

    # drdz
    result_dydx[0] = pr * (Δ / Σ)
    # dθdz
    result_dydx[1] = pθ * (1.0 / Σ)
    # dϕdz
    result_dydx[2] = (2.0 * a * r + (Σ - twor) * L / sin2θ) / ΣΔ
    # dprdz
    result_dydx[3] = ((-κ) * (r - 1.0) + twor * (r2 + a2) - twoa * L) / ΣΔ - (pr * pr) * (twor - 2.0) / Σ
    # dpθdz
    result_dydx[4] = (sinθ * cosθ) * (L2 / (sin2θ * sin2θ) - a2) / Σ

########################################################################################################################

# noinspection PyPep8Naming
@nb.jit(nopython = True)
def rkck(
    result_var: np.ndarray,
    result_err: np.ndarray,
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

        result_var[i] = var[i] + h * (C1 * dvdz[i] + C3 * ak3[i] + C4 * ak4[i] + C6 * ak6[i])

        result_err[i] = 0.0000 + h * (D1 * dvdz[i] + D3 * ak3[i] + D4 * ak4[i] + D5 * ak5[i] + D6 * ak6[i])

########################################################################################################################

# noinspection PyPep8Naming
@nb.jit(nopython = True)
def rkqs(
    var_tmp: np.ndarray,
    var_err: np.ndarray,
    #
    var: np.ndarray,
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
    tmp: np.ndarray
) -> typing.Tuple[float, float]:

    ####################################################################################################################

    rkck(var_tmp, var_err, var, a, L, C, dvdz, h, aks, tmp)

    ####################################################################################################################

    err_max = np.max(np.abs(var_err / scale)) / accuracy

    ####################################################################################################################

    if err_max <= 1.0:

        z += h

        var[:] = var_tmp

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
@nb.jit(nopython = True)
def odeint(
    var: np.ndarray,
    a: float,
    L: float,
    C: float,
    #
    z_end: float,
    accuracy: float,
    default_step: float,
) -> typing.Tuple[np.ndarray, int]:

    ####################################################################################################################

    dim = var.shape[0]

    ####################################################################################################################

    var_tmp = np.zeros(dim)
    var_err = np.zeros(dim)

    dvdz = np.zeros(dim)
    aks = np.zeros((5, dim))
    tmp = np.zeros(dim)

    ####################################################################################################################

    z = 0.000000000000000000000000000
    h = np.sign(z_end) * default_step

    for step in range(N_MAX_STEPS):

        model(dvdz, var, a, L, C)

        scale = np.abs(var) + np.abs(dvdz * h) + TINY

        z, h = rkqs(var_tmp, var_err, var, a, L, C, dvdz, z, h, accuracy, scale, aks, tmp)

        if z >= z_end:

            return var, step

########################################################################################################################

# noinspection PyPep8Naming
def integrate(
    r: float, θ: float, ϕ: float, t: float, pr: float, pθ: float,
    a: float, L: float, C: float,
    z_end: float = -1.0e+7,
    accuracy: float = +1.0e-5,
    default_step: float = +1.0e-2
) -> typing.Tuple[np.ndarray, int]:

    ####################################################################################################################

    var, step = odeint(np.array([r, θ, ϕ, t, pr, pθ]), a, L, C, z_end, accuracy, default_step)

    ####################################################################################################################

    θ = var[1]
    ϕ = var[2]

    θ = math.fmod(θ, TWO_PI)
    while θ < 0: θ += TWO_PI

    if θ > ONE_PI:
        θ = TWO_PI - θ
        ϕ = ONE_PI + ϕ

    while ϕ < 0: ϕ += TWO_PI
    ϕ = math.fmod(ϕ, TWO_PI)

    var[1] = θ
    var[2] = ϕ

    ####################################################################################################################

    return var, step

########################################################################################################################
