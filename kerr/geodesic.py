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

POSITIVE_GROWTH = -0.20
POSITIVE_SHRINK = -0.25

N_MAX_ITERS = 1_000

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit
def geodesic(
    out_dydx: np.ndarray,
    y: np.ndarray,
    a: float,
    L: float,
    κ: float
) -> None:

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

    r = y[0]
    θ = y[1]
    # = y[2]
    pr = y[3]
    pθ = y[4]

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

    if sinθ < 1.0e-8:
        sinθ = 1.0e-8
        sin2θ = 1.0e-16
        sin4θ = 1.0e-32

    ####################################################################################################################

    Δ = r2 - twor + a2

    ρ2 = r2 + a2 * cos2θ

    ####################################################################################################################

    # drdz
    out_dydx[0] = pr * (Δ / ρ2)
    # dθdz
    out_dydx[1] = pθ * (1 / ρ2)
    # dϕdz
    out_dydx[2] = (2.0 * a * r + (ρ2 - twor) * L / sin2θ) / (ρ2 * Δ)
    # dprdz
    out_dydx[3] = ((-κ) * (r - 1.0) + twor * (r2 + a2) - twoa * L) / (ρ2 * Δ) - (pr * pr) * (twor - 2.0) / ρ2
    # dpθdz
    out_dydx[4] = (sinθ * cosθ) * (L2 / sin4θ - a2) / ρ2

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit
def rkck(
    out_y: np.ndarray,
    out_e: np.ndarray,
    #
    y: np.ndarray,
    a: float,
    L: float,
    κ: float,
    #
    dvdz: np.ndarray,
    h: float,
    ak2: np.ndarray,
    ak3: np.ndarray,
    ak4: np.ndarray,
    ak5: np.ndarray,
    ak6: np.ndarray,
    tmp: np.ndarray
) -> None:

    dim = y.shape[0]

    ####################################################################################################################

    for i in range(dim):
        tmp[i] = y[i] + h * (B21 * dvdz[i])
    geodesic(ak2, tmp, a, L, κ)

    for i in range(dim):
        tmp[i] = y[i] + h * (B31 * dvdz[i] + B32 * ak2[i])
    geodesic(ak3, tmp, a, L, κ)

    for i in range(dim):
        tmp[i] = y[i] + h * (B41 * dvdz[i] + B42 * ak2[i] + B43 * ak3[i])
    geodesic(ak4, tmp, a, L, κ)

    for i in range(dim):
        tmp[i] = y[i] + h * (B51 * dvdz[i] + B52 * ak2[i] + B53 * ak3[i] + B54 * ak4[i])
    geodesic(ak5, tmp, a, L, κ)

    for i in range(dim):
        tmp[i] = y[i] + h * (B61 * dvdz[i] + B62 * ak2[i] + B63 * ak3[i] + B64 * ak4[i] + B65 * ak5[i])
    geodesic(ak6, tmp, a, L, κ)

    ####################################################################################################################

    for i in range(dim):

        out_y[i] = y[i] + h * (C1 * dvdz[i] + C3 * ak3[i] + C4 * ak4[i] + C6 * ak6[i])

        out_e[i] = 0.00 + h * (D1 * dvdz[i] + D3 * ak3[i] + D4 * ak4[i] + D5 * ak5[i] + D6 * ak6[i])

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit
def rkqs(
    inout_y: np.ndarray,
    a: float,
    L: float,
    κ: float,
    #
    dvdz: np.ndarray,
    z: float,
    h: float,
    accuracy: float,
    scale: np.ndarray,
    tmp_y: np.ndarray,
    tmp_e: np.ndarray,
    ak2: np.ndarray,
    ak3: np.ndarray,
    ak4: np.ndarray,
    ak5: np.ndarray,
    ak6: np.ndarray,
    tmp: np.ndarray
) -> typing.Tuple[float, float]:

    ####################################################################################################################

    rkck(tmp_y, tmp_e, inout_y, a, L, κ, dvdz, h, ak2, ak3, ak4, ak5, ak6, tmp)

    ####################################################################################################################

    err_max = np.max(np.abs(tmp_e / scale)) / accuracy

    ####################################################################################################################

    if err_max <= 1.0:

        z += h

        inout_y[:] = tmp_y

        if err_max > ERROR_THRESHOLD:

            h = 0.9 * h * pow(err_max, POSITIVE_GROWTH)
        else:
            h = 5.0 * h

    else:

        h = min(
            0.9 * h * pow(err_max, POSITIVE_SHRINK)
            ,
            0.1 * h
        )

    ####################################################################################################################

    return z, h

########################################################################################################################

# noinspection PyPep8Naming
@nb.njit
def odeint(
    inout_y: np.ndarray,
    a: float,
    L: float,
    κ: float,
    #
    z_end: float,
    accuracy: float,
    default_step: float,
) -> typing.Tuple[np.ndarray, int]:

    ####################################################################################################################

    dim = inout_y.shape[0]

    ####################################################################################################################

    dvdz = np.empty(dim, dtype = np.float64)

    tmp_y = np.empty(dim, dtype = np.float64)
    tmp_e = np.empty(dim, dtype = np.float64)

    ak2 = np.empty(dim, dtype = np.float64)
    ak3 = np.empty(dim, dtype = np.float64)
    ak4 = np.empty(dim, dtype = np.float64)
    ak5 = np.empty(dim, dtype = np.float64)
    ak6 = np.empty(dim, dtype = np.float64)

    tmp = np.empty(dim, dtype = np.float64)

    ####################################################################################################################

    z = 0.000000000000000000000000000
    h = np.sign(z_end) * default_step

    ####################################################################################################################

    for curr_iter in range(N_MAX_ITERS):

        geodesic(dvdz, inout_y, a, L, κ)

        scale = np.abs(inout_y) + np.abs(dvdz * h) + TINY

        z, h = rkqs(inout_y, a, L, κ, dvdz, z, h, accuracy, scale, tmp_y, tmp_e, ak2, ak3, ak4, ak5, ak6, tmp)

        if z <= z_end:

            return inout_y, curr_iter + 1

    ####################################################################################################################

    return inout_y, N_MAX_ITERS + 0

########################################################################################################################

@nb.njit
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
@nb.njit
def integrate(
    r: float, θ: float, ϕ: float, pr: float, pθ: float,
    a: float, L: float, κ: float,
    z_end: float = -1.0e+7,
    accuracy: float = +1.0e-5,
    default_step: float = +1.0e-2
) -> typing.Tuple[np.ndarray, int]:

    y, step = odeint(np.array([r, θ, ϕ, pr, pθ], dtype = np.float64), a, L, κ, z_end, accuracy, default_step)

    y[1], y[2] = wrap(float(y[1]), float(y[2])) # θ and ϕ

    if step == N_MAX_ITERS:

        y[0] = 0.0
        y[1] = 0.0
        y[2] = 0.0

    return y, step

########################################################################################################################
