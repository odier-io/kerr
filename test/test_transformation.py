#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

from kerr.coord import obs_to_bh, bh_to_obs

from kerr.coord import cartesian_to_boyer_lindquist, boyer_lindquist_to_cartesian

########################################################################################################################

N = 100

########################################################################################################################

def test_obs_to_bh():

    a_array = np.random.uniform(-0.999, +0.999, N)

    r_obs_array = np.random.uniform(0, 1000.000000, N)
    θ_obs_array = np.random.uniform(0, 1.0 * np.pi, N)
    ϕ_obs_array = np.random.uniform(0, 2.0 * np.pi, N)

    for i in range(N):

        a = a_array[i]

        r_obs = r_obs_array[i]
        θ_obs = θ_obs_array[i]
        ϕ_obs = ϕ_obs_array[i]

        x0 = np.random.uniform(-1000.0, +1000.0, N)
        y0 = np.random.uniform(-1000.0, +1000.0, N)
        z0 = np.random.uniform(-1000.0, +1000.0, N)

        x1, y1, z1 = obs_to_bh(a, r_obs, θ_obs, ϕ_obs, x0, y0, z0)

        x2, y2, z2 = bh_to_obs(a, r_obs, θ_obs, ϕ_obs, x1, y1, z1)

        assert np.allclose(x0, x2)
        assert np.allclose(y0, y2)
        assert np.allclose(z0, z2)

########################################################################################################################

def test_cartesian_to_boyer_lindquist():

    a_array = np.random.uniform(-0.999, +0.999, N)

    for i in range(N):

        a = a_array[i]

        x0 = np.random.uniform(-1000.0, +1000.0, N)
        y0 = np.random.uniform(-1000.0, +1000.0, N)
        z0 = np.random.uniform(-1000.0, +1000.0, N)

        r0, θ0, ϕ0 = cartesian_to_boyer_lindquist(a, x0, y0, z0)

        x1, y1, z1 = boyer_lindquist_to_cartesian(a, r0, θ0, ϕ0)

        assert np.allclose(x0, x1)
        assert np.allclose(y0, y1)
        assert np.allclose(z0, z1)

########################################################################################################################
