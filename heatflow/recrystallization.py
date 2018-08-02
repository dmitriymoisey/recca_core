import numpy as np
from numba import cuda, jit
from utils.constants import kB


def calc_probability(T_curr, T_sw1, T_sw2):
    T_sw_average = (T_sw1 + T_sw2) / 2.0

    P_sw_max = 1.0

    if (T_curr < T_sw1):
        return 0

    if (T_curr >= T_sw2):
        return 0

    if (T_curr >= T_sw1) and (T_curr < T_sw_average):
        return 2 * P_sw_max * (T_curr - T_sw1) / (T_sw2 - T_sw1)

    if (T_curr >= T_sw_average) and (T_curr < T_sw2):
        return 2 * P_sw_max * (T_sw2 - T_curr) / (T_sw2 - T_sw1)


calc_probability_gpu = cuda.jit(device=True)(calc_probability)


@cuda.jit
def calc_probability_kernel(cell_number, temperatures, T_sw1, T_sw2, probabilities):
    for i in range(cell_number):
        T_curr = temperatures[i]
        probabilities[i] = calc_probability_gpu(T_curr, T_sw1, T_sw2)


cell_number = 1000000

probabilities = np.ones(cell_number, dtype=np.float64)
temperatures = np.ones(cell_number, dtype=np.float64) * 253.0


def calc_relative_boundary_energy(y_HAGB, teta_IK, teta_HAGB):
    return y_HAGB * teta_IK / teta_HAGB * (1 - np.ln(teta_IK / teta_HAGB))


def force1(Q_i, Q_k, cell_volume):
    return (Q_i - Q_k) / cell_volume


def force2(E_i, E_k, cell_volume):
    return (E_i - E_k) / cell_volume


def force3(O_i, O_k, cell_volume):
    return (O_i - O_k) / cell_volume


# распараллелить
def calc_boundary_velocity(m1, m2, m3):
    first = m1 * force1(Q_i, Q_k, cell_volume)
    second = m2 * force2(E_i, E_k, cell_volume)
    third = m3 * force3(O_i, O_k, cell_volume)
    sum_ = first + second + third
    bound_energy = calc_relative_boundary_energy(y_HAGB, teta_IK, teta_HAGB)
    exp_ = np.exp((2.0 * surface_area * bound_energy) / kB * (T_i - T_k))
    return sum_ * exp_

# def qweasdzxc(u, a, b, c):
# 	alpha, beta, gamma = np.abs(np.cos(u, a)), np.abs(np.cos(u, b)), np.abs(np.cos(u, c))
# 	delta = np.max(alpha, beta, gamma)
