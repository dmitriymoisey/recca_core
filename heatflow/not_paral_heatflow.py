import numpy as np
from numba import jit, cuda


@jit
def heat_flow(index, neighbors, init_heatflow, init_temp, rasorientation_angles, k1, heat_conducivity):
    for neigh_id, value in enumerate(neighbors):
        if not (value == -1):
            heat_conduct = calc_heat_conductivity(index, int(value), heat_conducivity)
            init_heatflow += heat_conduct * (init_temp[int(value)] - init_temp[index]) * rasorientation_angles[index][
                neigh_id]
        else:
            init_heatflow += 0.0

    return init_heatflow * k1


@jit
def heat_flow_kernel(cell_number, total_neighbors, init_heatflow, new_heatflow, init_temp, new_temps,
                     rasorientation_angles, k1, k2, heat_conducivity):
    for i in range(cell_number):
        neighbors = total_neighbors[i]  # находим массив соседей
        init_heatflow_ = init_heatflow[i]
        new_heatflow[i] = heat_flow(i, neighbors, init_heatflow_, init_temp, rasorientation_angles, k1,
                                    heat_conducivity)  # новое значение притока тепловой энергии в i-й элемент
        new_temps[i] = init_temp[i] + k2 * new_heatflow[i]


def density():
    return 2900.0


def heat_capacity():
    return 930.0


@jit
def calc_heat_conductivity(i, k, heat_conds):
    return (heat_conds[i] + heat_conds[k]) / 2.0  # возвращаем среднее значение


@jit
def heat_expansion_coeff(cell_number):
    rv = np.ones(cell_number, dtype=np.float64) * 2.29E-5
    return rv


@jit
def calc_heat_expansion(cell_number, temps, prev_temps):
    coeff = heat_expansion_coeff(cell_number)
    return coeff * (temps - prev_temps)


@jit
def young_modul(cell_number):
    rv = np.ones(cell_number, dtype=np.float) * 0.4e+7  # Pa
    return rv


@jit
def calc_stresses(cell_number, temps, prev_temps):
    coeff = heat_expansion_coeff(cell_number)
    youngs = young_modul(cell_number)
    return coeff * youngs * (temps - prev_temps)


@jit
def calc_mechanical_energy(cell_number, cell_volume, prev_heat_expansion, delta_heat_expansion):
    youngs = young_modul(cell_number)
    a = np.sign(prev_heat_expansion + delta_heat_expansion)
    b = (prev_heat_expansion + delta_heat_expansion) ** 2
    c = np.sign(prev_heat_expansion) * (prev_heat_expansion ** 2)
    return youngs * cell_volume / 2.0 * (a * b - c)


@jit
def calc_heat_energy(prev_heat_energy, heat_capacity, temps, prev_temps, delta_mech_energy):
    rv = prev_heat_energy + heat_capacity * (temps - prev_temps) - delta_mech_energy
    return rv


@jit
def change_temps(temps, delta_mech_energy, heat_capacity):
    '''

    :param temps:
    :param delta_mech_energy:
    :param heat_capacity:
    :return:
    '''
    return temps - delta_mech_energy / heat_capacity
