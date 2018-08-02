import numpy as np
from numba import jit, cuda


def heat_flow(index, neighbors, init_temp, heat_flow, k1, k2, rasorientation_angles):
    """
    Метод для расчета притока тепла из соседних элементов

    Параметры:
    index - индекс рассматриваемого элемента
    neighbors - (1D массив, int) - индексы соседей на 1-й координационной сфере
    init_temp - (1D массив, float) - значения температур для каждого элемента на пред. временном шаге
    heat_flow - значение теплового потока q = 0.0
    k1 - постоянная составляющая закона Фурье
    k2 - постоянная составляющая дискретной модели теплопереноса
    rasorientation_angles - (1D массив, float) - значения углов разориентации
    """

    for neigh_id, value in enumerate(neighbors):  # перебираем массив соседей
        if not (value == -3):
            heat_flow += (init_temp[int(value)] - init_temp[index])  # *rasorientation_angles[index][neigh_id]
        else:
            heat_flow += 0.0

    heat_flow = k1 * heat_flow
    return init_temp[index] + k2 * heat_flow


# TES
def heat_flow_test(index, neighbors, init_heatflow, init_temp, rasorientation_angles):
    for neigh_id, value in enumerate(neighbors):
        if not (value == -3):
            init_heatflow += (init_temp[int(value)] - init_temp[index])  # *rasorientation_angles[index][neigh_id]
            print(f'INIT HEATFLOW : {init_heatflow}')
            input()
        else:
            init_heatflow += 0.0

    return init_heatflow * k1


@jit
def heat_flow_kernel(cell_number, total_neighbors, init_heatflow, new_heatflow, init_temp, new_temps,
                     rasorientation_angles, k1, k2):
    """
    Параллельная версия теплопереноса (CUDA)

    Параметры:
    total_neighbors - 2D массив [индексы элемента][соответсвующие индексы соседей (12)]
    init_temp - 1D массив значений температуры для каждого элемента
    new_temps - 1D массив для записи новых значений температуры
    """

    for i in range(cell_number):
        neighbors = total_neighbors[i]  # находим массив соседей
        init_heatflow_ = init_heatflow[i]
        new_heatflow[i] = heat_flow_gpu(i, neighbors, init_heatflow, init_temp,
                                        rasorientation_angles)  # новое значение притока тепловой энергии в i-й элемент
        new_temps[i] = init_temp[i] + k2 * new_heatflow[i]
