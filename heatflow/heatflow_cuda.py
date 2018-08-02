import numpy as np
from numba import jit, cuda


# @jit
def heat_flow(index, neighbors, init_heatflow, init_temp, rasorientation_angles, k1):
    '''
    Параметры:
    index -
    neighbors -
    init_heatflow -
    init_temp -
    rasorientation_angles -
    k1 -
    '''
    for neigh_id, value in enumerate(neighbors):
        if not (value == -3):
            init_heatflow += (init_temp[int(value)] - init_temp[index])  # *rasorientation_angles[index][neigh_id]
        else:
            init_heatflow += 0.0

    return init_heatflow * k1


heat_flow_gpu = cuda.jit(device=True)(heat_flow)


@cuda.jit
def heat_flow_kernel(cell_number, total_neighbors, init_heatflow, new_heatflow, init_temp, new_temps,
                     rasorientation_angles, k1, k2):
    """
    Параллельная версия теплопереноса (CUDA)

    Параметры:
    total_neighbors - 2D массив [индексы элемента][соответсвующие индексы соседей (12)]
    init_temp - 1D массив значений температуры для каждого элемента
    new_temps - 1D массив для записи новых значений температуры
    """

    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x;

    for i in range(startX, cell_number, gridX):
        neighbors = total_neighbors[i]  # находим массив соседей
        init_heatflow_ = init_heatflow[i]
        new_heatflow[i] = heat_flow_gpu(i, neighbors, init_heatflow_, init_temp, rasorientation_angles,
                                        k1)  # новое значение притока тепловой энергии в i-й элемент
        new_temps[i] = init_temp[i] + k2 * new_heatflow[i]
