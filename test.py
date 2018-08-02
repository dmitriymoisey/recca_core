import numpy as np
from numba import jit, cuda
from numba import *
import read_data
import sys
import os

task_name = sys.argv[-1]

seca_data = read_data.read_seca_file(f'./user/task_db/{task_name}.seca')
seca_data2 = read_data.read_seca_file(f'./user/task_db/{task_name}/{task_name}.seca')

CELL_NUMBER_X = int(float(seca_data2['cell_number_X']))
CELL_NUMBER_Y = int(float(seca_data2['cell_number_Y']))
CELL_NUMBER_Z = int(float(seca_data2['cell_number_Z']))

CELL_NUMBER = int(CELL_NUMBER_X * CELL_NUMBER_Y * CELL_NUMBER_Z)

TOTAL_TIME = float(seca_data['total_time'])
TIME_STEP = float(seca_data['time_step'])
NUM_OF_TIME_STEPS = int(seca_data['STEP_NUMBER'])
CELL_SIZE = float(seca_data['cell_size'])

print(f'number of cells : {CELL_NUMBER}')


def calc_triple_index(index, size_x, size_y, size_z):
    triple_index = {}
    if (index > -1) and (index < size_x * size_y * size_z):
        triple_index['i'] = int((index % (size_x * size_y)) % size_x)
        triple_index['j'] = int((index % (size_x * size_y)) / size_x)
        triple_index['k'] = int(index / (size_x * size_y))
    return triple_index


def add_neighbor(neighbors, i, j, k):
    neighbors.append({'i': i, 'j': j, 'k': k})


def set_neighbors(index, cell_number_x, cell_number_y, cell_number_z):
    indexes = calc_triple_index(index, cell_number_x, cell_number_y, cell_number_z)
    index1 = indexes['i']
    index2 = indexes['j']
    index3 = indexes['k']

    number_of_cells = cell_number_x * cell_number_y * cell_number_z

    neighbors = []

    if (index1 % 2 == 0) and (index3 % 3 == 0):
        add_neighbor(neighbors, index1 - 1, index2 - 1, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2, index3 - 1);
        add_neighbor(neighbors, index1, index2, index3 - 1);
        add_neighbor(neighbors, index1 - 1, index2 - 1, index3);
        add_neighbor(neighbors, index1 - 1, index2, index3);
        add_neighbor(neighbors, index1, index2 - 1, index3);
        add_neighbor(neighbors, index1, index2 + 1, index3);
        add_neighbor(neighbors, index1 + 1, index2 - 1, index3);
        add_neighbor(neighbors, index1 + 1, index2, index3);
        add_neighbor(neighbors, index1 - 1, index2, index3 + 1);
        add_neighbor(neighbors, index1, index2 - 1, index3 + 1);
        add_neighbor(neighbors, index1, index2, index3 + 1);

    elif (index1 % 2 == 0) and (index3 % 3 == 1):
        add_neighbor(neighbors, index1, index2, index3 - 1)
        add_neighbor(neighbors, index1, index2 + 1, index3 - 1)
        add_neighbor(neighbors, index1 + 1, index2, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2, index3)
        add_neighbor(neighbors, index1 - 1, index2 + 1, index3)
        add_neighbor(neighbors, index1, index2 - 1, index3)
        add_neighbor(neighbors, index1, index2 + 1, index3)
        add_neighbor(neighbors, index1 + 1, index2, index3)
        add_neighbor(neighbors, index1 + 1, index2 + 1, index3)
        add_neighbor(neighbors, index1 - 1, index2, index3 + 1)
        add_neighbor(neighbors, index1, index2, index3 + 1)
        add_neighbor(neighbors, index1, index2 + 1, index3 + 1)

    elif (index1 % 2 == 0) and (index3 % 3 == 2):
        add_neighbor(neighbors, index1, index2 - 1, index3 - 1)
        add_neighbor(neighbors, index1, index2, index3 - 1)
        add_neighbor(neighbors, index1 + 1, index2, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2 - 1, index3)
        add_neighbor(neighbors, index1 - 1, index2, index3)
        add_neighbor(neighbors, index1, index2 - 1, index3)
        add_neighbor(neighbors, index1, index2 + 1, index3)
        add_neighbor(neighbors, index1 + 1, index2 - 1, index3)
        add_neighbor(neighbors, index1 + 1, index2, index3)
        add_neighbor(neighbors, index1, index2, index3 + 1)
        add_neighbor(neighbors, index1 + 1, index2 - 1, index3 + 1)
        add_neighbor(neighbors, index1 + 1, index2, index3 + 1)
    elif (index1 % 2 == 1) and (index3 % 3 == 0):
        add_neighbor(neighbors, index1 - 1, index2, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2 + 1, index3 - 1)
        add_neighbor(neighbors, index1, index2, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2, index3)
        add_neighbor(neighbors, index1 - 1, index2 + 1, index3)
        add_neighbor(neighbors, index1, index2 - 1, index3)
        add_neighbor(neighbors, index1, index2 + 1, index3)
        add_neighbor(neighbors, index1 + 1, index2, index3)
        add_neighbor(neighbors, index1 + 1, index2 + 1, index3)
        add_neighbor(neighbors, index1 - 1, index2, index3 + 1)
        add_neighbor(neighbors, index1, index2, index3 + 1)
        add_neighbor(neighbors, index1, index2 + 1, index3 + 1)

    elif (index1 % 2 == 1) and (index3 % 3 == 1):
        add_neighbor(neighbors, index1, index2 - 1, index3 - 1)
        add_neighbor(neighbors, index1, index2, index3 - 1)
        add_neighbor(neighbors, index1 + 1, index2, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2 - 1, index3)
        add_neighbor(neighbors, index1 - 1, index2, index3)
        add_neighbor(neighbors, index1, index2 - 1, index3)
        add_neighbor(neighbors, index1, index2 + 1, index3)
        add_neighbor(neighbors, index1 + 1, index2 - 1, index3)
        add_neighbor(neighbors, index1 + 1, index2, index3)
        add_neighbor(neighbors, index1 - 1, index2, index3 + 1)
        add_neighbor(neighbors, index1, index2 - 1, index3 + 1)
        add_neighbor(neighbors, index1, index2, index3 + 1)

    elif (index1 % 2 == 1) and (index3 % 3 == 2):
        add_neighbor(neighbors, index1, index2, index3 - 1)
        add_neighbor(neighbors, index1, index2 + 1, index3 - 1)
        add_neighbor(neighbors, index1 + 1, index2, index3 - 1)
        add_neighbor(neighbors, index1 - 1, index2, index3)
        add_neighbor(neighbors, index1 - 1, index2 + 1, index3)
        add_neighbor(neighbors, index1, index2 - 1, index3)
        add_neighbor(neighbors, index1, index2 + 1, index3)
        add_neighbor(neighbors, index1 + 1, index2, index3)
        add_neighbor(neighbors, index1 + 1, index2 + 1, index3)
        add_neighbor(neighbors, index1, index2, index3 + 1)
        add_neighbor(neighbors, index1 + 1, index2, index3 + 1)
        add_neighbor(neighbors, index1 + 1, index2 + 1, index3 + 1)

    neighbors1S = []

    if (index < cell_number_x * cell_number_y * cell_number_z):
        for neigh_counter in range(12):
            neighbor_index_i = neighbors[neigh_counter]['i']
            neighbor_index_j = neighbors[neigh_counter]['j']
            neighbor_index_k = neighbors[neigh_counter]['k']

            if (neighbor_index_i > -1) and (neighbor_index_i < cell_number_x) and (neighbor_index_j > -1) and (
                    neighbor_index_j < cell_number_y) and (neighbor_index_k > -1) and (
                    neighbor_index_k < cell_number_z):
                neighbors1S.append(
                    neighbor_index_i + neighbor_index_j * cell_number_x + neighbor_index_k * cell_number_x * cell_number_y)
            else:
                neighbors1S.append(-3)
    else:
        for neigh_counter in range(12):
            neighbors1S.append(cell_number_x * cell_number_y * cell_number_z)
    return neighbors1S


def find_all_neighbors():
    neighbors_indeces = np.zeros((CELL_NUMBER, 12), dtype=np.int32)
    for i in range(CELL_NUMBER):
        neighbors_indeces[i] = set_neighbors(i, CELL_NUMBER_X, CELL_NUMBER_Y, CELL_NUMBER_Z)
    return neighbors_indeces


init_cond_file_data = read_data.read_init_cond_file(seca_data['init_cond_file'])
init_temp = np.array(init_cond_file_data['temperature'], dtype=np.float64)
grain_indeces = np.array(init_cond_file_data['grain_index'], dtype=np.int64)

grains_file_data = read_data.read_grains_file(seca_data['grains_file'])
x_angles = grains_file_data['x_euler_angle']
y_angles = grains_file_data['y_euler_angle']
z_angles = grains_file_data['z_euler_angle']


def calc_rasorientation_angle(index, neighbors, grain_indeces, x_euler_angles, y_euler_angles, z_euler_angles):
    """
    Метод для вычисления углов разориентации
    элемента с индексом index с его соседями
    на 1-й координацинной сфере

    Параметры:
    index - int индекс элемента для которого будем считать
            углы разориентации
    neighbors - (1D массив, int) соседей рассматриваемого элемента
    grain_indeces - (1D массив, int) индексов зерен для всех элементов

    x_euler_angles
    y_euler_angles - (1D массивы, float) углов ориентации кристаллической решетки для каждого зерна
    z_euler_angles

    """

    # создаем массив в который будем записывать посчитанные углы разориентации
    rasorientation_angles_ = np.zeros(12, dtype=np.float64)

    # индекс зерна рассматриваемого элемента
    gr_index_ = grain_indeces[index]

    # создаем массив индексов зерен для соседних элементов
    neigh_grain_indeces = [grain_indeces[i] for i in neighbors]

    for i, gr_index in enumerate(neigh_grain_indeces):
        rasorientation_angles_[i] = np.exp(
            -(np.abs(x_euler_angles[gr_index] - x_euler_angles[gr_index_]) +
              np.abs(y_euler_angles[gr_index] - y_euler_angles[gr_index_]) +
              np.abs(z_euler_angles[gr_index] - z_euler_angles[gr_index_])) / 3.0
        )
    return rasorientation_angles_


total_neighbors = find_all_neighbors()


def get_rasorientation_angles():
    """
    Метод для вычисления углов разориентации для всех
    элементов (exp( - |a[2]-[1]|+... / 3.0  ))

    return 2D массив размером (Кол-во элементов Х Кол-во соседей)
    """
    rasorientation_angles_ = np.ones((CELL_NUMBER, 12), dtype=np.float64)

    for i in range(CELL_NUMBER):
        rasorientation_angles_[i] = calc_rasorientation_angle(i, total_neighbors[i], grain_indeces, x_angles, y_angles,
                                                              z_angles)

    return rasorientation_angles_


rasorientation_angles = get_rasorientation_angles()

bound_cond_data = read_data.read_bound_cond_file(seca_data['bound_cond_file'])
bound_indeces = np.array(bound_cond_data['index'], dtype=np.int64)
bound_temperature = np.array(bound_cond_data['max_therm_param'], dtype=np.float64)


def set_boundary_conditions(temperature, bound_indeces, bound_temperature):
    """
    Метод для задания граничных условий

    Параметры:
    temperature - 1D массив температур для каждого элемента
    """
    for i, value in enumerate(bound_indeces):
        temperature[value] = bound_temperature[i]


CELL_SURFACE = 0.6938 * 6 * CELL_SIZE * CELL_SIZE
CELL_VOLUME = 0.6938 * 6 * (CELL_SIZE ** 3)

HEAT_CONDUCTIVITY = 237.0;
HEAT_CAPACITY = 930.0;
HEAT_EXPANSION_COEFF = 2.29E-5;
DENSITY = 2700.0;


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
        if not (value == -3):  # если индекс равен нулю то значит нет соседа (т.е пропускаем)
            heat_flow += k1 * (init_temp[int(value)] - init_temp[index]) * rasorientation_angles[index][neigh_id]

    return init_temp[index] + k2 * heat_flow


heat_flow_gpu = cuda.jit(device=True)(heat_flow)


@cuda.jit
def heat_flow_kernel(total_neighbors, init_temp, new_temps, T_neighbors, rasorientation_angles):
    """
    Параллельная версия теплопереноса (CUDA)

    Параметры:
    total_neighbors - 2D массив [индексы элемента][соответсвующие индексы соседей (12)]
    init_temp - 1D массив значений температуры для каждого элемента
    new_temps - 1D массив для записи новых значений температуры
    T_neighbors - 1D массив для температуры соседей
    """

    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x;

    k1 = HEAT_EXPANSION_COEFF * CELL_SURFACE * TIME_STEP / CELL_SIZE
    k2 = 1 / (HEAT_CAPACITY * DENSITY * CELL_VOLUME)

    for i in range(startX, CELL_NUMBER, gridX):
        heat_flow = 0.0
        neighbors = total_neighbors[i]  # находим массив соседей
        new_temps[i] = heat_flow_gpu(i, neighbors, init_temp, heat_flow, k1, k2,
                                     rasorientation_angles)  # записываем новое значение температуры


new_temps = np.zeros(CELL_NUMBER, dtype=np.float64)
T_neighbors = np.zeros(12, dtype=np.float64)

average_temps = []

for t_ in range(NUM_OF_TIME_STEPS):
    blockdim = (256, 64)  # задаем размер thread block
    griddim = (256, 128)  # задаем количество blockdim

    # запускаем программу
    heat_flow_kernel(total_neighbors, init_temp, new_temps, T_neighbors, rasorientation_angles)

    init_temp[:] = new_temps[:]  #
    set_boundary_conditions(init_temp, bound_indeces, bound_temperature)
    cuda.synchronize()

    average_temperature = np.average(new_temps)
    average_temps.append(average_temperature)

    print(f'Time step : {t_} , New Temperatures : {average_temperature}')

import matplotlib.pyplot as plt

plt.plot(range(NUM_OF_TIME_STEPS), average_temps)
plt.show()
