import numpy as np
from numba import jit

def get_cell_surface(cell_size):
    """
    Метод для расчета площади поверхности элемента
    """
    return 0.6938 * 6 * cell_size * cell_size

def get_cell_volume(cell_size):
    """
    Метод для расчета объема элемента
    """
    return 0.6938 * 6 * (cell_size ** 3)

def calc_triple_index(index, cell_number_X, cell_number_Y, cell_number_Z):
    triple_index = {}
    if (index > -1) and (index < cell_number_X * cell_number_Y * cell_number_Z):
        triple_index['i'] = int((index % (cell_number_X * cell_number_Y)) % cell_number_X)
        triple_index['j'] = int((index % (cell_number_X * cell_number_Y)) / cell_number_X)
        triple_index['k'] = int(index / (cell_number_X * cell_number_Y))
    return triple_index


def calc_coordinates(i, j, k):
    coords = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    radius = 0.5

    if (k % 3 == 0):

        if (i % 2 == 0):
            coords['x'] = radius * (1.0 + i * np.sqrt(3.0))
            coords['y'] = radius * (1.0 + j * 2.0)
        else:
            coords['x'] = radius * (1.0 + i * np.sqrt(3.0))
            coords['y'] = radius * (2.0 + j * 2.0)

    if (k % 3 == 1):

        if (i % 2 == 0):
            coords['x'] = radius * (1.0 + 1.0 / np.sqrt(3.0) + i * np.sqrt(3.0))
            coords['y'] = radius * (2.0 + j * 2.0)
        else:
            coords['x'] = radius * (1.0 + 1.0 / np.sqrt(3.0) + i * np.sqrt(3.0))
            coords['y'] = radius * (1.0 + j * 2.0)

    if (k % 3 == 2):

        if (i % 2 == 0):
            coords['x'] = radius * (1.0 + 2.0 / np.sqrt(3.0) + i * np.sqrt(3.0))
            coords['y'] = radius * (1.0 + j * 2.0)
        else:
            coords['x'] = radius * (1.0 + 2.0 / np.sqrt(3.0) + i * np.sqrt(3.0))
            coords['y'] = radius * (2.0 + j * 2.0)

    coords['z'] = radius * (1 + 2.0 * k * np.sqrt(2.0 / 3.0));

    return coords


def calc_all_coordinates(num_of_cells, cell_number_X, cell_number_Y, cell_number_Z):

    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(num_of_cells):
        triple = calc_triple_index(i, cell_number_X, cell_number_Y, cell_number_Z)
        coords = calc_coordinates(triple['i'], triple['j'], triple['k'])
        x_coords.append(coords['x'])
        y_coords.append(coords['y'])
        z_coords.append(coords['z'])

    return (x_coords, y_coords, z_coords)

def triple(i, j, k):
    return {'i': i, 'j': j, 'k': k}


def set_neighbors(index, num_of_cells, cell_number_X, cell_number_Y, cell_number_Z):
    """
    Метод для нахождения соседей для элемента
    ---
    return (1D массив[12], int)
    """

    indeces = calc_triple_index(index, cell_number_X, cell_number_Y, cell_number_Z)
    i = indeces['i']
    j = indeces['j']
    k = indeces['k']

    neighbs = [i for i in range(12)]

    if (i % 2 == 0) and (k % 3 == 0):
        neighbs[0] = triple(i - 1, j, k)
        neighbs[1] = triple(i - 1, j, k - 1)
        neighbs[2] = triple(i, j, k - 1)
        neighbs[3] = triple(i - 1, j - 1, k)
        neighbs[4] = triple(i - 1, j, k)
        neighbs[5] = triple(i, j - 1, k)
        neighbs[6] = triple(i, j + 1, k)
        neighbs[7] = triple(i + 1, j - 1, k)
        neighbs[8] = triple(i + 1, j, k)
        neighbs[9] = triple(i - 1, j, k + 1)
        neighbs[10] = triple(i, j - 1, k + 1)
        neighbs[11] = triple(i, j, k + 1)

    elif (i % 2 == 0) and (k % 3 == 1):
        neighbs[0] = triple(i, j, k - 1)
        neighbs[1] = triple(i, j + 1, k - 1)
        neighbs[2] = triple(i + 1, j, k - 1)
        neighbs[3] = triple(i - 1, j, k)
        neighbs[4] = triple(i - 1, j + 1, k)
        neighbs[5] = triple(i, j - 1, k)
        neighbs[6] = triple(i, j + 1, k)
        neighbs[7] = triple(i + 1, j, k)
        neighbs[8] = triple(i + 1, j + 1, k)
        neighbs[9] = triple(i - 1, j, k + 1)
        neighbs[10] = triple(i, j, k + 1)
        neighbs[11] = triple(i, j + 1, k + 1)

    elif (i % 2 == 0) and (k % 3 == 2):
        neighbs[0] = triple(i, j - 1, k - 1)
        neighbs[1] = triple(i, j, k - 1)
        neighbs[2] = triple(i + 1, j, k - 1)
        neighbs[3] = triple(i - 1, j - 1, k)
        neighbs[4] = triple(i - 1, j, k)
        neighbs[5] = triple(i, j - 1, k)
        neighbs[6] = triple(i, j + 1, k)
        neighbs[7] = triple(i + 1, j - 1, k)
        neighbs[8] = triple(i + 1, j, k)
        neighbs[9] = triple(i, j, k + 1)
        neighbs[10] = triple(i + 1, j - 1, k + 1)
        neighbs[11] = triple(i + 1, j, k + 1)

    elif (i % 2 == 1) and (k % 3 == 0):
        neighbs[0] = triple(i - 1, j, k - 1)
        neighbs[1] = triple(i - 1, j + 1, k - 1)
        neighbs[2] = triple(i, j, k - 1)
        neighbs[3] = triple(i - 1, j, k)
        neighbs[4] = triple(i - 1, j + 1, k)
        neighbs[5] = triple(i, j - 1, k)
        neighbs[6] = triple(i, j + 1, k)
        neighbs[7] = triple(i + 1, j, k)
        neighbs[8] = triple(i + 1, j + 1, k)
        neighbs[9] = triple(i - 1, j, k + 1)
        neighbs[10] = triple(i, j, k + 1)
        neighbs[11] = triple(i, j + 1, k + 1)

    elif (i % 2 == 1) and (k % 3 == 1):
        neighbs[0] = triple(i, j - 1, k - 1)
        neighbs[1] = triple(i, j, k - 1)
        neighbs[2] = triple(i + 1, j, k - 1)
        neighbs[3] = triple(i - 1, j - 1, k)
        neighbs[4] = triple(i - 1, j, k)
        neighbs[5] = triple(i, j - 1, k)
        neighbs[6] = triple(i, j + 1, k)
        neighbs[7] = triple(i + 1, j - 1, k)
        neighbs[8] = triple(i + 1, j, k)
        neighbs[9] = triple(i - 1, j, k + 1)
        neighbs[10] = triple(i, j - 1, k + 1)
        neighbs[11] = triple(i, j, k + 1)

    elif (i % 2 == 1) and (k % 3 == 2):
        neighbs[0] = triple(i, j, k - 1)
        neighbs[1] = triple(i, j + 1, k - 1)
        neighbs[2] = triple(i + 1, j, k - 1)
        neighbs[3] = triple(i - 1, j, k)
        neighbs[4] = triple(i - 1, j + 1, k)
        neighbs[5] = triple(i, j - 1, k)
        neighbs[6] = triple(i, j + 1, k)
        neighbs[7] = triple(i + 1, j, k)
        neighbs[8] = triple(i + 1, j + 1, k)
        neighbs[9] = triple(i, j, k + 1)
        neighbs[10] = triple(i + 1, j, k + 1)
        neighbs[11] = triple(i + 1, j + 1, k + 1)

    neighbors1S = np.zeros(12, dtype=np.int32)

    if (index < num_of_cells):
        for neigh_counter in range(12):
            neighbor_index_i = neighbs[neigh_counter]['i']
            neighbor_index_j = neighbs[neigh_counter]['j']
            neighbor_index_k = neighbs[neigh_counter]['k']

            if (neighbor_index_i > -1) and (neighbor_index_i < cell_number_X) and (neighbor_index_j > -1) and (
                    neighbor_index_j < cell_number_Y) and (neighbor_index_k > -1) and (
                    neighbor_index_k < cell_number_Z):
                neighbors1S[
                    neigh_counter] = neighbor_index_i + neighbor_index_j * cell_number_X + neighbor_index_k * cell_number_X * cell_number_Y
            else:
                neighbors1S[neigh_counter] = -3

    else:
        for neigh_counter in range(12):
            neighbors1S[neigh_counter] = num_of_cells

    return neighbors1S

def find_all_neighbors(num_of_cells, cell_number_X, cell_number_Y, cell_number_Z):
    """
    Метод для нахождения соседей для каждого элемента
    ---
    return (2D массив[кол-во элементов х 12 ], int)
    """
    neighbors_indeces = np.zeros((num_of_cells, 12), dtype=np.int32)
    for i in range(num_of_cells):
        neighbors_indeces[i] = set_neighbors(i, num_of_cells, cell_number_X, cell_number_Y, cell_number_Z)
    return neighbors_indeces

@jit
def set_heat_conductivity(cell_number, z_coords, heat_left, heat_right):
    heat_conductivity = np.zeros(cell_number, dtype=np.float64)
    middle_z = np.max(z_coords) / 2.0
    for i in range(cell_number):
        if (z_coords[i] > middle_z):
            heat_conductivity[i] = heat_right
        else:
            heat_conductivity[i] = heat_left

    return heat_conductivity
