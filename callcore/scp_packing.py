import numpy as np
from numba import jit
import recca_io.write as write
from tqdm import tqdm

class CellularAutomaton:
    """
    Класс в котором включено все необходимое для работы с клеточным автоматом
    """

    def __init__(self, cell_number_x, cell_number_y, cell_number_z,
                 cell_size):
        self.cell_number_x = cell_number_x  # кол-во элементов вдоль оси Х
        self.cell_number_y = cell_number_y  # кол-во элементов вдоль оси Y
        self.cell_number_z = cell_number_z  # кол-во элементов вдоль оси Z
        self.cell_size = cell_size  # размер элемента (м)
        self.number_of_cells = int(cell_number_x * cell_number_y * cell_number_z)  # общее число элементов

        # массив тройных индексов для каждого элемента
        print('Calculation triple indices for each element ...')
        self.triple_indexes = self.calc_triple_indexes()
        # массив соседей на 1-й координационной сфере для каждого элемента

        print('Calculation neighbors indices for each element ...')
        temp = self.get_total_neighbors()
        self.total_neighbors = temp[0]
        self.radius_vectors = temp[1]

        self.cell_surface = self.get_cell_surface()  # площадь грани между двумя элементами
        self.cell_volume = self.get_cell_volume()  # объем элемента

        self.location_type = np.zeros(self.number_of_cells, dtype=np.int32)

        print('Calculation coordinates for each element ... ')
        self.x_coords, self.y_coords, self.z_coords = self.get_coordinates()

        print('Finding the boundary elements ... ')
        boundary_elements = self.get_boundary_elements()  # находим глобольные индексы граничных элементов

        self.top_boundary_elements = np.array(boundary_elements['top'], dtype=np.int32)
        self.bottom_boundary_elements = np.array(boundary_elements['bottom'], dtype=np.int32)
        self.front_boundary_elements = np.array(boundary_elements['front'], dtype=np.int32)
        self.back_boundary_elements = np.array(boundary_elements['back'], dtype=np.int32)
        self.left_boundary_elements = np.array(boundary_elements['left'], dtype=np.int32)
        self.right_boundary_elements = np.array(boundary_elements['right'], dtype=np.int32)

        # массив индексов зерен
        self.prev_grain_indeces = np.zeros(self.number_of_cells, dtype=np.int32)
        self.grain_indeces = np.zeros(self.number_of_cells, dtype=np.int32)

        # массивы состояний для каждого элемента на (n-1) и n-м шагах [0,1]
        self.prev_states = np.zeros(self.number_of_cells, dtype=np.int8)
        self.current_states = np.zeros(self.number_of_cells, dtype=np.int8)

        # параметры материала
        self.heat_expansion_coefficient = np.ones(self.number_of_cells, dtype=np.float64)
        self.heat_capacity = np.ones(self.number_of_cells, dtype=np.float64)
        self.thermal_conductivity = np.ones(self.number_of_cells, dtype=np.float64)
        self.phonon_portion = np.ones(self.number_of_cells, dtype=np.float64)
        self.density = np.ones(self.number_of_cells, dtype=np.float64)

        self.rasorientation_angles = np.ones((self.number_of_cells, 26), dtype=np.float64)

    @jit
    def calc_triple_indexes(self):
        """
        Метод для нахождения тройных индексов для каждого элемента
        :return: 2D массив [кол-во элементов Х [i,j,k]]
        """
        triple_indexes = np.zeros((self.number_of_cells, 3), dtype=np.int32)  # массив тройных индексов
        triple_index = np.zeros(3, dtype=np.int32)  # [i,j,k]
        global_id = 0  # глобальный индекс элемента

        # находим тройные индексы для каждого элемента
        for k in range(self.cell_number_z):
            for j in range(self.cell_number_y):
                for i in range(self.cell_number_x):
                    triple_index[0] = i
                    triple_index[1] = j
                    triple_index[2] = k
                    triple_indexes[global_id] = triple_index
                    global_id += 1
        return triple_indexes

    @jit
    def get_total_neighbors(self):
        """
        Метод для нахождения индексов соседей на первой координационной сфере для каждого элемента
        :return:
        """
        total_neighbors = np.zeros((self.number_of_cells, 26), dtype=np.int32)  # массив соседей для каждого элемента
        total_radius_vectors = np.zeros((self.number_of_cells, 26, 3), dtype=np.int32)

        # в цикле для каждого элемента
        for global_id in range(self.number_of_cells):
            # находим индексы соседей
            temp = self.get_element_neighbors(index=global_id)
            total_neighbors[global_id] = temp[0]
            total_radius_vectors[global_id] = temp[1]

        return total_neighbors, total_radius_vectors

    @jit
    def get_element_neighbors(self, index):
        """
        Метод для нахождения индексов соседей
        на 1-й, 2-й, 3-й координационных сферах
        :param index: индекс рассматриваемого элемента
        :return:
        """
        # массив глобальных индексов соседних элементов на 1-й, 2-й, 3-й координационных сферах
        neighbors_global_indices = np.zeros(26, dtype=np.int32)

        # массив индексов [i,j,k] соседей на 1-й и 2-й координационных сферах
        neighbors = np.zeros((26, 3), dtype=np.int32)

        # тройной индекс рассматриваемого элемента
        triple_index = self.triple_indexes[index]
        i = triple_index[0]
        j = triple_index[1]
        k = triple_index[2]

        # находим всех соседей

        # ------- 1-я коор. сфера -------
        neighbors[0] = [i + 1, j, k]
        neighbors[1] = [i - 1, j, k]
        neighbors[2] = [i, j + 1, k]
        neighbors[3] = [i, j - 1, k]
        neighbors[4] = [i, j, k + 1]
        neighbors[5] = [i, j, k - 1]
        # -------------------------------

        # ------- 2-я коор. сфера -------
        neighbors[6] = [i, j - 1, k - 1]
        neighbors[7] = [i, j + 1, k - 1]
        neighbors[8] = [i - 1, j, k - 1]
        neighbors[9] = [i + 1, j, k - 1]
        neighbors[10] = [i + 1, j - 1, k]
        neighbors[11] = [i - 1, j - 1, k]
        neighbors[12] = [i + 1, j + 1, k]
        neighbors[13] = [i - 1, j + 1, k]
        neighbors[14] = [i + 1, j, k + 1]
        neighbors[15] = [i - 1, j, k + 1]
        neighbors[16] = [i, j + 1, k + 1]
        neighbors[17] = [i, j - 1, k + 1]
        # -------------------------------

        # ------- 3-я коор. сфера -------
        neighbors[18] = [i + 1, j - 1, k + 1]
        neighbors[19] = [i - 1, j - 1, k + 1]
        neighbors[20] = [i + 1, j - 1, k - 1]
        neighbors[21] = [i - 1, j - 1, k - 1]
        neighbors[22] = [i + 1, j + 1, k + 1]
        neighbors[23] = [i - 1, j + 1, k + 1]
        neighbors[24] = [i + 1, j + 1, k - 1]
        neighbors[25] = [i - 1, j + 1, k - 1]
        # -------------------------------

        radius_vectors = np.zeros((26, 3), dtype=np.float64)

        # далее ищем глобальный индекс каждого соседа
        for neighbor_id in range(26):

            # индексы i,j,k соседа
            neighbor_index_i = neighbors[neighbor_id][0]
            neighbor_index_j = neighbors[neighbor_id][1]
            neighbor_index_k = neighbors[neighbor_id][2]

            # если индексы выходят за пределы образца то глобальный индекс равен -1
            if neighbor_index_i >= self.cell_number_x or neighbor_index_i == -1 or \
                    neighbor_index_j >= self.cell_number_y or neighbor_index_j == -1 or \
                    neighbor_index_k >= self.cell_number_z or neighbor_index_k == -1:

                neighbor_index = -1
                radius_vectors[neighbor_id] = [0.0, 0.0, 0.0]

            else:  # если индекс принадлежит образцу то вычисляем значени глобального индекса
                neighbor_index = neighbor_index_i + neighbor_index_j * self.cell_number_x + \
                                 neighbor_index_k * self.cell_number_x * self.cell_number_y
                radius_vectors[neighbor_id] = neighbors[neighbor_id] - triple_index

            # сохраняем полученное значение в массив
            neighbors_global_indices[neighbor_id] = neighbor_index

        return neighbors_global_indices, radius_vectors

    def get_boundary_elements(self):
        """
        Метод для нахождения индексов граничных элементов
        :return:
        """
        top_boundary_elements = []
        bottom_boundary_elements = []
        front_boundary_elements = []
        back_boundary_elements = []
        left_boundary_elements = []
        right_boundary_elements = []

        for i in range(self.number_of_cells):
            triple_index = self.triple_indexes[i]
            if triple_index[0] == 0:
                back_boundary_elements.append(i)
                self.location_type[i] = 1
            if triple_index[0] == (self.cell_number_x - 1):
                front_boundary_elements.append(i)
                self.location_type[i] = 1
            if triple_index[1] == 0:
                left_boundary_elements.append(i)
                self.location_type[i] = 1
            if triple_index[1] == self.cell_number_y - 1:
                right_boundary_elements.append(i)
                self.location_type[i] = 1
            if triple_index[2] == 0:
                bottom_boundary_elements.append(i)
                self.location_type[i] = 1
            if triple_index[2] == self.cell_number_z - 1:
                top_boundary_elements.append(i)
                self.location_type[i] = 1

        return {'top': top_boundary_elements,
                'bottom': bottom_boundary_elements,
                'front': front_boundary_elements,
                'back': back_boundary_elements,
                'left': left_boundary_elements,
                'right': right_boundary_elements}

    def get_cell_volume(self):
        """
        Метод для вычисления объема элемента
        """
        return self.cell_size ** 3

    def get_cell_surface(self):
        """
        Метод для площади границы между двумя элементами
        """
        return self.cell_size ** 2

    @jit
    def set_material(self, material):
        """
        Метод для задания материала клеточного автомата
        """
        self.heat_expansion_coefficient = self.heat_expansion_coefficient * material.HEAT_EXPANSION_COEFF
        self.heat_capacity = self.heat_capacity * material.HEAT_CAPACITY
        self.thermal_conductivity = self.thermal_conductivity * material.HEAT_CONDUCTIVITY
        self.phonon_portion = self.phonon_portion * material.PHONON_PORTION
        self.density = self.density * material.DENSITY

    def create_grain_structure(self, number_of_grains, angle_range):
        """
        Метод для моделирования роста зеренной структуры
        """

        f_max = 1.0  # параметр регулирующий скорость роста зерен из зародешей
        grain_index = 1  # начальный индекс зерна

        # вероятность становления зародышем элемента
        probability_of_nucleation = float(number_of_grains / self.number_of_cells)

        # в цикле по всем активным элементам
        for global_id in range(self.number_of_cells):
            current_element_random_num = np.random.random()  # генерируем случайной число [0,1]
            # если данное число попадает в заданный промежуток, то данный элемент становиться зародышем нового зерна
            if 0.5 - probability_of_nucleation / 2.0 < current_element_random_num < 0.5 + probability_of_nucleation / 2.0:
                # для элемента который стал зародышем нового зерна
                self.prev_states[global_id] = 1  # задаем состояние 1
                self.prev_grain_indeces[global_id] = grain_index  # и присваем индекс зерна
                grain_index += 1

        self.actual_number_of_grains = grain_index

        print(f'Number of Grains {grain_index}')
        print('The creation of grain structure: START...')

        neighbor_grain_indices = np.zeros(26, dtype=np.int32)  # индексы зерен соседних элементов
        counter = 0  # счетчик итераций, росто зеренной структуры

        while 0 in self.prev_states:  # до тех пор, пока в автомате есть элемент с состоянием 0

            # в цикле для всех элементов
            for global_id, state in enumerate(self.prev_states):

                # если состояние рассматриваемого элемента равно 0
                if state == 0:

                    # массив глобальных индексов соседей
                    neighbors = self.total_neighbors[global_id]
                    # массив радиус векторов
                    radius_vectors = self.radius_vectors[global_id]

                    neighbors_states = []
                    neighbor_grain_indices = []
                    # в цикле для каждого соседа,
                    # находим состояние соседа, и индекс зерна
                    for index, neighbor_global_index in enumerate(neighbors):
                        if not neighbor_global_index == -1:
                            neighbors_states.append(self.prev_states[neighbor_global_index])
                            neighbor_grain_indices.append(self.prev_grain_indeces[neighbor_global_index])

                    neighbors_states = np.array(neighbors_states)
                    neighbor_grain_indices = np.array(neighbor_grain_indices)

                    # находим зерна с индексом на 1-й координационной сфере
                    grain_indices_1S = np.unique(neighbor_grain_indices[:5])  # новый массив без повторяющихся значений
                    grain_indices_1S = np.delete(grain_indices_1S, 0)  # удаляем 0

                    # если есть элемент на 1-й координационной сфере, который принадлежит зерну
                    if not grain_indices_1S.shape[0] == 0:
                        grain_index = get_most_frequent(grain_indices_1S)  # находим индекс зерна

                        radius_vectors_ = []  # лист радиус векторов элементов с индексом зерна grain_index
                        states_ = []  # лист состояний элементов с индексом зерна grain_index

                        # в цикле для соседей на 1,2,3 - й координационных сферах
                        # находим элементы принадлежащие зерну grain_index
                        for index, neighbor_global_index in enumerate(neighbors):
                            if not neighbor_global_index == -1:
                                if self.grain_indeces[neighbor_global_index] == grain_index:
                                    radius_vectors_.append(radius_vectors[index])
                                    states_.append(self.prev_states[neighbor_global_index])

                        # считаем векторную функцию
                        vector_function = self.calculate_vector_function(states_, radius_vectors_)
                        # длина векторной функцию
                        vector_function_modulus = np.linalg.norm(vector_function)

                        x = np.random.random()
                        # print(f'F = {vector_function_modulus}')
                        # print(f'Grain Index : {grain_index}')
                        # print(f'Radius Vectors : {radius_vectors_}')
                        # print(f'Neighbor state : {states_}')
                        # если случайно сгенерированное число попадает в заданным промежуток то элемент
                        # присоединяется к зерну
                        if (0.5 - vector_function_modulus / 2.0 / f_max) < x < (
                                0.5 + vector_function_modulus / 2.0 / f_max):
                            self.current_states[global_id] = 1
                            self.grain_indeces[global_id] = grain_index

            # присваем значения новых состояний
            for i, state in enumerate(self.prev_states):
                if state == 1:
                    self.current_states[i] = 1

            for i, grain_index in enumerate(self.prev_grain_indeces):
                if not grain_index == 0:
                    self.grain_indeces[i] = grain_index

            self.prev_states[:] = self.current_states[:]
            self.prev_grain_indeces[:] = self.grain_indeces[:]

            counter += 1

            print(f'Iteration #: {counter}')

        file_name = input('Save file:')
        if not file_name == 'no':
            write.create_file(file_name, self.location_type, self.grain_indeces, self.x_coords, self.y_coords,
                              self.z_coords)

        self.configure_crystal_lattice(self.actual_number_of_grains, angle_range)
        self.rasorientation_angles = self.get_rasorientation_angles()

    def calculate_vector_function(self, neighbors_states, radius_vectors):
        """
        Метод для вычисления векторной функции, определяющей рост зерна
        --------
        Параметры:
        neighbors_states - массив состояний соседей
        radius_vectors - массив радиус векторов из рассматриваемого элемента к соседям
        """

        f_x, f_y, f_z = 0.0, 0.0, 0.0  # компоненты вычисляемого вектора

        # коэффициенты для регулировки роста зеренной структуры
        k = np.ones(len(neighbors_states), dtype=np.float64)

        # в цикле для каждого соседа
        for i in range(len(neighbors_states)):
            r_length = float(np.sum(radius_vectors[i] ** 2))  # вычисляем длина радиус вектора

            # если длина радиус вектора равна 0
            if r_length == 0.0:
                f_x += 0.0
                f_y += 0.0
                f_z += 0.0

                # в противном случаем, вычисляем значения компонент вектора, путем суммирования по всем соседям
            else:
                f_x += - k[i] * neighbors_states[i] * radius_vectors[i][0] / r_length
                f_y += - k[i] * neighbors_states[i] * radius_vectors[i][1] / r_length
                f_z += - k[i] * neighbors_states[i] * radius_vectors[i][2] / r_length

        return np.array([f_x, f_y, f_z], dtype=np.float64)

    @jit
    def configure_crystal_lattice(self, number_of_grains, angle_range):
        """
        Метод для задания параметров кристаллической решетки
        """

        self.alphas = np.zeros(number_of_grains + 1, dtype=np.float64)  #
        self.betas = np.zeros(number_of_grains + 1, dtype=np.float64)  # углы ориентации кристаллической решетки
        self.gammas = np.zeros(number_of_grains + 1, dtype=np.float64)  #

        # случайным образом определяем углы кристаллической решетки для каждого зерна
        for i in range(1, number_of_grains + 1):
            self.alphas[i] = np.random.random() * angle_range
            self.betas[i] = np.random.random() * angle_range
            self.gammas[i] = np.random.random() * angle_range

    @jit
    def get_rasorientation_angles(self):
        """
        Метод для вычисления углов разориентации
        для каждого активного элемента
        """

        # 2D массив значений углов разориентации для каждого элемента
        rasorientation_angles = np.zeros((self.number_of_cells, 26), dtype=np.float64)

        # в цикле для каждого элемента
        for i in range(self.number_of_cells):
            neighbors = self.total_neighbors[i]  # глобальные индексы соседей
            rasorientation_angles[i] = self.calculate_rasorientation_angles(i,
                                                                            neighbors)  # вычисляем углы разориентации

        return rasorientation_angles

    @jit
    def calculate_rasorientation_angles(self, element_index, neighbors_indeces):
        """
        Метод для вычисления углов разориентации
        активного элемента с индексом element_index
        с его соседями
        """

        rv = np.zeros(26, dtype=np.float64)  # массив углов разориентации
        grain_index = self.grain_indeces[element_index]  # индекс зерна рассматриваемого элемента

        # в цикле для каждого соседа
        for i, neighbor_index in enumerate(neighbors_indeces):
            neighbor_grain_index = self.grain_indeces[neighbor_index]  # назодим индекс зерна соседа
            delta_alpha = np.abs(
                self.alphas[grain_index] - self.alphas[neighbor_grain_index])  # считаем разность углов
            delta_beta = np.abs(
                self.betas[grain_index] - self.betas[neighbor_grain_index])  # ориентации кристаллической решетки
            delta_gammas = np.abs(self.gammas[grain_index] - self.gammas[neighbor_grain_index])
            rv[i] = (delta_alpha + delta_beta + delta_gammas) / 3.0  # вычислем угол разориентации

        return rv

    def recrystallization_simulation(self, average_dislocation_density, dislocation_deviation, temperature):
        # ------------------------------
        # необходимые константы
        energy_hagb = 1.0E-8
        angle_limit_hagb = 30.0
        shear = 4.0e+10
        burgers_vector = 1.0
        young_modulus = np.ones(self.number_of_cells, dtype=np.float64) * 0.4e+7
        c = 10.0
        boltzmann = 1.38064852e-3
        # ------------------------------

        current_dislocation_density = np.zeros(self.number_of_cells, dtype=np.float64)
        prev_dislocation_density = np.zeros(self.number_of_cells, dtype=np.float64)

        random_numbers = np.zeros(self.actual_number_of_grains, dtype=np.float64)

        for i in range(self.actual_number_of_grains):
            random_numbers[i] = np.random.random()

        for index, grain in enumerate(self.grain_indeces):
            prev_dislocation_density[index] = random_numbers[grain] * 1.0e+16

        current_dislocation_density[:] = prev_dislocation_density[:]

        # вычисляем энергию на границе
        boundary_energy = np.ones((self.number_of_cells, 26), dtype=np.float64)

        for global_id in range(self.number_of_cells):
            neighbors = self.total_neighbors[global_id]
            local_boundary_energy = np.zeros(26, dtype=np.float64)
            orientation_angle = (self.alphas[self.grain_indeces[global_id]] +
                                 self.betas[self.grain_indeces[global_id]] +
                                 self.gammas[self.grain_indeces[global_id]]) / 3.0

            for neighbor_local_id, neighbor_global_id in enumerate(neighbors):
                if not neighbor_global_id == -1:
                    if self.grain_indeces[global_id] == self.grain_indeces[neighbor_global_id]:
                        local_boundary_energy[neighbor_local_id] = 0.0
                    else:
                        neighbor_orientation_angle = (self.alphas[self.grain_indeces[neighbor_global_id]] +
                                                      self.betas[self.grain_indeces[neighbor_global_id]] +
                                                      self.gammas[self.grain_indeces[neighbor_global_id]]) / 3.0
                        angle_diff = np.abs(neighbor_orientation_angle - orientation_angle)
                        local_boundary_energy[neighbor_local_id] = energy_hagb * angle_diff / angle_limit_hagb * \
                                                                   (1 - np.log(angle_diff / angle_limit_hagb))

            boundary_energy[global_id] = local_boundary_energy

        # вычисляем максимальные значения подвижности
        max_mobilities = np.zeros((self.number_of_cells, 26), dtype=np.float64)

        for global_id in range(self.number_of_cells):
            neighbors = self.total_neighbors[global_id]
            local_max_mobilities = np.zeros(26, dtype=np.float64)
            for neighbor_local_id, neighbor_global_id in enumerate(neighbors):
                if not neighbor_global_id == -1:
                    local_max_mobilities[neighbor_local_id] = c / (
                            young_modulus[global_id] * young_modulus[neighbor_global_id]) * np.exp(
                        -(young_modulus[global_id] - young_modulus[neighbor_global_id]) ** 2.0 / (
                                young_modulus[global_id] * young_modulus[neighbor_global_id]))
            max_mobilities[global_id] = local_max_mobilities

        # вычисляем значения подвижностей
        mobilities = np.zeros((self.number_of_cells, 26), dtype=np.float64)
        for global_id in range(self.number_of_cells):
            neighbors = self.total_neighbors[global_id]
            local_mobilities = np.zeros(26, dtype=np.float64)
            for neighbor_local_id, neighbor_global_id in enumerate(neighbors):
                if not neighbor_global_id == -1:
                    boundary_temperature = (temperature[global_id] + temperature[neighbor_global_id]) / 2.0
                    local_mobilities[neighbor_local_id] = max_mobilities[global_id][neighbor_local_id] * \
                                                          np.exp(-boundary_energy[global_id][
                                                              neighbor_local_id] / boltzmann / boundary_temperature)
            mobilities[global_id] = local_mobilities

        # вычисляем значения движущих сил на границе
        driving_force = np.zeros((self.number_of_cells, 26), dtype=np.float64)
        for global_id in range(self.number_of_cells):
            neighbors = self.total_neighbors[global_id]
            local_driving_force = np.zeros(26, dtype=np.float64)
            for neighbor_local_id, neighbor_global_id in enumerate(neighbors):
                if not neighbor_global_id == -1:
                    local_driving_force[neighbor_local_id] = 1 / 2.0 * (
                            prev_dislocation_density[neighbor_global_id] - prev_dislocation_density[
                        global_id]) * shear * burgers_vector ** 2
            driving_force[global_id] = local_driving_force

        # вычисляем скорости на границе
        boundary_velocities = np.zeros((self.number_of_cells, 26), dtype=np.float64)

        for global_id in range(self.number_of_cells):
            neighbors = self.total_neighbors[global_id]
            local_boundary_velocities = np.zeros(26, dtype=np.float64)
            local_mobilities = mobilities[global_id]
            local_driving_forces = driving_force[global_id]
            for neighbor_local_id, neighbor_global_id in enumerate(neighbors):
                if not neighbor_global_id == -1:
                    local_boundary_velocities[neighbor_local_id] = local_mobilities[neighbor_local_id] * \
                                                                   local_driving_forces[neighbor_local_id]
            boundary_velocities[global_id] = local_boundary_velocities

        print('RECRYSTALLIZATION TEST:')
        print(f'Boundary Energy: {np.average(boundary_energy)}')
        print(f'Max Mobilities: {np.average(max_mobilities)}')
        print(f'Mobilities: {np.average(mobilities)}')
        print(f'Driving Force: {np.average(driving_force)}')
        print(f'Boundary Velocity: {np.average(boundary_velocities)}')

        max_driving_force = np.max(current_dislocation_density)/2.0 * shear * burgers_vector ** 2.0
        max_velocity = mobilities * max_driving_force

    def get_coordinates(self):
        """
        Метод для вычисления координат x,y,z для всех элементов
        """

        # 2D массив для хранения значений координат
        x_coords = np.zeros(self.number_of_cells, dtype=np.float64)
        y_coords = np.zeros(self.number_of_cells, dtype=np.float64)
        z_coords = np.zeros(self.number_of_cells, dtype=np.float64)

        for i in range(self.number_of_cells):
            triple_index = self.triple_indexes[i]  # тройной индекс для i-го активного элемента
            x_coords[i], y_coords[i], z_coords[i] = self.calc_coordinates(triple_index)

        return x_coords, y_coords, z_coords

    def calc_coordinates(self, triple_index):
        """
        Метод для вычисления координат активного элемента
        """
        radius = 0.5

        i = triple_index[0]
        j = triple_index[1]
        k = triple_index[2]

        return radius*i, radius*j, radius*k


def random_color(size):
    """
    Метод для генерации случайных цветов для зерен
    """
    colors = []
    for _ in range(size):
        colors.append(list(np.random.choice(range(256), size=3)))
    return colors


def get_most_frequent(arr):
    """
    Метод для нахождения самого частого элемента в массиве
    """
    rv = np.bincount(arr)
    return np.argmax(rv)
