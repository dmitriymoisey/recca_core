import numpy as np
from numba import jit, cuda
import utils.constants as const


@jit
def determine_grain_index(grain_indices_, probs_):
    number_of_grains = np.unique(grain_indices_).shape[0]
    probabilities_ = np.zeros(number_of_grains, dtype=np.float64)
    for i in range(number_of_grains):
        a = np.where(grain_indices_ == np.unique(grain_indices_)[i])[0]
        sum_ = 0
        for j in range(a.shape[0]):
            sum_ += probs_[a[j]]
        probabilities_[i] = sum_
    return np.unique(grain_indices_)[np.argmax(probabilities_)]


@jit
def calculate_heat_transfer(number_of_cells, total_neighbors,
                            rasorientation_angles,
                            initial_heat_energy, current_heat_energy,
                            initial_temperature, current_temperature,
                            heat_conductivity, heat_capacity, density,
                            phonon_portion,
                            cell_size, cell_surface, cell_volume,
                            time_step):
    """
    Алгоритм ТЕПЛОПЕРЕНОСА
    :param number_of_cells:  общее число элементов
    :param total_neighbors: 2D массив [глобальный индекс, [глобальные индексы соседей]]
    :param rasorientation_angles: 2D массив [глобальный индекс, [значения углов разориентации с соседними элементами]]
    :param initial_heat_energy: 1D массив значений тепловой энергия для каждого элемента
    :param current_heat_energy: 1D массив значений тепловой энергия для каждого элемента
    :param initial_temperature: 1D массив значений температуры для каждого элемента
    :param current_temperature: 1D массив значений температуры для каждого элемента
    :param heat_conductivity: 1D массив значений теплопроводности для каждого элемента
    :param heat_capacity: 1D массив значений теплоемкости для каждого элемента
    :param density: 1D массив значений плотность для каждого элемента
    :param phonon_portion: 1D массив значений влияния фононов для каждого элемента
    :param cell_size: размер элемента
    :param cell_surface: площадь грани соприкосновения между двумя элементами
    :param cell_volume: объем элемента
    :param time_step: значение шага по времени
    :return:
    current_temperature: 1D массив текущих значений температуры для каждого элемента
    current_heat_energy: 1D массив текущих значений тепловой энергии для каждого элемента
    """

    # в цикле для каждого элемента
    for index in range(number_of_cells):
        neighbors = total_neighbors[index][:6]  # индексы соседей на 1-й коорд. сфере
        neighbor_heat_influxes = np.zeros(6, dtype=np.float64)  # массив тепловых потом от соседних элементов
        # в цикле по всем соседям на 1-й коорд. сфере
        for neighbor_id, global_id in enumerate(neighbors):
            # если элемент находится внутри образца, то считаем значение теплового потока от соседа
            if not global_id == -1:
                neighbor_heat_influxes[neighbor_id] = heat_conductivity[global_id] * cell_surface * (
                        initial_temperature[global_id] - initial_temperature[index]) * np.exp(
                    -rasorientation_angles[index][neighbor_id] * phonon_portion[global_id]) * time_step / cell_size

        # находим текущее значение тепловой энергии в элементе
        current_heat_energy[index] = initial_heat_energy[index] + np.sum(neighbor_heat_influxes)
        # находим текущее значение температуры в элементк
        current_temperature[index] = current_heat_energy[index] * (
                1 / (heat_capacity[index] * density[index] * cell_volume))

    return current_temperature, current_heat_energy


@jit
def calc_rasorientation_angle(element_grain_index, neighbor_grain_index, alphas, betas, gammas):
    element_grain_index = int(element_grain_index)
    neighbor_grain_index = int(neighbor_grain_index)
    return (np.abs(alphas[neighbor_grain_index] - alphas[element_grain_index]) +
            np.abs(betas[neighbor_grain_index] - betas[element_grain_index]) +
            np.abs(gammas[neighbor_grain_index] - gammas[element_grain_index])) / 3.0


@jit
def calc_boundary_energy(max_boundary_energy, max_angle, rasorientatiion_angle):
    if rasorientatiion_angle == 0.0:
        return 0.0
    else:
        return max_boundary_energy * rasorientatiion_angle / max_angle * \
               (1 - np.log(rasorientatiion_angle / max_angle))


@jit
def calc_mobility(max_mobility, boundary_energy, boundary_temperature):
    return max_mobility * np.exp(-boundary_energy / (const.kB * boundary_temperature))


@jit
def calc_driving_force(neighbor_disl_density, element_disl_density, shear_modulus, burgers_vector):
    return -1.0e+14 / 2.0 * (neighbor_disl_density - element_disl_density) * shear_modulus * (burgers_vector ** 2)


@jit
def calc_velocity(mobility, driving_force):
    return mobility * driving_force


@jit
def check_probability(probability):
    if 0.0 <= probability <= 1.0:
        return True
    else:
        return False


@jit
def recrystallization_simulation(number_of_cells, total_neighbors, grain_indices, alphas, betas, gammas,
                                 max_boundary_energy, max_angle, temperature, max_mobility,
                                 dislocation_density, shear_modulus, burgers_vector, time_step, cell_size):
    """
    Алгоритм РЕКРИСТАЛЛИЗАЦИИ
    :param number_of_cells:  общее число элементов
    :param total_neighbors: 2D массив [глобальный индекс, [глобальные индексы соседей]]
    :param alphas, betas, gammas: углы ориентации кристаллической решетки
    :param grain_indices: 1D массив значений индексов зерен для каждого элемента
    :param max_boundary_energy: максимальное значение энергии на границе
    :param max_angle: максимальный угол разориентации
    :param temperature: 1D массив значений температуры для каждого элемента
    :param max_mobility: максимальное значение подвижности
    :param dislocation_density: 1D массив значений плотностей дислокаций для каждого элемента
    :param shear_modulus: модуль сдвига
    :param burgers_vector: вектор Бюргерса
    :param cell_size: размер элемента
    :param time_step: значение шага по времени
    :return:
    new_grain_indeces: 1D массив текущих значений индексов зерен для каждого элемента
    new_dislocation_density: 1D массив текущих значений плотности дислокаций для каждого элемента
    """
    # массив для новых значений плотностей дислокаций
    new_dislocation_density = np.empty(number_of_cells, dtype=np.float64)
    new_dislocation_density[:] = dislocation_density[:]  # присваем значения на пред. шаге

    # массив для новых значений индексов зерен
    new_grain_indeces = np.empty(number_of_cells, dtype=np.int64)
    new_grain_indeces[:] = grain_indices[:]  # присваем значения на пред. шаге

    # в цикле для каждого элемента
    for global_id in range(number_of_cells):
        neighbors = total_neighbors[global_id][:6]  # массив индексов соседей на 1-й коорд. сфере

        probabilities = np.zeros(6, dtype=np.float64)
        grain_indices_ = np.zeros(6, dtype=np.int64)
        disl_dens_ = np.zeros(6, dtype=np.float64)

        # в цикле по соседям на 1-й коорд. сфере
        for neighbor_local_id, neighbor_global_id in enumerate(neighbors):

            if not neighbor_global_id == -1:
                rasorientation_angle = calc_rasorientation_angle(element_grain_index=grain_indices[global_id],
                                                                 neighbor_grain_index=grain_indices[neighbor_global_id],
                                                                 alphas=alphas,
                                                                 betas=betas,
                                                                 gammas=gammas)

                # вычисляем значения энергии на границе
                boundary_energy = calc_boundary_energy(max_boundary_energy=max_boundary_energy,
                                                       max_angle=max_angle,
                                                       rasorientatiion_angle=rasorientation_angle)

                # вычисляем среднее значение температуры между элементом и его соседом
                boundary_temperature = (temperature[global_id] + temperature[neighbor_global_id]) / 2.0

                # вычисляем значение подвижности
                mobility = calc_mobility(max_mobility=max_mobility,
                                         boundary_energy=boundary_energy,
                                         boundary_temperature=boundary_temperature)

                neighbor_disl_density = dislocation_density[neighbor_global_id]
                element_disl_density = dislocation_density[global_id]

                # вычисляем значение движущей силы рекристаллизации
                driving_force = calc_driving_force(neighbor_disl_density=neighbor_disl_density,
                                                   element_disl_density=element_disl_density,
                                                   shear_modulus=shear_modulus,
                                                   burgers_vector=burgers_vector)

                # вычисляем скорость движения на границе
                velocity = mobility * driving_force

                # вычисляем вероятность
                probability = velocity * time_step / cell_size

                # если вероятность < 1
                if check_probability(probability):
                    q = np.random.random()  # генерируем случайное число
                    if 0.5 - probability < q < 0.5 + probability:
                        probabilities[neighbor_local_id] = probability
                        grain_indices_[neighbor_local_id] = int(grain_indices[neighbor_global_id])
                        disl_dens_[neighbor_local_id] = dislocation_density[neighbor_global_id]

        if np.unique(probabilities).shape[0] == 2:
            new_grain_indeces[global_id] = grain_indices_[np.where(probabilities != 0.0)[0][0]]
            new_dislocation_density[global_id] = disl_dens_[np.where(probabilities != 0.0)[0][0]]
        elif np.unique(probabilities).shape[0] > 2:
            gr_ = determine_grain_index(grain_indices_, probabilities)
            new_grain_indeces[global_id] = gr_
            new_dislocation_density[global_id] = disl_dens_[np.where(grain_indices_ == gr_)[0][0]]

    return new_grain_indeces, new_dislocation_density

@Test
def recrystallization_simulation(number_of_cells, total_neighbors, grain_indices, alphas, betas, gammas,
                                 max_boundary_energy, max_angle, temperature, max_mobility,
                                 dislocation_density, shear_modulus, burgers_vector, time_step, cell_size):
    """
    Алгоритм РЕКРИСТАЛЛИЗАЦИИ
    :param number_of_cells:  общее число элементов
    :param total_neighbors: 2D массив [глобальный индекс, [глобальные индексы соседей]]
    :param alphas, betas, gammas: углы ориентации кристаллической решетки
    :param grain_indices: 1D массив значений индексов зерен для каждого элемента
    :param max_boundary_energy: максимальное значение энергии на границе
    :param max_angle: максимальный угол разориентации
    :param temperature: 1D массив значений температуры для каждого элемента
    :param max_mobility: максимальное значение подвижности
    :param dislocation_density: 1D массив значений плотностей дислокаций для каждого элемента
    :param shear_modulus: модуль сдвига
    :param burgers_vector: вектор Бюргерса
    :param cell_size: размер элемента
    :param time_step: значение шага по времени
    :return:
    new_grain_indeces: 1D массив текущих значений индексов зерен для каждого элемента
    new_dislocation_density: 1D массив текущих значений плотности дислокаций для каждого элемента
    """
    # массив для новых значений плотностей дислокаций
    new_dislocation_density = np.empty(number_of_cells, dtype=np.float64)
    new_dislocation_density[:] = dislocation_density[:]  # присваем значения на пред. шаге

    # массив для новых значений индексов зерен
    new_grain_indeces = np.empty(number_of_cells, dtype=np.int64)
    new_grain_indeces[:] = grain_indices[:]  # присваем значения на пред. шаге

    # в цикле для каждого элемента
    for global_id in range(number_of_cells):
        neighbors = total_neighbors[global_id][:6]  # массив индексов соседей на 1-й коорд. сфере

        probabilities = np.zeros(6, dtype=np.float64)
        grain_indices_ = np.zeros(6, dtype=np.int64)
        disl_dens_ = np.zeros(6, dtype=np.float64)

        # в цикле по соседям на 1-й коорд. сфере
        for neighbor_local_id, neighbor_global_id in enumerate(neighbors):

            if not neighbor_global_id == -1:
                rasorientation_angle = calc_rasorientation_angle(element_grain_index=grain_indices[global_id],
                                                                 neighbor_grain_index=grain_indices[neighbor_global_id],
                                                                 alphas=alphas,
                                                                 betas=betas,
                                                                 gammas=gammas)

                # вычисляем значения энергии на границе
                boundary_energy = calc_boundary_energy(max_boundary_energy=max_boundary_energy,
                                                       max_angle=max_angle,
                                                       rasorientatiion_angle=rasorientation_angle)

                # вычисляем среднее значение температуры между элементом и его соседом
                boundary_temperature = (temperature[global_id] + temperature[neighbor_global_id]) / 2.0

                # вычисляем значение подвижности
                mobility = calc_mobility(max_mobility=max_mobility,
                                         boundary_energy=boundary_energy,
                                         boundary_temperature=boundary_temperature)

                neighbor_disl_density = dislocation_density[neighbor_global_id]
                element_disl_density = dislocation_density[global_id]

                # вычисляем значение движущей силы рекристаллизации
                driving_force = calc_driving_force(neighbor_disl_density=neighbor_disl_density,
                                                   element_disl_density=element_disl_density,
                                                   shear_modulus=shear_modulus,
                                                   burgers_vector=burgers_vector)

                # вычисляем скорость движения на границе
                velocity = mobility * driving_force

                # вычисляем вероятность
                probability = velocity * time_step / cell_size

                # если вероятность < 1
                if check_probability(probability):
                    q = np.random.random()  # генерируем случайное число
                    if 0.5 - probability < q < 0.5 + probability:
                        probabilities[neighbor_local_id] = probability
                        grain_indices_[neighbor_local_id] = int(grain_indices[neighbor_global_id])
                        disl_dens_[neighbor_local_id] = dislocation_density[neighbor_global_id]

        if np.unique(probabilities).shape[0] == 2:
            new_grain_indeces[global_id] = grain_indices_[np.where(probabilities != 0.0)[0][0]]
            new_dislocation_density[global_id] = disl_dens_[np.where(probabilities != 0.0)[0][0]]
        elif np.unique(probabilities).shape[0] > 2:
            gr_ = determine_grain_index(grain_indices_,probabilities)
            new_grain_indeces[global_id] = gr_
            new_dislocation_density[global_id] = disl_dens_[np.where(grain_indices_ == gr_)[0][0]]

    return new_grain_indeces, new_dislocation_density

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
def calc_stresses(cell_number, current_temperature, prev_temperature):
    coeff = heat_expansion_coeff(cell_number)
    youngs = young_modul(cell_number)
    return coeff * youngs * (current_temperature - prev_temperature)


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
