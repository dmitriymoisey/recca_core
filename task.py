# coding=utf-8
from callcore.scp_packing import *
from db import DBUtils
import sqlite3 as sqlite
import numpy as np
import DBCommon


class Task:
    def __init__(self, specimen_name, task_name):
        """
        Конструктор задачи
        """

        self.db = DBUtils()
        self.data = self.db.read_db(specimen_name)

        self.material_data = self.data['Material']
        self.initial_conditions_data = self.data['Initial Conditions']
        self.boundary_conditions_data = self.data['Boundary Conditions']
        self.task_data = self.data['Task']

        self.name = specimen_name  # имя образца
        self.task_name = task_name  # имя задачи

        self.time_step = self.task_data['Time Step']  # значение временного шага
        self.total_time = self.task_data['Total Time']  # общее время
        self.num_of_time_steps = int(self.total_time / self.time_step)  # кол-во шагов по времени

        self.cell_number_X = self.data['Cell Number X']  # кол-во элементов вдоль оси X
        self.cell_number_Y = self.data['Cell Number Y']  # кол-во элементов вдоль оси Y
        self.cell_number_Z = self.data['Cell Number Z']  # кол-во элементов вдоль оси Z
        self.cell_size = self.data['Cell Size']  # размер элемента
        self.number_of_grains = self.data['Number Of Grains']  # кол-во зерен
        self.angle_range = self.data['Angle Range']  # углы ориентации кристаллической решетки от 0 до angle_range

        self.is_stochastic = False
        if int(self.data[DBCommon.TYPE_OF_GRAIN_DISTRIB]) == 1:
            self.is_stochastic = True

        print(self.is_stochastic)

        self.automat = CellularAutomaton(self.cell_number_X, self.cell_number_Y, self.cell_number_Z, self.cell_size)

        try:
            self.structure_data = self.db.read_structure_db(specimen_name)
            self.automat.location_type = self.structure_data['Location Type']
            self.automat.grain_indeces = self.structure_data['Grain Index']
            self.automat.x_coords = self.structure_data['X Coords']
            self.automat.y_coords = self.structure_data['Y Coords']
            self.automat.z_coords = self.structure_data['Z Coords']
            self.automat.actual_number_of_grains = np.max(self.automat.grain_indeces) + 1
            self.automat.configure_crystal_lattice(self.automat.actual_number_of_grains,
                                                   self.angle_range)
            self.automat.rasorientation_angles = self.automat.get_rasorientation_angles()
            self.automat.prev_dislocation_density = self.automat.get_dislocation_density(
                number_of_grains=self.automat.actual_number_of_grains,
                min_value=1.0e+13,
                deviation=1.0e+5)
            self.automat.current_dislocation_density[:] = self.automat.prev_dislocation_density[:]

        except sqlite.DatabaseError as err:
            print(err)
            self.automat.create_grain_structure(number_of_grains=self.number_of_grains,
                                                angle_range=self.angle_range,
                                                is_stochastic=self.is_stochastic)

            self.db.write_structure_data_to_db(specimen_name=self.name,
                                               location_types=self.automat.location_type,
                                               grain_indices=self.automat.grain_indeces,
                                               x_coords=self.automat.x_coords,
                                               y_coords=self.automat.y_coords,
                                               z_coords=self.automat.z_coords,
                                               number_of_cells=self.automat.number_of_cells)

        self.db.create_results_table(specimen_name=self.name,
                                     task_name=self.task_name)

        # Material Data
        self.heat_conductivity = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Heat Conductivity']
        self.density = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data['Density']
        self.heat_expansion = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Heat Expansion']
        self.heat_capacity = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Heat Capacity']
        self.phonon_portion = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Phonon Portion']
        self.angle_limit_hagb = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Angle Limit HAGB']
        self.energy_hagb = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data['Energy HAGB']
        self.max_mobility = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Max. Mobility']
        self.lattice_parameter = np.ones(self.automat.number_of_cells, dtype=np.float64) * self.material_data[
            'Lattice Parameter']

        self.initial_heat_energy = np.ones(self.automat.number_of_cells, dtype=np.float64)
        self.current_heat_energy = np.ones(self.automat.number_of_cells, dtype=np.float64)

        self.initial_temperature = np.ones(self.automat.number_of_cells, dtype=np.float64)
        self.current_temperature = np.ones(self.automat.number_of_cells, dtype=np.float64)

        self.initial_dislocation_density = np.empty(self.automat.number_of_cells, dtype=np.float64)
        self.current_dislocation_density = np.empty(self.automat.number_of_cells, dtype=np.float64)
        self.set_dislocation_density(number_of_grains=self.automat.actual_number_of_grains,
                                     min_value=1.0E+13,
                                     deviation=1.0E+5,
                                     grain_indeces=self.automat.grain_indeces)

        # изменение деформации вследствие теплового расширения на (n)-м шаге
        self.delta_deformation = np.zeros(self.automat.number_of_cells, dtype=np.float64)
        # деформации вследствие теплового расширения на (n-1)-м шаге
        self.current_deformation = np.zeros(self.automat.number_of_cells, dtype=np.float64)

        # изменения напряжения на (n)-м шаге
        self.delta_stress = np.zeros(self.automat.number_of_cells, dtype=np.float64)
        # изменения механической энергии на (n)-м шаге
        self.delta_mechanical_energy = np.zeros(self.automat.number_of_cells, dtype=np.float64)

    def __repr__(self):
        rv = f'Task Name : {self.name}\nElement Size : ' \
             f'{self.cell_size} \n' \
             f'Specimen Size: {self.cell_number_X}X{self.cell_number_Y}X{self.cell_number_Z}\n' \
             f'Number of Cells: {self.automat.number_of_cells}\n' \
             f'Total Time: {self.total_time}\n' \
             f'Time Step: {self.time_step}\n' \
             f'Number Of Time Steps: {self.num_of_time_steps}'
        return rv

    def set_material_params(self):
        pass

    def set_initial_conditions(self):
        self.initial_heat_energy = self.initial_heat_energy * self.heat_capacity * self.density \
                                   * self.automat.cell_volume * self.initial_conditions_data['Temperature']
        self.current_heat_energy = self.current_heat_energy * self.heat_capacity * self.density \
                                   * self.automat.cell_volume * self.initial_conditions_data['Temperature']

        self.initial_temperature = self.initial_temperature * self.initial_conditions_data['Temperature']
        self.current_temperature = self.current_temperature * self.initial_conditions_data['Temperature']

    def set_boundary_conditions(self):
        for bound_cond in self.boundary_conditions_data:
            if bound_cond['Facet'] == 'Top':
                for boundary_element in self.automat.top_boundary_elements:
                    if not boundary_element == -1:
                        self.set_boundary_value(bound_cond, boundary_element)

            if bound_cond['Facet'] == 'Bottom':
                for boundary_element in self.automat.bottom_boundary_elements:
                    if not boundary_element == -1:
                        self.set_boundary_value(bound_cond, boundary_element)

    def set_dislocation_density(self, number_of_grains, min_value, deviation, grain_indeces):
        random_numbers = np.zeros(number_of_grains, dtype=np.float64)
        for i in range(number_of_grains):
            random_numbers[i] = np.random.random()

        for i, grain_index in enumerate(grain_indeces):
            self.current_dislocation_density[i] = min_value + deviation * random_numbers[grain_index]

        self.initial_dislocation_density[:] = self.current_dislocation_density[:]

    def set_boundary_value(self, bound_condition, boundary_element):
        self.initial_temperature[boundary_element] = bound_condition['Temperature Average']
        self.current_temperature[boundary_element] = bound_condition['Temperature Average']

        self.initial_heat_energy[boundary_element] = bound_condition['Temperature Average'] * \
                                                     self.heat_capacity[boundary_element] * \
                                                     self.density[boundary_element] * \
                                                     self.automat.cell_volume
        self.current_heat_energy[boundary_element] = bound_condition['Temperature Average'] * \
                                                     self.heat_capacity[boundary_element] * \
                                                     self.density[boundary_element] * \
                                                     self.automat.cell_volume

    def cpu_start(self):
        """
        Метод для запуска ядра на центральном процессоре
        """

        import callcore.cpu_calculation as cpu_calc

        tw = self.get_time_steps()

        self.set_initial_conditions()
        self.set_boundary_conditions()

        # TODO: change this later
        max_angle = 90.0
        max_boundary_energy = 1.0E-18
        shear_modulus = 4.0E+10
        max_mobility = 1.0E-2
        burgers_vector = 3.615E-10

        for time_step_ in range(self.num_of_time_steps + 1):

            self.current_temperature, self.current_heat_energy \
                = cpu_calc.calculate_heat_transfer(number_of_cells=self.automat.number_of_cells,
                                                   total_neighbors=self.automat.total_neighbors,
                                                   rasorientation_angles=self.automat.rasorientation_angles,
                                                   initial_heat_energy=self.initial_heat_energy,
                                                   current_heat_energy=self.current_heat_energy,
                                                   initial_temperature=self.initial_temperature,
                                                   current_temperature=self.current_temperature,
                                                   heat_conductivity=self.heat_conductivity,
                                                   heat_capacity=self.heat_capacity,
                                                   density=self.density, cell_size=self.automat.cell_size,
                                                   phonon_portion=self.phonon_portion,
                                                   cell_surface=self.automat.cell_surface,
                                                   cell_volume=self.automat.cell_volume,
                                                   time_step=self.time_step)

            new_grain_indices, new_disl_dens \
                = cpu_calc.recrystallization_simulation(number_of_cells=self.automat.number_of_cells,
                                                        total_neighbors=self.automat.total_neighbors,
                                                        alphas=self.automat.alphas,
                                                        betas=self.automat.betas,
                                                        gammas=self.automat.gammas,
                                                        grain_indices=self.automat.grain_indeces,
                                                        max_boundary_energy=max_boundary_energy,
                                                        max_angle=max_angle,
                                                        temperature=self.current_temperature,
                                                        max_mobility=max_mobility,
                                                        dislocation_density=self.initial_dislocation_density,
                                                        shear_modulus=shear_modulus,
                                                        burgers_vector=burgers_vector,
                                                        time_step=self.time_step,
                                                        cell_size=self.automat.cell_size)

            self.automat.grain_indeces[:] = new_grain_indices[:]
            self.initial_dislocation_density[:] = new_disl_dens[:]

            self.set_boundary_conditions()

            self.initial_heat_energy[:] = self.current_heat_energy[:]
            self.initial_temperature[:] = self.current_temperature[:]

            print(f'Time Step : {time_step_}/{self.num_of_time_steps}')

            if time_step_ in tw:
                print(f'Time Step : {time_step_}')
                print(f'Average Temperature : {np.average(self.initial_temperature)}')
                self.db.write_result_to_db(specimen_name=self.name,
                                           time_step=time_step_,
                                           task_name=self.task_name,
                                           location_types=self.automat.location_type,
                                           grain_indices=self.automat.grain_indeces,
                                           x_coords=self.automat.x_coords,
                                           y_coords=self.automat.y_coords,
                                           z_coords=self.automat.z_coords,
                                           temperature=self.initial_temperature,
                                           number_of_cells=self.automat.number_of_cells)

        # self.delta_deformation = cpu_calc.calc_heat_expansion(self.automat.number_of_cells,
        #                                                       self.current_temperature,
        #                                                       self.initial_temperature)
        #
        # self.delta_stress = cpu_calc.calc_stresses(self.automat.number_of_cells,
        #                                            self.current_temperature,
        #                                            self.initial_temperature)
        #
        # self.current_deformation = self.current_temperature + self.delta_deformation
        #
        # self.delta_mechanical_energy = cpu_calc.calc_mechanical_energy(self.automat.number_of_cells,
        #                                                                self.automat.cell_volume,
        #                                                                self.current_deformation,
        #                                                                self.delta_deformation)
        #
        # self.current_heat = self.current_heat - self.delta_mechanical_energy
        #
        # # self.current_temperature = self.current_temperature - self.delta_mechanical_energy / self.heat_capacity
        #
        # self.initial_heat[:] = self.current_heat[:]
        # self.initial_temperature[:] = self.current_temperature[:]

        # self.set_boundary_conditions('left', 1000.0)
        # self.set_boundary_conditions('right', 1000.0)

        # print(f'Heat Expansion : {np.average(self.current_deformation)}')
        # print(f'Total stresses : {np.average(self.delta_stress)}')
        # print(f'Mechanical Energy : {np.average(self.delta_mechanical_energy)}')

    def get_time_steps(self):
        time_steps = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000,
                      60000, 70000, 80000, 90000, 100000]
        return time_steps
