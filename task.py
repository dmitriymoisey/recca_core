# coding=utf-8
from callcore.elements import *
from utils.material import Material
from callcore.scp_packing import *
from recca_io.write import *
from db import DBUtils


class Task:
    def __init__(self, specimen_name, task_name):
        """
        Конструктор задачи
        """

        self.db = DBUtils()

        self.material_name = 'titanium'  # имя материала

        self.name = specimen_name  # имя образца
        self.task_name = task_name  # имя задачи

        self.cell_number_X = 5  # кол-во элементов вдоль оси X
        self.cell_number_Y = 50  # кол-во элементов вдоль оси Y
        self.cell_number_Z = 50  # кол-во элементов вдоль оси Z
        self.cell_size = 1.0e-6  # размер элемента
        self.number_of_grains = 100  # кол-во зерен
        self.angle_range = 10  # углы ориентации кристаллической решетки от 0 до angle_range

        self.automat = CellularAutomaton(self.cell_number_X, self.cell_number_Y, self.cell_number_Z, self.cell_size)
        self.automat.create_grain_structure(self.number_of_grains, angle_range=30.0)

        self.db.write_structure_data_to_db(specimen_name=self.name,
                                           location_types=self.automat.location_type,
                                           grain_indices=self.automat.grain_indeces,
                                           x_coords=self.automat.x_coords,
                                           y_coords=self.automat.y_coords,
                                           z_coords=self.automat.z_coords,
                                           number_of_cells=self.automat.number_of_cells)

        self.db.create_results_table(specimen_name=self.name,
                                     task_name=self.task_name)

        self.time_step = 1.0e-9  # значение временного шага
        self.total_time = 1.0e-3  # общее время
        self.num_of_time_steps = int(self.total_time / self.time_step)  # кол-во шагов по времени

        # self.automat.recrystallization_simulation(average_dislocation_density=10.0,
        #                                           dislocation_deviation=0.5,
        #                                           temperature=self.initial_temperature)

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

    def check_task_params(self):
        '''
        Метод для проверки параметров задачи
        :return:
        '''
        pass

    def set_boundary_conditions(self, facet, arr, value):
        if facet == 'top':
            for boundary_element in self.automat.top_boundary_elements:
                if not boundary_element == -1:
                    arr[boundary_element] = value

        if facet == 'bottom':
            for boundary_element in self.automat.bottom_boundary_elements:
                if not boundary_element == -1:
                    arr[boundary_element] = value

        if facet == 'front':
            for boundary_element in self.automat.front_boundary_elements:
                if not boundary_element == -1:
                    arr[boundary_element] = value

        if facet == 'back':
            for boundary_element in self.automat.back_boundary_elements:
                if not boundary_element == -1:
                    arr[boundary_element] = value

        if facet == 'left':
            for boundary_element in self.automat.left_boundary_elements:
                if not boundary_element == -1:
                    arr[boundary_element] = value

        if facet == 'right':
            for boundary_element in self.automat.right_boundary_elements:
                if not boundary_element == -1:
                    arr[boundary_element] = value

    def material_params(self):
        material = Material(self.material_name)

        self.heat_capacity = material.HEAT_CAPACITY
        self.density = material.DENSITY
        self.heat_conductivity = np.ones(self.automat.number_of_cells, dtype=np.float64) * material.HEAT_CONDUCTIVITY
        self.heat_expansion_coeff = material.HEAT_EXPANSION_COEFF
        self.phonon_portion = 0.666666666666667

    def cpu_start(self):
        """
        Метод для запуска ядра на центральном процессоре
        """

        import callcore.cpu_calculation as cpu_calc

        self.material_params()

        tw = self.get_time_steps()

        initial_heat_energy = np.ones(self.automat.number_of_cells,
                                      dtype=np.float64) * self.heat_capacity * self.density * self.automat.cell_volume * 300.0
        current_heat_energy = np.ones(self.automat.number_of_cells,
                                      dtype=np.float64) * self.heat_capacity * self.density * self.automat.cell_volume * 300.0

        initial_temperature = np.ones(self.automat.number_of_cells, dtype=np.float64) * 300.0
        current_temperature = np.ones(self.automat.number_of_cells, dtype=np.float64) * 300.0

        self.set_boundary_conditions(facet='top', arr=initial_temperature, value=1300.0)
        self.set_boundary_conditions(facet='top', arr=initial_heat_energy,
                                     value=1300.0 * self.heat_capacity * self.density * self.automat.cell_volume)

        self.set_boundary_conditions(facet='bottom', arr=initial_temperature, value=300.0)
        self.set_boundary_conditions(facet='bottom', arr=initial_heat_energy,
                                     value=300.0 * self.heat_capacity * self.density * self.automat.cell_volume)

        self.set_boundary_conditions(facet='top', arr=current_temperature, value=1300.0)
        self.set_boundary_conditions(facet='top', arr=current_heat_energy,
                                     value=1300.0 * self.heat_capacity * self.density * self.automat.cell_volume)

        self.set_boundary_conditions(facet='bottom', arr=current_temperature, value=300.0)
        self.set_boundary_conditions(facet='bottom', arr=current_heat_energy,
                                     value=300.0 * self.heat_capacity * self.density * self.automat.cell_volume)

        for time_step_ in range(self.num_of_time_steps + 1):

            current_temperature, current_heat_energy = cpu_calc.calculate_heat_transfer(
                number_of_cells=self.automat.number_of_cells,
                total_neighbors=self.automat.total_neighbors,
                rasorientation_angles=self.automat.rasorientation_angles,
                initial_heat_energy=initial_heat_energy,
                current_heat_energy=current_heat_energy,
                initial_temperature=initial_temperature,
                current_temperature=current_temperature,
                heat_conductivity=self.heat_conductivity,
                heat_capacity=self.heat_capacity,
                density=self.density, cell_size=self.automat.cell_size,
                phonon_portion=self.phonon_portion,
                cell_surface=self.automat.cell_surface,
                cell_volume=self.automat.cell_volume,
                time_step=self.time_step)

            self.set_boundary_conditions(facet='top', arr=current_temperature, value=1300.0)
            self.set_boundary_conditions(facet='top', arr=current_heat_energy,
                                         value=1300.0 * self.heat_capacity * self.density * self.automat.cell_volume)

            self.set_boundary_conditions(facet='facet', arr=current_temperature, value=300.0)
            self.set_boundary_conditions(facet='facet', arr=current_heat_energy,
                                         value=300.0 * self.heat_capacity * self.density * self.automat.cell_volume)

            initial_heat_energy[:] = current_heat_energy[:]
            initial_temperature[:] = current_temperature[:]

            if time_step_ in tw:
                print(f'Time Step : {time_step_}')
                print(f'Average Temperature : {np.average(initial_temperature)}')
                create_res_file(file_name=f'task1_{time_step_}.res',
                                location_type=self.automat.location_type,
                                x=self.automat.x_coords,
                                y=self.automat.y_coords,
                                z=self.automat.z_coords,
                                temperature=current_temperature)

                self.db.write_result_to_db(specimen_name=self.name,
                                           time_step=time_step_,
                                           task_name=self.task_name,
                                           location_types=self.automat.location_type,
                                           grain_indices=self.automat.grain_indeces,
                                           x_coords=self.automat.x_coords,
                                           y_coords=self.automat.y_coords,
                                           z_coords=self.automat.z_coords,
                                           temperature=initial_temperature,
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
            #

    def get_time_steps(self):
        time_steps = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000,
                      60000, 70000, 80000, 90000, 100000]
        return time_steps
