import sqlite3


class DataBaseUtils:
    def __init__(self, task_name):
        self.db_name = ""

        try:
            self.connection = sqlite3.connect(self.db_name)
        except sqlite3.DatabaseError:
            print('Failed to connect to database')

        self.cursor = self.connection.cursor()

    def read_task_data(self):
        query = """"""
        self.cursor.execute(query)
        task_data = self.cursor.fetchall()
        return task_data

    def write_structure_data(self, specimen_name, number_of_cells, location_type, grain_index, x, y, z):
        create_table_query = f"""
        CREATE TABLE IF NOT EXIST {specimen_name}_Structure
        ('ElementIndex' INT, 
        'LocationType' INT,
        'GrainIndex' INT,
        'X_Coordinate' REAL,
        'Y_Coordinate' REAL,
        'Z_Coordinate' REAL
        );
        """
        try:
            self.cursor.execute(create_table_query)
        except sqlite3.DatabaseError as err:
            print(err)
            print('Failed to create specimen structure table')

        for i in range(number_of_cells):
            self.cursor.execute(f"""INSERT INTO {specimen_name}_Structure VALUES 
            ({i},{location_type[i]},{grain_index[i]},{x[i]},{y[i]},{z[i]}); """)

    def write_result_to_data(self, result):
        table_name = f'{result.specimen_name}_Result_{result.time_step}'
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        ('ElementIndex' INT,
        'LocationType' INT,
        'GrainIndex' INT,
        'X_Coordinate' REAL,
        'Y_Coordinate' REAL,
        'Z_Coordinate' REAL,
        'Temperature' REAL,
        'ElasticEnergy' REAL,
        'DislocationDensity' REAL,
        'MomentX' REAL,
        'MomentY' REAL,
        'MomentZ' REAL
        );
        """

        try:
            self.cursor.execute(create_table_query)
            print(f'New Results Table : {table_name} is created!')
        except sqlite3.DatabaseError as err:
            print(f'Error : {err}')
            print(f'Failed to create new results table {table_name}')

        for i in range(result.number_of_elements):
            self.cursor.execute(f"""
            INSERT INTO {table_name} 
            VALUES({result.element_index[i]},{result.location_type[i]},
            {result.grain_index[i]},{result.x_coords[i]},{result.y_coords[i]},
            {result.temperature[i]},{result.elastic_energy[i]},
            {result.moment_x[i]},{result.moment_y[i]},{result.moment_z[i]});
            """)

    def close_connection(self):
        self.connection.close()


import numpy as np


class Result:

    def __init__(self, specimen_name, time_step,
                 element_index, location_type, grain_index,
                 x_coords, y_coords, z_coords,
                 temperature, elastic_energy, dislocation_density,
                 moment_x, moment_y, moment_z):
        self.specimen_name = specimen_name
        self.time_step = time_step
        self.location_type = location_type
        self.grain_index = grain_index
        self.element_index = element_index
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.z_coords = z_coords
        self.temperature = temperature
        self.elastic_energy = elastic_energy
        self.dislocation_density = dislocation_density
        self.moment_x = moment_x
        self.moment_y = moment_y
        self.moment_z = moment_z

        self.number_of_elements = np.max(element_index)
        self.average_temperature = np.average(self.temperature)
        self.average_elastic_energy = np.average(self.elastic_energy)
        self.average_dislocation_density = np.average(self.dislocation_density)
        self.average_moment_x = np.average(self.moment_x)
        self.average_moment_y = np.average(self.moment_y)
        self.average_moment_z = np.average(self.moment_z)

    def __repr__(self):
        return f'Time Step = {self.time_step};\nAverage Temperature = {self.average_temperature}'
