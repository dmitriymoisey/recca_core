import sqlite3 as sqlite
import sys


class DBUtils:
    def __init__(self):
        self.db_name = "TEST.db"

        try:
            self.connection = sqlite.connect(self.db_name)

        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to connect to database')
        else:
            print('Connection to database established')

        self.cursor = self.connection.cursor()

    def read_db(self, specimen_name):

        data = {}

        try:
            self.cursor.execute(f"""SELECT * FROM specimens WHERE name='{specimen_name}';""")
        except sqlite.DatabaseError as err:
            print(err)
            print('No such specimen in database')
        else:
            print('Reading Specimen Data from Database...')
        result = self.cursor.fetchall()

        data['Specimen Name'] = result[0][1]
        data['Material'] = result[0][2]
        data['Cell Number X'] = result[0][3]
        data['Cell Number Y'] = result[0][4]
        data['Cell Number Z'] = result[0][5]
        data['Number Of Grains'] = result[0][6]
        data['Angle Range'] = result[0][7]
        data['Initial Conditions'] = result[0][8]
        data['Boundary Conditions'] = result[0][9]
        data['Task'] = result[0][10]

        # считываем информацию о материале
        try:
            self.cursor.execute(f"""SELECT * FROM materials WHERE Name='{data['Material']}'""")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to read material table')
        else:
            print('Reading material data from database...')

        result = self.cursor.fetchall()[0]

        material_data = {}
        material_data['Name'] = result[1]
        material_data['Heat Conductivity'] = result[2]
        material_data['Density'] = result[3]
        material_data['Heat Expansion'] = result[4]
        material_data['Heat Capacity'] = result[5]
        material_data['Phonon Portion'] = result[6]
        material_data['Angle Limit HAGB'] = result[7]
        material_data['Energy HAGB'] = result[8]
        material_data['Max. Mobility'] = result[9]
        material_data['Lattice Parameter'] = result[10]

        # считываем информацию о начальных условиях
        init_cond_table_name = f"{specimen_name}_InitialCondition"
        try:
            self.cursor.execute(f"""SELECT * FROM {init_cond_table_name} WHERE Name='{data['Initial Conditions']}'""")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to read initial conditions table')
        else:
            print('Reading initial conditions data from database...')

        result = self.cursor.fetchall()[0]

        initial_condition_data = {}
        initial_condition_data['Name'] = result[1]
        initial_condition_data['Temperature'] = result[2]
        initial_condition_data['Elastic Energy'] = result[3]
        initial_condition_data['Dislocation Density'] = result[4]
        initial_condition_data['Moment X'] = result[5]
        initial_condition_data['Moment Y'] = result[6]
        initial_condition_data['Moment Z'] = result[7]

        # считываем информацию о граничных условиях
        bound_cond_table_name = f"{specimen_name}_BoundaryConditions_{data['Boundary Conditions']}"
        try:
            self.cursor.execute(f"""SELECT * FROM {bound_cond_table_name}""")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to read boundary conditions table')
        else:
            print('Reading boundary conditions data from database...')

        boundary_condition_data = []

        for row in self.cursor.fetchall():
            facet_data = {}

            facet_data['Facet'] = row[1]
            facet_data['Temperature Average'] = row[2]
            facet_data['Temperature Deviation'] = row[3]
            facet_data['Temperature Load Time'] = row[4]
            facet_data['Elastic Energy Average'] = row[5]
            facet_data['Elastic Energy Deviation'] = row[6]
            facet_data['Elastic Energy Load Time'] = row[7]
            facet_data['Dislocation Density Average'] = row[8]
            facet_data['Dislocation Density Deviation'] = row[9]
            facet_data['Dislocation Density Load Time'] = row[10]
            facet_data['Moment X Average'] = row[11]
            facet_data['Moment X Deviation'] = row[12]
            facet_data['Moment X Load Time'] = row[13]
            facet_data['Moment Y Average'] = row[14]
            facet_data['Moment Y Deviation'] = row[15]
            facet_data['Moment Y Load Time'] = row[16]
            facet_data['Moment Z Average'] = row[17]
            facet_data['Moment Z Deviation'] = row[18]
            facet_data['Moment Z Load Time'] = row[19]

            boundary_condition_data.append(facet_data)

        # считываем информацию о параметрах задачи
        task_table_name = f"{specimen_name}_Task"
        try:
            self.cursor.execute(f"""SELECT * FROM {task_table_name}""")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to read task table')
        else:
            print('Reading task data from database...')

        result = self.cursor.fetchall()[0]

        task_data = {}

        task_data['Name'] = result[1]
        task_data['Time Step'] = result[2]
        task_data['Total Time'] = result[3]

        data['Material'] = material_data
        data['Initial Conditions'] = initial_condition_data
        data['Boundary Conditions'] = boundary_condition_data
        data['Task'] = task_data

        return data

    def create_results_table(self, specimen_name, task_name):
        table_name = f"{specimen_name}_{task_name}"
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        ('TimeStep' INT);
        """

        try:
            self.cursor.execute(query)
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to create results table')
        else:
            print('Table is successfully created')
            self.connection.commit()

    def write_result_to_db(self, specimen_name, time_step, task_name,
                           location_types, grain_indices,
                           x_coords, y_coords, z_coords, temperature, number_of_cells):

        try:
            self.cursor.execute(f"INSERT INTO {specimen_name}_{task_name} VALUES({time_step});")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to write time step to task database')
        else:
            print('Time Step is added to database')
            self.connection.commit()


        table_name = f"{specimen_name}_{task_name}_{time_step}"
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        ('LocationType' INT,
        'GrainIndex' INT,
        'CoordinateX' REAL, 
        'CoordinateY' REAL, 
        'CoordinateZ' REAL,
        'Temperature' REAL);
        """

        try:
            self.cursor.execute(create_table_query)
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to create new results table')
        else:
            print('New Results Table is created')

        print('Writing data to database...')

        for i in range(number_of_cells):
            query = f"""
            INSERT INTO {table_name}
            VALUES(
            {location_types[i]},{grain_indices[i]},
            {x_coords[i]},{y_coords[i]},{z_coords[i]},
            {temperature[i]});
            """
            self.cursor.execute(query)
        try:
            self.connection.commit()
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to write result to database')
        else:
            print(f'{time_step} is added to database')


    def write_structure_data_to_db(self, specimen_name, location_types, grain_indices,
                                   x_coords, y_coords, z_coords, number_of_cells):

        table_name = f"{specimen_name}_StructureData"
        create_structure_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        ('LocationType' INT,
        'GrainIndex' INT,
        'CoordinateX' REAL, 'CoordinateY' REAL, 'CoordinateZ' REAL);
		"""

        try:
            self.cursor.execute(create_structure_table_query)
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to create new structure data table')
        else:
            print('New Structure Data Table is created')

        print('Writing structure data to database...')

        for i in range(number_of_cells):
            query = f"""
            INSERT INTO {table_name} 
            VALUES(
            {location_types[i]},
            {grain_indices[i]},
            {x_coords[i]},
            {y_coords[i]},
            {z_coords[i]}
            );
            """
            self.cursor.execute(query)

        self.connection.commit()
