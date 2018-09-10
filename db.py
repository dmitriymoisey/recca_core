import sqlite3 as sqlite
import DBCommon

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

        try:
            self.cursor.execute(f"""SELECT * FROM specimens WHERE name='{specimen_name}';""")
        except sqlite.DatabaseError as err:
            print(err)
            print('No such specimen in database')
        else:
            print('Reading Specimen Data from Database...')
        result = self.cursor.fetchall()

        data = {'Specimen Name': result[0][0], 'Cell Number X': result[0][1], 'Cell Number Y': result[0][2],
                'Cell Number Z': result[0][3], 'Cell Size': result[0][4], 'Number Of Grains': result[0][5],
                'Angle Range': result[0][6], DBCommon.TYPE_OF_GRAIN_DISTRIB: result[0][7],
                'Material': result[0][8], 'Initial Conditions': result[0][9],
                'Boundary Conditions': result[0][10], 'Task': result[0][11]}

        # считываем информацию о материале
        try:
            self.cursor.execute(f"""SELECT * FROM {DBCommon.MATERIALS} WHERE {DBCommon.NAME}='{data['Material']}'""")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to read material table')
        else:
            print('Reading material data from database...')

        result = self.cursor.fetchall()[0]

        material_data = {'Name': result[0], 'Heat Conductivity': result[1], 'Density': result[2],
                         'Heat Expansion': result[3], 'Heat Capacity': result[4], 'Phonon Portion': result[5],
                         'Angle Limit HAGB': result[6], 'Energy HAGB': result[7], 'Max. Mobility': result[8],
                         'Lattice Parameter': result[9], 'Shear Modulus': result[11]}

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

        initial_condition_data = {'Name': result[0], 'Temperature': result[1], 'Elastic Energy': result[2],
                                  'Dislocation Density': result[3], 'Moment X': result[4], 'Moment Y': result[5],
                                  'Moment Z': result[6]}

        # считываем информацию о граничных условиях
        bound_cond_table_name = f"{specimen_name}_BoundaryCondition"
        try:
            self.cursor.execute(f"""SELECT * FROM {bound_cond_table_name}""")
        except sqlite.DatabaseError as err:
            print(err)
            print('Failed to read boundary conditions table')
        else:
            print('Reading boundary conditions data from database...')

        boundary_condition_data = []

        for row in self.cursor.fetchall():
            facet_data = {'Facet': row[1], 'Temperature Average': row[2], 'Temperature Deviation': row[3],
                          'Temperature Load Time': row[4], 'Elastic Energy Average': row[5],
                          'Elastic Energy Deviation': row[6], 'Elastic Energy Load Time': row[7],
                          'Dislocation Density Average': row[8], 'Dislocation Density Deviation': row[9],
                          'Dislocation Density Load Time': row[10], 'Moment X Average': row[11],
                          'Moment X Deviation': row[12], 'Moment X Load Time': row[13], 'Moment Y Average': row[14],
                          'Moment Y Deviation': row[15], 'Moment Y Load Time': row[16], 'Moment Z Average': row[17],
                          'Moment Z Deviation': row[18], 'Moment Z Load Time': row[19]}

            boundary_condition_data.append(facet_data)

        print(boundary_condition_data)

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

        task_data = {'Name': result[0], 'Time Step': result[1], 'Total Time': result[2]}

        data['Material'] = material_data
        data['Initial Conditions'] = initial_condition_data
        data['Boundary Conditions'] = boundary_condition_data
        data['Task'] = task_data

        return data

    def create_results_table(self, specimen_name, task_name):
        table_name = f"{specimen_name}_{task_name}"
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        ({DBCommon.TIME_STEP} INT);
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
        ({DBCommon.LOCATION_TYPE} INT,
        {DBCommon.GRAIN_INDEX} INT,
        {DBCommon.COORDINATE_X} REAL, 
        {DBCommon.COORDINATE_Y} REAL, 
        {DBCommon.COORDINATE_Z} REAL,
        {DBCommon.TEMPERATURE} REAL);
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
        ({DBCommon.LOCATION_TYPE} INT,
        {DBCommon.GRAIN_INDEX} INT,
        {DBCommon.COORDINATE_X} REAL, 
        {DBCommon.COORDINATE_Y} REAL, 
        {DBCommon.COORDINATE_Z} REAL);
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
        print('Done writing structure to Database')

    def read_structure_db(self, specimen_name):
        table_name = f"{specimen_name}_StructureData"
        sql_query = f"""SELECT * FROM {table_name}"""
        self.cursor.execute(sql_query)
        location_type = []
        grain_index = []
        x_coords = []
        y_coords = []
        z_coords = []
        for row in self.cursor.fetchall():
            location_type.append(row[0])
            grain_index.append(row[1])
            x_coords.append(row[2])
            y_coords.append(row[3])
            z_coords.append(row[4])

        return {'Location Type': location_type,
                'Grain Index': grain_index,
                'X Coords': x_coords,
                'Y Coords': y_coords,
                'Z Coords': z_coords}

