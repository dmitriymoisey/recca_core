# Методы для чтения информации о задании

def read_seca_file(task_name):
    print(f'Reading : {task_name}')
    d = {}
    with open(task_name) as seca_file:
        for line in seca_file:
            if not (line.startswith('#') or line == '\n'):
                (key, val) = line.split(' = ')
                key, val = key.replace(' ', ''), val.replace(' ', '')
                val = val.replace('\n', '')
                d[key] = val
        seca_file.close()
    return d


def read_res(path):
    print(f'Reading : {path}')
    z_coords = []
    temperature = []
    with open(path) as f:
        for line in f:
            items = line.split(' ')
            z_coords.append(float(items[2]))
            temperature.append(float(items[4]))
        f.close()
    return {'z coordinates': z_coords, 'temperature': temperature}


def read_grains_file(path):
    f = open(path)

    index = []
    material = []
    x_euler_angle = [0.0]
    y_euler_angle = [0.0]
    z_euler_angle = [0.0]
    disloc_dens = []
    aver_disl_dens = []
    max_deviation = []
    first_type = []
    second_type = []
    third_type = []

    for line in f:
        if not (line.startswith('#')):
            items = line.split(' ')
            index.append(int(items[0]))
            material.append(items[1])
            x_euler_angle.append(float(items[2]))
            y_euler_angle.append(float(items[3]))
            z_euler_angle.append(float(items[4]))
            disloc_dens.append(float(items[5]))
            aver_disl_dens.append(float(items[6]))
            max_deviation.append(float(items[7]))
            first_type.append(int(items[8]))
            second_type.append(int(items[9]))
            third_type.append(int(items[10]))

    data_ = {
        'index': index,
        'material': material,
        'x_euler_angle': x_euler_angle,
        'y_euler_angle': y_euler_angle,
        'z_euler_angle': z_euler_angle,
        'disloc_dens': disloc_dens,
        'aver_disl_dens': aver_disl_dens,
        'max_deviation': max_deviation,
        'first_type': first_type,
        'second_type': second_type,
        'third_type': third_type
    }
    return data_


def read_init_cond_file_geometry(path):
    lines = []

    f = open(path)

    for line in f.readlines():
        if not line.startswith('#'):
            lines.append(line)

    first = lines[0].split(' ')

    first = {
        'number_of_inner_cells': int(first[0]),
        'spec_size_X': float(first[1]),
        'spec_size_Y': float(first[2]),
        'spec_size_Z': float(first[3]),
        'number_of_grains': int(first[4])
    }

    grain_index = []
    x_coord = []
    y_coord = []
    z_coord = []
    location_type = []

    for line in lines[1:]:
        items = line.split(' ')
        grain_index.append(int(items[0]))
        x_coord.append(float(items[1]))
        y_coord.append(float(items[2]))
        z_coord.append(float(items[3]))
        location_type.append(int(items[4]))

    data = {
        'general_data': first,
        'grain_indeces': grain_index,
        'x_coord': x_coord,
        'y_coord': y_coord,
        'z_coord': z_coord,
        'location_type': location_type
    }

    return data
