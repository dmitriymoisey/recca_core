
def create_file(file_name, location_type, grain_indeces, x, y, z):
    file = open(file_name, 'w+')
    print(f'Writing data in {file_name}: START...')
    for i in range(grain_indeces.shape[0]):
        line = f'{location_type[i]} {grain_indeces[i]} {x[i]} {y[i]} {z[i]}\n'
        file.write(line)
    file.close()
    print(f'Writing data in {file_name}: DONE.')


def create_res_file(file_name, location_type, x, y, z, temperature):
    file = open(file_name, 'w+')
    print(f'Writing data in {file_name}: START...')
    for i in range(temperature.shape[0]):
        line = f'{location_type[i]} {x[i]} {y[i]} {z[i]} {temperature[i]}\n'
        file.write(line)
    file.close()
    print(f'Writing data in {file_name}: DONE.')