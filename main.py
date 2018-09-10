import sys
from task import Task

task_name = sys.argv[-1]
specimen_name = sys.argv[-2]

task = Task(specimen_name=specimen_name,
            task_name=task_name)

print(task)
task.cpu_start()

from interf.plot_builder import show_temperature_distribution

# data1 = read_res('task1_100.res')
# data2 = read_res('task1_1000.res')
# data3 = read_res('task1_5000.res')
# data4 = read_res('task1_10000.res')
# data5 = read_res('task1_30000.res')
# data6 = read_res('task1_60000.res')
# data7 = read_res('task1_90000.res')
# show_temperature_distribution(data1['z coordinates'], data1['temperature'])
# show_temperature_distribution(data2['z coordinates'], data2['temperature'])
# show_temperature_distribution(data3['z coordinates'], data3['temperature'])
# show_temperature_distribution(data4['z coordinates'], data4['temperature'])
# show_temperature_distribution(data5['z coordinates'], data5['temperature'])
# show_temperature_distribution(data6['z coordinates'], data6['temperature'])
# show_temperature_distribution(data7['z coordinates'], data7['temperature'])
# plt.ylabel('Temperature (K)')
# plt.xlabel('length (element radius)')
# plt.legend(['1e-10', '1e-9', '5e-9', '1e-8', '3e-8', '6e-8', '9e-8'])
# plt.title('Specimen size : 10X10X60 ( 1e-7 m.)\nBound Cond: T(left) = 1000.0K , T(right) = 300.0K \nTime Step = 1e-12')
# plt.show()
#
