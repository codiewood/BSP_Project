# import OS_model
# from numpy import random
# import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

rc('text', usetex=True)
rc('font', family='serif')

# Generate data

time_step = 0.005
end_time = 100
simulation_count = 10

times = np.linspace(0, end_time, int(end_time / time_step) + 1)

# Generate Data
# def generate_data(k_pert):
#     file_name = str(k_pert).replace('.', '_') + '_frac_length_data.txt'
#     f = open(file_name, 'a')
#     fractional_lengths = np.zeros((simulation_count, len(times)))
#     for i in range(simulation_count):
#         random.seed(i + 17)
#         x = OS_model.Monolayer()
#         x.set_k_pert(k_pert)
#         fractional_lengths[i] = x.measure_sorting(end_time)[1]
#     fractional_length_mean = np.mean(fractional_lengths, axis=0)
#     fractional_length_variance = np.var(fractional_lengths, axis=0)
#     np.savetxt(f, fractional_length_mean, fmt='%1.3f', newline=", ")
#     f.write("\n")
#     np.savetxt(f, fractional_length_variance, fmt='%1.3f', newline=", ")
#     f.write("\n")
#     f.close()
#
#
# if __name__ == '__main__':
#     jobs = []
#     for k_pert in (2, 4, 8, 16, 32):
#         p = multiprocessing.Process(target=generate_data, args=(k_pert,))
#         jobs.append(p)
#         p.start()

# k_pert_vals = (0.01,0.1,0,1,10,100)
# file_name = str(k_pert).replace('.', '_') + '_frac_length_data.txt'


plot_every = 20  # Determines how frequently we pull data points to plot.

plot_times = times[0::plot_every]
palette = ['tab:cyan', 'tab:blue', 'tab:purple', 'k', 'tab:orange', 'tab:red']
fig = plt.figure(1)
for index, k_pert in enumerate((0, 0.01, 0.1, 1)):
    file_name = str(k_pert).replace('.', '_') + '_frac_length_data.txt'
    fractional_length_data = genfromtxt(file_name, delimiter=',')
    plot_fractional_lengths = fractional_length_data[:, 0::plot_every]
    plt.plot(plot_times, plot_fractional_lengths[0], label=str(k_pert))
    std_dev = np.sqrt(plot_fractional_lengths[1])
    plt.fill_between(plot_times, plot_fractional_lengths[0] + std_dev, plot_fractional_lengths[0] - std_dev, alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Average fractional length')
plt.legend(bbox_to_anchor=(1, 1, 1, 0), loc='upper left', title=r'$k_\textrm{pert}$')
fig.savefig('file.pdf', bbox_inches='tight', )
