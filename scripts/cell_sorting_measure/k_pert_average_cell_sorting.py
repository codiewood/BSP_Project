import OS_model
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from numpy import genfromtxt

# Generate data

# time_step = 0.005
# end_time = 100
# simulation_count = 10
#
# times = np.linspace(0, end_time, int(end_time / time_step))
#
# f = open('k_pert_frac_length_data_new.txt', 'a')
# np.savetxt(f, times, fmt='%1.3f', newline=", ")
# f.write("\n")
#
# for k_pert in (0.01, 0.1, 1, 10, 100):
#     fractional_lengths = np.zeros((simulation_count, len(times)+1))
#     for i in range(simulation_count):
#         random.seed(i+17)
#         x = OS_model.Monolayer()
#         x.set_k_pert(k_pert)
#         fractional_lengths[i] = x.measure_sorting(end_time, time_step)[1]
#     fractional_length_mean = np.mean(fractional_lengths, axis=0)
#     fractional_length_variance = np.var(fractional_lengths, axis=0)
#     np.savetxt(f, fractional_length_mean, fmt='%1.3f', newline=", ")
#     f.write("\n")
#     np.savetxt(f, fractional_length_variance, fmt='%1.3f', newline=", ")
#     f.write("\n")
#
# f.close()

fractional_length_data = genfromtxt('C:/Users/Codie/Google Drive/University/MMath/Year 3/BSP/BSP_Project/scripts/cell_sorting_measure/k_pert_frac_length_data_new.txt', delimiter=',')

# plot_fractional_lengths = fractional_length_data[:,0::20] # determines how often we plot a point
# palette = ['tab:blue','tab:purple','k','tab:orange','tab:red']
# fig = plt.figure(1)
# for index, k_pert in enumerate((0.01, 0.1, 1, 10, 100)):
#     plt.plot(plot_fractional_lengths[0], plot_fractional_lengths[index + 1], color=palette[index], label=str(k_pert))
# plt.xlabel('Time')
# plt.ylabel('Average fractional length')
# plt.legend(bbox_to_anchor=(1, 1, 1, 0), loc='upper left')
# fig.savefig('frac_length', bbox_inches='tight')
