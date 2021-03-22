import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

time_step = 0.005
end_time = 100
simulation_count = 10

times = np.linspace(0, end_time, int(end_time / time_step) + 1)

rc('text', usetex=True)
rc('font', family='serif')

plot_every = 20  # Determines how frequently we pull data points to plot.
plot_times = times[0::plot_every]

for col_param in (0, 0.01, 10, 100):
    param_vals = (0, 0.01, 10, 100, 1, 0.1, col_param)
    fig, ax = plt.subplots()
    ax.set_ylim([-0.1, 1.3])
    ax.grid(color='lightgray', linestyle='--', alpha=0.7)
    for index, param in enumerate(param_vals):
        file_name = str(param).replace('.', '_') + 'lambda_frac_length_spaced_data.txt'
        fractional_length_data = genfromtxt(file_name, delimiter=',')
        plot_fractional_lengths = fractional_length_data[:, 0::plot_every]
        if param == col_param:
            colour = 'mediumpurple'
            opacity = 1
            std_dev = np.sqrt(plot_fractional_lengths[1])
            plt.fill_between(plot_times, plot_fractional_lengths[0] + std_dev, plot_fractional_lengths[0] - std_dev,
                             alpha=0.1, color=colour)
        elif param == 0.1:
            colour = 'royalblue'
        elif param == 1:
            colour = 'plum'
        # elif param == 0.1:
        #     colour = 'darkgray'
        else:
            colour = 'lightgray'
        plt.plot(plot_times, plot_fractional_lengths[0], label=str(param), color=colour)
    plt.xlabel('Time')
    plt.ylabel('Fractional length')
    # plt.legend(bbox_to_anchor=(1, 1, 1, 0), loc='upper left', title=r'$k_\textrm{pert}$')
    plt.show()
    plot_name = str(col_param).replace('.', '_') + 'lambda_frac_plot.pdf'
    fig.savefig(plot_name, bbox_inches='tight', )
