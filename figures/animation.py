import OS_model
from matplotlib import pyplot as plt
#from matplotlib import patches as mpatches
from matplotlib import animation
from numpy import random

random.seed(1234)
show_interactions = False
time_step = 0.005
monolayer = OS_model.Monolayer()
# monolayer.manual_cell_placement(((2.0, 2.0), (2.0, 3.0)), [0, 0])

vmax = monolayer.size  # Set up axes and plot
radius = 0.5
r_max, mag, drag = monolayer.sim_params
fig, ax = monolayer.generate_axes()


def build_plot(time):
    robin = plt.Circle((vmax / 2, vmax / 2), radius=vmax, facecolor='w')
    fig.gca().add_artist(robin)
    monolayer.simulate(0.25 * time, time_step)
    pos = monolayer.positions
    cell_colour = ['plum', 'royalblue']
    for xi, yi, cell_type in zip(pos[:, 0], pos[:, 1], monolayer.cell_types):
        if show_interactions:
            interaction_zone = plt.Circle((xi, yi), radius=r_max,
                                          facecolor='grey', edgecolor='k', alpha=0.15)
            fig.gca().add_artist(interaction_zone)
        cell = plt.Circle((xi, yi), radius=radius, facecolor=cell_colour[cell_type], edgecolor='k')
        fig.gca().add_artist(cell)
    plt.title('Cells at ' + str(round(monolayer.sim_time, 1)) + ' hours')


anim = animation.FuncAnimation(fig, build_plot, frames=400, interval=100)
anim.save('default.gif')
