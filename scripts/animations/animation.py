import OS_model
from matplotlib import pyplot as plt
from matplotlib import animation
from numpy import random

# Set up desired conditions

random.seed(1234)
show_interactions = False
time_step = 0.005
end_time = 50
division_rate = 1 / 24
monolayer = OS_model.Monolayer(space=True)
monolayer.set_division_timer(division_rate)

vmax = monolayer.size  # Set up axes and plot
radius = 0.5
r_max, mag, drag = monolayer.r_max, monolayer.mag, monolayer.drag
spacing = monolayer.space + 1 + division_rate * end_time * vmax
fig, ax = OS_model.generate_axes(radius=radius, size=vmax, spacing=spacing)


def build_plot(time):
    robin = plt.Circle((vmax / 2, vmax / 2), radius=500, facecolor='w')
    fig.gca().add_artist(robin)
    monolayer.simulate(0.25 * time)
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


anim = animation.FuncAnimation(fig, build_plot, frames=4 * end_time, interval=100)
anim.save('file_name.gif')
