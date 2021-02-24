import OS_model
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import animation
from numpy import random

random.seed(1234)
show_interactions = False
time_step = 0.005
monolayer = OS_model.Monolayer(size=5)
monolayer.manual_cell_placement(((2.0,2.0),(2.0,3.0)),[0,0])

vmax = monolayer.size  # Set up axes and plot
radius = 0.5
r_max, mag, drag = monolayer.sim_params
fig, ax = plt.subplots()
ax.set_xlim(-radius, vmax + radius)
ax.set_ylim(-radius, vmax + radius)
ax.set_aspect(1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plum_patch = mpatches.Patch(facecolor='plum', edgecolor='k', label='Type 0')
blue_patch = mpatches.Patch(facecolor='royalblue', edgecolor='k', label='Type 1')
leg = [plum_patch, blue_patch]
if show_interactions:
    int_patch = mpatches.Patch(facecolor='grey', edgecolor='k', alpha=0.15, label='Interaction')
    leg.append(int_patch)
plt.legend(handles=leg, bbox_to_anchor=((3 - len(leg)) / 6, -0.15, len(leg) / 3, .102), loc='upper left',
           ncol=len(leg), mode="expand", borderaxespad=0.)

def build_plot(time):
    robin = plt.Circle((vmax / 2, vmax / 2), radius=vmax, facecolor='w')
    fig.gca().add_artist(robin)
    monolayer.simulate(0.25*time, time_step)
    pos = monolayer.positions
    cell_colour = ['plum', 'royalblue']
    for xi, yi, cell_type in zip(pos[:, 0], pos[:, 1], monolayer.cell_types):
        if show_interactions:
            interaction_zone = plt.Circle((xi, yi), radius=r_max,
                                          facecolor='grey', edgecolor='k', alpha=0.15)
            fig.gca().add_artist(interaction_zone)
        cell = plt.Circle((xi, yi), radius=radius, facecolor=cell_colour[cell_type], edgecolor='k')
        fig.gca().add_artist(cell)
    plt.title('Cells at ' + str(round(monolayer.sim_time,1)) + ' hours')

anim = animation.FuncAnimation(fig, build_plot, frames=400, interval=100)
anim.save('2_same_cells.gif')