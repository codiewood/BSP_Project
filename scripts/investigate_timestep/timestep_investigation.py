import OS_model
from numpy import random

# import numpy as np

time_step_old = 0.005
time_step_new = 0.01

random.seed(72)
x = OS_model.Monolayer(size=5)
x.set_k_pert(k_pert=0)  # Remove randomness so that simulations are 'the same'
x.show_cells()
# Writing data file
# f = open('timestep_forces_data.txt', 'a')

# f.write("Initial positions \n")
# np.savetxt(f, x.positions, fmt='%1.3f', newline=", ")

# f.write("\n Initial forces with old time_step \n")
# np.savetxt(f, x.interaction_forces(), fmt='%1.3f', newline=", ")

x.show_cells(0.01, time_step=time_step_old)  # Run simulation

# Writing data file
# f.write("\n Forces at 0.01 with old time_step \n")
# np.savetxt(f, x.interaction_forces(), fmt='%1.3f', newline=", ")

# f.write("\n Positions at 0.01 with old time_step \n")
# np.savetxt(f, x.positions, fmt='%1.3f', newline=", ")

# Create identical y
random.seed(72)
y = OS_model.Monolayer(size=5)
y.set_k_pert(k_pert=0)

# f.write("\n Initial forces with new time_step \n")
# np.savetxt(f, y.interaction_forces(), fmt='%1.3f', newline=", ")

y.show_cells(time=0.01, time_step=time_step_new)  # Run simulation

# f.write("\n Forces at 0.01 with new time_step \n")
# np.savetxt(f, y.interaction_forces(), fmt='%1.3f', newline=", ")

# f.write("\n Positions at 0.01 with new time_step \n")
# np.savetxt(f, y.positions, fmt='%1.3f', newline=", ")

# f.close()

old_pos = x.positions
new_pos = y.positions
dif = old_pos - new_pos
print(dif)  # We see that differences in the calculated position of molecules is negligible

x.show_cells(1, time_step=time_step_old)  # Continue simulation until 1 hour
y.show_cells(1, time_step=time_step_new)

old_pos2 = x.positions
new_pos2 = y.positions
dif2 = old_pos2 - new_pos2
print(dif2)  # The difference grows as we run the simulation for longer
