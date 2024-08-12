from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
from agent.utils.dynamical_system_operations import denormalize_state
import numpy as np

# Parameters
params_name = '2nd_order_2D_boundary'
results_base_directory = './'

simulation_length = 2000

# Initial states
x_t_init = np.array([[0.5, 0.6], [-0.75, 0.9], [0.9, -0.9], [-0.9, -0.9], [0.9, 0.9], [0.9, 0.3], [-0.9, -0.1],
                     [-0.9, 0.0], [0.4, 0.4], [0.9, -0.1], [-0.9, -0.5], [0.9, -0.5]])


# Obstacles
# obstacles = {'centers': [[0.0, -20]],
#              'axes': [[5, 5]],
#              'safety_margins': [[1.0, 1.0]]}

obstacles = {'centers': [[250, 272]],
             'axes': [[15, 15]],
             'safety_margins': [[1.0, 1.0]]}


# Load parameters
Params = getattr(importlib.import_module('params.' + params_name), 'Params')
params = Params(results_base_directory)
params.results_path += params.selected_primitives_ids + '/'
params.load_model = True

# Extend state if second order
if params.dynamical_system_order == 2:
    x_t_init = np.concatenate([x_t_init, np.zeros([x_t_init.shape[0], params.manifold_dimensions])], axis=1)

# Initialize framework
learner, _, data = initialize_framework(params, params_name, verbose=False)

# Initialize dynamical system

dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda())

# Initialize trajectory plotter
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
fig.show()
x_t_init_denorm = denormalize_state(x_t_init[:, :params.manifold_dimensions].T, x_min=data['x min'].reshape(-1, 1), x_max=data['x max'].reshape(-1, 1))
trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init_denorm, pause_time=1e-5, goal=data['goals'][0])

# Get min max real data in tensors
x_min = torch.FloatTensor(data['x min']).reshape(1, -1).cuda()
x_max = torch.FloatTensor(data['x max']).reshape(1, -1).cuda()

# Simulate dynamical system and plot
for i in range(simulation_length):
    # Do transition
    x_t = dynamical_system.transition(space='task', obstacles=obstacles)['desired state']

    # Denormalize data
    x_t_denorm = denormalize_state(x_t[:, :params.manifold_dimensions], x_min=x_min, x_max=x_max)

    # Update plot
    trajectory_plotter.update(x_t_denorm.T.cpu().detach().numpy())

