import numpy as np
import torch
import torch.nn.functional as F
from agent.neural_network import NeuralNetwork
from agent.utils.ranking_losses import TripletLoss, TripletAngleLoss, TripletAngleLossSquared
from agent.dynamical_system import DynamicalSystem
from agent.utils.dynamical_system_operations import normalize_state


class StableImitation:
    """
    Computes PUMA losses and optimizes Neural Network
    """
    def __init__(self, data, params):
        # Params file parameters
        self.dim_manifold = params.manifold_dimensions
        self.dynamical_system_order = params.dynamical_system_order
        self.space_type = params.space_type
        self.dim_ambient_space = params.ambient_space_dimension
        self.dim_state = self.dim_ambient_space * params.dynamical_system_order
        self.imitation_window_size = params.imitation_window_size
        self.batch_size = params.batch_size
        self.save_path = params.results_path
        self.multi_motion = params.multi_motion
        self.generalization_window_size = params.stabilization_window_size
        self.imitation_loss_weight = params.imitation_loss_weight
        self.stabilization_loss_weight = params.stabilization_loss_weight
        self.boundary_loss_weight = params.boundary_loss_weight
        self.load_model = params.load_model
        self.results_path = params.results_path
        self.interpolation_sigma = params.interpolation_sigma
        self.delta_t = 1  # used for training, can be anything (in principle)

        # Parameters data processor
        self.primitive_ids = np.array(data['demonstrations primitive id'])
        self.n_primitives = data['n primitives']
        self.goals_tensor = torch.FloatTensor(data['goals training']).cuda()
        self.demonstrations_train = data['demonstrations train']
        self.n_demonstrations = data['n demonstrations']
        self.demonstrations_length = data['demonstrations length']
        self.min_vel = torch.from_numpy(data['vel min train'].reshape([1, self.dim_ambient_space])).float().cuda()
        self.max_vel = torch.from_numpy(data['vel max train'].reshape([1, self.dim_ambient_space])).float().cuda()
        if data['acc min train'] is not None:
            min_acc = torch.from_numpy(data['acc min train'].reshape([1, self.dim_ambient_space])).float().cuda()
            max_acc = torch.from_numpy(data['acc max train'].reshape([1, self.dim_ambient_space])).float().cuda()
        else:
            min_acc = None
            max_acc = None

        # Dynamical-system-only params
        self.params_dynamical_system = {'saturate transition': params.saturate_out_of_boundaries_transitions,
                                        'x min': data['x min'],
                                        'x max': data['x max'],
                                        'vel min train': self.min_vel,
                                        'vel max train': self.max_vel,
                                        'acc min train': min_acc,
                                        'acc max train': max_acc}

        # Initialize Neural Network losses
        self.mse_loss = torch.nn.MSELoss()
        if params.triplet_type == 'euclidean':
            self.triplet_loss = TripletLoss(margin=params.triplet_margin)
        elif params.triplet_type == 'spherical':
            self.triplet_loss = TripletAngleLoss(margin=params.triplet_margin)
        elif params.triplet_type == 'spherical squared':
            self.triplet_loss = TripletAngleLossSquared(margin=params.triplet_margin)
        else:
            raise ValueError('triplet type not valid, options: euclidean, spherical.')

        # Initialize Neural Network
        self.model = NeuralNetwork(dim_state=self.dim_state,
                                   dynamical_system_order=self.dynamical_system_order,
                                   n_primitives=self.n_primitives,
                                   multi_motion=self.multi_motion,
                                   latent_space_dim=params.latent_space_dim,
                                   neurons_hidden_layers=params.neurons_hidden_layers).cuda()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=params.learning_rate,
                                           weight_decay=params.weight_decay)

        # Load Neural Network if requested
        if self.load_model:
            self.model.load_state_dict(torch.load(self.results_path + 'model'), strict=False)

        # Initialize latent goals
        self.model.update_goals_latent_space(self.goals_tensor)

    def init_dynamical_system(self, initial_states, primitive_type=None, delta_t=1):
        """
        Creates dynamical system using the parameters/variables of the learning policy
        """
        # If no primitive type, assume single-model learning
        if primitive_type is None:
            primitive_type = torch.FloatTensor([1])

        # Create dynamical system
        dynamical_system = DynamicalSystem(x_init=initial_states,
                                           space_type=self.space_type,
                                           model=self.model,
                                           primitive_type=primitive_type,
                                           order=self.dynamical_system_order,
                                           min_state_derivative=[self.params_dynamical_system['vel min train'],
                                                                 self.params_dynamical_system['acc min train']],
                                           max_state_derivative=[self.params_dynamical_system['vel max train'],
                                                                 self.params_dynamical_system['acc max train']],
                                           saturate_transition=self.params_dynamical_system['saturate transition'],
                                           dim_state=self.dim_state,
                                           delta_t=delta_t,
                                           x_min=self.params_dynamical_system['x min'],
                                           x_max=self.params_dynamical_system['x max'])

        return dynamical_system

    def imitation_loss(self, state_sample, primitive_type_sample):
        """
        Imitation learning loss
        """
        # Create dynamical system
        dynamical_system = self.init_dynamical_system(initial_states=state_sample[:, :, 0],
                                                      primitive_type=primitive_type_sample)

        # Compute imitation error for transition window
        imitation_error_accumulated = 0

        for i in range(self.imitation_window_size - 1):
            # Compute transition
            x_t_d = dynamical_system.transition()['desired state']

            # Compute and accumulate error
            imitation_error_accumulated += self.mse_loss(x_t_d[:, :self.dim_manifold], state_sample[:, :self.dim_manifold, i + 1].cuda())

        imitation_error_accumulated = imitation_error_accumulated / (self.imitation_window_size - 1)

        return imitation_error_accumulated * self.imitation_loss_weight

    def triplet_stability_loss(self, state_sample, primitive_type_sample):
        """
        Triplet stability loss
        """
        # Create dynamical systems
        dynamical_system_task = self.init_dynamical_system(initial_states=state_sample,
                                                           primitive_type=primitive_type_sample)

        # Compute cost over trajectory
        triplet_stability_cost = 0

        for i in range(self.generalization_window_size):
            # Do transition
            y_t_task_prev = dynamical_system_task.y_t
            y_t_task = dynamical_system_task.transition()['latent state']

            if i > 0:  # we need at least one iteration to have a previous point to push the current one away from
                # Transition matching cost
                y_goal = self.model.get_goals_latent_space_batch(primitive_type_sample)
                triplet_stability_cost += self.triplet_loss(y_goal, y_t_task, y_t_task_prev)

        triplet_stability_cost = triplet_stability_cost / (self.generalization_window_size - 1)

        return triplet_stability_cost * self.stabilization_loss_weight

    def boundary_constrain_loss(self, state_sample, primitive_type_sample):
        # Force states to start at the boundary
        selected_axis = torch.randint(low=0, high=self.dim_state, size=[self.batch_size])
        selected_limit = torch.randint(low=0, high=2, size=[self.batch_size])
        limit_options = torch.FloatTensor([-1, 1])
        limits = limit_options[selected_limit]
        replaced_samples = torch.arange(start=0, end=self.batch_size)
        state_sample[replaced_samples, selected_axis] = limits.cuda()

        # Create dynamical systems
        self.params_dynamical_system['saturate transition'] = False
        dynamical_system = self.init_dynamical_system(initial_states=state_sample,
                                                      primitive_type=primitive_type_sample)
        self.params_dynamical_system['saturate transition'] = True

        # Do one transition at the boundary and get velocity
        transition_info = dynamical_system.transition()
        x_t_d = transition_info['desired state']
        dx_t_d = transition_info['desired velocity']

        # Iterate through every dimension
        epsilon = 5e-2
        loss = 0
        if self.space_type == 'sphere':
            return 0
        elif self.space_type == 'euclidean_sphere':
            states_boundary = 3
        else:
            states_boundary = self.dim_ambient_space

        for i in range(states_boundary):
            distance_upper = torch.abs(x_t_d[:, i] - 1)
            distance_lower = torch.abs(x_t_d[:, i] + 1)

            # Get velocities for points in the boundary
            dx_axis_upper = dx_t_d[distance_upper < epsilon]
            dx_axis_lower = dx_t_d[distance_lower < epsilon]

            # Compute normal vectors for lower and upper limits
            normal_upper = torch.zeros(dx_axis_upper.shape).cuda()
            normal_upper[:, i] = 1
            normal_lower = torch.zeros(dx_axis_lower.shape).cuda()
            normal_lower[:, i] = -1

            # Compute dot product between boundary velocities and normal vectors
            dot_product_upper = torch.bmm(dx_axis_upper.view(-1, 1, self.dim_ambient_space),
                                          normal_upper.view(-1, self.dim_ambient_space, 1)).reshape(-1)

            dot_product_lower = torch.bmm(dx_axis_lower.view(-1, 1, self.dim_ambient_space),
                                          normal_lower.view(-1, self.dim_ambient_space, 1)).reshape(-1)

            # Concat with zero in case no points sampled in boundaries, to avoid nans
            dot_product_upper = torch.cat([dot_product_upper, torch.zeros(1).cuda()])
            dot_product_lower = torch.cat([dot_product_lower, torch.zeros(1).cuda()])

            # Compute losses
            loss += F.relu(dot_product_upper).mean()
            loss += F.relu(dot_product_lower).mean()

        loss = loss / (2 * self.dim_ambient_space)

        return loss * self.boundary_loss_weight

    def demo_sample(self):
        """
        Samples a batch of windows from the demonstrations
        """

        # Select demonstrations randomly
        selected_demos = np.random.choice(range(self.n_demonstrations), self.batch_size)

        # Get random points inside trajectories
        i_samples = []
        for i in range(self.n_demonstrations):
            selected_demo_batch_size = sum(selected_demos == i)
            demonstration_length = self.demonstrations_train.shape[1]
            i_samples = i_samples + list(np.random.randint(0, demonstration_length, selected_demo_batch_size, dtype=int))

        # Get sampled positions from training data
        position_sample = self.demonstrations_train[selected_demos, i_samples]
        position_sample = torch.FloatTensor(position_sample).cuda()

        # Create empty state
        state_sample = torch.empty([self.batch_size, self.dim_state, self.imitation_window_size]).cuda()

        # Fill first elements of the state with position
        state_sample[:, :self.dim_ambient_space, :] = position_sample[:, :, (self.dynamical_system_order - 1):]

        # Fill rest of the elements with velocities for second order systems
        if self.dynamical_system_order == 2:
            velocity = (position_sample[:, :, 1:] - position_sample[:, :, :-1]) / self.delta_t
            velocity_norm = normalize_state(velocity,
                                            x_min=self.min_vel.reshape(1, self.dim_ambient_space, 1),
                                            x_max=self.max_vel.reshape(1, self.dim_ambient_space, 1))
            state_sample[:, self.dim_ambient_space:, :] = velocity_norm

        # Finally, get primitive ids of sampled batch (necessary when multi-motion learning)
        primitive_type_sample = self.primitive_ids[selected_demos]
        primitive_type_sample = torch.FloatTensor(primitive_type_sample).cuda()

        return state_sample, primitive_type_sample

    def space_sample(self):
        """
        Samples a batch of windows from the state space
        """
        with torch.no_grad():
            # Sample state
            state_sample_gen = torch.Tensor(self.batch_size, self.dim_state).uniform_(-1, 1).cuda()

            # Choose sampling methods
            if not self.multi_motion:
                primitive_type_sample_gen = torch.randint(0, self.n_primitives, (self.batch_size,)).cuda()
            else:
                # If multi-motion learning also sample in interpolation space
                # sigma of the samples are in the demonstration spaces
                encodings = torch.eye(self.n_primitives).cuda()
                primitive_type_sample_gen_demo = encodings[torch.randint(0, self.n_primitives, (round(self.batch_size * self.interpolation_sigma),)).cuda()]

                # 1 - sigma  of the samples are in the interpolation space
                primitive_type_sample_gen_inter = torch.rand(round(self.batch_size * (1 - self.interpolation_sigma)), self.n_primitives).cuda()

                # Concatenate both samples
                primitive_type_sample_gen = torch.cat((primitive_type_sample_gen_demo, primitive_type_sample_gen_inter), dim=0)

        return state_sample_gen, primitive_type_sample_gen

    def compute_loss(self, state_sample_IL, primitive_type_sample_IL, state_sample_gen, primitive_type_sample_gen):
        """
        Computes total cost
        """
        loss_list = []  # list of losses
        losses_names = []

        # Learning from demonstrations outer loop
        if self.imitation_loss_weight != 0:
            imitation_cost = self.imitation_loss(state_sample_IL, primitive_type_sample_IL)
            loss_list.append(imitation_cost)
            losses_names.append('Imitation')

        # Triplet stability loss
        if self.stabilization_loss_weight != 0:
            triplet_stability_cost = self.triplet_stability_loss(state_sample_gen, primitive_type_sample_gen)
            loss_list.append(triplet_stability_cost)
            losses_names.append('Stability')

        # Boundary loss
        if self.boundary_loss_weight != 0:
            state_sample_gen_bound = torch.clone(state_sample_gen)
            boundary_cost = self.boundary_constrain_loss(state_sample_gen_bound, primitive_type_sample_gen)
            loss_list.append(boundary_cost)
            losses_names.append('Boundary')

        # Sum losses
        loss = 0
        for i in range(len(loss_list)):
            loss += loss_list[i]

        return loss, loss_list, losses_names

    def update_model(self, loss):
        """
        Updates Neural Network with computed cost
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update goal in latent space
        self.model.update_goals_latent_space(self.goals_tensor)

    def train_step(self):
        """
        Samples data and trains Neural Network
        """
        # Sample from space
        state_sample_gen, primitive_type_sample_gen = self.space_sample()

        # Sample from trajectory
        state_sample_IL, primitive_type_sample_IL = self.demo_sample()

        # Get loss from PUMA
        loss, loss_list, losses_names = self.compute_loss(state_sample_IL,
                                                          primitive_type_sample_IL,
                                                          state_sample_gen,
                                                          primitive_type_sample_gen)

        # Update model
        self.update_model(loss)

        return loss, loss_list, losses_names