import matplotlib.pyplot as plt
from evaluation.evaluate import Evaluate
from matplotlib.patches import Ellipse
import numpy as np
from agent.utils.dynamical_system_operations import denormalize_state


class Evaluate2DO1(Evaluate):
    """
    Class for evaluating first-order two-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.show_plot = params.show_plot

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration, save_path):
        """
        Computes qualitative results
        """
        # Get velocities
        vel = self.get_vector_field(sim_results['initial states grid'], primitive_id)

        # Plot
        if save_path is None:
            save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pdf'

        self.plot_dynamical_system(sim_results, vel, attractor, primitive_id,
                                   title='Dynamical System',
                                   save_path=save_path)
        return True

    def plot_dynamical_system(self, sim_results, vel, attractor, primitive_i, title, save_path, obstacles=None):
        """
        Plots demonstrations, simulated trajectories, attractor and vector field
        """
        # Update plot params
        # plt.rcParams.update({'font.size': 14,
        #                      'figure.figsize': (8, 9)})
        plt.rcParams.update({'figure.figsize': (8, 9)})

        # Obstacle avoidance
        if obstacles is not None:
            obstacle_avoidance = True
            ax = plt.gca()
        else:
            obstacle_avoidance = False

        # Get denormalized states equally-spaced grid
        grid_x1 = denormalize_state(sim_results['grid'][0], self.x_min[0], self.x_max[0])
        grid_x2 = denormalize_state(sim_results['grid'][1], self.x_min[1], self.x_max[1])

        # Plot vector field
        plt.streamplot(grid_x1, grid_x2, vel[:, :, 0], vel[:, :, 1],
                       linewidth=1.0, density=1.5, arrowstyle='fancy',
                       arrowsize=1, color='black', cmap=plt.cm.Greys)

        # Plot contour with norm of the velocities
        norm_vel = np.linalg.norm(vel, axis=2)  # compute norm velocities
        CS = plt.contourf(grid_x1, grid_x2, norm_vel, cmap='viridis', levels=50)
        cbar = plt.colorbar(CS, location='bottom')
        cbar.ax.set_xlabel('speed (mm/s)')

        # Plot demonstrations (we need loop because they could have different lengths)
        for i in range(self.n_trajectories):
            if [self.primitive_ids == primitive_i][0][i]:
                plt.scatter(self.demonstrations_eval[i][0], self.demonstrations_eval[i][1], color='white', alpha=0.5)

        # Plot trajectories that start from the same points as the demonstrations
        plt.plot(denormalize_state(sim_results['visited states demos'][:, :self.n_trajectories, 0],
                                   self.x_min[0], self.x_max[0]),
                 denormalize_state(sim_results['visited states demos'][:, :self.n_trajectories, 1],
                                   self.x_min[1], self.x_max[1]),
                 linewidth=4, color='red', zorder=11)

        # Plot attractors
        plt.scatter(attractor[:, 0], attractor[:, 1], linewidth=4, color='blue', zorder=12)

        # Plot ellipse when obstacle avoidance
        if obstacle_avoidance:
            for i in range(len(obstacles['centers'])):
                ellipse = Ellipse(xy=(obstacles['centers'][i][0], obstacles['centers'][i][1]),
                                  width=obstacles['axes'][i][0] * 2,
                                  height=obstacles['axes'][i][1] * 2,
                                  edgecolor='mlearnera',
                                  fc='mlearnera',
                                  lw=2,
                                  zorder=10)
                ax.add_artist(ellipse)

        # Plot details/info
        plt.xlim([self.x_min[0], self.x_max[0]])
        plt.ylim([self.x_min[1], self.x_max[1]])
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(title)

        # Save
        print('Saving image to %s...' % save_path)
        plt.savefig(save_path, bbox_inches='tight')

        if self.show_plot:
            plt.show()

        # Close
        plt.clf()
        plt.close()

        return True