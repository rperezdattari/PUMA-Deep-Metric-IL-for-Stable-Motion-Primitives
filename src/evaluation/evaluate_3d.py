from evaluation.evaluate import Evaluate
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from agent.utils.dynamical_system_operations import denormalize_state


class Evaluate3D(Evaluate):
    """
    Class for evaluating three-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.show_plotly = params.show_plotly

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration):
        """
        Computes qualitative results
        """
        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pickle'
        # Get velocities
        vel = self.get_vector_field(sim_results['initial states grid'], primitive_id)

        self.plot_DS_plotly(sim_results['visited states demos'], sim_results['visited states grid'], sim_results['grid'], vel, save_path)
        return True

    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        # Not implemented
        return False

    def plot_DS_plotly(self, visited_states, visited_states_grid, grid, vel, save_path):
        """
        Plots demonstrations and simulated trajectories when starting from demos initial states
        """
        plot_data = []
        # # Plot random trajectories
        # for i in range(visited_states_grid.shape[1]):
        #     # Denorm states
        #     denorm_visited_states = denormalize_state(visited_states_grid, self.x_min, self.x_max)
        #
        #     # Plot network executions
        #     marker_data_executed = go.Scatter3d(
        #         x=denorm_visited_states[:, i, 0],
        #         y=denorm_visited_states[:, i, 1],
        #         z=denorm_visited_states[:, i, 2],
        #         marker=dict(size=0.01, color='blue'),
        #         line=dict(color='blue', width=10),
        #         opacity=0.1,
        #         # mode='markers'
        #     )
        #     plot_data.append(marker_data_executed)

        for i in range(self.n_trajectories):
            # Plot datasets
            marker_data_demos = go.Scatter3d(
                x=self.demonstrations_eval[i][0],
                y=self.demonstrations_eval[i][1],
                z=self.demonstrations_eval[i][2],
                #marker=go.scatter3d.Marker(size=3, color='white'),
                marker=dict(size=1, color='white'),
                line=dict(color='white', width=10),
                opacity=0.9,
                #mode='markers',
                name='demonstration %i' % i,
            )
            plot_data.append(marker_data_demos)

            # Plot network executions
            denorm_visited_states = denormalize_state(visited_states, self.x_min, self.x_max)
            marker_data_executed = go.Scatter3d(
                x=denorm_visited_states[:, i, 0]*1.01,  # we lift a bit network executions for plotting purposes
                y=denorm_visited_states[:, i, 1]*1.01,
                z=denorm_visited_states[:, i, 2]*1.01,
                #marker=go.scatter3d.Marker(size=3, color='red'),
                marker=dict(size=1, color='red'),
                line=dict(color='red', width=10),
                opacity=1.0,
                #mode='markers',
                name='CONDOR %i' % i,
            )
            plot_data.append(marker_data_executed)

        # Create sphere
        norm_vel = np.linalg.norm(vel, axis=2)
        #colors_sphere = np.zeros(shape=norm_vel.shape)
        sphere = go.Surface(x=grid[0], y=grid[1], z=grid[2], opacity=1.0, surfacecolor=norm_vel, colorscale='Viridis')

        # Create the arrow traces. We remove the cone in the goal and add a very small cone to make them black
        normalized_vel = vel / norm_vel.reshape(self.density, self.density, 1)
        arrows = go.Cone(x=np.append(grid[0].reshape(-1)[1:], 0),
                         y=np.append(grid[1].reshape(-1)[1:], 0),
                         z=np.append(grid[2].reshape(-1)[1:], 0),
                         u=np.append(normalized_vel[:, :, 0].reshape(-1)[1:], 1e-5),
                         v=np.append(normalized_vel[:, :, 1].reshape(-1)[1:], 1e-5),
                         w=np.append(normalized_vel[:, :, 2].reshape(-1)[1:], 1e-5),
                         sizemode='absolute', sizeref=0.5, showscale=False, colorscale='Greys', opacity=0.6)

        # Layout info
        layout = go.Layout(autosize=True,
                           # scene=dict(
                           #     xaxis_title='x (m)',
                           #     yaxis_title='y (m)',
                           #     zaxis_title='z (m)'),
                           scene=dict(xaxis=dict(visible=False),
                                      yaxis=dict(visible=False),
                                      zaxis=dict(visible=False)),
                           margin=dict(l=65, r=50, b=65, t=90),
                           showlegend=False,
                           font=dict(family='Time New Roman', size=15))
        # plot data
        plot_data.append(sphere)
        plot_data.append(arrows)
        fig = go.Figure(data=plot_data, layout=layout)
        plot_data = {'3D_plot': fig}

        # Save
        print('Saving image data to %s...' % save_path)
        pickle.dump(plot_data, open(save_path, 'wb'))
        pio.write_image(fig, save_path + '.pdf', width=1300, height=1300)

        if self.show_plotly:
            fig.show()

        return True
