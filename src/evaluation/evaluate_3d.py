from evaluation.evaluate import Evaluate
import pickle
import plotly.graph_objects as go
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
        self.plot_DS_plotly(sim_results['visited states demos'], sim_results['visited states grid'], save_path)
        return True

    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        # Not implemented
        return False

    def plot_DS_plotly(self, visited_states, visited_states_grid, save_path):
        """
        Plots demonstrations and simulated trajectories when starting from demos initial states
        """
        plot_data = []
        # Plot random trajectories
        for i in range(visited_states_grid.shape[1]):
            # Denorm states
            denorm_visited_states = denormalize_state(visited_states_grid, self.x_min, self.x_max)

            # Plot network executions
            marker_data_executed = go.Scatter3d(
                x=denorm_visited_states[:, i, 0],
                y=denorm_visited_states[:, i, 1],
                z=denorm_visited_states[:, i, 2],
                marker=dict(size=0.01, color='blue'),
                line=dict(color='blue', width=10),
                opacity=0.1,
                # mode='markers'
            )
            plot_data.append(marker_data_executed)

        for i in range(self.n_trajectories):
            # Plot datasets
            marker_data_demos = go.Scatter3d(
                x=self.demonstrations_eval[i][0],
                y=self.demonstrations_eval[i][1],
                z=self.demonstrations_eval[i][2],
                marker=go.scatter3d.Marker(size=3, color='red'),
                opacity=0.5,
                mode='markers',
                name='demonstration %i' % i,
            )
            plot_data.append(marker_data_demos)

            # Plot network executions
            denorm_visited_states = denormalize_state(visited_states, self.x_min, self.x_max)
            marker_data_executed = go.Scatter3d(
                x=denorm_visited_states[:, i, 0],
                y=denorm_visited_states[:, i, 1],
                z=denorm_visited_states[:, i, 2],
                marker=go.scatter3d.Marker(size=3, color='green'),
                opacity=0.5,
                mode='markers',
                name='CONDOR %i' % i,
            )
            plot_data.append(marker_data_executed)

        layout = go.Layout(autosize=True,
                           scene=dict(
                               xaxis_title='x (m)',
                               yaxis_title='y (m)',
                               zaxis_title='z (m)'),
                           margin=dict(l=65, r=50, b=65, t=90),
                           showlegend=False,
                           font=dict(family='Time New Roman', size=15))
        fig = go.Figure(data=plot_data, layout=layout)

        plot_data = {'3D_plot': fig}

        # Save
        print('Saving image data to %s...' % save_path)
        pickle.dump(plot_data, open(save_path, 'wb'))

        if self.show_plotly:
            fig.show()

        return True
