from evaluation.evaluate_nd import EvaluateND
import pickle
import plotly.graph_objects as go
from agent.utils.dynamical_system_operations import denormalize_state
from scipy.spatial.transform import Rotation


class Evaluate7DR3S3(EvaluateND):
    """
    Class for evaluating three-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.show_plotly = params.show_plotly

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration, save_path):
        """
        Computes qualitative results
        """
        # Run eval nd
        super().compute_quali_eval(sim_results, attractor, primitive_id, iteration, save_path)

        # Create 3D plost TODO: uncomment eventually
        #save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pickle'
        #self.plot_DS_plotly_angle(sim_results['visited states demos'][:, :, 3:], sim_results['visited states grid'][:, :, 3:], save_path)
        #self.plot_DS_plotly_pos(sim_results['visited states demos'][:, :, :3], sim_results['visited states grid'][:, :, :3], save_path)
        return True

    def plot_DS_plotly_angle(self, visited_states, visited_states_grid, save_path):
        """
        Plots demonstrations and simulated trajectories when starting from demos initial states
        """
        plot_data = []

        # Denorm states
        denorm_visited_states_grid = denormalize_state(visited_states_grid, self.x_min[3:], self.x_max[3:])
        denorm_visited_states = denormalize_state(visited_states, self.x_min[3:], self.x_max[3:])

        # Map to euler angles
        rot = Rotation.from_quat(denorm_visited_states_grid.reshape(-1, 4))
        visited_states_grid_eul = rot.as_euler('xyz').reshape(-1, self.density**self.dim_manifold, 3)
        rot = Rotation.from_quat(denorm_visited_states.reshape(-1, 4))
        visited_states_eul = rot.as_euler('xyz').reshape(-1, self.n_trajectories, 3)
        demonstrations_eval_eul = []
        for i in range(self.n_trajectories):
            rot = Rotation.from_quat(self.demonstrations_eval[i][3:].T)
            eul = rot.as_euler('xyz')
            demonstrations_eval_eul.append(eul.T)

        # Plot random trajectories
        for i in range(visited_states_grid.shape[1]):

            # Plot network executions
            marker_data_executed = go.Scatter3d(
                x=visited_states_grid_eul[:, i, 0],
                y=visited_states_grid_eul[:, i, 1],
                z=visited_states_grid_eul[:, i, 2],
                marker=go.scatter3d.Marker(size=1, color='blue'),
                line=dict(color='blue', width=10),
                opacity=0.3,
                mode='markers'
            )
            plot_data.append(marker_data_executed)

        for i in range(self.n_trajectories):

            # Plot datasets
            marker_data_demos = go.Scatter3d(
                x=demonstrations_eval_eul[i][0, :],
                y=demonstrations_eval_eul[i][1, :],
                z=demonstrations_eval_eul[i][2, :],
                marker=go.scatter3d.Marker(size=3, color='red'),
                opacity=0.5,
                mode='markers',
                name='demonstration %i' % i,
            )
            plot_data.append(marker_data_demos)

            # Plot network executions
            marker_data_executed = go.Scatter3d(
                x=visited_states_eul[:, i, 0],
                y=visited_states_eul[:, i, 1],
                z=visited_states_eul[:, i, 2],
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

    def plot_DS_plotly_pos(self, visited_states, visited_states_grid, save_path):
        """
        Plots demonstrations and simulated trajectories when starting from demos initial states
        """
        plot_data = []

        # Denorm states
        denorm_visited_states_grid = denormalize_state(visited_states_grid, self.x_min[:3], self.x_max[:3])
        denorm_visited_states = denormalize_state(visited_states, self.x_min[:3], self.x_max[:3])

        # # Map to euler angles
        # rot = Rotation.from_quat(denorm_visited_states_grid.reshape(-1, 4))
        # visited_states_grid_eul = rot.as_euler('xyz').reshape(-1, self.density**self.dim_manifold, 3)
        # rot = Rotation.from_quat(denorm_visited_states.reshape(-1, 4))
        # visited_states_eul = rot.as_euler('xyz').reshape(-1, self.n_trajectories, 3)
        # demonstrations_eval_eul = []
        # for i in range(self.n_trajectories):
        #     rot = Rotation.from_quat(self.demonstrations_eval[i][3:].T)
        #     eul = rot.as_euler('xyz')
        #     demonstrations_eval_eul.append(eul.T)

        # Plot random trajectories
        for i in range(visited_states_grid.shape[1]):

            # Plot network executions
            marker_data_executed = go.Scatter3d(
                x=denorm_visited_states_grid[:, i, 0],
                y=denorm_visited_states_grid[:, i, 1],
                z=denorm_visited_states_grid[:, i, 2],
                marker=go.scatter3d.Marker(size=1, color='blue'),
                line=dict(color='blue', width=10),
                opacity=0.3,
                mode='markers'
            )
            plot_data.append(marker_data_executed)

        for i in range(self.n_trajectories):

            # Plot datasets
            marker_data_demos = go.Scatter3d(
                x=self.demonstrations_eval[i][0, :],
                y=self.demonstrations_eval[i][1, :],
                z=self.demonstrations_eval[i][2, :],
                marker=go.scatter3d.Marker(size=3, color='red'),
                opacity=0.5,
                mode='markers',
                name='demonstration %i' % i,
            )
            plot_data.append(marker_data_demos)

            # Plot network executions
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
