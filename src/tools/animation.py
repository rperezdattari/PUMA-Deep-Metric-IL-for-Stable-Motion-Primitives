"""
Authors:
    Lekan Molu <patlekno@icloud.com>
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
    Initial version of this code extracted from:
    https://github.com/robotsorcerer/LyapunovLearner/blob/master/scripts/visualization/traj_plotter.py
"""

import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


class PointManager:
    def __init__(self, max_size, D):
        self.max_size = max_size
        self.D = D
        self._xi = np.empty((0, D))       # Initialize empty arrays
        self._xi_dot = np.empty((0, D))   # Initialize empty arrays

    def append_point(self, xi):
        # Append the new point to _xi
        self._xi = np.append(self._xi, xi[0].reshape(1, self.D), axis=0)
        self._xi_dot = np.append(self._xi_dot, xi[1].reshape(1, self.D), axis=0)

        # Check if the array exceeds max size, if so remove the oldest points
        if len(self._xi) > self.max_size:
            self._xi = self._xi[1:]  # Remove the first row (oldest point)
            self._xi_dot = self._xi_dot[1:]  # Remove the first row (oldest point)


class TrajectoryPlotter():
    def __init__(self, fig, fontdict=None, pause_time=1e-3, labels=None, x0=None, goal=None):
        """
        Class TrajectoryPlotter:
        This class expects to be constantly given values to plot in realtime.
        """
        self._fig = fig
        self.save = False
        plt.ion()
        self._ax = plt.subplot()

        self._labels = labels
        self._init = False
        self.Xinit = x0
        self._fontdict  = fontdict
        self._labelsize = 16
        self.goal = goal
        self.pause_time = pause_time
        self.i = 0

        plt.rcParams['toolbar'] = 'None'
        for key in plt.rcParams:
            if key.startswith('keymap.'):
                plt.rcParams[key] = ''

        if np.any(self.Xinit):
            self.init(x0.shape[-1])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def init(self, data_len):
        """
        Initialize plots based off the length of the data array.
        """
        max_size = 100  # Maximum number of points
        D = self.Xinit.shape[-1]  # Number of dimensions
        self.manager = PointManager(max_size, D)
        
        self._data_len = data_len

        # Plot it
        self._plots = [np.nan for _ in range(self.Xinit.shape[-1])]
        for i in range(D):
            self._plots[i] = [self._ax.plot(self.Xinit[0, i], self.Xinit[1, i], 'o', color='C'+str(i), markersize=0)[0]]
            self._plots[i] += [self._ax.plot([], [], color='C'+str(i), linewidth=2.5)[0]]

        # Show attractor
        self.targ, = self._ax.plot(self.goal[0], self.goal[1], 'g*', markersize=15, linewidth=3, label='Target')
        self._ax.set_xlabel('$x$', fontdict=self._fontdict)
        self._ax.set_ylabel('$y$', fontdict=self._fontdict)

        self._ax.xaxis.set_tick_params(labelsize=self._labelsize)
        self._ax.yaxis.set_tick_params(labelsize=self._labelsize)

        self._ax.grid('on')
        self._ax.legend(loc='best')
        self._ax.set_title('Dynamical System', fontdict=self._fontdict)

        self._init = True

    def update(self, xi):
        """
        Update the plots with new data x. Assumes x is a one-dimensional array.
        """
        D = xi.shape[-1]  # number of conditions
        if not self._init:
            self.init(xi.shape[0])

        assert xi.shape[1] == self._data_len, f'xi of shape {xi.shape}has to be of shape {self._data_len}'

        self.manager.append_point([xi[0], xi[1]])

        for idx, traj_plotter in enumerate(self._plots):
            traj_plotter[-1].set_data(self.manager._xi[:, idx], self.manager._xi_dot[:, idx])

        self.draw()
        time.sleep(self.pause_time)

    def draw(self):
        for plots in self._plots:
            for each_plot in plots:
                self._ax.draw_artist(each_plot)

        if self.save:
            self._fig.savefig('frames/%i.png' % self.i)
            self.i += 1

        self._fig.canvas.flush_events()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.show()
    x = np.array([[0, 0]]).T
    traj_plotter = TrajectoryPlotter(fig, x0=x, goal=[0, 0])
    traj_plotter.init(data_len=1)
    for i in range(1000):
        x = np.array([[np.sin(i*0.007), np.sin(i*0.014)]]).T
        traj_plotter.update(x)
