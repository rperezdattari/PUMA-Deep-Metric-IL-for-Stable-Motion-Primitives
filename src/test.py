import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a function to generate the sphere
def gen_sphere(radius):
    # Generate points in spherical coordinates
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2*np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)

    # Convert to cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return x, y, z


# Define a function to compute the exponential map of a tangent vector
def exp_map(p, v, radius):
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return p

    u = v / norm_v
    return np.cos(norm_v*radius)*p + np.sin(norm_v*radius)*u


def parallel_transport(p, v_north, q):
    """
    Apply parallel transport of the tangent vector v_north from the tangent plane at the north pole of a sphere to the tangent plane at point q on the sphere.

    Args:
        p (np.ndarray): A 1D array of shape (3,) representing the point on the sphere where the tangent plane is defined.
        v_north (np.ndarray): A 1D array of shape (3,) representing the tangent vector defined in the tangent plane at the north pole of the sphere.
        q (np.ndarray): A 1D array of shape (3,) representing the point on the sphere to which the tangent vector is to be transported.

    Returns:
        np.ndarray: A 1D array of shape (3,) representing the transported tangent vector in the tangent plane at point q on the sphere.
    """
    # Define the north pole

    # Project the tangent vector onto the tangent plane at the north pole
    v_north_proj = v_north - np.dot(v_north, p) * p

    # Compute the angle between the north pole and point q
    theta = np.arccos(np.dot(p, q))

    # Compute the axis of rotation for the parallel transport
    axis = np.cross(p, q)
    if np.linalg.norm(axis) == 0:
        return v_north_proj
    axis = axis / np.linalg.norm(axis)

    # Compute the transported tangent vector
    v_transport = np.cos(theta) * v_north_proj + np.sin(theta) * np.cross(axis, v_north_proj)

    return v_transport


# Define the point on the sphere where the tangent vector is defined
p = np.array([0.5, 0.5, 0.7])
p = p / np.linalg.norm(p)

# Define the north pole
north_pole = np.array([0, 0, 1])

# Define the vector in the tangent plane at the north pole
v_north = np.array([1, 1, 0])
v_north = v_north - np.dot(v_north, north_pole)*north_pole
v_north = v_north / np.linalg.norm(v_north)

# Perform parallel transport of the tangent vector from the tangent plane at the north pole to the tangent plane at point p
v = parallel_transport(north_pole, v_north, p)

# Define the radius of the sphere
radius = 1

# Compute the exponential map of the tangent vector
q = exp_map(p, v, radius)

# Generate the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = gen_sphere(radius)
ax.plot_surface(x, y, z, alpha=0.5)

# Plot the original point p and the mapped point q
ax.scatter(p[0], p[1], p[2], color='b', label='p')
ax.scatter(q[0], q[1], q[2], color='g', label='exp_p(v)')

# Plot the tangent vector v
ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color='r', label='v')

# Plot the tangent vector v_north at the north pole
v_north_transport = parallel_transport(north_pole, p, v_north)
ax.quiver(north_pole[0], north_pole[1], north_pole[2], v_north[0], v_north[1], v_north[2], color='b', label='v')

#plt.legend()
plt.show()

import plotly.graph_objects as go
import numpy as np

# Define the equation for a sphere
a, b, c = 0, 0, 0  # center of the sphere
r = 1  # radius of the sphere
theta, phi = np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50)
THETA, PHI = np.meshgrid(theta, phi)
X = a + r * np.sin(THETA) * np.cos(PHI)
Y = b + r * np.sin(THETA) * np.sin(PHI)
Z = c + r * np.cos(THETA)
colors_sphere = np.zeros(shape=Z.shape)
sphere = go.Surface(x=X, y=Y, z=Z, opacity=1.0, surfacecolor=colors_sphere, colorscale='Viridis')

# Define the arrow positions and directions
arrow_x = [1, -1, 0, 0, 0, 0]
arrow_y = [0, 0, 1, -1, 0, 0]
arrow_z = [0, 0, 0, 0, 1,  1]
arrow_u = [0, 0, 1, -1, 0, 0.001]
arrow_v = [1, 0, 0, 0, 1, 0]
arrow_w = [0, 1, 0, 0, 0, 0]

# Create the arrow traces
arrows = go.Cone(x=arrow_x, y=arrow_y, z=arrow_z, u=arrow_u, v=arrow_v, w=arrow_w,
                 sizemode='absolute', sizeref=0.10, showscale=False, colorscale='Greys')

# Create the figure and add the surface and arrow traces
fig = go.Figure(data=[sphere, arrows])

# Set the layout of the plot
fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

# Show the plot
fig.show()