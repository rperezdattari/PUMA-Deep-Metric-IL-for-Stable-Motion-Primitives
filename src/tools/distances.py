import matplotlib.pyplot as plt
import numpy as np
plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,
                     "font.family": "Times New Roman"})
# Define parameters
radius = 1
num_points = 500
theta = np.linspace(-2*np.pi, 2*np.pi, num_points)  # theta values for circumference points


# chordal distance
theta_distances = np.linspace(-2*np.pi, 2*np.pi, num_points)
theta_distances[theta_distances < 0] *= -1
chordal_distances = 2*np.sin(theta_distances/2)

# spherical distance
spherical_distance = []
for theta_i in theta_distances:
    # if theta_i < 0:
    #     theta_i *= -1
    if theta_i < np.pi:
        spherical_distance.append(theta_i)
    else:
        spherical_distance.append(2*np.pi - theta_i)
spherical_distance = np.array(spherical_distance)
#spherical_distance[spherical_distance > np.pi] = 2*np.pi - spherical_distance[spherical_distance > np.pi]

# Plot 2: Length of lines as function of distance from north pole
plt.plot(theta, chordal_distances**2, linewidth=8, color='C0')
plt.plot(theta, spherical_distance**2, linewidth=8, color='C1')
plt.title('Distance to goal', fontsize=40, fontweight="bold")
plt.xlabel('angle [rad]', fontsize=33, fontweight="bold")
plt.ylabel('distance [dist./radius]', fontsize=33, fontweight="bold")
plt.xlim([-2*np.pi, 2*np.pi])
#plt.ylim([0, 2.1])
#plt.legend(fontsize=20)
plt.grid(linestyle='--', linewidth=1)

# Set the linewidth for the x-axis and y-axis
width = 1.5
plt.gca().spines['bottom'].set_linewidth(width)
plt.gca().spines['top'].set_linewidth(width)
plt.gca().spines['left'].set_linewidth(width)
plt.gca().spines['right'].set_linewidth(width)
plt.tick_params(axis='x', which='both', width=width, labelsize=20)  # x-axis
plt.tick_params(axis='y', which='both', width=width, labelsize=20)  # y-axis

plt.tight_layout()
plt.show()