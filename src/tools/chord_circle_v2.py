import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,
                     "font.family": "Times New Roman"})
# Define parameters
radius = 1
num_points = 30
theta = np.linspace(np.pi, 0, num_points)  # theta values for circumference points
theta += np.pi / 2

# Calculate coordinates
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Calculate distances from north pole (0, radius) to each point
distances = 2*np.sin(theta/2) #np.sqrt((x - 0)**2 + (y - radius)**2)

# Order distances starting from the North Pole and going around
# north_pole_index = np.argmin(distances)
# distances = np.roll(distances, -north_pole_index)

# Plot 1: Circumference with connecting lines
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# Add red point at the North Pole
plt.plot(0, radius, 'ro', markersize=15, label='goal', zorder=100, markeredgecolor='black', markeredgewidth=3)

# Draw lines from north pole to points
div = 4
i = 0
for _ in range(30//div):
    if i == 0:
        plt.plot([0, x[i]], [radius, y[i]], 'C0--', linewidth=4, label='$d_{\\mathrm{chord}}$')
    else:
        plt.plot([0, x[i]], [radius, y[i]], 'C0--', linewidth=4)

    i += div

plt.plot(x[18:], y[18:], 'C1--', label='$d_{\\mathrm{g.c.}}$', linewidth=5, zorder=3)

# Draw the circumference
plt.plot(x, y, label='trajectory', linewidth=10, color='black')

plt.axis('equal')
# Create Line2D instances for the legend
line1 = mlines.Line2D([], [], color='red', marker='o', markersize=15, linestyle='None',
                       markeredgecolor='black', markeredgewidth=3, label='goal')
line2 = mlines.Line2D([], [], color='C0', linewidth=4, label='$d_{\\mathrm{chord}}$')
line3 = mlines.Line2D([], [], color='C1', linewidth=5, label='$d_{\\mathrm{g.c.}}$')
line4 = mlines.Line2D([], [], color='black', linewidth=10, label='trajectory')

# Add the legend manually
plt.legend(handles=[line1, line2, line3, line4], fontsize=25, loc='lower left')
plt.axis('off')  # Removes gridlines, axis labels, and ticks

# Plot 2: Length of lines as function of distance from north pole
plt.subplot(1, 2, 2)
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
plt.plot(theta, spherical_distance, linewidth=6, color='C1')
plt.plot(theta, chordal_distances, linewidth=6, color='C0')
plt.title('Distance to goal', fontsize=40, fontweight="bold")
plt.xlabel('angle [rad]', fontsize=38, fontweight="bold")
plt.ylabel('$d$ [dist./radius]', fontsize=38, fontweight="bold")
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
