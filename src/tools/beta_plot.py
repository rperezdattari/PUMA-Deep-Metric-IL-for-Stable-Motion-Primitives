import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,
                     "font.family": "Times New Roman",
                     "font.size": 20},)
# Constant value
delta = 1
max_t = 6
# Time values
t = np.linspace(0, max_t, 5000)


# Function
def d(t):
    return ((np.sin(2 * np.pi * t / delta)) ** 2 + 0.2 * (np.cos(2 * np.pi * t / delta)) ** 2) * np.exp(-0.75 * t)


y = d(t)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='$d(t)$', linewidth=4, color='black')
plt.title('Surrogate Conditions on $d(t)$')
plt.xlabel('time $(t)$')
plt.ylabel('$d(t)$')
plt.grid(True)

# Calculating the upper bound
y_upper = np.empty(len(t))
arg_max_b = []
for i, t_i in enumerate(t):
    if t_i > (max_t - delta):
        b_range = t[i:]
    else:
        b_range = t[(t >= t_i) & (t <= t_i + delta)]

    max_b = np.max(d(b_range))
    arg_max_b.append(np.argmax(d(b_range)))

    y_upper[i] = max_b  # d(t_i) if d(t_i) > max_b else max_b

plt.plot(t, y_upper, label='$\\beta_{t}(t)$', linewidth=4, color='C0')

# Initial conditions
initial_conditions = [0, 0.07, 0.1, 0.13, 0.17, 0.235]

# Colors for each initial condition scatter plot
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
# colors = ['green', 'green', 'green',' green', 'green', 'green']
i=0
for idx, init_cond in enumerate(initial_conditions):
    # Time samples every 0.5 seconds from initial condition to 10
    t_samples = np.arange(init_cond, max_t + 0.1, delta)

    # Function values at these times
    y_samples = d(t_samples)

    # Scatter plot for these time samples
    # plt.scatter(t_samples, y_samples, color=colors[idx], label=f'Start t={init_cond}', s=20)
    if i == 0:
        plt.scatter(t_samples, y_samples, color='C3', label='$d(n \cdot \Delta t + \\alpha)$', s=30, zorder=100)
    else:
        plt.scatter(t_samples, y_samples, color='C3', s=30, zorder=100)
    # Connect the dots with lines
    # plt.plot(t_samples, y_samples, color=colors[idx], linestyle='dashed', linewidth=1.0)
    plt.plot(t_samples, y_samples, color='C3', linestyle='dashed', linewidth=1.5)

    i+=1

plt.legend()
plt.grid(linestyle='--', linewidth=1)
width = 0
# plt.gca().spines['bottom'].set_linewidth(width)
# plt.gca().spines['top'].set_linewidth(width)
# plt.gca().spines['left'].set_linewidth(width)
# plt.gca().spines['right'].set_linewidth(width)
plt.xlim([0, 6])
plt.ylim([-0.01, 0.87])
plt.tight_layout()

plt.show()