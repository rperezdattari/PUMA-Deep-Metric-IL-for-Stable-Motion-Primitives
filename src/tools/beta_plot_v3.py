import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,
                     "font.family": "Times New Roman",
                     "font.size": 20},)
# Constant value
delta = 1
max_t = 8
n_samples = 5000
# Time values
t = np.linspace(0, max_t, n_samples)
time_step = max_t / n_samples
phase = 0  # np.pi / 2

# Function
def d(t):
    #return np.exp(t)
    return ((np.sin((2 * np.pi * t / delta) + phase)) ** 2 + 0.2 * (np.cos((2 * np.pi * t / delta) + phase)) ** 2) * np.exp(-0.75 * t)


y = d(t)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='$\delta$', linewidth=2.5, color='black')
plt.title('Surrogate conditions on $\delta$')
plt.xlabel('time $(t)$')
plt.ylabel('$||y_{\mathrm{g}} - y_{t}||$')
plt.grid(True)

# Calculating the upper bound
y_upper = np.empty(len(t))
K = 3.0
max_b_list = []

for i, t_i in enumerate(t):
    if t_i > (max_t - delta):
        b_range = t[i:]
    else:
        b_range = t[(t >= t_i) & (t <= t_i + delta)]

    max_b = np.max(d(b_range))
    max_b_list.append(max_b)
    argmax_b = np.argmax(d(b_range))
    t_max = b_range[argmax_b]
    tb_ti = t_max - t_i

    if i == 0:
        y_upper_init = d(0) + K * t_max
        y_upper[i] = y_upper_init
        #m = (y_upper_init - max_b) / delta
    else:
        m = (y_upper[i-1] - max_b) * 2# / (delta)  #(tb_ti + time_step)
        y_upper[i] = y_upper[i-1] - m * time_step

plt.plot(t, np.array(max_b_list), label='$\delta^{\\textrm{max}}_{t + \Delta t}$', linewidth=2, color='C1', linestyle='--')
plt.plot(t, y_upper, label='$\\beta$', linewidth=2, color='C0', linestyle='--')

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
        #plt.scatter(t_samples, y_samples, color='C3', label='$\\delta(y_{0}, \\bar{t} + n \cdot \Delta t)$', s=20, zorder=100)
        plt.scatter(t_samples, y_samples, color='C3', s=20, zorder=100)
    else:
        plt.scatter(t_samples, y_samples, color='C3', s=30, zorder=100)
    # Connect the dots with lines
    # plt.plot(t_samples, y_samples, color=colors[idx], linestyle='dashed', linewidth=1.0)
    plt.plot(t_samples, y_samples, color='C3', linestyle=':', linewidth=1.5)

    i+=1

plt.legend()
plt.grid(linestyle='--', linewidth=1)
width = 0
# plt.gca().spines['bottom'].set_linewidth(width)
# plt.gca().spines['top'].set_linewidth(width)
# plt.gca().spines['left'].set_linewidth(width)
# plt.gca().spines['right'].set_linewidth(width)
plt.xlim([0, max_t])
plt.ylim([-0.01, 0.93])
plt.tight_layout()

plt.show()