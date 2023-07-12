import numpy as np
import matplotlib.pyplot as plt

# Constants
delta = 1
max_t = 10
n_points = 10000

# Time values
t = np.linspace(0, max_t, n_points)
delta_t = max_t / n_points
phase = 0 # np.pi / 2

# Function to plot
def d(t):
    return ((np.sin((2 * np.pi * t / delta) + phase)) ** 2 + 0.2 * (np.cos((2 * np.pi * t / delta) + phase)) ** 2) * np.exp(-0.75 * t)

# Calculating upper bound
d_values = d(t)
upper_bound = np.empty_like(d_values)
A_list = []
B_list = []
t_interval_list = []
K = 3

for i, t_val in enumerate(t):
    # Find the t values for the interval t - delta * t and t + delta * t
    t_lower = max(t_val - delta, 0)
    t_upper = min(t_val + delta, max_t)

    # Find indices of t values within the interval
    indices_lower = np.where((t >= t_lower) & (t <= t_val))
    indices_upper = np.where((t <= t_upper) & (t >= t_val))

    # Find the maximum d(t) within the interval
    A = d_values[indices_lower].max()
    A_i = d_values[indices_lower].argmax()
    A_t = t[indices_lower[0][A_i]]
    # t_interval = t_val - t[indices_lower[0][A_i]]
    # if np.abs(t_interval - 1) < 0.001:
    #     t_interval = 0
    #
    # if t_val > 0.3 and t_interval != 0:
    #     t_interval -= 0.6
    B = d_values[indices_upper].max()
    B_i = d_values[indices_upper].argmax()
    B_t = t[indices_upper[0][B_i]]

    init_bound = d(0) + K * B_t  # 1.0
    if t_lower == 0:
        A = init_bound
        A_t = 0

    # Define D and m
    #D = A - B
    if i == 0:
        D = init_bound - B
    else:
        D = upper_bound[i-1] - B
    m = D / delta
    #delta_interval = B_t - A_t
    #m = D / delta_interval
    A_list.append(A)
    B_list.append(B)
    #t_interval_list.append(t_interval)

    # Define f(t)
    #upper_bound[i] = (A + np.abs(D)*0) - m * t_interval
    #upper_bound[i] = (A + np.abs(D)) - m * t_interval
    #upper_bound[i] = (A + B) / 2 - 0.1*m * t_val
    if i == 0:
        upper_bound[i] = init_bound
    else:
        upper_bound[i] = upper_bound[i-1] - m * delta_t

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(t, d_values, label='d(t)')
plt.plot(t, upper_bound, label='Upper Bound')
#plt.plot(t, np.array(A_list), label='A')
#plt.plot(t, np.array(B_list), label='B')
#plt.plot(t, np.array(t_interval_list), label='t interval')
plt.title("Function Plot with Upper Bound")
plt.xlabel("t")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.show()
