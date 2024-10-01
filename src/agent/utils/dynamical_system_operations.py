import torch


"""""""""""""""""""""""""""""""""""""""""""""""
       Dynamical System useful functions
"""""""""""""""""""""""""""""""""""""""""""""""

def normalize_state(state, x_min, x_max):
    """
    Normalize state
    """
    state = (((state - x_min) / (x_max - x_min)) - 0.5) * 2

    return state


def get_derivative_normalized_state(dx, x_min, x_max):
    """
    Computes derivative of normalized state from derivative of unnormalized state
    """
    dx_x_norm = (2 * dx) / (x_max - x_min)
    return dx_x_norm


def denormalize_state(state, x_min, x_max):
    """
    Denormalize state
    """
    state = ((state / 2) + 0.5) * (x_max - x_min) + x_min
    return state


def denormalize_derivative(dx_t, max_state_derivative):
    """
    Denormalize state derivative
    """
    dx_t_denormalized = dx_t *  max_state_derivative
    return dx_t_denormalized


def normalize_derivative(dx_t, max_state_derivative):
    """
    Normalize state derivative
    """
    dx_t_normalized = dx_t / max_state_derivative
    return dx_t_normalized


def euler_integration(x_t, dx_t, delta_t):
    """
    Euler integration and get x_{t+1}
    """
    x_t_next = x_t + dx_t * delta_t
    return x_t_next


def euler_diff(x_t_next, x_t, delta_t):
    """
    Euler differentiation and get dx_t
    """
    dx = (x_t_next - x_t) / delta_t
    return dx


def batch_dot_product(x, y):
    """
    Compute dot product for each pair of vectors in the batch.
    """
    return torch.sum(x * y, dim=-1, keepdim=True)


def map_points_to_sphere(x_t, radius=1):
    """
    Projects points to sphere surface
    """
    norm = torch.linalg.norm(x_t, dim=1).reshape(-1, 1)
    x_t = (radius / norm) * x_t
    return x_t


def project_to_manifold(x_init, space_type):
    """
    Project state to dynamical systems' manifold
    """
    if space_type == 'sphere':
        x_init_projected = map_points_to_sphere(x_init)
    elif space_type == 'euclidean_sphere':
        # Hardcode dimensions unique to euclidean_sphere
        cartesian_position_dim = 3
        total_position_dim = 7

        # Only project part of the space that is spherical
        projected_points = map_points_to_sphere(x_init[:, cartesian_position_dim:total_position_dim])
        x_init_projected = torch.cat([x_init[:, :cartesian_position_dim], projected_points, x_init[:, total_position_dim:]], dim=1)  # pytorch doesn't like inplace operations
    else:
        x_init_projected = x_init

    return x_init_projected


def project_point_onto_plane(p, n, r=0):
    """
    Projects a point p onto a plane defined by point r and normal vector n in R^{n+1}
    """

    # Compute the vector from r to p
    v = p - r

    # Calculate the dot product of v and n
    dot_v_n = batch_dot_product(v, n)

    # Calculate the dot product of n with itself
    dot_n_n = batch_dot_product(n, n)

    # Calculate the projection of v onto n
    proj_v_onto_n = (dot_v_n / dot_n_n) * n

    # Calculate the projection of p onto the plane
    p_plane = p - proj_v_onto_n

    return p_plane


def exp_map_sphere(p, v):
    """
    Computes the sphere's exponential map at a given point p with with coordinate v in the tangent plane's reference frame
    """
    v_norm = v.norm(dim=1, keepdim=True)
    mapped_point = torch.cos(v_norm) * p + torch.sin(v_norm) * (v / v_norm)
    return mapped_point


def euler_non_euclidean_1st_order(x_t, vel_t_d_free, delta_t):
    """
    Integrates and projects point onto spherical manifold
    """
    # Project velocity to tangent space
    vel_t_d_tangent = project_point_onto_plane(vel_t_d_free, x_t)

    # Integrate
    x_t_d_tangent = euler_integration(x_t*0, vel_t_d_tangent, delta_t)  # x_t multiplied by zero because tangent plane is w.r.t. this value

    # Compute exponential map
    x_t_d = exp_map_sphere(x_t, x_t_d_tangent)
    return x_t_d, vel_t_d_tangent


def euler_non_euclidean_2nd_order(x_t, vel_t_free, acc_t_d_free, delta_t):
    """
    Integrates (2nd order) and projects point onto spherical manifold
    """
    # Project velocity and acceleration to tangent plane
    vel_t_tangent = project_point_onto_plane(vel_t_free, x_t)
    acc_t_d_tangent = project_point_onto_plane(acc_t_d_free, x_t)

    # Integrate acc in tangent space to velocity in tangent space
    vel_t_d_tangent = euler_integration(vel_t_tangent, acc_t_d_tangent, delta_t)

    # Compute exponential map
    x_t_d_tangent = euler_integration(x_t*0, vel_t_d_tangent, delta_t)  # x_t multiplied by zero because tangent plane is w.r.t. this value
    x_t_d = exp_map_sphere(x_t, x_t_d_tangent)
    return x_t_d, vel_t_d_tangent, acc_t_d_tangent
