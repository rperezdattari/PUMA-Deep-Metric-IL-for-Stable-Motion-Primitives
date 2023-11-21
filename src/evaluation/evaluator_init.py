from evaluation.evaluate_2d_o1 import Evaluate2DO1
from evaluation.evaluate_2d_o2 import Evaluate2DO2
from evaluation.evaluate_3d import Evaluate3D
from evaluation.evaluate_4d_S3 import Evaluate4DS3
from evaluation.evaluate_7d_R3S3 import Evaluate7DR3S3
from evaluation.evaluate_nd import EvaluateND


def evaluator_init(learner, data, params, verbose=True):
    """
    Selects and initializes evaluation class
    """
    cartesian_space_dim = params.manifold_dimensions
    if params.space == 'sphere':
        cartesian_space_dim += 1

    if cartesian_space_dim == 2 and params.dynamical_system_order == 1:
        return Evaluate2DO1(learner, data, params, verbose)
    elif cartesian_space_dim == 2 and params.dynamical_system_order == 2:
        return Evaluate2DO2(learner, data, params, verbose)
    elif cartesian_space_dim == 3 and params.space == 'sphere':
        return Evaluate3D(learner, data, params, verbose)
    elif cartesian_space_dim == 4 and params.space == 'sphere':
        return Evaluate4DS3(learner, data, params, verbose)
    elif params.space == 'euclidean_sphere':
        return Evaluate7DR3S3(learner, data, params, verbose)
    else:
        return EvaluateND(learner, data, params, verbose)
