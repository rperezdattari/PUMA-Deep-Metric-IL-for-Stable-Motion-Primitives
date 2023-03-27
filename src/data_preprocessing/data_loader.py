from datasets.dataset_keys import LASA, LASA_S2, LAIR, optitrack, interpolation, joint_space, lieflows, ABB, ABB_R3S3
from scipy.spatial.transform import Rotation
from spatialmath import SO3, UnitQuaternion
import os
import pickle
import numpy as np
import scipy.io as sio
import json


def load_demonstrations(dataset_name, selected_primitives_ids):
    """
    Loads demonstrations
    """
    # Get names of primitives in dataset
    dataset_primitives_names = get_dataset_primitives_names(dataset_name)

    # Get names of selected primitives for training
    primitives_names, primitives_save_name = select_primitives(dataset_primitives_names, selected_primitives_ids)

    # Get number of selected primitives
    n_primitives = len(primitives_names)

    # Get loading path
    dataset_path = 'datasets/' + dataset_name + '/'

    # Get data loader
    data_loader = get_data_loader(dataset_name)

    # Load
    demonstrations, demonstrations_primitive_id, delta_t_eval = data_loader(dataset_path, primitives_names)

    # Out dictionary
    loaded_info = {'demonstrations raw': demonstrations,
                   'demonstrations primitive id': demonstrations_primitive_id,
                   'n primitives': n_primitives,
                   'delta t eval': delta_t_eval}
    return loaded_info


def get_dataset_primitives_names(dataset_name):
    """
    Chooses primitives keys
    """
    if dataset_name == 'LASA':
        dataset_primitives_names = LASA
    elif dataset_name == 'LASA_S2':
        dataset_primitives_names = LASA_S2
    elif dataset_name == 'lieflows_robot':
        dataset_primitives_names = lieflows
    elif dataset_name == 'LAIR':
        dataset_primitives_names = LAIR
    elif dataset_name == 'optitrack':
        dataset_primitives_names = optitrack
    elif dataset_name == 'interpolation':
        dataset_primitives_names = interpolation
    elif dataset_name == 'joint_space':
        dataset_primitives_names = joint_space
    elif dataset_name == 'ABB':
        dataset_primitives_names = ABB
    elif dataset_name == 'ABB_R3S3':
        dataset_primitives_names = ABB_R3S3
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return dataset_primitives_names


def select_primitives(dataset, selected_primitives_ids):
    """
    Gets selected primitives
    """
    selected_primitives_names = []
    selected_primitives_save_name = ''
    selected_primitives_ids = list(map(int, selected_primitives_ids.split(',')))  # map from string to list
    for id in selected_primitives_ids:
        selected_primitives_names.append(dataset[id])
        selected_primitives_save_name += str(id) + '_'

    return selected_primitives_names, selected_primitives_save_name[:-1]


def get_data_loader(dataset_name):
    """
    Chooses data loader depending on the data type
    """
    if dataset_name == 'LASA':
        data_loader = load_LASA
    elif dataset_name == 'LAIR' or dataset_name == 'optitrack' or dataset_name == 'interpolation':
        data_loader = load_numpy_file
    elif dataset_name == 'joint_space':
        data_loader = load_from_dict
    elif dataset_name == 'LASA_S2':
        data_loader = load_LASA_S2
    elif dataset_name == 'lieflows_robot':
        data_loader = load_lieflows_S3
    elif dataset_name == 'ABB':
        data_loader = load_ABB
    elif dataset_name == 'ABB_R3S3':
        data_loader = load_ABB_S3R3
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return data_loader


def load_LASA(dataset_dir, demonstrations_names):
    """
    Load LASA matlab models
    """
    s_x, s_y, demos, primitive_id, dt = [], [], [], [], []
    for i in range(len(demonstrations_names)):
        mat_file = sio.loadmat(dataset_dir + demonstrations_names[i])
        data = mat_file['demos']

        for j in range(data.shape[1]):  # iterate through demonstrations
            s_x = data[0, j]['pos'][0, 0][0]
            s_y = data[0, j]['pos'][0, 0][1]
            s = [s_x, s_y]
            demos.append(s)
            dt.append(data[0, j]['dt'][0, 0][0, 0])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_numpy_file(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in numpy files
    """
    demos, primitive_id = [], []
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        for demo_primitive in demos_primitive:
            data = np.load(dataset_dir + demonstrations_names[i] + '/' + demo_primitive)
            if data.shape[0] == 1:
                # if extra dimension in demo, remove
                data = data[0]
            demos.append(data.T)
            primitive_id.append(i)

    dt = 1
    return demos, primitive_id, dt


def load_from_dict(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries
    """
    demos, primitive_id, dt = [], [], []

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            demos.append(data['q'].T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_ABB(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries for ABB
    """
    demos, primitive_id, dt = [], [], []

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            rot = Rotation.from_matrix(np.array(data['x_rot']))
            eul = rot.as_euler('xyz') * 30
            #plot_points_3d(eul)
            rot = Rotation.from_euler('xyz', eul)
            quat = rot.as_quat()
            demos.append(quat.T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_ABB_S3R3(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries for ABB
    """
    demos, primitive_id, dt = [], [], []

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            #rot = Rotation.from_matrix(np.array(data['x_rot']))
            #eul = rot.as_euler('xyz') * 30
            #plot_points_3d(eul)
            #rot = Rotation.from_euler('xyz', eul)
            #quat = rot.as_quat()
            prev_quat = None
            quats = []
            for numpy_rot_mat in data['x_rot']:
                # Get quatenion array from data
                quat = UnitQuaternion(SO3(numpy_rot_mat)).A

                # Check if quaternion flip
                if prev_quat is None:
                    prev_quat = quat

                dist_quats = np.linalg.norm(quat - prev_quat)

                if dist_quats > 0.5:
                    quat *= -1

                quats.append(quat)
                prev_quat = quat

            #quat = [UnitQuaternion(SO3(numpy_rot_mat)).A for numpy_rot_mat in data['x_rot']]
            demo = np.concatenate([np.array(data['x_pos']), np.array(quats)], axis=1).T
            demos.append(demo)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_LASA_S2(dataset_path, primitives_names):
    """
    Load LASA S2 models
    """
    demos, primitive_id = [], []
    for i in range(len(primitives_names)):
        path = dataset_path + primitives_names[i] + '.txt'
        # Read the data from the file
        with open(path) as f:
            data = f.read()

        # Reconstruct the data as a dictionary
        data = json.loads(data)

        # Iterate through demonstrations
        for j in range(len(data['xyz'])):
            s = np.array(data['xyz'][j]).T
            demos.append(s)
            primitive_id.append(i)

    dt = 1
    return demos, primitive_id, dt


def load_lieflows_S3(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in numpy file from lieflow
    """
    dt_value = 0.01
    demos, primitive_id, dt = [], [], []
    for i in range(len(demonstrations_names)):
        demos_primitive_names = os.listdir(dataset_dir + demonstrations_names[i])

        for demo_primitive_name in demos_primitive_names:
            data = np.load(dataset_dir + demonstrations_names[i] + '/' + demo_primitive_name, allow_pickle=True)
            for j in range(data.shape[0]):
                if j == 10:  # TODO: remove, just too many demos
                    break
                euler_angles = data[j].T[3:]
                quaternions = euler_to_quaternion_batch(euler_angles)
                demos.append(quaternions)
                primitive_id.append(i)
                dt_demo = np.zeros(data[j].shape[0]) + dt_value
                dt.append(dt_demo)
    return demos, primitive_id, dt


def euler_to_quaternion_batch(euler_angles):
    """
    Convert a batch of Euler angles to quaternions.

    Parameters:
        euler_angles (numpy.ndarray): Array of shape (3, N) containing the batch of Euler angles
                                       [roll, pitch, yaw] in radians. N is the batch size.

    Returns:
        numpy.ndarray: Array of shape (4, N) containing the batch of quaternions [w, x, y, z].

    By ChatGPT
    """
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Calculate quaternion elements
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    """
    Converts quaternions to Euler angles (in radians) in the ZYX convention.
    Assumes the quaternions are normalized.
    By ChatGPT
    """
    yaw = np.arctan2(2*(q[:, :, 1]*q[:, :, 2] + q[:, :, 0]*q[:, :, 3]), q[:, :, 0]**2 + q[:, :, 1]**2 - q[:, :, 2]**2 - q[:, :, 3]**2)
    pitch = np.arcsin(2*(q[:, :, 0]*q[:, :, 2] - q[:, :, 1]*q[:, :, 3]))
    roll = np.arctan2(2*(q[:, :, 0]*q[:, :, 1] + q[:, :, 2]*q[:, :, 3]), q[:, :, 0]**2 - q[:, :, 1]**2 - q[:, :, 2]**2 + q[:, :, 3]**2)
    return np.stack([yaw, pitch, roll], axis=-1)


import matplotlib.pyplot as plt
def plot_points_3d(points):
    """
    Plots a batch of 3D points using matplotlib.

    Parameters:
        points (numpy.ndarray): Array of shape (batch_size, 3) containing the batch of 3D points.

    Returns:
        None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
