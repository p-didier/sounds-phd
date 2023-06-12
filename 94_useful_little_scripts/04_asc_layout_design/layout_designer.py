# Purpose of script:
# Acoustic scenario (ASC) designer script.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



BASE_EXPORT_PATH = '94_useful_little_scripts/04_asc_layout_design/layouts'
PARAMS = {
    'rd': [5, 5, 5],
    'Mk': [6, 6, 6, 6],  # number of sensors per node
    'node_arrangement': 'polygon',
    'node_spacing': 1.0,  # [m]
    'sensor_arrangement': 'within_sphere',  # 'linear', 'circular', 'within_sphere'
    'sensor_spacing': 0.2,  # [m]
    'N_targets': 1,
    'N_interferers': 8, # number of interferers
    'interferer_arrangement': 'circular',  # 'linear', 'circular', 'within_sphere'
    'interferer_spacing': 2,  # [m]
    'same_height': True,  # if True, all elements are at the same height
    #
    'export_file_name': f'{BASE_EXPORT_PATH}/asc1.yaml'
}

def main(p=PARAMS):
    """Main function (called by default when running script)."""
    room = build_room(p)

    # # Plot room
    # plot_room(room, p)

    # Export coordinates as YAML
    export_yaml(room, p)

    # # Try and re-import YAML
    # room2 = import_yaml(p['export_file_name'])

    # # Check that both rooms are the same
    # assert room == room2, 'Rooms are not the same!'


def import_yaml(file_name):
    """Function that imports room coordinates from a single YAML file."""
    # file_name: file name of YAML file
    # room: room coordinates

    # Import
    with open(file_name, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # Create dictionary
    room = {
        'rd': data['rd'],
        'Mk': data['Mk'],
        'sensor_positions': data['sensor_positions'],
        'target_positions': data['target_positions'],
        'interferer_positions': data['interferer_positions'],
    }

    return room


def export_yaml(room, p):
    """Function that exports room coordinates as a single YAML file."""
    # room: room coordinates
    # p: parameters
    
    # Check that folder exists
    if not os.path.exists(BASE_EXPORT_PATH):
        os.makedirs(BASE_EXPORT_PATH)

    # Export
    with open(p['export_file_name'], 'w') as f:
        f.write('rd: [%.2f, %.2f, %.2f]\n' % (p['rd'][0], p['rd'][1], p['rd'][2]))
        f.write('Mk: [')
        for Mk in p['Mk']:
            f.write('%d, ' % Mk)
        f.write(']\n')
        f.write('sensor_positions:\n')
        for sensor_position in room['sensor_positions']:
            f.write('  - [%.2f, %.2f, %.2f]\n' % (sensor_position[0], sensor_position[1], sensor_position[2]))
        f.write('target_positions:\n')
        for target_position in room['target_positions']:
            f.write('  - [%.2f, %.2f, %.2f]\n' % (target_position[0], target_position[1], target_position[2]))
        f.write('interferer_positions:\n')
        for interferer_position in room['interferer_positions']:
            f.write('  - [%.2f, %.2f, %.2f]\n' % (interferer_position[0], interferer_position[1], interferer_position[2]))


def plot_room(room, p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot sensors
    sensor_positions = np.array(room['sensor_positions'])
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2], c='b', marker='o')

    # Plot targets
    target_positions = np.array(room['target_positions'])
    ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], c='g', marker='d')

    # Plot interferers
    interferer_positions = np.array(room['interferer_positions'])
    ax.scatter(interferer_positions[:, 0], interferer_positions[:, 1], interferer_positions[:, 2], c='k', marker='+')

    # Plot room
    ax.set_xlim([0, p['rd'][0]])
    ax.set_ylim([0, p['rd'][1]])
    ax.set_zlim([0, p['rd'][2]])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # If same height, plot a plane and set view
    if p['same_height']:
        xx, yy = np.meshgrid(np.arange(0, p['rd'][0], 0.1), np.arange(0, p['rd'][1], 0.1))
        zz = np.ones(xx.shape) * p['rd'][2] / 2
        ax.plot_surface(xx, yy, zz, alpha=0.2)
        ax.view_init(elev=90, azim=0)

    # add legend
    ax.legend(['sensors', 'targets', 'interferers'])
    plt.show()


def build_room(p):

    # Place nodes
    if p['node_arrangement'] == 'polygon':
        node_positions = polygon(p['rd'], p['node_spacing'], len(p['Mk']))
    else:
        raise ValueError('Unknown node arrangement.')
    
    # Place sensors at each node
    sensor_positions = []
    for ii, node_position in enumerate(node_positions):
        if p['sensor_arrangement'] == 'linear':
            sensor_positions.append(linear(p['Mk'][ii], p['sensor_spacing'], node_position))
        elif p['sensor_arrangement'] == 'circular':
            sensor_positions.append(circular(p['Mk'][ii], p['sensor_spacing'], node_position))
        elif p['sensor_arrangement'] == 'within_sphere':
            sensor_positions.append(within_sphere(p['Mk'][ii], p['sensor_spacing'], node_position))
        else:
            raise ValueError('Unknown sensor arrangement.')
        
    # Flatten list
    sensor_positions = [item for sublist in sensor_positions for item in sublist]
        
    # Place targets
    target_positions = []
    for i in range(p['N_targets']):
        target_positions.append([
            p['rd'][0] * np.random.rand(),
            p['rd'][1] * np.random.rand(),
            p['rd'][2] * np.random.rand()
        ])

    # Place interferers
    if p['interferer_arrangement'] == 'linear':
        interferer_positions = linear(p['N_interferers'], p['interferer_spacing'], [p['rd'][0] / 2, p['rd'][1] / 2, p['rd'][2] / 2])
    elif p['interferer_arrangement'] == 'circular':
        interferer_positions = circular(p['N_interferers'], p['interferer_spacing'], [p['rd'][0] / 2, p['rd'][1] / 2, p['rd'][2] / 2])
    elif p['interferer_arrangement'] == 'within_sphere':
        interferer_positions = within_sphere(p['N_interferers'], p['interferer_spacing'], [p['rd'][0] / 2, p['rd'][1] / 2, p['rd'][2] / 2])
    else:
        raise ValueError('Unknown interferer arrangement.')

    # Adjust height of elements
    if p['same_height']:
        fixedHeight = p['rd'][2] / 2
        for i in range(len(node_positions)):
            node_positions[i][2] = fixedHeight
        for i in range(len(sensor_positions)):
            sensor_positions[i][2] = fixedHeight
        for i in range(len(target_positions)):
            target_positions[i][2] = fixedHeight
        for i in range(len(interferer_positions)):
            interferer_positions[i][2] = fixedHeight
        
    # Create dictionary
    room = {
        'rd': p['rd'],
        'Mk': p['Mk'],
        'sensor_positions': sensor_positions,
        'target_positions': target_positions,
        'interferer_positions': interferer_positions,
    }

    return room


def linear(N_sensors, sensor_spacing, node_position):
    """Function that arrange sensors in a linear array."""
    # N_sensors: number of sensors
    # sensor_spacing: distance between sensors
    # node_position: node position [x, y, z]
    # sensor_positions: sensor positions [x, y, z]

    # Sensor positions
    sensor_positions = []
    for i in range(N_sensors):
        sensor_positions.append([node_position[0] + i * sensor_spacing, node_position[1], node_position[2]])

    # Center array with respect to node position
    sensor_positions = np.array(sensor_positions)
    sensor_positions[:, 0] -= np.mean(sensor_positions[:, 0]) - node_position[0]
    
    return sensor_positions


def circular(N_sensors, sensor_spacing, node_position):
    """Function that arrange sensors in a circular array."""
    # N_sensors: number of sensors
    # sensor_spacing: distance between sensors
    # node_position: node position [x, y, z]
    # sensor_positions: sensor positions [x, y, z]

    # Sensor positions
    sensor_positions = []
    for i in range(N_sensors):
        sensor_positions.append([
            node_position[0] + sensor_spacing * np.cos(2 * np.pi * i / N_sensors),
            node_position[1] + sensor_spacing * np.sin(2 * np.pi * i / N_sensors),
            node_position[2]
        ])

    # Center array with respect to node position
    sensor_positions = np.array(sensor_positions)
    sensor_positions[:, 0] -= np.mean(sensor_positions[:, 0]) - node_position[0]

    return sensor_positions


def within_sphere(N_sensors, sensor_spacing, node_position):
    """Function that arrange sensors within a sphere."""
    # N_sensors: number of sensors
    # sensor_spacing: distance between sensors
    # node_position: node position [x, y, z]
    # sensor_positions: sensor positions [x, y, z]

    # Sensor positions
    sensor_positions = []
    for i in range(N_sensors):
        sensor_positions.append([
            node_position[0] + sensor_spacing * np.cos(2 * np.pi * i / N_sensors),
            node_position[1] + sensor_spacing * np.sin(2 * np.pi * i / N_sensors),
            node_position[2]
    ])
        
    # Center array with respect to node position
    sensor_positions = np.array(sensor_positions)
    sensor_positions[:, 0] -= np.mean(sensor_positions[:, 0]) - node_position[0]

    return sensor_positions


# Function that arrange nodes in a square grid
def square_grid(rd, node_spacing):
    """Function that arrange nodes in a square grid."""
    # rd: room dimensions [x, y, z]
    # node_spacing: distance between nodes
    # node_positions: node positions [x, y, z]

    # Number of nodes in each direction
    N_nodes = [int(rd[0] / node_spacing), int(rd[1] / node_spacing), int(rd[2] / node_spacing)]

    # Node positions
    node_positions = []
    for i in range(N_nodes[0]):
        for j in range(N_nodes[1]):
            for k in range(N_nodes[2]):
                node_positions.append([i * node_spacing, j * node_spacing, k * node_spacing])

    return node_positions


# Function that arrange nodes in a polygon
def polygon(rd, node_spacing, N_nodes):
    """Function that arrange nodes in a `N_nodes`-edged polygon
    of radius `node_spacing`."""
    # rd: room dimensions [x, y, z]
    # node_spacing: distance between nodes
    # N_nodes: number of nodes
    # node_positions: node positions [x, y, z]

    # Node positions
    node_positions = []
    for i in range(N_nodes):
        node_positions.append([
            rd[0] / 2 + node_spacing * np.cos(2 * np.pi * i / N_nodes),
            rd[1] / 2 + node_spacing * np.sin(2 * np.pi * i / N_nodes),
            rd[2] / 2
        ])

    return node_positions

if __name__ == '__main__':
    sys.exit(main())