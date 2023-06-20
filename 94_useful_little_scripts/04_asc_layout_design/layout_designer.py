# Purpose of script:
# Acoustic scenario (ASC) designer script.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
import sys
import yaml
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



BASE_EXPORT_PATH = '94_useful_little_scripts/04_asc_layout_design/layouts'
PARAMS = {
    'rd': [5, 5, 5],
    'Mk': [3, 3],  # number of sensors per node
    'node_arrangement': 'polygon',
    'node_spacing': 1.5,  # [m]
    'sensor_arrangement': 'within_sphere',  # 'linear', 'circular', 'within_sphere'
    'sensor_spacing': 0.2,  # [m]
    'N_targets': 1,
    'N_interferers': 2, # number of interferers
    'interferer_arrangement': 'circular',  # 'linear', 'circular', 'within_sphere'
    'interferer_spacing': 2,  # [m]
    'same_height': True,  # if True, all elements are at the same height
    #
    'export_file_name': f'{BASE_EXPORT_PATH}/ascsmall1.yaml'
}

def main(p=PARAMS):
    """Main function (called by default when running script)."""
    room = build_room(p)

    # # Plot room

    # Export coordinates as YAML
    export_yaml(room, p)
    plot_room(room, p, export=True)

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
        'sensorCoords': data['sensorCoords'],
        'targetCoords': data['targetCoords'],
        'interfererCoords': data['interfererCoords'],
    }

    return room


def export_yaml(room, p):
    """Function that exports room coordinates as a single YAML file."""
    # room: room coordinates
    # p: parameters
    
    # Check that folder exists
    if not os.path.exists(BASE_EXPORT_PATH):
        os.makedirs(BASE_EXPORT_PATH)

    # Add datetime to export name
    p['export_file_name'] = p['export_file_name'].replace(
        '.yaml',
        f'_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
    )

    # Export
    with open(p['export_file_name'], 'w') as f:
        f.write('rd: [%.2f, %.2f, %.2f]\n' % (p['rd'][0], p['rd'][1], p['rd'][2]))
        f.write('Mk: [')
        for Mk in p['Mk']:
            f.write('%d, ' % Mk)
        f.write(']\n')
        f.write('sensorCoords:\n')
        for sensor_position in room['sensorCoords']:
            f.write('  - [%.2f, %.2f, %.2f]\n' % (sensor_position[0], sensor_position[1], sensor_position[2]))
        f.write('targetCoords:\n')
        for target_position in room['targetCoords']:
            f.write('  - [%.2f, %.2f, %.2f]\n' % (target_position[0], target_position[1], target_position[2]))
        f.write('interfererCoords:\n')
        for interferer_position in room['interfererCoords']:
            f.write('  - [%.2f, %.2f, %.2f]\n' % (interferer_position[0], interferer_position[1], interferer_position[2]))


def plot_room(room, p, export=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot sensors
    sensorCoords = np.array(room['sensorCoords'])
    ax.scatter(sensorCoords[:, 0], sensorCoords[:, 1], sensorCoords[:, 2], c='b', marker='o')

    # Plot targets
    targetCoords = np.array(room['targetCoords'])
    ax.scatter(targetCoords[:, 0], targetCoords[:, 1], targetCoords[:, 2], c='g', marker='d')

    # Plot interferers
    interfererCoords = np.array(room['interfererCoords'])
    ax.scatter(interfererCoords[:, 0], interfererCoords[:, 1], interfererCoords[:, 2], c='k', marker='+')

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
    
    if export:
        fig.savefig(p['export_file_name'].replace('.yaml', '.png'), dpi=300)
    else:
        plt.show()


def build_room(p):

    # Place nodes
    if p['node_arrangement'] == 'polygon':
        node_positions = polygon(p['rd'], p['node_spacing'], len(p['Mk']))
    else:
        raise ValueError('Unknown node arrangement.')
    
    # Place sensors at each node
    sensorCoords = []
    for ii, node_position in enumerate(node_positions):
        if p['sensor_arrangement'] == 'linear':
            sensorCoords.append(linear(p['Mk'][ii], p['sensor_spacing'], node_position))
        elif p['sensor_arrangement'] == 'circular':
            sensorCoords.append(circular(p['Mk'][ii], p['sensor_spacing'], node_position))
        elif p['sensor_arrangement'] == 'within_sphere':
            sensorCoords.append(within_sphere(p['Mk'][ii], p['sensor_spacing'], node_position))
        else:
            raise ValueError('Unknown sensor arrangement.')
        
    # Flatten list
    sensorCoords = [item for sublist in sensorCoords for item in sublist]
        
    # Place targets
    targetCoords = []
    for i in range(p['N_targets']):
        targetCoords.append([
            p['rd'][0] * np.random.rand(),
            p['rd'][1] * np.random.rand(),
            p['rd'][2] * np.random.rand()
        ])

    # Place interferers
    if p['interferer_arrangement'] == 'linear':
        interfererCoords = linear(p['N_interferers'], p['interferer_spacing'], [p['rd'][0] / 2, p['rd'][1] / 2, p['rd'][2] / 2])
    elif p['interferer_arrangement'] == 'circular':
        interfererCoords = circular(p['N_interferers'], p['interferer_spacing'], [p['rd'][0] / 2, p['rd'][1] / 2, p['rd'][2] / 2])
    elif p['interferer_arrangement'] == 'within_sphere':
        interfererCoords = within_sphere(p['N_interferers'], p['interferer_spacing'], [p['rd'][0] / 2, p['rd'][1] / 2, p['rd'][2] / 2])
    else:
        raise ValueError('Unknown interferer arrangement.')

    # Adjust height of elements
    if p['same_height']:
        fixedHeight = p['rd'][2] / 2
        for i in range(len(node_positions)):
            node_positions[i][2] = fixedHeight
        for i in range(len(sensorCoords)):
            sensorCoords[i][2] = fixedHeight
        for i in range(len(targetCoords)):
            targetCoords[i][2] = fixedHeight
        for i in range(len(interfererCoords)):
            interfererCoords[i][2] = fixedHeight
        
    # Create dictionary
    room = {
        'rd': p['rd'],
        'Mk': p['Mk'],
        'sensorCoords': sensorCoords,
        'targetCoords': targetCoords,
        'interfererCoords': interfererCoords,
    }

    return room


def linear(N_sensors, sensor_spacing, node_position):
    """Function that arrange sensors in a linear array."""
    # N_sensors: number of sensors
    # sensor_spacing: distance between sensors
    # node_position: node position [x, y, z]
    # sensorCoords: sensor positions [x, y, z]

    # Sensor positions
    sensorCoords = []
    for i in range(N_sensors):
        sensorCoords.append([node_position[0] + i * sensor_spacing, node_position[1], node_position[2]])

    # Center array with respect to node position
    sensorCoords = np.array(sensorCoords)
    sensorCoords[:, 0] -= np.mean(sensorCoords[:, 0]) - node_position[0]
    
    return sensorCoords


def circular(N_sensors, sensor_spacing, node_position):
    """Function that arrange sensors in a circular array."""
    # N_sensors: number of sensors
    # sensor_spacing: distance between sensors
    # node_position: node position [x, y, z]
    # sensorCoords: sensor positions [x, y, z]

    # Sensor positions
    sensorCoords = []
    for i in range(N_sensors):
        sensorCoords.append([
            node_position[0] + sensor_spacing * np.cos(2 * np.pi * i / N_sensors),
            node_position[1] + sensor_spacing * np.sin(2 * np.pi * i / N_sensors),
            node_position[2]
        ])

    # Center array with respect to node position
    sensorCoords = np.array(sensorCoords)
    sensorCoords[:, 0] -= np.mean(sensorCoords[:, 0]) - node_position[0]

    return sensorCoords


def within_sphere(N_sensors, sensor_spacing, node_position):
    """Function that arrange sensors within a sphere."""
    # N_sensors: number of sensors
    # sensor_spacing: distance between sensors
    # node_position: node position [x, y, z]
    # sensorCoords: sensor positions [x, y, z]

    # Sensor positions
    sensorCoords = []
    for i in range(N_sensors):
        sensorCoords.append([
            node_position[0] + sensor_spacing * np.cos(2 * np.pi * i / N_sensors),
            node_position[1] + sensor_spacing * np.sin(2 * np.pi * i / N_sensors),
            node_position[2]
    ])
        
    # Center array with respect to node position
    sensorCoords = np.array(sensorCoords)
    sensorCoords[:, 0] -= np.mean(sensorCoords[:, 0]) - node_position[0]

    return sensorCoords


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