import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

JOINTS = [
"nose", 
"left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder","right_shoulder", 
"left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", 
"left_knee", "right_knee", "left_ankle", "right_ankle"
]

def generate_arrows(a):
    # These are the joint connections pulled in by hand from joint definitions
    connections = [[0, 1], [0, 2], [1, 3], [2, 4], [11, 12], [0, 5], [0, 6], [5, 7],\
                   [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]]


    arrow_locs = []
    arrow_dirs = []
    for connect in connections:
        arrow_locs.append(a[connect[0]])
        arrow_dirs.append(a[connect[1]] - a[connect[0]])
    return np.stack(arrow_locs), np.stack(arrow_dirs)


def display_3d_joints(joints3DList):
    arrow_locs, arrow_dirs = generate_arrows(joints3DList)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.quiver(
    	arrow_locs[:, 0],
		-arrow_locs[:, 2],
		arrow_locs[:, 1],
		arrow_dirs[:, 0],
		-arrow_dirs[:, 2],
		arrow_dirs[:, 1],
		arrow_length_ratio=.01
	)
    return fig, ax
