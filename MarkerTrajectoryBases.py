#    MIT License
#
#    Copyright (c) 2018 Olaf Haag
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

"""
Implementation of marker trajectory bases (MTB) joint localization.
From:
Hang, J., Lasenby, J., & Li, A. (2015, December).
An Automatic Solution Framework for Robust and Computationally Efficient Joint Estimation in Optical Motion Capture.
In Computational Science and Computational Intelligence (CSCI), 2015 International Conference on (pp. 1-6). IEEE.

Limitations:
- needs cleaned data (no missing data!)
- without hybrid JMV Optimization can only solve joints for body segments with 3 or 4 markers (or 2 plus solved joint).
"""
# %% Imports
import os
from math import factorial

import numpy as np
import c3d
from scipy.optimize import minimize
from scipy.spatial import distance


# %% auxiliary marker
def auxiliary_marker(m1, m2, m3):
    """Marker that is perpendicular to plane spanned by 3 markers.
    Works for single frame or whole trajectory.
    
    :param m1: marker 1
    :type m1: np.array
    :param m2: marker 2
    :type m2: np.array
    :param m3: marker 3
    :type m3: np.array
    :return: auxiliary marker
    :rtype: np.array
    """
    m4 = m3 + np.cross(m1 - m3, m2 - m3)
    return m4


# %% joint location
def joint_from_markers(m1, m2, m3, m4, par1, par2, par3):
    """Joint location estimate by linear combination of marker base vectors
    
    :param m1: marker 1
    :type m1: np.array
    :param m2: marker 2
    :type m2: np.array
    :param m3: marker 3
    :type m3: np.array
    :param m4: marker 4
    :type m4: np.array
    :param par1: weight of marker1
    :param par2: weight of marker2
    :param par3: weight of marker3
    :return:  joint trajectory
    :rtype: np.array
    """
    j = par1 * m1 + par2 * m2 + par3 * m3 + (1 - par1 - par2 - par3) * m4
    return j


# %% Average joint to marker distance.
def avg_joint_marker_distance(joint, marker) -> float:
    """ Average distance between a marker and a joint over all frames.
    
    :param joint: joint trajectory
    :type joint: np.array
    :param marker: marker trajectory
    :type marker: np.array
    :return: average distance
    :rtype: float
    """
    avg = np.linalg.norm(joint - marker, axis=1).sum() / len(marker)  # length of the vectors equals number of frames.
    return avg


# %% variance in joint-marker distance
def joint_marker_distance_variance(joint, marker) -> float:
    """Computes variance in joint-marker distance.
    
    :param joint: joint trajectory
    :type joint: np.array
    :param marker: marker trajectory
    :type marker: np.array
    :return: variance
    :rtype: float
    """
    sig = np.square(np.linalg.norm(joint - marker, axis=1) - avg_joint_marker_distance(joint, marker)).sum()
    sig /= len(marker)
    return sig


# %% Checks
def are_coplanar(markers) -> bool:
    """Checks if given marker positions are co-planar.
    
    :param markers: list of markers, each an array with single x,y,z coordinates.
    :type markers: list
    :return: Whether the markers are co-planar or not.
    :rtype: bool
    """
    # Less than 4 markers are co-planar (if not superimposed, but this is practically not possible with markers).
    if len(markers) > 4:
        return True
    
    # Calculate the volume of the tetrahedron formed by the 4 markers.
    # If this volume is zero, then they must be coplanar.
    
    # β_ij = |v_i - v_k|²
    markers = np.asarray(markers, dtype=float)
    sq_distance = distance.pdist(markers, metric='sqeuclidean')
    
    # Add border.
    n_vertices = distance.num_obs_y(sq_distance)
    bordered = np.concatenate((np.ones(n_vertices), sq_distance))
    
    # Make matrix and find volume.
    sq_distance_matrix = distance.squareform(bordered)
    
    coeff = - (-2) ** (n_vertices - 1) * factorial(n_vertices - 1) ** 2
    volume_squared = np.linalg.det(sq_distance_matrix) / coeff
    
    if volume_squared <= 0:
        return True
    #print("Volume formed by markers:", np.sqrt(volume_squared))
    
    return False


def are_collinear(markers) -> bool:
    """Checks if given marker positions are collinear.
    
    :param markers: list of markers, each an array with single x,y,z coordinates.
    :type markers: list
    :return: Whether the markers are collinear or not.
    :rtype: bool
    """
    # Less than 3 markers are collinear (if not superimposed, but this is practically not possible with markers).
    if len(markers) < 3:
        return True
    
    # take cross-product of vectors and compare to 0.
    v1 = markers[1] - markers[0]
    v2 = markers[2] - markers[0]
    if np.cross(v1, v2).any():
        return False
    else:
        if len(markers) == 4:
            # Check against 4th marker vector.
            return not np.cross(v1, markers[3] - markers[0]).any()
        else:
            return True


# %% cost function
def cost_func(x0, *args) -> float:
    """
    
    :param x0: 3 lambda weights for linear combination of marker vectors to retrieve joint location.
    :type x0: np.array
    :param args: markers belonging to rigid body 1 & rigid body 2, distance penalty weight factor.
    :type args: tuple
    :return: cost
    :rtype: float
    """
    rigid1 = args[0]
    rigid2 = args[1]
    penalty = float(args[2])
    # First, construct the joint trajectory from rigid body 1 and weights.
    j = joint_from_markers(*rigid1, *x0)
    q = 0.0
    all_markers = rigid1 + rigid2
    # Todo: parallelize?
    for m in all_markers:
        q += joint_marker_distance_variance(j, m) + penalty * avg_joint_marker_distance(j, m)
    
    q /= len(rigid1) + len(all_markers)
    return q


# %% C3D file
def humanize_time(secs):
    ms = secs % int(secs) * 1000
    mins, secs = divmod(int(secs), 60)
    hours, mins = divmod(mins, 60)
    return "{:02d} hours {:02d} minutes {:02d} seconds ~{:d} milliseconds".format(hours, mins, secs, int(ms))


def read_c3d_file(file_path, output_fps=30):
    """
    
    :param file_path:
    :type file_path: str
    :param output_fps:
    :return: marker data
    """
    with open(file_path, 'rb') as file_handle:
        reader = c3d.Reader(file_handle)
        marker_labels = reader.point_labels
        print("Marker Labels:", ",".join(marker_labels))
        first_frame = reader.first_frame()
        last_frame = reader.last_frame()
        print("First Frame:", first_frame, "Last Frame:", last_frame)
        fps = reader.header.frame_rate
        print("FPS:", fps)
        n_frames = last_frame - first_frame + 1
        total_length = n_frames / fps
        print("Clip length in total:", humanize_time(total_length))
        # Extract Positionsfor each frame.
        pos_array = np.empty([n_frames, len(marker_labels), 3])
        pos_array.fill(np.NAN)
        cond_array = np.empty([n_frames, len(marker_labels)])
        cond_array.fill(np.NAN)
        print("Reading frames...")
        for i, points, _ in reader.read_frames():
            # pos_array[frame, marker], e.g. pos_array[:,11] all frames for 12th marker
            # points are mirrored/different coordinate system somehow.
            pos_array[i - first_frame, :, :] = np.vstack([-1.0 * points[:, 0], -1.0 * points[:, 2], -1.0 * points[:, 1]]).T
            cond_array[i - first_frame, :] = points[:, 3]
            if n_frames is not None and i - first_frame >= n_frames:
                break
        
        # There might be a lot of frames. To speed up optimization use only a subset.
        nth_frame = int(fps / output_fps)
        frames_indices = np.arange(0, n_frames, nth_frame)
        #scale = 0.001  # convert mm to m
        pos_subset = pos_array[frames_indices]
        cond_subset = cond_array[frames_indices]
        
        # Todo: handle missing/bad data
        return pos_subset, cond_subset
    
    
# %% Optimize
if __name__ == "__main__":
    # Set Data folder path
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    except NameError:
        data_path = os.path.join(os.getcwd(), "Data")
    
    c3d_filepath = os.path.join(data_path, "arm-4-4-4_30fps.c3d")
    out_fps = 30
    markers, conditionals = read_c3d_file(c3d_filepath, output_fps=out_fps)
    # todo: set marker groups by file/spectral clustering or other
    markers_rb1 = list()
    for i in range(4):
        markers_rb1.append(markers[:, i])
    markers_rb2 = list()
    for i in range(4, 8):
        markers_rb2.append(markers[:, i])
    markers_rb3 = list()
    for i in range(8, 12):
        markers_rb3.append(markers[:, i])
        
    # Todo: check co-planarity/collinearity
        
    x0 = np.array([1.0, 1.0, 1.0])  # initial lambda weights.
    optim_rb_sets = [(markers_rb1, markers_rb2), (markers_rb2, markers_rb3)]
    points = list()
    for rb_set in optim_rb_sets:
        solution = minimize(cost_func, x0, args=(*rb_set, 0.2))
        if solution.success:
            # Extract estimated parameters
            est_lambda1 = solution.x[0]
            est_lambda2 = solution.x[1]
            est_lambda3 = solution.x[2]
            print("Cost Q:", solution.fun)
            print("number of iterations:", solution.nit)
            print("Estimated weight parameters:\n1={}\n2={}\n3={}".format(est_lambda1, est_lambda2, est_lambda3))
            j = joint_from_markers(*rb_set[0], est_lambda1, est_lambda2, est_lambda3)
            point = np.hstack((j, np.zeros((j.shape[0], 2), dtype=j.dtype)))
            points.append(point)
        else:
            print("ERROR: Optimization was not successful!")
    
    # Write joint trajectories to file.
    points = np.array(points)
    writer = c3d.Writer(point_rate=float(out_fps))
    for i in range(points.shape[1]):
        writer.add_frames([(points[:, i], np.array([[]]))])
    with open(os.path.join(data_path, 'test.c3d'), 'wb') as h:
        writer.write(h)
