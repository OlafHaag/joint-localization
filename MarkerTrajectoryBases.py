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
from itertools import combinations
import warnings

import numpy as np
import c3d
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from MarkerGroups import read_c3d_file, avg_marker_pair_distance, marker_pair_distance_variance, compute_cluster, best_groups_from_clusters


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
    all_markers = rigid1 + rigid2
    q = np.array([marker_pair_distance_variance(j, m) + penalty * avg_marker_pair_distance(j, m) for m in all_markers]).sum()
    q /= len(all_markers)
    return q


# %% Optimize
if __name__ == "__main__":
    # Set Data folder path
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    except NameError:
        data_path = os.path.join(os.getcwd(), "Data")
    
    c3d_filepath = os.path.join(data_path, "arm-4-4-4_clean_30fps.c3d")
    out_fps = 30
    markers, conditionals = read_c3d_file(c3d_filepath, output_fps=out_fps)
    # Find marker groups by spectral clustering multiple times using several different samplings.
    print("Finding rigid bodies from marker trajectories through spectral clustering...")
    n_samples = 10
    # Todo: parallelize.
    clusters = [compute_cluster(markers, min_groups=3, max_groups=3) for i in range(n_samples)]
    marker_groups = best_groups_from_clusters(clusters)
    # Put together marker data according to groups.
    # List of rigid bodies that contain a list of their respective marker data.
    rigid_bodies = [[markers[:, marker_idx] for marker_idx in group] for group in marker_groups]
    
    # Todo: check co-planarity/collinearity
    
    x0 = np.array([1.0, 1.0, 1.0])  # initial lambda weights.
    rb_pairs_indices = list(combinations(range(len(marker_groups)), 2))
    # Map actual rigid body marker collection to index pairs.
    rb_pairs = dict([(idx, (rigid_bodies[idx[0]], rigid_bodies[idx[1]])) for idx in rb_pairs_indices])
    # Create a NxN matrix to hold edge weights for a fully connected graph of rigid bodies.
    edge_weights = np.zeros((len(rb_pairs_indices), len(rb_pairs_indices)))
    # Create dictionary to hold new trajectory for each point connecting a rigid body pair.
    points = dict()
    # Todo: parallelize?
    for idx_pair, rb_set in rb_pairs.items():
        print("\nOptimizing rigid body pair {}...".format(idx_pair))
        solution = minimize(cost_func, x0, args=(*rb_set, 0.2))  # Adjust: penalty factor on average distance.
        if solution.success:
            # Extract estimated parameters
            est_lambda1 = solution.x[0]
            est_lambda2 = solution.x[1]
            est_lambda3 = solution.x[2]
            # Use cost as edge weight for computing the minimum spanning tree.
            edge_weights[idx_pair[0], idx_pair[1]] = solution.fun
            print("Cost Q:", solution.fun)
            print("number of iterations:", solution.nit)
            print("Estimated weight parameters:\n1={}\n2={}\n3={}".format(est_lambda1, est_lambda2, est_lambda3))
            j = joint_from_markers(*rb_set[0], est_lambda1, est_lambda2, est_lambda3)
            point = np.hstack((j, np.zeros((j.shape[0], 2), dtype=j.dtype)))
            points[idx_pair] = point
        else:
            print("ERROR: Optimization was not successful!")
    # Make graph from edge weights
    rb_graph = csr_matrix(edge_weights)
    print("\nFully connected graph:\n", rb_graph.toarray())
    # Which rigid bodies are connected?
    tree_csr = minimum_spanning_tree(rb_graph)
    print("Minimum spanning tree:\n", tree_csr.toarray().astype(float))
    # Todo: relate non-zero data to marker_groups.
    connected_rb_indices = np.transpose(np.nonzero(tree_csr.toarray())).tolist()
    connected_rb_indices = [tuple(idx) for idx in connected_rb_indices]
    
    # Write joint trajectories to file. Write only those points that connect rigid bodies in minimum spanning tree.
    mst_points = np.array([trajectory for idx, trajectory in points.items() if idx in connected_rb_indices])
    writer = c3d.Writer(point_rate=float(out_fps))
    for i in range(mst_points.shape[1]):
        writer.add_frames([(mst_points[:, i], np.array([[]]))])
    with open(os.path.join(data_path, 'test.c3d'), 'wb') as file_handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore UserWarning: missing parameter ANALOG:DESCRIPTIONS/LABELS
            writer.write(file_handle)
