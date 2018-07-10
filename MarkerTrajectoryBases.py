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
import sys
from math import factorial
from itertools import combinations
import warnings
from multiprocessing import Pool, freeze_support
import time

import numpy as np
import c3d
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.metrics.pairwise import paired_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from MarkerGroups import read_c3d_file, compute_cluster, best_groups_from_clusters, validate


# %% auxiliary marker
def auxiliary_marker(m1, m2, m3):
    """Marker that is perpendicular to plane spanned by 3 markers.
    Works for single frame or whole trajectory.
    
    :param m1: marker 1
    :type m1: numpy.ndarray
    :param m2: marker 2
    :type m2: numpy.ndarray
    :param m3: marker 3
    :type m3: numpy.ndarray
    :return: auxiliary marker
    :rtype: numpy.ndarray
    """
    m4 = m3 + np.cross(m1 - m3, m2 - m3)
    return m4


# %% joint location
def joint_from_markers(marker_trajectories, weights):
    """Joint location estimate by linear combination of marker base vectors.
    
    :param marker_trajectories: multidimensional array with 4 marker trajectories.
    :type marker_trajectories: numpy.ndarray
    :param weights: weights for markers 1-3
    :type weights: numpy.ndarray
    :return: joint trajectory
    :rtype: numpy.ndarray
    """
    # Add fourth weight and reshape array.
    weights = np.append(weights, 1 - weights.sum())[:, np.newaxis]
    j = np.multiply(marker_trajectories, weights).sum(axis=1)
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
    """ Cost function to optimize weights from which the best trajectory for a joint is calculated.
    
    :param x0: 3 lambda weights for linear combination of marker vectors to retrieve joint location.
    :type x0: numpy.ndarray
    :param args: marker trajectories matrix,
                 marker indices belonging to rigid body 1 & rigid body 2, distance penalty weight factor.
    :type args: tuple
    :return: cost
    :rtype: float
    """
    trajectories = args[0]
    rigid1_indices = args[1]
    rigid2_indices = args[2]
    penalty = float(args[3])
    # First, construct the joint trajectory from rigid body 1 and weights.
    j = joint_from_markers(trajectories[:, rigid1_indices, :], x0)
    all_marker_indices = rigid1_indices + rigid2_indices
    # Todo: Is there a faster way? Distances of all markers to joint in parallel. Or use n_jobs for speedup?
    # Then calculate cost q.
    distances_to_joint = np.array([paired_distances(t, j, n_jobs=-1) for t in np.swapaxes(trajectories[:, all_marker_indices],0,1)])
    mean_distances = np.mean(distances_to_joint, axis=1)
    var_distances = np.var(distances_to_joint, axis=1)
    q = (var_distances + penalty * mean_distances).sum()/len(all_marker_indices)
    return q


def get_marker_groups(markers,
                      n_clusters=10,
                      n_processes=0,
                      nth_frame=15,
                      rnd_offset=5,
                      min_groups=2,
                      max_groups=20,
                      ground_truth=None):
    """ Find marker groups by spectral clustering multiple times using several different samplings.

    :param markers: trajectories
    :param n_clusters: Compute this many clusters.
    :param n_processes: Number of Python subprocesses to start for cluster computation.
    :param nth_frame: sample every n-th frame
    :param rnd_offset: offset n-th frame by random range in -/+ rnd_rnd_offset
    :param min_groups: Minimum number of rigid bodies to look for.
    :param max_groups: Maximum number of rigid bodies to look for.
    :param ground_truth: List of lists of marker indices you'd expect.
    :type ground_truth: list
    :return: marker groups
    :rtype: list
    """
    # FixMe: ValueError: shapes (12,12) and (13,13) not aligned: 12 (dim 1) != 13 (dim 0)
    if n_processes > 0:
        processes = min(n_clusters, n_processes)
        print('Creating pool with %d processes\n' % processes)
        with Pool(processes) as pool:
            print("Computing {} clusters...".format(n_clusters))
            args = [[markers, nth_frame, rnd_offset, min_groups, max_groups]] * n_clusters
            clusters = pool.starmap(compute_cluster, args)
        # Make list from generator
        clusters = list(clusters)
    else:
        # Alternative serial computation. Faster for small files without multiprocess overhead..
        clusters = [compute_cluster(markers, min_groups=min_groups, max_groups=max_groups) for i in range(n_clusters)]
    groups = best_groups_from_clusters(clusters)
    
    if ground_truth:
        print("Comparing clusters to ground truth... ", end="", flush=True)
        validated = validate(clusters, ground_truth)
        print("Done.")
        print("N ground truth found in {} sampled clusters: {}".format(n_clusters, validated))
    return groups


def get_rigid_body_connections(edge_weights):
    """ Compute minimum spanning tree from edge weight matrix and return found index pairs."""
    # Make graph from edge weights
    rb_graph = csr_matrix(edge_weights)
    print("\nFully connected graph:\n", rb_graph.toarray())
    # Which rigid bodies are connected?
    tree_csr = minimum_spanning_tree(rb_graph)
    print("Minimum spanning tree:\n", tree_csr.toarray().astype(float))
    # Relate non-zero data in minimum spanning tree to marker_groups.
    connections = np.transpose(np.nonzero(tree_csr.toarray())).tolist()
    connections = [tuple(idx) for idx in connections]
    return connections


def save_to_c3d_file(file_path, points, fps=30):
    writer = c3d.Writer(point_rate=float(fps))
    for i in range(points.shape[1]):
        writer.add_frames([(points[:, i], np.array([[]]))])
    try:
        with open(file_path, 'wb') as file_handle:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore UserWarning: missing parameter ANALOG:DESCRIPTIONS/LABELS
                writer.write(file_handle)
    except IOError as e:
        print("Failed to write file. Reason:", e)
        

def process_c3d_file(in_file,
                     out_fps=30,
                     n_clusters=10,
                     n_cluster_processes=4,
                     min_rigid_bodies=3,
                     max_rigid_bodies=20):
    """ Compute rigid bodies and their joint trajectories for C3D file and save them to file.
    :param n_cluster_processes: Adjust number of processes to your CPU and data size. 0 = no multiprocessing.
    """
    print("Processing file:", in_file)
    t0 = time.time()
    markers, conditionals = read_c3d_file(in_file, output_fps=out_fps)
    print("Finding rigid bodies from marker trajectories through spectral clustering...")
    marker_groups = get_marker_groups(markers,
                                      n_clusters=n_clusters,
                                      n_processes=n_cluster_processes,
                                      min_groups=min_rigid_bodies,
                                      max_groups=max_rigid_bodies)
    
    # Todo: check co-planarity/collinearity within groups.

    # Generate all possible rigid body pairings using their indices from marker_groups.
    rb_idx_pairs = list(combinations(range(len(marker_groups)), 2))
    # Create a NxN matrix to hold edge weights for a fully connected graph of rigid bodies.
    edge_weights = np.zeros((len(rb_idx_pairs),) * 2)
    # Create dictionary to hold new trajectory for each point connecting a rigid body pair.
    points = dict()
    x0 = np.array([1.0, 1.0, 1.0])  # initial lambda weights.
    # Todo: parallelize?
    for idx_pair in rb_idx_pairs:
        rb1_marker_indices = marker_groups[idx_pair[0]]
        rb2_marker_indices = marker_groups[idx_pair[1]]
        print("\nOptimizing connection for marker groups {} & {}.".format(rb1_marker_indices, rb2_marker_indices))
        solution = minimize(cost_func, x0, args=(markers,             # trajectories for all markers.
                                                 rb1_marker_indices,  # marker indices belonging to rigid body
                                                 rb2_marker_indices,  # marker indices belonging to rigid body
                                                 0.2))                # Adjust: penalty factor on average distance.
        if solution.success:
            # Extract estimated parameters
            final_weights = solution.x
            # Use cost as edge weight for computing the minimum spanning tree.
            edge_weights[idx_pair[0], idx_pair[1]] = solution.fun
            print("Cost Q:", solution.fun)
            print("number of iterations:", solution.nit)
            print("Estimated weight parameters: {}".format(final_weights))
            # Calculate joint trajectory with final weights.
            joint_trajectory = joint_from_markers(markers[:, rb1_marker_indices, :], solution.x)
            # Add columns for residuals and camera contribution.
            point = np.hstack((joint_trajectory, np.zeros((joint_trajectory.shape[0], 2), dtype=joint_trajectory.dtype)))
            points[idx_pair] = point
        else:
            print("ERROR: Optimization was not successful!")
    if not edge_weights.any():
        print("No connections could be found between marker groups.")
        sys.exit()
    
    connected_rb_indices = get_rigid_body_connections(edge_weights)
    for idx in connected_rb_indices:
        print("marker group {} is connected to group {}".format(marker_groups[idx[0]], marker_groups[idx[1]]))
        
    # Write joint trajectories to file. Write only those points that connect rigid bodies in minimum spanning tree.
    mst_points = np.array([trajectory for idx, trajectory in points.items() if idx in connected_rb_indices])
    out_file_path = in_file[:-4] + '-joints.c3d'
    print("Saving trajectories to {}".format(out_file_path))
    save_to_c3d_file(out_file_path, mst_points, out_fps)
    print("elapsed time: {} seconds".format(time.time()-t0))


# %% Optimize
if __name__ == "__main__":
    freeze_support()

    # Set Data folder path
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    except NameError:
        data_path = os.path.join(os.getcwd(), "Data")
    
    c3d_filepath = os.path.join(data_path, "arm-4-4-4_clean_30fps.c3d")
    process_c3d_file(c3d_filepath,
                     out_fps=30,
                     n_clusters=10,
                     n_cluster_processes=0,  # No multiprocessing. Takes twice as long.
                     min_rigid_bodies=3,
                     max_rigid_bodies=3)
