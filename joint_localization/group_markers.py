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
To determine marker groups, the method clusters based on the standard
deviation in distance over time between all pairs of markers.
To address the speed and error issues, the method calculates this quantity
only over a jittered uniform sampling of frames.

A. Kirk et al., “Skeletal parameter estimation from optical motion capture data,”
in CVPR 2005. IEEE Computer Society Conference on, vol. 2, 2005, pp. 1185 vol. 2–.
"""
import os
import warnings
from itertools import combinations
from multiprocessing import Pool, freeze_support
import argparse

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import c3d

from joint_localization.stsc import self_tuning_spectral_clustering


def humanize_time(seconds):
    """Convert time in seconds to (better) human readable format.
    
    :param seconds: seconds
    :type seconds: float
    :return: Human readable time in h:m:s:ms
    :rtype: str
    """
    ms = seconds % int(seconds) * 1000
    mins, seconds = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return "{:02d} hours {:02d} minutes {:02d} seconds ~{:d} milliseconds".format(hours, mins, seconds, int(ms))


def read_c3d_file(file_path, output_fps=30):
    """Read INTEL format C3D file and return a subsample of marker positions and conditionals.
    Also prints information about the file content.

    :param file_path: Path to C3D file.
    :type file_path: str
    :param output_fps: out sample rate. If the file has 480fps and output_fps is 30, every 16th frame will be sampled.
    :type output_fps: int
    :return: marker data
    :rtype: (list, numpy.ndarray, numpy.ndarray)
    """
    try:
        with open(file_path, 'rb') as file_handle:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore UserWarning: missing parameter ANALOG:DESCRIPTIONS/LABELS
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
            # Extract positions for each frame.
            pos_array = np.empty([n_frames, len(marker_labels), 3])
            pos_array.fill(np.NAN)
            cond_array = np.empty([n_frames, len(marker_labels)])
            cond_array.fill(np.NAN)
            print("Reading frames... ", end="", flush=True)
            for i, points, _ in reader.read_frames():
                # pos_array[frame, marker], e.g. pos_array[:,11] all frames for 12th marker
                # Convert to a mirrored coordinate system.
                pos_array[i - first_frame, :, :] = np.vstack([-1.0 * points[:, 0],
                                                              -1.0 * points[:, 2],
                                                              -1.0 * points[:, 1]]).T
                cond_array[i - first_frame, :] = points[:, 3]
                if n_frames is not None and i - first_frame >= n_frames:
                    break
    except OSError:
        print("ERROR: Could not read file.")
        raise
    print("Done.")
    
    # There might be a lot of frames. To speed up optimization use only a subset.
    print("Subsampling frames to {} frames per second... ".format(output_fps), end="", flush=True)
    nth_frame = int(fps / output_fps)
    frames_indices = np.arange(0, n_frames, nth_frame)
    pos_subset = pos_array[frames_indices]
    cond_subset = cond_array[frames_indices]
    print("Done.")
    
    # Todo: handle missing/bad data
    # The data for a given marker typically contains large errors
    # just before the system loses track of that marker and for a
    # short period after the marker is rediscovered. To eliminate
    # problems with “ghost” markers and erroneous position data
    # during those periods, the first few frames are trimmed off
    # the beginning and end of each marker’s data segment, and any
    # marker with a maximum number of consecutive frames less than
    # one half second is ignored.
    return marker_labels, pos_subset, cond_subset
    
    
def sample_marker_positions(markers, delta, rnd_offset):
    """Returns a subset of the marker data.
    
    :param markers: ndarray of marker data
    :type markers: numpy.ndarray
    :param delta: number of frames to skip.
    :type delta: int
    :param rnd_offset: maximum offset forward or backward to regular frame sampling.
    :type rnd_offset: int
    :return: randomly sampled subset of marker data.
    :rtype: numpy.ndarray
    """
    n_frames = len(markers)
    frames_indices = np.arange(0, n_frames, delta)
    randomize = lambda x: x + np.random.randint(-rnd_offset, rnd_offset)
    frames_indices = np.fromiter((randomize(x) for x in frames_indices), frames_indices.dtype)
    # Make sure indices are valid.
    frames_indices[np.where(frames_indices < 0)] = 0
    frames_indices[np.where(frames_indices >= n_frames)] = n_frames - np.random.randint(1, rnd_offset)
    subset = markers[frames_indices]
    return subset


def get_cost_matrix(markers_sample):
    """Standard deviation in distance between marker pairs.
    Define a cost matrix, A, such that element A ij is the
    standard deviation in distance between markers i and j for a
    particular sampling of frames.
    
    :param markers_sample: positions of markers over a sample of frames.
    :type markers_sample: numpy.ndarray
    :return: matrix of standard deviations with shape (markers, markers).
    :rtype: numpy.ndarray
    """
    distances = np.array([euclidean_distances(x) for x in markers_sample])
    std_distances = np.std(distances, axis=0)
    
    return std_distances


def sum_distance_deviations(marker_indices, cost_matrix):
    """Sum up standard deviations of distances for given marker indices.
    To avoid penalizing large marker groups, the standard deviation within
    a group is normalized by the number of markers in the group.
    
    :param marker_indices: marker indices to sum pairwise
    :type marker_indices: list
    :param cost_matrix: NxN matrix containing standard deviations of pairwise distances.
    :type cost_matrix: numpy.ndarray
    :return: Wighted sum of standard deviations for pairwise distances.
    :rtype: float
    """
    pairs = list(combinations(marker_indices, 2))
    return cost_matrix[tuple(zip(*pairs))].sum() / len(marker_indices)


def get_affinity_matrix(cost_matrix):
    """Transforms a cost matrix into affinity by inversion.
    
    :param cost_matrix: cost matrix
    :type cost_matrix: numpy.ndarray
    :return: Inverted cost matrix.
    :rtype: numpy.ndarray
    """
    affinity = cost_matrix.copy()
    np.fill_diagonal(affinity, 1.0)
    affinity = 1 / affinity
    return affinity


def compute_stsc_cluster(markers, sample_nth_frame=15, rnd_frame_offset=5, min_groups=2, max_groups=20):
    """Segments the markers into rigid body groups.
    
    :param markers: Marker trajectories
    :param sample_nth_frame: sample rate for subsampling the marker data (for computational efficiency).
    :param rnd_frame_offset: semi-randomize regular sampling rate by +/- random max offset to counter periodic errors.
    :param min_groups: Minimum number of groups to consider.
    :param max_groups: Maximum number of groups to consider.
    :return: {'groups': list of lists with marker indices, 'sum_dev': float}
    :rtype: dict
    """
    assert sample_nth_frame < len(markers), "Sampling rate exceeds number of frames! No markers to sample."
    # Samples are selected over all possible frames at intervals
    # of one half second, plus or minus a few frames.
    # This jitter ensures that any periodic errors do not affect the segmentation.
    marker_subset = sample_marker_positions(markers, sample_nth_frame, rnd_frame_offset)
    costs = get_cost_matrix(marker_subset)
    # The costs need to be sensibly inverted, so that low costs are closer to 1 and high costs close to zero.
    affinity = get_affinity_matrix(costs)
    #print("Commencing self tuning spectral clustering. min: {}, max:{}".format(min_groups, max_groups))
    groups = self_tuning_spectral_clustering(affinity, min_n_cluster=min_groups, max_n_cluster=max_groups)  # FixMe: spectral clustering is way too slow.
    # Sum standard deviation of distances over all marker pairs in each group.
    sum_deviations = np.array([sum_distance_deviations(group, costs) for group in groups]).sum()
    return {'groups': groups, 'sum_dev': sum_deviations}


def best_groups_from_clusters(clusters):
    """From among the multiple clusterings, select the one which
    minimizes the sum standard deviation of distances over all
    marker pairs in each group, for all clusterings.
    
    :param clusters: List of dictionaries [{'groups': list, 'sum_dev': float},]
    :type clusters: list
    :return: List for marker groups.
    :rtype: list
    """
    std_sums = [cluster['sum_dev'] for cluster in clusters]
    best_idx = np.argmin(std_sums)
    best_groups = clusters[best_idx]['groups']
    print("rigid body groups:", best_groups)
    print("weighted distances deviation sum:", std_sums[best_idx])
    return best_groups


def validate(clusters, ground_truth):
    """Validate if the ground truth is within the clusters.

    :param clusters: sampled clusters
    :type clusters: list
    :param ground_truth: list of prior known marker groups.
    :type ground_truth: list
    :return: Number of times ground truth was found in clusters.
    :rtype: int
    """
    is_valid = np.array([[group in cluster['groups'] for group in ground_truth] for cluster in clusters])
    is_valid = is_valid.all(axis=1).sum()
    return is_valid
    
    
def group_markers_stsc(marker_trajectories,
                       n_clusters=10,
                       nth_frame=15,
                       rnd_offset=5,
                       min_groups=2, max_groups=20):
    """Find groups of markers by self-tuning spectral clustering.
    
    :param marker_trajectories: Marker trajectories
    :param n_clusters: Compute this many clusters.
    :param nth_frame:
    :param rnd_offset:
    :param min_groups: Minimum number of rigid bodies to look for.
    :param max_groups: Maximum number of rigid bodies to look for.
    :return: marker groups
    :rtype: list
    """
    # Compute clusters in parallel.
    processes = min(n_clusters, 4)  # Adjust number of processes to your CPU.
    print("Creating pool with {} processes\n".format(processes))
    with Pool(processes) as pool:
        print("Computing {} clusters...".format(n_clusters))
        args = [[marker_trajectories, nth_frame, rnd_offset, min_groups, max_groups]] * n_clusters
        clusters = pool.starmap(compute_stsc_cluster, args)
    # Make list from generator.
    clusters = list(clusters)
    marker_groups = best_groups_from_clusters(clusters)
    # Todo: Move validation to tests.
    """
    if ground_truth:
        print("Comparing clusters to ground truth... ", end="", flush=True)
        validated = validate(clusters, ground_truth)
        print("Done.")
        print("N ground truth found in {} sampled clusters: {}".format(n_clusters, validated))
    """
    return marker_groups


def group_markers_kmeans(marker_trajectories, k):
    """Find groups of markers by k-means clustering.
    
    :param marker_trajectories:
    :param k: Number of clusters to fit markers to.
    :return: marker groups
    :rtype: list
    """
    raise NotImplementedError


def assign_labels_to_groups(marker_labels, groups):
    labeled_groups = {k: [marker_labels[idx] for idx in group] for k, group in enumerate(groups)}
    return labeled_groups


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="""Find groups of markers in C3D file which behave like rigid bodies.""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-v", "--ver", action='version', version='%(prog)s 0.1a')
    parser.add_argument("-m", "--method", type=str, choices=['spectral', 'k-means'],
                        default="spectral", help="Method to use for clustering.")
    parser.add_argument("-k", type=int, default=17, help="Desired number of groups.")
    parser.add_argument("input.c3d", type=str, help="C3D file (Intel format) with marker data.")
    args = vars(parser.parse_args())
    c3d_filepath = args['input.c3d']
    k_groups = args['k']
    cluster_method = args['method']

    labels, markers, conditionals = read_c3d_file(c3d_filepath, output_fps=30)
    
    if cluster_method == 'spectral':
        groups_indices = group_markers_stsc(markers,
                                            n_clusters=10,
                                            nth_frame=15,
                                            min_groups=k_groups,
                                            max_groups=k_groups)
    elif cluster_method == 'k-means':
        groups_indices = group_markers_kmeans(markers, k_groups)
    
    # Assign IDs back to labels.
    groups_labeled = assign_labels_to_groups(labels, groups_indices)
    print("Grouping of markers by labels:")
    for group_idx, marker_labels in groups_labeled.items():
        print("Group {}: {}".format(group_idx, ", ".join(marker_labels)))
    
    
    # Todo: Move to tests
    #ground_truth=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    '''
    # Next case.
    file_name = "arm-4-2-3_clean_120fps.c3d"
    groups2 = group_markers_stsc(c3d_filepath,
                               resample_fps=60,
                               n_clusters=60,
                               nth_frame=1,
                               min_groups=3,
                               max_groups=3,
                               ground_truth=[[0, 1, 2, 3], [4, 5], [6, 7, 8]])

    # Takes a long time (> 0.5h).
    # Next case: full body
    file_name = "fullbody_44Markers_clean_120fps.c3d"
    print("\nProcessing file:", file_name)
    c3d_filepath = os.path.join(data_path, file_name)
    groups_fullbody = group_markers_stsc(c3d_filepath,
                                       resample_fps=24,
                                       n_clusters=4,
                                       nth_frame=12,
                                       min_groups=15,
                                       max_groups=20,
                                       ground_truth=[[0, 1, 2, 3],  # head
                                                     [4, 5, 6, 7],  # torso
                                                     [8, 9, 34, 36],  # hips
                                                     [11, 12, 14], [10, 13, 21],  # shoulders
                                                     [22], [15],  # upper arms
                                                     [16, 17], [23, 24],  # lower arms
                                                     [18, 19, 20], [25, 26, 27],  # hands
                                                     [33, 35], [37, 43],  # upper legs
                                                     [31, 32], [38, 39],  # lower legs
                                                     [28, 29, 30], [40, 41, 42],  # feet
                                                     ])
    '''
