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
# %% Imports
import os
import warnings
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
import c3d

from stsc import self_tuning_spectral_clustering


# %% C3D file
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
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
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
        print("Reading frames...")
        for i, points, _ in reader.read_frames():
            # pos_array[frame, marker], e.g. pos_array[:,11] all frames for 12th marker
            # Points are mirrored. Different coordinate system somehow.
            pos_array[i - first_frame, :, :] = np.vstack([-1.0 * points[:, 0], -1.0 * points[:, 2], -1.0 * points[:, 1]]).T
            cond_array[i - first_frame, :] = points[:, 3]
            if n_frames is not None and i - first_frame >= n_frames:
                break
        
        # There might be a lot of frames. To speed up optimization use only a subset.
        nth_frame = int(fps / output_fps)
        frames_indices = np.arange(0, n_frames, nth_frame)
        # scale = 0.001  # convert mm to m
        pos_subset = pos_array[frames_indices]
        cond_subset = cond_array[frames_indices]
        
        # Todo: handle missing/bad data
        # The data for a given marker typically contains large errors
        # just before the system loses track of that marker and for a
        # short period after the marker is rediscovered. To eliminate
        # problems with “ghost” markers and erroneous position data
        # during those periods, the first few frames are trimmed off
        # the beginning and end of each marker’s data segment, and any
        # marker with a maximum number of consecutive frames less than
        # one half second is ignored.
        return pos_subset, cond_subset
    
    
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


# %% Average marker pair distance.
def avg_marker_pair_distance(marker1, marker2) -> float:
    """Average distance between a pair of markers over all frames.

    :param marker1: marker trajectory
    :type marker1: np.array
    :param marker2: marker trajectory
    :type marker2: np.array
    :return: average distance
    :rtype: float
    """
    avg = np.linalg.norm(marker1 - marker2, axis=1).sum() / len(marker1)  # length of vectors equals number of frames.
    return avg


# %% Variance in distance for a marker pair.
def marker_pair_distance_variance(marker1, marker2) -> float:
    """Computes variance in marker-marker distance.

    :param marker1: marker trajectory
    :type marker1: np.array
    :param marker2: marker trajectory
    :type marker2: np.array
    :return: variance
    :rtype: float
    """
    sig = np.square(np.linalg.norm(marker1 - marker2, axis=1) - avg_marker_pair_distance(marker1, marker2)).sum()
    sig /= len(marker2)
    return sig


# %% Cost Matrix
def cost_matrix(markers_sample):
    """Standard deviation (squared) in distance between marker pairs.
    Define a cost matrix, A, such that element A ij is the
    standard deviation in distance between markers i and j for a
    particular sampling of frames.
    
    :param markers_sample: positions of markers over a sample of frames.
    :type markers_sample: numpy.ndarray
    :return: matrix with standard deviations for markers x markers
    :rtype: numpy.ndarray
    """
    # Make pairs of markers, no repetitions (AB, but not BA).
    n_markers = markers_sample.shape[1]
    marker_indices = [idx for idx in range(n_markers)]
    pairs = list(combinations(marker_indices, 2))
    #print("Computing variances in distances for {} marker pairs...".format(len(pairs)))
    # Create matrix with NxN
    matrix = np.empty((n_markers, n_markers))
    # Set each value for a marker paired with itself to zero.
    np.fill_diagonal(matrix, 0.0)
    # Compute variance for each pair.
    for pair in pairs:
        variance = marker_pair_distance_variance(markers_sample[:, pair[0]], markers_sample[:, pair[1]])
        matrix[pair, pair[::-1]] = variance
        #print("{} var={}, std={}".format(pair, variance, np.sqrt(variance)))
    return matrix


def sum_distance_deviations(group, cost_matrix):
    pairs = list(combinations(group, 2))
    # Sum up variances within the group.
    # To avoid penalizing large marker groups, the variance within
    # a group is normalized by the number of markers in the group.
    return cost_matrix[list(zip(*pairs))].sum() / len(group)


#%% Picking best groups configuration.
def compute_cluster(markers, sample_nth_frame=15, rnd_frame_offset=5, min_groups=2, max_groups=20):
    """Segements the markers into rigid body groups.
    
    :param markers: Marker trajectories
    :param sample_nth_frame: sample rate for subsampling the marker data (for computational efficiency).
    :param rnd_frame_offset: semi-randomize regular sampling rate by +/- random max offset to counter periodic errors.
    :param min_groups: Minimum number of groups to consider.
    :param max_groups: Maximum number of groups to consider.
    :return: {'groups': list of lists with marker indices, 'sum_dev': float}
    :rtype: dict
    """
    # Samples are selected over all possible frames at intervals
    # of one half second, plus or minus a few frames.
    # This jitter ensures that any periodic errors do not affect the segmentation.
    marker_subset = sample_marker_positions(markers, sample_nth_frame, rnd_frame_offset)
    costs = cost_matrix(marker_subset)
    # The costs need to be sensibly inverted, so that low costs are closer to 1 and high costs close to zero.
    affinity = costs.copy()
    np.fill_diagonal(affinity, 1.0)
    affinity = 1 / affinity
    groups = self_tuning_spectral_clustering(affinity, min_n_cluster=min_groups, max_n_cluster=max_groups)
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
    print("distances deviation sum:", std_sums[best_idx])
    return best_groups
    
    
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

    # Rather than using one sampling of frames, find marker groups
    # by clustering multiple times using several different samplings.
    n_samples = 10
    clusters = [compute_cluster(markers, min_groups=3, max_groups=3) for i in range(n_samples)]
    marker_groups = best_groups_from_clusters(clusters)
