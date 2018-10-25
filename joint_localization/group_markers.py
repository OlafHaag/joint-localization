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
import warnings
from itertools import combinations
from multiprocessing import Pool, freeze_support
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster.k_means_ import KMeans
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


def read_c3d_file(file_path):
    """Read INTEL format C3D file and return a subsample of marker positions and conditionals.
    Also prints information about the file content.

    :param file_path: Path to C3D file.
    :type file_path: str
    :return: marker data: {'marker_labels': list, 'trajectories': shape(frames, markers, coordinates),
                           'conditionals': shape(frames, conditionals), 'frame_rate': int}
    :rtype: dict
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
    
    data = {'marker_labels': marker_labels,
            'trajectories': pos_array,
            'conditionals': cond_array,
            'frame_rate': fps,
            }
    return data
    
# Todo: handle missing/bad data.
# The data for a given marker typically contains large errors
# just before the system loses track of that marker and for a
# short period after the marker is rediscovered. To eliminate
# problems with “ghost” markers and erroneous position data
# during those periods, the first few frames are trimmed off
# the beginning and end of each marker’s data segment, and any
# marker with a maximum number of consecutive frames less than
# one half second is ignored.


def subsample_marker_data(marker_data, delta=1, random_offset=0, frames=None):
    """Returns a subset of the marker data and the frames used to sample.
    The default parameter values do not subsample.
    
    :param marker_data: Array of marker data (trajectories or conditionals).
    :type marker_data: numpy.ndarray
    :param delta: Sample every nth frame.
    :type delta: int
    :param random_offset: maximum offset forward or backward to regular frame sampling.
    :type random_offset: int
    :param frames: Frame indices to use for subsample. If not None, delta and random_offset are ignored.
    :type frames: list|numpy.ndarray
    :return: Tuple(subsample of marker data, indices of frames used).
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    if delta <= 0:
        raise ValueError("Delta value for frames subsampling must be > 0!")
    elif delta <= random_offset:
        raise ValueError("Delta value for frames subsampling must be greater than random offset.\n"
                         "Otherwise frame order might get jumbled.")
    n_frames = marker_data.shape[0]
    if not frames:
        frames_indices = np.arange(0, n_frames, delta)
    else:
        frames_indices = frames
    if len(frames_indices) > n_frames:
        raise ValueError("Number of frames to subsample must be less than number of frames!")
    
    if (random_offset == 0) or frames:
        pass  # Do not change frames_indices.
    else:
        randomize = lambda x: x + np.random.randint(-random_offset, random_offset)
        frames_indices = np.fromiter((randomize(x) for x in frames_indices), frames_indices.dtype)
        # Make sure indices are valid.
        frames_indices[np.where(frames_indices < 0)] = 0  # No negative frames.
        # Do not overshoot.
        frames_indices[np.where(frames_indices >= n_frames)] = n_frames - np.random.randint(1, random_offset)
    subset = marker_data[frames_indices]
    return subset, frames_indices


def get_distance_deviations(markers_sample):
    """Standard deviation in distance between marker pairs.
    Define a distance matrix, A, such that element A ij is the
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


def sum_distance_deviations(marker_indices, distance_matrix):
    """Sum up standard deviations of distances for given marker indices.
    To avoid penalizing large marker groups, the standard deviation within
    a group is normalized by the number of markers in the group.
    
    :param marker_indices: marker indices to sum pairwise
    :type marker_indices: list
    :param distance_matrix: NxN matrix containing standard deviations of pairwise distances.
    :type distance_matrix: numpy.ndarray
    :return: Wighted sum of standard deviations for pairwise distances.
    :rtype: float
    """
    pairs = list(combinations(marker_indices, 2))
    return distance_matrix[tuple(zip(*pairs))].sum() / len(marker_indices)


def get_affinity_matrix(dist_matrix, delta=4.0):
    """Transforms a distance matrix into an affinity matrix by applying the Gaussian (RBF, heat) kernel.
    
    :param dist_matrix: Distance matrix, for which 0 means identical elements,
    and high values means very dissimilar elements.
    :type dist_matrix: numpy.ndarray
    :param delta: Width of the Gaussian kernel.
    :type delta: float
    :return: Affinity matrix, for which 1 means identical elements,
    and low values means very dissimilar elements.
    :rtype: numpy.ndarray
    """
    affinity = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
    return affinity


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
    :param data: A 2D numpy array of shape (N,M)
    :param row_labels: A list or array of length N with the labels
                     for the rows
    :param col_labels: A list or array of length M with the labels
                     for the columns
    Optional arguments:
    :param ax: A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
    :param cbar_kw: A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
    :param cbarlabel: The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = add_colorbar(im, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=None, threshold=None, **textkw):
    """A function to annotate a heatmap.

    Arguments:
    :param im: The AxesImage to be labeled.
    Optional arguments:
    :param data: Data used to annotate. If None, the image's data is used.
    :param valfmt:  The format of the annotations inside the heatmap.
                    This should either use the string format method, e.g.
                    "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
    :param textcolors:  A list or array of two color specifications. The first is
                        used for values below a threshold, the second for those above.
    :param threshold:   Value in data units according to which the colors from
                        textcolors are applied. If None (the default) uses the
                        middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if textcolors is None:
        textcolors = ["black", "white"]
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
    

def show_matrix_plot(matrix, labels, matrix_type='affinity'):
    """Plot an affinity or distance matrix.
    
    :param matrix: NxN Matrix with markers' pairwise distance deviation/affinity.
    :param labels: marker labels
    :type labels: list
    :param matrix_type: Label of color bar and plot title.
    :type matrix_type: str
    """
    size = tuple((int(0.5*x) for x in matrix.shape))
    fig, ax = plt.subplots(figsize=size, tight_layout=True)
    im, cbar = heatmap(matrix, labels, labels, ax=ax,
                       cmap="tab20b", cbarlabel=matrix_type)
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    ax.set_title("{} Matrix".format(matrix_type.title()))
    plt.show()
    
    
def compute_stsc_cluster(marker_trajectories, sample_nth_frame=15, rnd_frame_offset=5, min_groups=2, max_groups=20):
    """Segments the markers into rigid body groups.
    
    :param marker_trajectories: Marker trajectories
    :param sample_nth_frame: sample rate for subsampling the marker data (for computational efficiency).
    :param rnd_frame_offset: semi-randomize regular sampling rate by +/- random max offset to counter periodic errors.
    :param min_groups: Minimum number of groups to consider.
    :param max_groups: Maximum number of groups to consider.
    :return: {'groups': list of lists with marker indices, 'sum_dev': float}
    :rtype: dict
    """
    assert sample_nth_frame < len(marker_trajectories), "Sampling rate exceeds number of frames! No markers to sample."
    # Samples are selected over all possible frames at intervals
    # of one half second, plus or minus a few frames.
    # This jitter ensures that any periodic errors do not affect the segmentation.
    marker_subset, _ = subsample_marker_data(marker_trajectories, sample_nth_frame, rnd_frame_offset)
    dist_mat = get_distance_deviations(marker_subset)
    # The distance deviations need to be sensibly inverted for the algorithm,
    # so that low values are closer to 1 and high values close to zero.
    affinity = get_affinity_matrix(dist_mat, delta=4.0)  # Adjust delta
    #print("Commencing self tuning spectral clustering. min: {}, max:{}".format(min_groups, max_groups))
    groups = self_tuning_spectral_clustering(affinity, min_n_cluster=min_groups, max_n_cluster=max_groups)  # FixMe: spectral clustering is way too slow.
    # Sum standard deviation of distances over all marker pairs in each group.
    sum_deviations = np.array([sum_distance_deviations(group, dist_mat) for group in groups]).sum()
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
    
    return marker_groups
    
    
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


def n_groups_sanity_check(group_func):
    def group_func_check(*args, **kwargs):
        try:
            num_markers = args[0].shape[1]
        except IndexError:
            raise ValueError("Clustering function must take marker trajectories as the first positional argument.")
        try:
            num_groups = args[1]
        except IndexError:
            raise ValueError("Clustering function must take number of groups as the second positional argument.")
        if num_groups == 1:
            print("WARNING: Desired groups is 1. All markers are in one and the same group.")
            groups = [np.arange(num_markers).tolist()]
        elif num_groups >= num_markers:
            print("WARNING: Desired groups are greater or equal number of markers. Each marker is a group on its own.")
            groups = [[_] for _ in np.arange(num_markers)]
        elif num_groups >= 2:
            groups = group_func(*args, **kwargs)
        else:
            raise ValueError("Number of desired groups must be greater than 0.")
        return groups
    return group_func_check


@n_groups_sanity_check
def group_markers_kmeans(marker_trajectories, n_groups):
    """Find groups of markers by k-means clustering.
    
    :param marker_trajectories:
    :param n_groups: Number of clusters to fit markers to.
    :return: marker groups
    :rtype: list
    """
    # FixMe: Doesn't work well on fullbody. Try other metric?
    dist_matrix = get_distance_deviations(marker_trajectories)
    affinity = get_affinity_matrix(dist_matrix, delta=4.0)
    kmeans = KMeans(n_clusters=n_groups).fit(affinity)
    print("Ran k-means clustering with {} iterations.\n"
          "Sum of squared distances of samples to their closest cluster center: {}".format(kmeans.n_iter_,
                                                                                           kmeans.inertia_))
    group_ids = kmeans.labels_
    marker_indices = group_ids.argsort()
    groups = [marker_indices[group_ids == group].tolist() for group in np.unique(group_ids)]
    return groups


def assign_labels_to_groups(marker_labels, groups):
    labeled_groups = {k: [marker_labels[idx] for idx in group] for k, group in enumerate(groups)}
    return labeled_groups


def process_c3d_file(file_path, cluster_method="spectral", num_groups=17, sample_frame_rate=None):
    # Do we support the method?
    if cluster_method not in ["spectral", "k-means"]:
        raise NotImplementedError("Chosen method '{}' clustering not supported.".format(cluster_method))
    # Gather data.
    data = read_c3d_file(file_path)
    labels = data['marker_labels']
    trajectories = data['trajectories']
    #conditionals = data['conditionals']
    c3d_fps = data['frame_rate']
    # Subsample.
    if sample_frame_rate is None:
        sample_frame_rate = 0
    if sample_frame_rate < 0:
        raise ValueError("Subsample frame rate must be greater than Zero!")
    elif sample_frame_rate > 0:
        nth_frame = int(c3d_fps / sample_frame_rate)
        trajectories, frame_indices = subsample_marker_data(trajectories, delta=nth_frame)
        #conditionals, frame_indices = subsample_marker_data(conditionals, frames=frame_indices)
    
    if cluster_method == 'spectral':
        groups_indices = group_markers_stsc(trajectories,
                                            n_clusters=10,
                                            nth_frame=15,  # FixMe: This should be frame rate independent.
                                            min_groups=num_groups,
                                            max_groups=num_groups)  # Todo: Support min/max parameter.
    elif cluster_method == 'k-means':
        groups_indices = group_markers_kmeans(trajectories, num_groups)
    
    # Assign IDs back to labels.
    groups_labeled = assign_labels_to_groups(labels, groups_indices)
    return groups_labeled
    
    
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
    parser.add_argument("-s", "--subsample-rate", type=int, default=0, help="Sub-sample data to a lower frame rate.\n"
                                                                            "0 to disable subsampling, "
                                                                            "or enter desired frame rate.")
    parser.add_argument("input.c3d", type=str, help="C3D file (Intel format) with marker data.")
    args = vars(parser.parse_args())
    c3d_filepath = args['input.c3d']
    num_groups = args['k']
    cluster_method = args['method']
    subsample_frame_rate = args['subsample_rate']

    groups = process_c3d_file(c3d_filepath,
                              cluster_method=cluster_method,
                              num_groups=num_groups,
                              sample_frame_rate=subsample_frame_rate)
    print("Grouping of markers by labels (using {} clustering):".format(cluster_method))
    for group_idx, marker_labels in groups.items():
        print("Group {}: {}".format(group_idx, ", ".join(marker_labels)))
