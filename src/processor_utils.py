import math
import skimage
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class AnomalyCluster:
    id: int
    """Cluster ID (label)"""

    sensor_cell_indezes: np.array
    """Contains the array of affected sensor cells identified by their positional index in the raw data array."""

    avg_load: float
    """Average load of the sensor cells."""

def find_contact_points(sensor_shape: tuple, data: np.array, background_label: int = 0) -> np.array:
    """Identifies anomalies in the data and clusters them, returns the cluster labels as an array.
        :param sensor_shape: tuple describing the sensor's shape.
        :param data: one-dimensional array containing the image data.
        :param background_label: the label that is assigned to background pixels.
        :return: one-dimensional array with the corresponding labess of the input data.
    """

    # Apply a threshold filter to convert the continuous data into binary image data (either 0 for background, or 1).
    threshold = skimage.filters.threshold_otsu(data)
    # Check that the threshold is not 0, it would catch all the noise
    if math.isclose(threshold, 0):
        return np.repeat(background_label, np.multiply(*sensor_shape))
    binary_image = data >= threshold


    # Apply connected components analysis (CCA) on boolean image.
    # The algorithm takes a two-dimensional array (the nrows x ncols matrix) as input.
    data_matrix = binary_image.reshape(sensor_shape)
    labels, _ = skimage.measure.label(
        data_matrix,
        background=background_label,
        return_num=True,
        connectivity=None # Full-connectivity (8-connectivity)
    )
    return labels.flatten()


def collect_anomaly_clusters(data: np.array, cluster_labels: np.array, background_cluster_label = 0) -> list[AnomalyCluster]:
    """Returns the sensor cells in bundles of anomaly clusters.

    :param data: one-dimensional data array containing the measurements.
    :param cluster_labels: one-dimensional array with the same length as the data array, containing cluster memberships.
    :param background_cluster_label: the label assigned to the background, defaults to 0.
    """
    clusters = list()
    unique_cluster_ids = np.unique(cluster_labels)
    # Do not cluster background cells
    cluster_ids_wo_background = unique_cluster_ids[unique_cluster_ids != background_cluster_label]
    for cluster_id in cluster_ids_wo_background:
        affected_sensor_cell_idx = np.where(cluster_labels == cluster_id)[0]
        affected_sensor_cell_values = data[affected_sensor_cell_idx]
        new_cluster = AnomalyCluster(
            id=cluster_id,
            sensor_cell_indezes=affected_sensor_cell_idx,
            avg_load=np.nanmean(affected_sensor_cell_values)
        )
        clusters.append(new_cluster)
    return clusters

def jaccard_index(a: list|set, b: list|set) -> float:
    """Calculates the Jaccard similarity score for the given sets.

    :param a: first set of values
    :param b: second set of values
    :return: Jaccard similarity score, 0 if the sets are disjoint"""
    s1 = set(a)
    s2 = set(b)
    overlaping = len(s1.intersection(s2))
    union = len(s1.union(s2))

    if overlaping == 0 or union == 0:
        return 0
    return overlaping / union

def find_similar_cluster(cluster: AnomalyCluster, clusters: list[AnomalyCluster]) -> Optional[AnomalyCluster]:
    """Find the most similar cluster via the Jaccard similarity coefficient.

    :param cluster: Cluster to find a match for
    :type cluster: AnomalyCluster
    :param clusters: List of clusters to find a match in
    :type clusters: list[AnomalyCluster]
    :return: Cluster with the highest similarity score, otherwise None
    """
    similarity_scores = np.zeros(len(clusters))

    for i, other_cluster in enumerate(clusters):
        jaccard_score = jaccard_index(cluster.sensor_cell_indezes, other_cluster.sensor_cell_indezes)
        similarity_scores[i] = jaccard_score

    # Find maximum similarity cluster
    most_similar_cluster_idx = np.argmax(similarity_scores)
    highest_similarity_score = similarity_scores[most_similar_cluster_idx]
    if math.isclose(highest_similarity_score, 0):
        return None

    return clusters[most_similar_cluster_idx]


def get_cluster_labels(clusters: list[AnomalyCluster]) -> list:
    """Return the cluster IDs from the list of clusters.

    :param clusters: list of clusters
    :type clusters: list[AnomalyCluster]
    :return: list of cluster IDs
    """
    return list(map(lambda c: c.id, clusters))

def track_clusters(prev_clusters: list[AnomalyCluster], next_clusters: list[AnomalyCluster]):
    """Track moving clusters from one point in time to another by assigning similar clusters the same ID.
    This function iterates over the new (next) clusters and assigns the label of the best-matching previous cluster.
    If no best-matching previous cluster exists, the cluster gets a new label assigned,\
    continuing with the highest available label of the next clusters.
    Furthermore the function ensures that clusters are uniquely labeled in sequential order: 1, 2, 3, ....

    :param prev_clusters: list of clusters from the previous point in time
    :type prev_clusters: list[AnomalyCluster]
    :param next_clusters: list of clusters from the current point in time
    :type next_clusters: list[AnomalyCluster]
    """
    if len(prev_clusters) == 0 or len(next_clusters) == 0:
        return

    # Iterate over next clusters and assign best-matching label from previous cluster, or new label.
    cluster_labels = get_cluster_labels(next_clusters)
    max_cluster_label = np.max(cluster_labels)
    already_assigned_labels = []
    for idx, next_cluster in enumerate(next_clusters):
        similar_cluster = find_similar_cluster(next_cluster, prev_clusters)
        # Find the new cluster label
        new_cluster_label = None
        if similar_cluster is None:
            new_cluster_label = max_cluster_label
            max_cluster_label += 1
        else:
            new_cluster_label = similar_cluster.id

        # Ensure label has not been assigned before
        if new_cluster_label in already_assigned_labels:
            new_cluster_label += 1
            max_cluster_label = max(new_cluster_label, max_cluster_label)

        # Assign new cluster label
        next_cluster.id = new_cluster_label
        already_assigned_labels.append(new_cluster_label)

    # Ensure cluster labels start at `1` but retain sequential order
    # In some cases (e.g. a cluster splits into two) a offset could occur.
    next_clusters.sort(key = lambda c: c.id)
    for idx, cluster in enumerate(next_clusters):
        cluster.id = idx + 1
