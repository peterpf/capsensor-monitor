import pytest
import numpy as np
from processor_utils import AnomalyCluster, track_clusters

@pytest.mark.parametrize("prev_clusters,next_clusters, expected_cluster_labels", [
    pytest.param(
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
        ],
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
        ],
        [1],
        id="does not change cluster label on same clusters"
    ),
    pytest.param(
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
            AnomalyCluster(
                id=2,
                sensor_cell_indezes=np.array([4, 5, 6]),
                avg_load=5
            ),
        ],
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([4, 5, 6]),
                avg_load=5
            ),
            AnomalyCluster(
                id=2,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
       ],
        [1, 2],
        id="correctly assigns the cluster label of the highest similarity"
    ),
    pytest.param(
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([]),
                avg_load=5
            ),
            AnomalyCluster(
                id=2,
                sensor_cell_indezes=np.array([]),
                avg_load=5
            ),
            AnomalyCluster(
                id=3,
                sensor_cell_indezes=np.array([]),
                avg_load=5
            ),
            AnomalyCluster(
                id=4,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
            AnomalyCluster(
                id=5,
                sensor_cell_indezes=np.array([4, 5, 6]),
                avg_load=5
            ),
        ],
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
            AnomalyCluster(
                id=2,
                sensor_cell_indezes=np.array([4, 5, 6]),
                avg_load=5
            ),
       ],
        [1, 2],
        id="uses cluster labels from previous clusters"
    ),
    pytest.param(
        [
            AnomalyCluster(
                id=2,
                sensor_cell_indezes=np.array([4, 5, 6]),
                avg_load=5
            ),
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([1, 2, 3]),
                avg_load=5
            ),
        ],
        [
            AnomalyCluster(
                id=1,
                sensor_cell_indezes=np.array([4, 5, 6]),
                avg_load=5
            ),
        ],
        [1],
        id="correctly assigns the cluster label of the highest similarity"
    ),
])
def test_track_cluster(prev_clusters: list[AnomalyCluster], next_clusters: list[AnomalyCluster], expected_cluster_labels: list):
    # Act
    track_clusters(prev_clusters, next_clusters)

    for idx, next_cluster in enumerate(next_clusters):
        assert next_clusters[idx].id == expected_cluster_labels[idx]
