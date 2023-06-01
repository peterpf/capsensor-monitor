import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from processor_utils import AnomalyCluster
import processor_utils
from exceptions import ProcessorException


@dataclass
class ProcessorResult:
    """Results of a single processor run"""

    filtered_data: np.array
    """Data that has been filtered and preprocessed for further analysis"""

    interaction_cluster_labels: np.array
    """Holds the labels for each sensor cell clustered by anomalic regions."""

    anomaly_clusters: list[AnomalyCluster]
    """Holds the anomalic regions bundled in clusters"""


class Processor:
    """Preprocesses the sensor data and runs interaction detection on it:

    - calculates baseline for no-stress data (data with no interaction, just noise)
    - applies a threshold filter that sets everything below the threshold to 0


    :param sensor_shape: Shape of the sensor
    :type sensor_shape: tuple[int, int]
    :param sensitivity: Sensitivity of the threshold filter: mean + sensitivity * var(data)
    :type sensitivity: float
    """

    _sensor_shape: tuple
    """Dimension of the sensor given in (#rows, #cols)."""

    _sensitivity: float = 5.0
    """The sensitivity regarding noise filtering and is directly proportional to the aggressivness of filtering,
    a higher value means more noise removal which also affects the target signal.
    """

    _baseline_mean: np.array = None
    """The baseline mean calculated from the buffered samples.
    The array has nrows*ncols entries, where each entry corresponds to a sensor cell.
    """

    _baseline_variance: np.array = None
    """The baseline variance calculated from the buffered samples.
    The array has nrows*ncols entries, where each entry corresponds to a sensor cell.
    """

    _prev_anomaly_clusters: list[AnomalyCluster] = []
    """The last known processor result"""

    def __init__(self, sensor_shape: tuple, sensitivity: float):
        self._sensor_shape = sensor_shape
        self._sensitivity = sensitivity

    def update_baseline(self, samples: list[np.array]):
        """Calculates baseline measurements on the given samples.
        The samples-array has the following form:

        ```
        [
            [r1c1, r1c2, ...],
            [r1c1, r1c2, ...],
        ]
        ```

        The metrics are calculated over the first axis, meaning that the functions apply column-wise -- outputing a nrows*ncols array,
        e.g. mean over all `r1c1` values, mean over all `r1c2`, ...

        :param samples: list of one-dimensional sample arrays (of size nrows*ncols) to calculate the baseline from
        :type samples: list[np.array]
        """

        buffer_array = np.asarray(samples)
        self._baseline_mean = np.apply_over_axes(np.nanmean, buffer_array, 0)[0]
        self._baseline_variance = np.apply_over_axes(np.nanvar, buffer_array, 0)[0]

    def is_baseline_computed(self) -> bool:
        """Returns true if all baseline features are calculated.

        :return: boolean indicating whether baseline has been fully calculated
        """
        return self._baseline_mean is not None and self._baseline_variance is not None

    def _adjust_data_to_baseline(self, data, baseline_mean, baseline_var: np.array) -> np.array:
        """Removes noise by setting values in the data array to zero if they fall below the threshold = mean + sensitivity * var.
        The data is baseline-adjusted and should be closer to zero.
        """
        threshold = baseline_mean + self._sensitivity * baseline_var
        # Retrieve a boolean mask to find indeces where data points are below the threshold
        below_threshold_mask = data < threshold
        # Apply mask and set "true" positions to 0
        grounded_data = data.copy()
        grounded_data[below_threshold_mask] = 0

        return grounded_data

    def process(self, data: np.array) -> ProcessorResult:
        """Process the data by removing the baseline and detecting interactions.

        :param data: one dimensional data array of length (nrows*ncols)
        :type data: np.array
        :return: ProcessorResult
        """

        if not self.is_baseline_computed():
            raise ProcessorException("cannot process data, baseline not computed")

        # Adjust data to baseline (remove noise and level data to 0)
        baseline_adjusted_data = self._adjust_data_to_baseline(data, self._baseline_mean, self._baseline_variance)

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(baseline_adjusted_data.reshape(-1, 1)).flatten()

        cluster_labels = processor_utils.find_contact_points(self._sensor_shape, scaled_data)
        anomaly_clusters = processor_utils.collect_anomaly_clusters(baseline_adjusted_data, cluster_labels)

        # Track cluster movement and ideally assign the same labels.
        if len(self._prev_anomaly_clusters) > 0 and len(anomaly_clusters) > 0:
            processor_utils.track_clusters(self._prev_anomaly_clusters, anomaly_clusters)
        self._prev_anomaly_clusters = anomaly_clusters

        result = ProcessorResult(
            filtered_data=baseline_adjusted_data,
            interaction_cluster_labels=cluster_labels,
            anomaly_clusters=anomaly_clusters
        )

        return result
