import time
import numpy as np
from processor import Processor
from processor_utils import AnomalyCluster
from dataclasses import dataclass
import collections
from exceptions import PipelineException


@dataclass
class PipelineResult:
    """Holds the result of a pipeline run"""

    raw_data: np.array
    """The raw data."""

    filtered_data: np.array
    """Contains the filtered (output) data array."""

    scaled_data: np.ndarray
    """Data that was filtered and then scaled with a MinMaxScaler."""

    binary_data: np.ndarray
    """Data that was filtered and scaled, then converted to a binary image."""

    anomaly_clusters: list[AnomalyCluster]
    """List of anomaly clusters which may reflect contact points on the sensor."""

    anomaly_cluster_labels: np.array
    """Holds the cluster labels for every sensor cell"""

    processing_duration: int
    """Time (in nanoseconds) it took to process the sample."""


class Pipeline:
    """Preprocessing pipeline. The constructor takes the following arguments:

    :param processor: the processor instance
    :type processor: Processor
    :param sensor_shape: the sensor shape given as a tuple of (nrows, ncols), defaults to (12, 21)
    :type sensor_shape: tuple[int, int]
    :param buffer_size: maximum number of samples to store in the buffer, defaults to 10
    :type buffer_size: int
    :param num_skip_samples: The number of samples to skip that may be unstable readings, default to 5
    :type num_skip_samples: int
    """

    sensor_shape: tuple
    """Defines the sensor shape with (nrows, ncols)."""

    _buffer_size: int
    """Defines the number of samples stored in the buffered_samples array."""

    _num_skip_samples: int
    """Number of samples to skip at the beginnging to ensure stable measurements."""

    _sample_buffer: collections.deque
    """Contains samples which are used to calculate running-metrics (e.g. running average)
    Every entry in the list is a single observation of the sensor values,
    a one-dimensional array with nrows * ncols entries.
    """

    _processor: Processor
    """The sample processor."""

    _processed_sample_count: int = 0
    """Counts the number of processed samples, required for the sample-skipping feature."""

    def __init__(self, processor: Processor, sensor_shape: tuple, buffer_size: int = 10, num_skip_samples: int = 5):
        self._processor = processor
        self.sensor_shape = sensor_shape
        self._buffer_size = buffer_size
        self._num_skip_samples = num_skip_samples
        self._sample_buffer = collections.deque(maxlen=self._buffer_size)

    def process(self, data: np.ndarray) -> PipelineResult:
        """Runs the data processing task:

        - skip samples at the beginning to ensure stabilized measurements
        - buffer samples and compute a baseline for noise-removal
        - preprocess the samples to plot a heatmap
        - perform interaction detection

        :param data: two-dimensional data array of dimension (nrows, ncols) that is processed
        :type data: np.ndarray
        :return: PipelineResult
        """

        time_start = time.perf_counter_ns()
        self._processed_sample_count += 1

        # Wait until measurements have stabilized, skip data processing
        if self._processed_sample_count <= self._num_skip_samples:
            raise PipelineException(f"skipping sample {self._processed_sample_count}/{self._num_skip_samples}")

        # update baseline measurements when necessary, must be placed BEFORE adding samples to buffer.
        if len(self._sample_buffer) + 1 == self._buffer_size:
            self._processor.update_baseline(list(self._sample_buffer))

        # Register sample in buffer
        self._sample_buffer.append(data)

        # Verify that baseline is ready to be used
        if not self._processor.is_baseline_computed():
            raise PipelineException(f"not enough samples for baseline: {len(self._sample_buffer)}/{self._buffer_size}")

        processor_result = self._processor.process(data)
        self._prev_processor_result = processor_result

        time_end = time.perf_counter_ns()
        duration = time_end - time_start

        pipeline_result = PipelineResult(
            raw_data=data,
            filtered_data=processor_result.filtered_data,
            scaled_data=processor_result.scaled_data,
            binary_data=processor_result.binary_data,
            anomaly_clusters=processor_result.anomaly_clusters,
            anomaly_cluster_labels=processor_result.interaction_cluster_labels,
            processing_duration=duration,
        )

        return pipeline_result
