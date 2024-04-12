import numpy as np
import collections
import serial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import Animation
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.gridspec import GridSpec
import utils as utils
from input_parser import decode_and_parse
import exceptions
import plot_utils

from config import AppConfig

from pipeline import Pipeline
from processor import Processor
from serial_iterator import SerialIterator
from file_writer import FileWriter, PerformanceData

logger = utils.configure_logger(__name__)

matplotlib.use("TkAgg")


class Application:
    """The main application. Sets up all relevant resources such as logging, serial connection, and visualization.
    A fileWriter to log raw data is instantiated when logging is enabled via the AppConfig.

    :param config: The application configuration
    :type config: AppConfig
    """

    _config: AppConfig
    """Application configuration."""

    _filewriter: FileWriter = None
    """If data logging is enabled, store it in files."""

    _default_data_matrix: np.ndarray
    """The default data matrix which is displayed when no data is available."""

    _fig: Figure
    """The sole figure used in this application to visualize the data."""

    _axis: list[Axis]
    """List of axis used to plot animations, the first entry is the main axis, subsequent axis represent the top-k measures."""

    _animation: Animation
    """Animation-reference to keep it alive throughout the application lifecycle."""

    _serial_iterator: SerialIterator
    """Wraps around the serial object to iterate over readline."""

    _serial_obj: serial.Serial
    """Serial object."""

    _pipeline: Pipeline

    _data_queue: collections.deque
    """Holds the last n received samples (filtered data)."""

    def __init__(self, config: AppConfig):
        if config.log_data:
            self._filewriter = FileWriter(config.data_directory)
            self._filewriter.write_configuration(config)

        self._config = config
        sensor_shape = self._config.sensor_shape
        self._default_data_matrix = np.zeros(sensor_shape)
        self._connect_serial()
        self._init_pipeline()
        self._init_data_queues()

    def _init_pipeline(self):
        """Initialze the processing pipeline for the given sensor shape,
        sensitivity, and buffer size."""
        processor = Processor(sensor_shape=self._config.sensor_shape, sensitivity=self._config.sensitivity)
        self._pipeline = Pipeline(processor=processor, sensor_shape=self._config.sensor_shape, buffer_size=self._config.buffer_size)

    def _init_data_queues(self):
        """Initialzes the sample buffer to track received samples over time."""
        buffer_size = self._config.top_k_buffer_size
        empty_sample = np.tile(np.array([np.nan]), (buffer_size, self._config.nrows * self._config.ncols))
        self._data_queue = collections.deque(empty_sample, buffer_size)

    def _connect_serial(self):
        """Establish a connection to the serial device as specified in the configuration."""
        serial_obj = serial.Serial(self._config.serial_port, self._config.baudrate, timeout=self._config.connection_timeout)
        self._serial_obj = serial_obj
        self._serial_iterator = SerialIterator(serial_obj)

    def _update_animation(self, serial_data, *fargs):
        """The main animation loop.

        :param serial_data: encoded byte data from the SerialIterator
        """

        ax_main = self._axis[0]
        try:
            # Parse data from command line into data matrix
            raw_data = decode_and_parse(serial_data, self._config.inverse_sensor_shape)

            # Process raw data
            pipeline_result = self._pipeline.process(raw_data)

            # Track resulting data over time
            self._data_queue.append(raw_data)

            # Generate a heatmap image from the processed data
            main_artists = plot_utils.draw_annotated_matplotlib_heatmap(ax_main, self._config.sensor_shape, pipeline_result)
            top_k_artists = plot_utils.draw_top_k_timeseries(np.array(list(self._data_queue)), self._axis[1:], pipeline_result)

            # Store data in logfile if enabled
            if self._filewriter is not None:
                performance_data = PerformanceData(
                    overall_processing_duration=pipeline_result.processing_duration,
                    num_clusters=len(pipeline_result.anomaly_clusters),
                )
                self._filewriter.write_performance_data(performance_data)
                self._filewriter.write_pipeline_result(pipeline_result)

            artists = [*main_artists, *top_k_artists]
            return artists

        except Exception as e:
            if isinstance(e, exceptions.InputParserException):
                # Ignore this exception
                logger.debug(f"ignored exception: {e}")
            elif isinstance(e, exceptions.PipelineException):
                # Ignore this exception
                logger.debug(f"ignored exception: {e}")
            else:
                raise e

        # Set animation to default values
        return plot_utils.draw_default_heatmap(ax_main, self._default_data_matrix)

    def run(self):
        """Run the application: starts the rendering loop."""

        self._fig = plt.figure(layout="constrained")
        gs = GridSpec(nrows=3, ncols=2, figure=self._fig, width_ratios=[0.3, 0.7])
        # Main plot
        ax_main = self._fig.add_subplot(gs[:, 0])

        # top-k plots
        ax_top1 = self._fig.add_subplot(gs[0, 1])
        ax_top2 = self._fig.add_subplot(gs[1, 1])
        ax_top3 = self._fig.add_subplot(gs[2, 1])
        self._axis = [ax_main, ax_top1, ax_top2, ax_top3]
        # Initialize axis bounds
        for i, ax in enumerate(self._axis[1:]):
            ax.set_title(f"Cluster {i+1}")
            ax.set_xlim(self._config.top_k_buffer_size, 0)
            ax.text(max(ax.get_xlim()) / 2, 0.5, "No data", va="center", ha="center")
            ax.set_yticklabels([])

        # Set up the animation,
        # use the SerialIterator to feed the serial lines into the animation function.
        self._animation = animation.FuncAnimation(
            self._fig,
            func=self._update_animation,
            frames=self._serial_iterator,  # Provide frame data via SerialIterator
            interval=0,  # We don't want any artificial delay between frames, set to 0.
            blit=True,  # Optimize rendering of animation
            cache_frame_data=False,
        )

        plt.show()

    def teardown(self):
        """Clean up all resources."""

        # Close serial connection
        if self._serial_obj is not None:
            self._serial_obj.close()

        # Stop filewriter
        if self._filewriter is not None:
            self._filewriter.close()
