import sys
import click
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Generator, Any, Tuple, Callable

import matplotlib
import matplotlib.pyplot as plt

import plot_utils as plot_utils
import processor_utils
from pipeline import PipelineResult
import utils as utils
from dataclasses import dataclass


logger = utils.configure_logger(__name__)

matplotlib.use("TkAgg")


@dataclass
class ParsedValues:
    """Holds the values for a line from the source file paths."""

    raw: np.ndarray
    filtered: np.ndarray
    scaled: np.ndarray
    binary: np.ndarray
    clustered: np.ndarray

@dataclass
class SourceFilepaths:
    """Holds the location of the log data source files,
    and provides an interface to read the values line-by-line."""

    raw: Path
    """Path to the raw data."""

    filtered: Path
    """Path to the filtered data."""

    scaled: Path
    """Path to the scaled data."""

    binary: Path
    """Path to the binary data."""

    cluster_labels: Path
    """Path to the cluster-label data."""


    def parse_lines(self, sensor_shape: Tuple[int]) -> Generator[ParsedValues, Any, Any]:
        """Parses the data from the given source files.
        Each source file must have the exact same lenght,
        and each entry must correspond to the same observation across all source files.
        """
        with (
            open(self.raw, "r") as raw_fh,
            open(self.filtered, "r") as filtered_fh,
            open(self.scaled, "r") as scaled_fh,
            open(self.binary, "r") as binary_fh,
            open(self.cluster_labels, "r") as clustered_fh,
        ):
            zipped = zip(raw_fh, filtered_fh, scaled_fh, binary_fh, clustered_fh, strict=False)
            for raw_text, filtered_text, scaled_text, binary_text, clustered_text in zipped:
                yield ParsedValues(
                    raw=_parse_text_to_array(raw_text, sensor_shape),
                    filtered=_parse_text_to_array(filtered_text, sensor_shape),
                    scaled=_parse_text_to_array(scaled_text, sensor_shape),
                    binary=_parse_text_to_array(binary_text, sensor_shape),
                    clustered=_parse_text_to_array(clustered_text, sensor_shape).astype(int),
                    )

class ProcessingStage(Enum):
    RAW = "raw"
    """Raw data."""

    FILTERED = "filtered"
    """Filtered data."""

    SCALED = "scaled"
    """Scaled data."""

    BINARY = "binary"
    """Binary data."""

    CLUSTERED = "clustered"
    """Clustered data."""


class DataViewer():
    _source_filepaths: SourceFilepaths
    """The file paths to the source files."""

    _max_frame_idx: int
    """The maximum number of frames."""

    _current_frame_idx: int
    """The current frame that is displayed. Defaults to 0."""

    _sensor_shape: Tuple[int]
    """The shape of the sensor given by a tuple."""

    _application_running: bool
    """Flag that indicates whether the application is running. Defaults to True, duh."""

    _animation_running: bool
    """A flag indicating whether the animation is running. Defaults to False."""

    _frame_delay: float
    """The delay in seconds between two consecutive frames. Defaults to 0.1 seconds."""

    def __init__(
            self,
            source_folder: Path,
            sensor_shape: Tuple[int],
            visualization_stage: ProcessingStage,
            start_frame_idx: int = 0,
            frame_delay: float = 0.1,
            is_static_image: bool = False
    ):
        self._sensor_shape = sensor_shape
        self._frame_delay = frame_delay
        self._application_running = True
        self._source_filepaths = _get_files_from_source_folder(source_folder)
        self._max_frame_idx = _get_num_lines(self._source_filepaths.raw)
        self._current_frame_idx = np.min([start_frame_idx, self._max_frame_idx])
        self._visualize_frame = self._get_frame_visualization_func(visualization_stage)
        self._init_ui(is_static_image)

        if not is_static_image:
            self._animation_running = True
            self._run_animation()
            input("press [enter] to close...")

    def _init_ui(self, is_static_image: bool):
        fig, axis = plt.subplots()
        self._fig = fig
        self._axis = axis
        # Set the very first frame.
        self._update_visual(self._current_frame_idx)
        # Adjust bottom margin to make space for the buttons.
        fig.subplots_adjust(bottom=0.2)

        # Add buttons to interact with the plot.
        # axis_btn_next = fig.add_axes([0.3, 0.05, 0.1, 0.075])
        # axis_btn_prev = fig.add_axes([0.5, 0.05, 0.1, 0.075])
        # axis_btn_playpause = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        # btn_next = Button(axis_btn_next, "Next")
        # btn_next.on_clicked(self.on_button_next)
        # btn_prev = Button(axis_btn_prev, "Prev")
        # btn_prev.on_clicked(self.on_button_prev)
        # btn_playpause = Button(axis_btn_playpause, "Play/Pause")
        # btn_playpause.on_clicked(self.on_button_playpause)

        # Set up callback event when the matplotlib window is closed -- shutdown app.
        fig.canvas.mpl_connect("close_event", self._on_window_close)

        plt.tight_layout()
        plt.show(block=is_static_image)

    def _get_frame_visualization_func(self, stage: ProcessingStage) -> Callable:
        if stage == ProcessingStage.RAW:
            def func(axis: Any, data: ParsedValues, sensor_shape: Any):
                plot_utils.draw_raw_signal_heatmap(axis, sensor_shape, data.raw)
        elif stage == ProcessingStage.FILTERED:
            def func(axis: Any, data: ParsedValues, sensor_shape: Any):
                plot_utils.draw_raw_signal_heatmap(axis, sensor_shape, data.filtered)
        elif stage == ProcessingStage.SCALED:
            def func(axis: Any, data: ParsedValues, sensor_shape: Any):
                plot_utils.draw_raw_signal_heatmap(axis, sensor_shape, data.scaled)
        elif stage == ProcessingStage.BINARY:
            def func(axis: Any, data: ParsedValues, sensor_shape: Any):
                plot_utils.draw_raw_signal_heatmap(axis, sensor_shape, data.binary)
        elif stage == ProcessingStage.CLUSTERED:
            func = _visualize_frame_stage_clustered
        else:
            raise Exception("Unsupported stage selected: %s", stage.name)
        return func

    def start_animation(self):
        """Start the animation."""
        self._animation_running = True

    def stop_animation(self):
        """Stop the animation."""
        self._animation_running = False

    def _on_window_close(self, *args):
        """Event handler when the matplotlib window is closed,
        handles graceful shutdown of application.
        """
        logger.info("Shutting down application...")
        self.stop_animation()
        self._application_running = False
        sys.exit(0)


    def _run_animation(self):
        """Iterater over the frames as long as `_animation_running` is set to `True`."""
        while self._current_frame_idx < self._max_frame_idx and self._animation_running:
            self._update_visual(self._current_frame_idx)
            plt.pause(self._frame_delay)
            self._current_frame_idx += 1

    def on_button_next(self, *args):
        """Goes to the next frame."""
        self._current_frame_idx += 1
        self._update_visual(self._current_frame_idx)

    def on_button_prev(self, *args):
        """Goes to the previous frame."""
        self._current_frame_idx -= 1
        self._update_visual(self._current_frame_idx)

    def on_button_playpause(self, *args):
        """Starts or stops the animation."""
        self._animation_running = not self._animation_running
        logger.info("Animation running: %s", self._animation_running)

    def _update_visual(self, frame_idx: int):
        """Update the visualization with the current frame."""
        data = self._get_frame_data(frame_idx)
        self._visualize_frame(self._axis, data, self._sensor_shape)
        self._fig.canvas.manager.set_window_title(f"Frame: {self._current_frame_idx}/{self._max_frame_idx - 1}")

    def _get_frame_data(self, frame_idx: int) -> ParsedValues:
        """Get the data for a given frame index."""
        data_parser_generator = self._source_filepaths.parse_lines(self._sensor_shape)
        for idx, data in enumerate(data_parser_generator):
            if idx == frame_idx:
                return data

@click.command()
@click.argument("source", type=click.Path(exists=True))  # The path to the source folder containing the log files.
@click.option("--start-frame", default=0, help="The start frame index.")
@click.option("--snapshot", is_flag=True, help="Whether to view the frame as a static image.")
@click.option("--stage", default=ProcessingStage.CLUSTERED.name,
              type=click.Choice([item.name for item in ProcessingStage]),
              help="Set which preprocessing stage to visualize.")
@click.option("--sensor-shape", nargs=2, default=(6, 6), help="The shape of the sensor.")
def cli(source: str, start_frame: int, snapshot: bool, stage: str, sensor_shape: Tuple[int]):
    source_filepath = Path(source)
    source_filepaths = _get_files_from_source_folder(source_filepath)
    _get_num_lines(source_filepaths.raw)
    processing_stage = ProcessingStage[stage]

    logger.info("Visualizing data in processing stage: %s", processing_stage)
    DataViewer(source_filepath, sensor_shape, processing_stage, start_frame_idx = start_frame, is_static_image=snapshot)


def _get_files_from_source_folder(source: Path) -> SourceFilepaths:
    """Return the filepaths to relevant data files:
    - raw.csv
    - filtered.csv
    - scaled.csv
    - binary.csv
    - cluster_labels.csv
    """

    return SourceFilepaths(
        raw=source / "raw.csv",
        filtered=source / "filtered.csv",
        scaled=source / "scaled.csv",
        binary=source / "binary.csv",
        cluster_labels=source / "cluster_labels.csv",
    )


def _visualize_frame_stage_clustered(axis: any, data: ParsedValues, sensor_shape: Tuple[int]):
    """Visualizes a single frame for the given data and sensor shape."""
    anomaly_clusters = processor_utils.collect_anomaly_clusters(data.filtered, data.clustered)
    pipeline_result = PipelineResult(
        raw_data=data.raw,
        filtered_data=data.filtered,
        scaled_data=data.scaled,
        binary_data=data.binary,
        anomaly_clusters=anomaly_clusters,
        anomaly_cluster_labels=data.clustered,
        processing_duration=0,
    )
    plot_utils.draw_annotated_matplotlib_heatmap(axis, sensor_shape, pipeline_result)


# TODO: Make it clear that flip=True is the default value,
# this should actually happen in the data acquisition system via a configuration parameter.
def _parse_text_to_array(
        raw_text: str,
        sensor_shape: Tuple[int],
        flip: bool = True) -> np.array:
    """Converts the raw text to a numpy array with the dimensions of `sensor_shape`.
    When `flip=True` it will reverse the array,
    this should be set when using the soft-sensor integration to produce correctly oriented images."""
    text = raw_text.rstrip("\n")
    values = np.array(text.split(","))
    casted_values = values.astype(float)
    if flip:
        return np.flip(casted_values)
    return casted_values


def _get_num_lines(file_path: Path) -> int:
    """Counts the number of lines in a file.

    Args:
        file_path: The path to the file.

    Returns:
        The number of lines in the file.
    """

    with open(file_path, "r") as f:
        count = sum(1 for _ in f)
        return count


if __name__ == "__main__":
    cli()
