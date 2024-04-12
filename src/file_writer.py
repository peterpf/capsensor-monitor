import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from config import AppConfig
from dataclasses import asdict
from pipeline import PipelineResult
from dataclasses import dataclass

@dataclass
class PerformanceData:
    overall_processing_duration: int
    """The overall processing duration in nanoseconds."""

    num_clusters: int
    """The number of clusters appearing in the sample."""

    def get_header() -> list[str]:
        """Returns a list of headers that should be written to a CSV file."""
        return ["overall_processing_duration[ns]", "num_clusters"]

    def get_row(self) -> list:
        """Return a list of all values from all members of this class.
        Order of elements must correspond to the return value of `get_header()`.
        """
        return [self.overall_processing_duration, self.num_clusters]


class FileWriter:
    """Manages (creates and writes) log files.

    :param logdir_parent_folder: The directory where a new folder (based on current time) is created that holds the log files.
    :type logdir_parent_folder: str
    """

    _logging_directory_path: Path
    """The directory path to store log files."""

    _logfile_performance_data: any
    """The logfile for performance data (CSV)."""

    _logfile_raw_data: any
    """The logfile for raw data (CSV)."""

    _logfile_thresholded_data: any
    """The logfile for thresholded data (CSV)."""

    _logfile_clustered_data: any
    """The logfile for clustered data (CSV)."""

    def __init__(self, logdir_parent_folder: Path | str):
        logdir_parent_folder = logdir_parent_folder if isinstance(logdir_parent_folder, Path) else Path(logdir_parent_folder)
        self._setup_log_folder(logdir_parent_folder)
        self._logfile_performance_data = self._create_logfile("performance.csv", header=PerformanceData.get_header())
        self._logfile_raw_data = self._create_logfile("raw.csv")
        self._logfile_filtered_data = self._create_logfile("filtered.csv")
        self._logfile_scaled_data = self._create_logfile("scaled.csv")
        self._logfile_binary_data = self._create_logfile("binary.csv")
        self._logfile_clustered_data = self._create_logfile("cluster_labels.csv")

    def _setup_log_folder(self, parent_folder: Path, current_time=datetime.now()):
        """Creates the logging directory if it does not already exist under the parent folder.
        The log folder has the current time as name.

        :param parent_folder: The directory path to create
        :type parent_folder: str
        :param current_time: The current system time to be used as the log-folder name
        :type current_time: datetime.datetime
        """
        formatted_datetime = current_time.strftime("%Y%m%dT%H%M%S")
        self._logging_directory_path = parent_folder / formatted_datetime
        os.makedirs(self._logging_directory_path, exist_ok=True)

    def _create_logfile(self, filename: str, header: list[str] = None):
        """Create the log file with the given filename, including file extension.

        :param filename: The filename of the logfile
        :type filename: str
        :param header: list of headers to write as first line in the logfile
        :type header: list[str]
        """
        filepath = self._logging_directory_path / filename
        # Open file in append mode
        file = open(filepath, "a")

        # Write header
        if header is not None:
            file.write(",".join(header) + "\n")

        return file

    def write_configuration(self, config: AppConfig):
        """Write the application configuration to a JSON file in the logging directory.

        :param config: The application configuration
        :type config: AppConfig
        """
        serializable_config = asdict(config)
        json_data = json.dumps(serializable_config)
        config_filepath = self._logging_directory_path / "config.json"
        with open(config_filepath, "w") as fh:
            fh.write(json_data)

    def write_performance_data(self, data: PerformanceData):
        """Write the performance measurement to a file."""

        self._write_data(self._logfile_performance_data, data.get_row())

    def write_pipeline_result(self, result: PipelineResult):
        """Write the pipeline result to the corresponding log files.

        :param result: The pipeline result.
        :type result: PipelineResult
        """
        self._write_data(self._logfile_raw_data, result.raw_data)
        self._write_data(self._logfile_filtered_data, result.filtered_data)
        self._write_data(self._logfile_scaled_data, result.scaled_data)
        self._write_data(self._logfile_binary_data, result.binary_data)
        self._write_data(self._logfile_clustered_data, result.anomaly_cluster_labels)

    def _write_data(self, logfile, data: np.array):
        """Appends the data to the given log file.

        :param logfile: the logfile handle
        :type logfile: any
        :param data: the data to write to the logfile, must have the same number of columns as the headers
        :type data: np.array
        """
        array_str = ",".join([str(v) for v in data])
        logfile.write(array_str + "\n")

    def close(self):
        """Close all open files."""
        self._logfile_raw_data.close()
        self._logfile_filtered_data.close()
        self._logfile_clustered_data.close()
