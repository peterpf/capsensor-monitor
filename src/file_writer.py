import os
import numpy as np

class FileWriter:
    """Manages (creates and writes) log files.

    :param logging_directory_path: The directory of the logfile
    :type logging_directory_path: str
    """

    _logging_directory_path: str
    """The directory path to store log files."""

    _raw_data_logfile: any
    """The logfile for raw data (CSV)."""

    def __init__(self, logging_directory_path: str):
        self._ensure_directory_exists(logging_directory_path)
        self._logging_directory_path = logging_directory_path

        self._raw_data_logfile = self._create_logfile("raw_data.csv")


    def _ensure_directory_exists(self, directory: str):
        """Creates the directory if it does not already exist.

        :param directory: The directory path to create
        :type directory: str
        """
        os.makedirs(directory, exist_ok=True)

    def _create_logfile(self, filename: str, header: list[str] = None):
        """Create the log file with the given filename, including file extension.

        :param filename: The filename of the logfile
        :type filename: str
        :param header: list of headers to write as first line in the logfile
        :type header: list[str]
        """
        # Open file in append mode
        filepath = os.path.join(self._logging_directory_path, filename)
        file = open(filepath, "a")

        # Write header
        if header is not None:
            file.write(",".join(header) + "\n")

        return file

    def write_raw_data(self, data: np.array):
        """Appends the data to the raw data log file.

        :param data: the data to write to the logfile, must have the same number of columns as the headers
        :type data: np.array
        """
        array_str = ",".join([str(v) for v in data])
        self._raw_data_logfile.write(array_str + "\n")

    def close(self):
        """Close all open files."""
        self._raw_data_logfile.close()
