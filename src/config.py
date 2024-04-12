from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Application configuration"""

    serial_port: str
    """Serial port."""

    baudrate: int
    """Baudrate"""

    connection_timeout: float
    """Connection timeout in seconds"""

    buffer_size: int
    """Number of samples to buffer"""

    top_k_buffer_size: int
    """Sample buffer size for the top-k feature"""

    nrows: int
    """Number of rows"""

    ncols: int
    """Number of columns"""

    sensor_shape: tuple = field(init=False)
    """Sensor dimension as tuple (#rows, #columns)."""

    inverse_sensor_shape: tuple = field(init=False)
    """Inverted sensor shape to convert arrays back (#columns, #rows)."""

    sensitivity: float
    """The sensitivity regarding noise filtering and is directly proportional to the aggressivness of filtering.
    """

    log_data: bool
    """Set to true and define the logging directory to write data to CSV files."""

    data_directory: str
    """Directory to store data files when log_data is enabled."""

    def __post_init__(self):
        self.inverse_sensor_shape = (self.ncols, self.nrows)
        self.sensor_shape = (self.nrows, self.ncols)
