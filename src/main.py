from app import Application
import utils as utils
from config import AppConfig

logger = utils.configure_logger(__name__)

# Number of samples which have been read from the serial connection.
num_samples_read = 0

if __name__ == "__main__":
    config = AppConfig(
        serial_port="/dev/ttyACM0",
        baudrate=115200,
        connection_timeout=0.1,
        buffer_size=50,
        top_k_buffer_size=80,
        nrows=6,
        ncols=6,
        sensitivity=5.0,
        log_data=True,
        data_directory="tmp/capsense",
    )

    app = Application(config)

    try:
        app.run()
    except Exception as e:
        logger.debug(f"error during run(): {e}")
    finally:
        logger.debug("cleaning up resources...")
        app.teardown()
