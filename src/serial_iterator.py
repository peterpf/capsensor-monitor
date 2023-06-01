import serial

class SerialIterator:
    """Wrapper around the serial object to iterate over the serial readline.

    :param serial_obj: The serial object
    :type serial_obj: serial.Serial
    """

    _serial_obj: serial.Serial
    """The serial object"""

    def __init__(self, serial_obj: serial.Serial):
        self._serial_obj = serial_obj

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next line from the serial, stops iteration when the serial connection is closed."""
        # Verify that the serial connection is active
        if self._serial_obj is None or not self._serial_obj.is_open:
            raise StopIteration()

        return self._serial_obj.readline()
