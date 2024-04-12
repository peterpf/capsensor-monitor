import numpy as np
from exceptions import InputParserException


def decode_and_parse(serial_data: any, output_dim: tuple) -> str:
    """Decode the next line from the serial. Raises an InputParserException if something goes wrong.

    :param serial_data: serial data received from the serial connection
    :param output_dim: output dimensions of the data array (nrows, ncols)
    :type output_dim: tuple
    :return: flattened numpy array of the transformed input data
    """
    try:
        decoded_data = serial_data.decode()
    except Exception as e:
        raise InputParserException(f"decoding failed, reason: {e}")

    return parse_string_input_to_numpy_array(decoded_data, output_dim)


def parse_string_input_to_numpy_array(input: str, output_dim: tuple) -> np.array:
    """Converts the string to a numpy array, and reshapes it to follow the form:
    To match the sensor orientation with the visualization, the dimensions are first flipped, then transposed.
    Raises an InputParserException if something goes wrong.

    ```
    [r1c1, r1c2, r1c3, r2c1, r2c2, r2c3, r3c1, r3c2, r3c3]
    ```

    :param data: a one-dimensional data array containing nrows * ncols entries
    :type data: str
    :param output_dim: output dimensions of the data array (nrows, ncols)
    :type output_dim: tuple
    :return: flattened numpy array of the transformed input data
    """
    try:
        string_values = input.split(",")
        cell_values = np.array([int(v) for v in string_values])
        data_matrix = np.reshape(cell_values, output_dim).transpose()
        return data_matrix.flatten()
    except Exception as e:
        raise InputParserException(f"failed to parse data with reason: {e} for input {input}")
