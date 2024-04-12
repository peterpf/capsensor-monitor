import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.artist import Artist
from pipeline import PipelineResult


def draw_timeseries(ax: Axis, data: np.ndarray):
    """Draw time series data for a single top-k figure.

    :param ax: axis to draw the data on
    :type ax: matplotlib.axis.Axis
    :param data: two-dimensional array containing the sensor cell values over time
    :type data: np.ndarray
    """
    # Clear axis from previous draw.
    ax.clear()

    # Plot data
    for i in range(data.shape[1]):
        ax.plot(data[:, i])

    # Adjust y-axis limits with some offset so all data is in the view.
    ymin, ymax = (np.nanmin(data), np.nanmax(data))
    y_offset = 0.1 * np.abs(ymax - ymin)
    ymin -= y_offset
    ymax += y_offset
    ax.set_ylim(ymin, ymax)
    # Hide y-axis labels because they don't show up in the animation anyways.
    ax.set_yticklabels([])


def draw_top_k_timeseries(top_k_data: np.ndarray, axis: list[Axis], pipeline_result: PipelineResult) -> list[Artist]:
    """Draw timeseries graphs for the top-k clusters.

    :param top_k_data: two-dimensional array with dimension (buffer_size, nrows*ncols) containing timeseries data of the top-k clusters
    :type top_k_data: np.ndarray
    :param axis: Axis to use for plotting.
    :type axis: list[Axis]
    :param pipeline_result: The pipeline result used to assign the cells to clusters
    :type pipeline_restult: PipelineResult
    :return: list of Artist objects to animate
    """
    clusters = pipeline_result.anomaly_clusters

    # Draw the affected sensor cells on each axis
    clusters.sort(key=lambda c: c.id)
    for idx, cluster in enumerate(clusters):
        # Assure we have enough axis to draw the clusters
        if idx >= len(axis):
            break
        ax = axis[idx]
        data = top_k_data[:, cluster.sensor_cell_indezes]
        draw_timeseries(ax, data)

    return axis


def draw_annotated_matplotlib_heatmap(axis: Axis, sensor_shape: tuple, pipeline_result: PipelineResult) -> list[Artist]:
    """Renders the frame with a heatmap which visualizes the contact points.
    Each contact point is its own cluster with a unique label.
    A sequential color scale is used, it only references the normalized cell values from the clusters.
    Background cells are ignored and drawn as black.
    This means that the highest value over all clusters has the darkest color,\
    while the lowest value over all clusters has the brightest color.

    :param axis: Axis object to draw on
    :type axis: matplotlib.axis.Axis
    :param sensor_shape: Shape of the sensor
    :type sensor_shape: tuple[int, int]
    :param pipeline_result: The pipeline result containing the raw/filtered data and clustered anomaly-regions.
    :type pipeline_result: PipelineResult
    :return: List of Artist objects to animate
    """
    # Clear the axis from previous draw
    axis.clear()

    # Don't draw anything if there is no interaction
    if len(pipeline_result.anomaly_clusters) == 0:
        return draw_default_heatmap(axis, np.zeros(sensor_shape))

    filtered_data = pipeline_result.filtered_data.copy()
    # Create a data mask that selects valid data points (values != 0 as set by the processor)
    valid_data_mask = filtered_data != 0
    filtered_data[valid_data_mask]

    # Holds data used for drawing the heatmap
    heatmap_data = np.repeat(np.nan, len(filtered_data))

    # Insert raw data from the contact point clusters
    for cluster in pipeline_result.anomaly_clusters:
        cell_ids = cluster.sensor_cell_indezes
        heatmap_data[cell_ids] = pipeline_result.raw_data[cell_ids]

    # Normalize heatmap_data to [0, 1] to map onto color scale.
    normalizer = plt.Normalize(vmin=np.nanmin(heatmap_data), vmax=np.nanmax(heatmap_data))
    heatmap_data = normalizer(heatmap_data)

    # Select color scale and set color of invalid data to `black`.
    cmap = plt.cm.OrRd
    cmap.set_bad(color="black")

    # Convert heatmap data to a matrix with the same dimensions as the sensor shape.
    heatmap_data_matrix = np.reshape(heatmap_data, sensor_shape)
    heatmap = axis.imshow(cmap(heatmap_data_matrix), cmap=cmap, aspect="equal")

    # Create a matrix containing the sensor cell indezes
    cell_matrix_mask = np.arange(np.multiply(*sensor_shape)).reshape(sensor_shape)
    for cluster in pipeline_result.anomaly_clusters:
        cluster_label = cluster.id
        cell_idxs = cluster.sensor_cell_indezes
        # Iterate over sensor shape matrix and check if value is in the list of affected-cell-indeces.
        for i in range(cell_matrix_mask.shape[0]):
            for j in range(cell_matrix_mask.shape[1]):
                if cell_matrix_mask[i, j] in cell_idxs:
                    axis.text(j, i, cluster_label, ha="center", va="center", color="black")

    return heatmap, axis

def draw_raw_signal_heatmap(axis: Axis, sensor_shape: tuple, raw_data: np.array, cmap=plt.cm.Greys_r):
    """Renders a heatmap of the data for the given sensor_shape.
    By default it uses the reversed 'Greys' color scale."""
    # Clear the axis from previous draw
    axis.clear()

    # Holds data used for drawing the heatmap
    heatmap_data = raw_data.copy()

    # Normalize heatmap_data to [0, 1] to map onto color scale.
    normalizer = plt.Normalize(vmin=np.nanmin(heatmap_data), vmax=np.nanmax(heatmap_data))
    heatmap_data = normalizer(heatmap_data)

    # Convert heatmap data to a matrix with the same dimensions as the sensor shape.
    heatmap_data_matrix = np.reshape(heatmap_data, sensor_shape)
    heatmap = axis.imshow(cmap(heatmap_data_matrix), cmap=cmap, aspect="equal")

    return heatmap, axis

def draw_filtered_signal_heatmap(axis: Axis, sensor_shape: tuple, filtered_data: np.array):
    # Clear the axis from previous draw
    axis.clear()

    # Holds data used for drawing the heatmap
    invalid_data_mask = filtered_data == 0
    heatmap_data = filtered_data.copy()
    heatmap_data[invalid_data_mask] = np.nan

    # Normalize heatmap_data to [0, 1] to map onto color scale.
    normalizer = plt.Normalize(vmin=np.nanmin(heatmap_data), vmax=np.nanmax(heatmap_data))
    heatmap_data = normalizer(heatmap_data)

    # Select color scale and set color of invalid data to `black`.
    cmap = plt.cm.OrRd
    cmap.set_bad(color="black")

    # Convert heatmap data to a matrix with the same dimensions as the sensor shape.
    heatmap_data_matrix = np.reshape(heatmap_data, sensor_shape)
    heatmap = axis.imshow(cmap(heatmap_data_matrix), cmap=cmap, aspect="equal")

    return heatmap, axis


def draw_default_heatmap(axis: Axis, data_matrix: np.ndarray, cmap="gray") -> list[Artist]:
    """Renders a default black heatmap.

    :param axis: The axis to render the image.
    :type axis: matplotlib.axis.Axis
    :param data_matrix: The default data matrix to draw
    :type data_matrix: np.ndarray
    :return: List of Artists to animate
    """

    default_image = axis.imshow(data_matrix, cmap=cmap, animated=False)
    return (default_image,)
