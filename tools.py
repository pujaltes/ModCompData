import numpy as np
import matplotlib.pyplot as plt


def gen_linemesh(x, y):
    """
    Creates a NumPy array of lines connecting every point to its neighbors
    in a Cartesian grid pattern.

    Args:
        x (np.ndarray): 1D array representing the x-coordinates of the meshgrid.
        y (np.ndarray): 1D array representing the y-coordinates of the meshgrid.

    Returns:
        np.ndarray: 3D array where each 2D slice represents a line connecting
                    a point to its neighbor in the Cartesian grid.
    """
    # Create the meshgrid
    X, Y = np.meshgrid(x, y)

    # Reshape to create a set of 2D points
    points = np.column_stack((X.ravel(), Y.ravel()))

    # Calculate the number of rows and columns
    num_rows, num_cols = X.shape

    # Initialize the line array
    num_lines = (num_rows * (num_cols - 1)) + ((num_rows - 1) * num_cols)
    lines = np.zeros((num_lines, 3, 2))

    # Populate the line array
    line_idx = 0
    for row in range(num_rows):
        for col in range(num_cols - 1):
            lines[line_idx] = np.array(
                [points[row * num_cols + col], points[row * num_cols + col + 1], [np.nan, np.nan]]
            )
            line_idx += 1

    for col in range(num_cols):
        for row in range(num_rows - 1):
            lines[line_idx] = np.array(
                [points[row * num_cols + col], points[(row + 1) * num_cols + col], [np.nan, np.nan]]
            )
            line_idx += 1
    return lines.reshape((num_lines * 3, 2)).T


def gen_rot_matrix(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def remove_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def matrix_plot(matrix, lines, ax, arrows=True):
    if ax is None:
        fig, ax = plt.subplots()
    lims = (
        np.nanmin(lines[0]) + -0.5,
        np.nanmax(lines[0]) + 0.5,
        np.nanmin(lines[1]) - 0.5,
        np.nanmax(lines[1]) + 0.5,
    )
    lines = matrix @ lines

    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.plot(*lines, color="black")

    if arrows:
        # plot arrows for scaled (to be able to see them) canonical basis vectors rotated by the matrix
        e0 = matrix @ np.array([2, 0])
        e1 = matrix @ np.array([0, 2])

        ax.quiver(0, 0, e0[0], e0[1], angles="xy", scale_units="xy", scale=1, color="red", width=15e-3, zorder=2)
        ax.quiver(0, 0, e1[0], e1[1], angles="xy", scale_units="xy", scale=1, color="blue", width=15e-3, zorder=2)

    ax.set_aspect("equal")
    # remove the axis and box
    remove_axes(ax)


def wobbly(linemesh):
    x, y = linemesh
    x = x + 0.5 * np.sin(y)
    y = y + 0.5 * np.cos(x)
    return np.stack([x, y])


def vortex(linemesh):
    x, y = linemesh
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x = r * np.cos(theta + 0.5 * r)
    y = r * np.sin(theta + 0.5 * r)
    return np.stack([x, y])


def gaussian(x, mean, variance):
    # note that variance is the standard deviation squared (sigma^2)
    numerator = np.exp(-0.5 * (x - mean) ** 2 / variance)
    denominator = np.sqrt(2 * np.pi * variance)

    return numerator / denominator


def exponential(x, beta):
    return beta * np.exp(-beta * x)
