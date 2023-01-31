from scipy.interpolate import PchipInterpolator
from scipy.interpolate import UnivariateSpline
import numpy as np
import math
from scipy.ndimage import uniform_filter
from scipy.ndimage.morphology import distance_transform_edt


def tz_stretch(x_cube, y_cube, wb):
    """Apply stretch/squeeze to simulate time-depth conversion

    Arguments:
        x_cube {ndarray} -- 4D array containing seismic data, 1st dimension angles
        y_cube {ndarray} -- 3D array containing hydrocarbon labels
        wb {ndarray} -- 2D array containing seabed horizon in samples

    Returns:
        [ndarray, ndarray] -- Stretched/squeezed seismic data and hydrocarbon labels
    """
    n_cubes, x_shape, y_shape, z_shape = x_cube.shape
    seafloor_map = uniform_filter(wb, size=(31, 31))
    seafloor_min = int(seafloor_map.min())

    max_squeeze_depth = np.random.triangular(
        seafloor_map.min() * 1.2,
        (seafloor_map.min() + z_shape) / 2.0,
        z_shape * 0.8,
    )
    max_squeeze_pct = np.random.triangular(1.05, 1.15, 1.25)

    # build function between ascending indices and stretched/squeezed indices
    # - hang from min depth for seafloor
    raw_depth_indices = np.arange(0.0, z_shape)
    stretched_depth_indices = raw_depth_indices.copy()

    # get depth (as index) and percentage to squeeze point that is midway
    # between seafloor and bottom of trace
    fit_indices = np.array(
        [seafloor_min, seafloor_min + 1.0, max_squeeze_depth, z_shape - 1, z_shape]
    )
    fit_depths = np.array(
        (
            seafloor_min,
            seafloor_min + 1.0,
            max_squeeze_depth / max_squeeze_pct,
            z_shape - 1,
            z_shape,
        )
    )
    spline_interp = UnivariateSpline(fit_indices, fit_depths)
    stretched_depth_indices[seafloor_min + 1 : z_shape - 1] = spline_interp(
        raw_depth_indices[seafloor_min + 1 : z_shape - 1]
    )

    output_x_cube = np.zeros_like(x_cube)
    output_y_cube = np.zeros_like(y_cube)
    for k in range(n_cubes):
        for i in range(x_shape):
            for j in range(y_shape):
                stretch_times = raw_depth_indices.copy()
                fit_indices_ij = fit_indices - seafloor_min + seafloor_map[i, j]
                fit_depths_ij = fit_depths - seafloor_min + seafloor_map[i, j]
                seafloor_ind = int(seafloor_map[i, j]) + 1
                spline_interp = UnivariateSpline(fit_depths_ij, fit_indices_ij)
                stretch_times[seafloor_ind : z_shape - 1] = spline_interp(
                    raw_depth_indices[seafloor_ind : z_shape - 1]
                )
                output_trace = np.interp(
                    stretch_times, raw_depth_indices, x_cube[k, i, j, :]
                )
                output_trace[stretch_times >= z_shape - 1.0] = 0.0
                output_x_cube[k, i, j, :] = output_trace.copy()

                if k == 0:
                    # Apply to labels
                    output_trace = np.interp(
                        stretch_times, raw_depth_indices, y_cube[i, j, :]
                    )
                    output_trace[stretch_times >= z_shape - 1.0] = 0.0
                    output_y_cube[i, j, :] = output_trace.copy()

    return output_x_cube, output_y_cube


def uniform_stretch(x_cube, y_cube):
    x_max, y_max, t_max = x_cube.shape[1:]

    x_in = np.arange(x_max)
    y_in = np.arange(y_max)
    t_in = np.arange(t_max)

    squeeze_min = 0.85
    squeeze_max = 1.15
    # Uniform r.v. squeeze factors
    x_squeeze_factor = np.random.rand() * (squeeze_max - squeeze_min) + squeeze_min
    y_squeeze_factor = np.random.rand() * (squeeze_max - squeeze_min) + squeeze_min
    t_squeeze_factor = np.random.rand() * (squeeze_max - squeeze_min) + squeeze_min

    x_out = x_in * x_squeeze_factor
    if x_squeeze_factor > 1:
        roll_x_by = math.floor(np.sum(x_out > x_max - 1) / 2)
        # print(roll_x_by)
        x_out = np.roll(x_out, roll_x_by)

    y_out = y_in * y_squeeze_factor
    if y_squeeze_factor > 1:
        roll_y_by = math.floor(np.sum(y_out > y_max - 1) / 2)
        # print(roll_y_by)
        y_out = np.roll(y_out, roll_y_by)

    t_out = t_in * t_squeeze_factor
    if t_squeeze_factor > 1:
        roll_t_by = math.floor(np.sum(t_out > t_max - 1) / 2)
        # print(roll_t_by)
        t_out = np.roll(t_out, roll_t_by)

    x_cube_sqz = np.apply_along_axis(
        lambda w: np.interp(x_out, x_in, w, left=0, right=0), axis=1, arr=x_cube
    )
    y_cube_sqz = np.apply_along_axis(
        lambda w: np.interp(x_out, x_in, w, left=0, right=0), axis=0, arr=y_cube
    )
    y_cube_sqz = np.round(y_cube_sqz)

    x_cube_sqz = np.apply_along_axis(
        lambda w: np.interp(y_out, y_in, w, left=0, right=0), axis=2, arr=x_cube_sqz
    )
    y_cube_sqz = np.apply_along_axis(
        lambda w: np.interp(y_out, y_in, w, left=0, right=0), axis=1, arr=y_cube_sqz
    )
    y_cube_sqz = np.round(y_cube_sqz)

    x_cube_sqz = np.apply_along_axis(
        lambda w: np.interp(t_out, t_in, w, left=0, right=0), axis=3, arr=x_cube_sqz
    )
    y_cube_sqz = np.apply_along_axis(
        lambda w: np.interp(t_out, t_in, w, left=0, right=0), axis=2, arr=y_cube_sqz
    )
    y_cube_sqz = np.round(y_cube_sqz).astype(
        "uint8"
    )  # convert to uint8 for writing to disk

    return x_cube_sqz, y_cube_sqz
