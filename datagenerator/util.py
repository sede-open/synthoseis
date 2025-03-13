import os
import numpy as np


def find_line_with_most_voxels(volume, voxel_thresholds, cfg):
    """
    Find the line with the most voxels
    ----------------------------------

    Finds the line wtih the most voxels.

    Loop over a volume and locate the inline which has the highest number of
    voxels which meet the threshold criteria

    If multiple thresholds are used, first is used as a lower bound,
    and second is used as an upper bound

    Parameters
    ----------
    volume : np.ndarray
        3D volume (ndarray)
    voxel_thresholds: list
        Used for thresholding
    cfg : dict
        model parameters

    Returns
    -------
    inline_index : int
        Inline containing highest number of voxels meeting criteria.
    """
    max_voxels = 0
    inline_index = int(cfg.cube_shape[0] / 2)
    for ii in range(volume.shape[0]):
        voxels = volume[ii, ...].copy()
        try:  # if two values provided...
            voxels = np.where(
                (voxels >= voxel_thresholds[0] & voxels < voxel_thresholds[1]),
                1.0,
                0.0,
            )
        except TypeError:  # Just use 1 threshold
            voxels = np.where(voxels > voxel_thresholds, 1.0, 0.0)
        if voxels[voxels == 1.0].size > max_voxels:
            max_voxels = voxels[voxels == 1.0].size
            inline_index = int(ii)

    return inline_index


def plot_voxels_not_in_regular_layers(
    volume: np.ndarray, threshold: float, title: str, png_name: str, cfg
) -> None:
    """
    Plot voxels not in regular layers

    Analyze voxel values not in regular layers by plotting a
    histogram of voxels above a given threshold.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (ndarray)
    threshold : float
        Threshold for voxel values.
    title : str
        Title for plot.
    png_name : str
        Name of png file to save.
    cfg : dict
        Model configurations.

    Returns
    -------
    None
    """
    plt = import_matplotlib()
    voxels = volume[volume > threshold]
    plt.figure(1, figsize=(20, 15))
    plt.clf()
    plt.title(title)
    plt.hist(voxels, 81)
    plt.savefig(os.path.join(cfg.work_subfolder, png_name), format="png")
    plt.close(1)


def plot_xsection(volume, maps, line_num, title, png_name, cfg, cmap="prism") -> None:
    """
    Plot cross section.

    Parameters
    ----------
    volume : np.ndarray
        The 3D volume as a numpy array
    maps : _type_
        _description_
    line_num : _type_
        _description_
    title : _type_
        _description_
    png_name : _type_
        _description_
    cfg : _type_
        _description_
    cmap : str, optional
        _description_, by default "prism"

    Returns
    -------
    None
    """
    plt = import_matplotlib()
    plt.clf()
    plt.title(f"{title}\nInline: {line_num}")
    plt.figure(1, figsize=(20, 15))
    # If depth maps is infilled and volume is not, update the Z axis of the volume
    # Pick the 75%'th horizon to check
    if (
        np.max(maps[:, :, -int(maps.shape[-1] / 4)]) > volume.shape[-1]
    ):  # don't use last horizon
        plt.imshow(
            np.fliplr(
                np.rot90(
                    volume[line_num, :, : (volume.shape[-1] * cfg.infill_factor) - 1], 3
                )
            ),
            aspect="auto",
            cmap=cmap,
        )
    else:
        plt.imshow(
            np.fliplr(np.rot90(volume[line_num, ...], 3)), aspect="auto", cmap=cmap
        )
    plt.colorbar()
    plt.ylim((volume.shape[-1], 0))
    for i in range(0, maps.shape[-1], 1):
        plt.plot(range(cfg.cube_shape[1]), maps[line_num, :, i], "k-", lw=0.3)
    plt.savefig(os.path.join(cfg.work_subfolder, png_name), format="png")
    plt.close(1)


def plot_3D_faults_plot(cfg, faults, plot_faults=True, plot_throws=True) -> None:
    """
    Plot 3D faults plot
    -------------------

    Plots the faults in a 3D plot.

    Parameters
    ----------
    cfg : _type_
        The configuration.
    faults : np.ndarray
        The faults a numpy object.
    plot_faults : bool, optional
        Whether to plot the faults or not, by default True
    plot_throws : bool, optional
        Whether to plot the fault throws or not, by default True

    Returns
    -------
    None
    """
    from plotly.offline import plot
    import plotly.graph_objects as go

    fi1 = faults.fault_planes
    decimation_factor = 2
    x1, y1, z1 = np.where(
        fi1[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5
    )
    x1 *= decimation_factor
    y1 *= decimation_factor
    z1 *= -decimation_factor
    trace1 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        name="fault_segments",
        mode="markers",
        marker=dict(size=2.0, color="blue", opacity=0.025),
    )

    if plot_faults:
        fi2 = faults.fault_intersections
        x2, y2, z2 = np.where(
            fi2[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.25
        )
        x2 *= decimation_factor
        y2 *= decimation_factor
        z2 *= -decimation_factor
        trace2 = go.Scatter3d(
            x=x2,
            y=y2,
            z=z2,
            name="fault_intersection_segments",
            mode="markers",
            marker=dict(
                size=1.5,
                color="red",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.025,
            ),
        )

        data = [trace1, trace2]
        layout = go.Layout(
            title=go.layout.Title(
                text="Fault Segments & Intersections"
                + "<br>"
                + '<span style="font-size: 12px;">'
                + cfg.work_subfolder
                + "</span>",
                xref="paper",
                x=0,
            )
        )
        camera = dict(eye=dict(x=1.25 / 1.2, y=1.25 / 1.2, z=0.75 / 1.2))
        fig3 = go.Figure(data=data, layout=layout)
        fig3.update_layout(
            scene_camera=camera,
            autosize=False,
            width=960,
            height=720,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                zaxis=dict(
                    nticks=4,
                    range=[-1260, 0],
                ),
            ),
            margin=dict(r=20, l=10, b=10, t=25),
        )
        output_filename = os.path.join(
            cfg.work_subfolder,
            f"qcplot_fault_segments_and_intersections_3D__{cfg.date_stamp}",
        )
        plot(fig3, filename=f"{output_filename}.html", auto_open=False)
        try:
            fig3.write_image(f"{output_filename}.png")
        except:
            print("png file saving failed")
        print(
            "\n   ... util/plot_3D_faults_plot finished creating 3D plot at:\n       "
            + output_filename
        )

    if plot_throws:
        fi3 = faults.fault_plane_throw
        z3 = fi3[x1, y1, -z1]
        trace3 = go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            name="fault_intersection_segments",
            mode="markers",
            marker=dict(
                size=4.0,
                color=z3,  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.05,
            ),
        )
        data = [trace3]
        layout = go.Layout(
            title=go.layout.Title(
                text="Fault Throw along fault plane"
                + "<br>"
                + '<span style="font-size: 12px;">'
                + cfg.work_subfolder
                + "</span>",
                xref="paper",
                x=0,
            )
        )
        camera = dict(eye=dict(x=1.25 / 1.2, y=1.25 / 1.2, z=0.75 / 1.2))
        fig4 = go.Figure(data=data, layout=layout)
        fig4.update_layout(
            scene_camera=camera,
            autosize=False,
            width=960,
            height=720,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                zaxis=dict(
                    nticks=4,
                    range=[-1260, 0],
                ),
            ),
            margin=dict(r=20, l=10, b=10, t=25),
        )
        output_filename = os.path.join(
            cfg.work_subfolder, f"qcplot_fault_segments_throw_3D__{cfg.date_stamp}"
        )
        plot(fig4, filename=f"{output_filename}.html", auto_open=False)
        try:
            fig4.write_image(f"{output_filename}.png")
        except:
            print("png file saving failed")
        print(
            "\n   ... util/plot_3D_faults_plot finished creating 3D plot at:\n       "
            + output_filename
        )


def plot_3D_closure_plot(
    cfg, closures, plot_closures=True, plot_strat_closures=True
) -> None:
    """
    Plot 3D closures
    ----------------

    Plot the closures in 3D.

    Parameters
    ----------
    cfg : dict
        The configuration.
    closures : np.ndarray
        Closures array.
    plot_closures : bool, optional
        Whether to plot the closures or not, by default True
    plot_strat_closures : bool, optional
        Whether to plot statigraphic closures or not, by default True

    Returns
    -------
    None
    """
    from plotly.offline import plot
    import plotly.graph_objects as go

    fi1 = closures.gas_closures
    decimation_factor = 2
    x1, y1, z1 = np.where(
        fi1[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5
    )
    x1 *= decimation_factor
    y1 *= decimation_factor
    z1 *= -decimation_factor

    fi2 = closures.oil_closures
    x2, y2, z2 = np.where(
        fi2[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.25
    )
    x2 *= decimation_factor
    y2 *= decimation_factor
    z2 *= -decimation_factor

    fi3 = closures.brine_closures
    x3, y3, z3 = np.where(
        fi3[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5
    )
    x3 *= decimation_factor
    y3 *= decimation_factor
    z3 *= -decimation_factor

    size = 2.0
    opacity = 0.15

    if plot_closures:
        trace1 = go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            name="gas_segments",
            mode="markers",
            marker=dict(size=size, color="red", opacity=opacity),
        )

        trace2 = go.Scatter3d(
            x=x2,
            y=y2,
            z=z2,
            name="oil_segments",
            mode="markers",
            marker=dict(
                size=size,
                color="green",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=opacity,
            ),
        )

        trace3 = go.Scatter3d(
            x=x3,
            y=y3,
            z=z3,
            name="brine_segments",
            mode="markers",
            marker=dict(
                size=size,
                color="blue",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=opacity / 5.0,
            ),
        )

        data = [trace1, trace2, trace3]
        layout = go.Layout(
            title=go.layout.Title(
                text="Closure Segments (Gas, Oil, Brine)"
                + "<br>"
                + '<span style="font-size: 12px;">'
                + cfg.work_subfolder
                + "</span>",
                xref="paper",
                x=0,
            )
        )
        camera = dict(eye=dict(x=1.25 / 1.2, y=1.25 / 1.2, z=0.75 / 1.2))
        fig4 = go.Figure(data=data, layout=layout)
        fig4.update_layout(
            scene_camera=camera,
            autosize=False,
            width=960,
            height=720,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                zaxis=dict(
                    nticks=4,
                    range=[-1260, 0],
                ),
            ),
            # width=650,
            margin=dict(r=20, l=10, b=10, t=25),
        )
        output_filename = os.path.join(
            cfg.work_subfolder, f"qcplot_closure_segments_3D_{cfg.date_stamp}"
        )
        plot(fig4, filename=f"{output_filename}.html", auto_open=False)
        try:
            fig4.write_image(f"{output_filename}.png")
        except ValueError:
            print("png file saving failed")
        print(
            "\n   ... util/plot_3D_closure_plot finished creating 3D plot at:\n       "
            + output_filename
        )

    if plot_strat_closures:
        ###-------------------------------------------------------------------------
        ### Make 3D plot of strat traps (onlap/pinch-out traps)
        ###-------------------------------------------------------------------------
        fi4 = closures.strat_closures
        decimation_factor = 2
        x4, y4, z4 = np.where(
            (fi4[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5)
            & (fi1[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5)
        )
        x4 *= decimation_factor
        y4 *= decimation_factor
        z4 *= -decimation_factor

        x5, y5, z5 = np.where(
            (fi4[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5)
            & (fi2[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5)
        )
        x5 *= decimation_factor
        y5 *= decimation_factor
        z5 *= -decimation_factor

        x6, y6, z6 = np.where(
            (fi4[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5)
            & (fi3[::decimation_factor, ::decimation_factor, ::decimation_factor] > 0.5)
        )
        x6 *= decimation_factor
        y6 *= decimation_factor
        z6 *= -decimation_factor

        size = 2.0
        opacity = 0.15

        trace1 = go.Scatter3d(
            x=x4,
            y=y4,
            z=z4,
            name="gas_strat",
            mode="markers",
            marker=dict(size=size, color="red", opacity=opacity),
        )

        trace2 = go.Scatter3d(
            x=x5,
            y=y5,
            z=z5,
            name="oil_strat",
            mode="markers",
            marker=dict(
                size=size,
                color="green",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=opacity,
            ),
        )

        trace3 = go.Scatter3d(
            x=x6,
            y=y6,
            z=z6,
            name="brine_strat",
            mode="markers",
            marker=dict(
                size=size,
                color="blue",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=opacity / 5.0,
            ),
        )

        data = [trace1, trace2, trace3]
        layout = go.Layout(
            title=go.layout.Title(
                text="Stratigraphic Closure Segments (Gas, Oil, Brine)"
                + "<br>"
                + '<span style="font-size: 12px;">'
                + cfg.work_subfolder
                + "</span>",
                xref="paper",
                x=0,
            )
        )
        camera = dict(eye=dict(x=1.25 / 1.2, y=1.25 / 1.2, z=0.75 / 1.2))
        fig5 = go.Figure(data=data, layout=layout)
        fig5.update_layout(
            scene_camera=camera,
            autosize=False,
            width=960,
            height=720,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[0, 300],
                ),
                zaxis=dict(
                    nticks=4,
                    range=[-1260, 0],
                ),
            ),
            # width=650,
            margin=dict(r=20, l=10, b=10, t=25),
        )
        output_filename = os.path.join(
            cfg.work_subfolder, f"qcplot_strat_closure_segments_3D_{cfg.date_stamp}"
        )
        plot(fig5, filename=f"{output_filename}.html", auto_open=False)
        try:
            fig5.write_image(f"{output_filename}.png")
        except:
            print("png file saving failed")
        print(
            "\n   ... util/plot_3D_strat_closure_plot finished creating 3D plot at:\n       "
            + output_filename
        )


def plot_facies_qc(cfg, lith, seismic, facies, maps) -> None:
    """
    Plot Facies for QC
    ------------------

    Plots the facies to aid in QC of the model.

    Parameters
    ----------
    cfg : dict
        The configuration used in the model.
    lith : np.ndarray
        The lithology model.
    seismic : np.ndarray
        The seismic model.
    facies : np.ndarray
        The facies model.
    maps : dict
        The maps used in the model.

    Returns
    -------
    None
    """
    plt = import_matplotlib()
    from itertools import groupby
    import matplotlib as mpl

    # Plot Centre Inline of facies and seismic
    iline = lith.shape[0] // 2
    avg_sand_unit_thickness = np.mean(
        [b for a, b in [(k, sum(1 for i in g)) for k, g in groupby(facies)] if a == 1.0]
    )
    textstr = "\n".join(
        (
            f"Sand % Input: {cfg.sand_layer_pct:.1%}",
            f"Sand % Actual: {facies[facies == 1.0].size / facies.size:.1%}",
            f"Sand thickness (layers) Input: {cfg.sand_layer_thickness}",
            f"Sand thickness (layers) Actual: {avg_sand_unit_thickness:.1f}",
            f"Sand Voxel % in model: {lith[lith[:] == 1].size / lith[lith[:] >= 0].size:.1%}",
        )
    )
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 15), sharey=True)
    fig.suptitle("Facies QC Plot")
    axs[0].set_ylim(cfg.cube_shape[-1])
    if cfg.include_salt and np.max(lith[iline, ...]) > 1:
        axs[0].imshow(
            np.fliplr(np.rot90(lith[iline, ...], 3)),
            aspect="auto",
            cmap=mpl.colors.ListedColormap(["blue", "saddlebrown", "gold", "grey"]),
        )
    else:
        axs[0].imshow(
            np.fliplr(np.rot90(lith[iline, ...], 3)),
            aspect="auto",
            cmap=mpl.colors.ListedColormap(["blue", "saddlebrown", "gold"]),
        )
    _img = axs[1].imshow(
        np.fliplr(np.rot90(seismic[iline, ...], 3)),
        aspect="auto",
        cmap="Greys",
        vmin=-300,
        vmax=300,
    )
    # fig.colorbar(_img, ax=axs[1])
    props = dict(boxstyle="round", alpha=0.5)
    # Add textbox with textstr to facies subplot
    axs[0].text(
        0.05,
        0.95,
        textstr,
        transform=axs[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    for i in range(maps.shape[-1]):
        axs[0].plot(range(cfg.cube_shape[0]), maps[iline, :, i], "k-", lw=0.3)

    fig.savefig(
        os.path.join(cfg.work_subfolder, "QC_plot__Facies_FullStackCumulativeSum.png")
    )
    plt.close()


def infill_surface(surface: np.ndarray) -> np.ndarray:
    """
    Infill Surface
    --------------

    Infill surfaces.

    Fill holes in input 2D surface.
    Holes have value of either nan or zero.
    Fill using replacement with interpolated (nearest-neighbor) value.

    Parameters
    ----------
    surface : np.ndarray
        2D numpy array

    Returns
    -------
    surface_infilled : np.ndarray
        (2D) surface_infilled
    """
    from scipy.interpolate import NearestNDInterpolator

    # get x,y,z indices for non-nan points on surface
    xx, yy = np.meshgrid(
        range(surface.shape[0]), range(surface.shape[1]), sparse=False, indexing="ij"
    )
    x_no_nan = xx[~np.isnan(surface)]
    y_no_nan = yy[~np.isnan(surface)]
    z_surface = surface[~np.isnan(surface)]

    if z_surface.flatten().shape[0] > 2:
        # set up nearest-neighbor interpolator function
        xy = zip(x_no_nan, y_no_nan)
        nn = NearestNDInterpolator(xy, z_surface)
        # interpolate at every x,y on regular grid
        surface_nn = nn(xx, yy)
        # replace input surface with interpolated (nearest-neighbor)
        # - if input value is either nan or zero
        surface_infilled = surface.copy()
        surface_infilled[np.isnan(surface)] = surface_nn[np.isnan(surface)]
        surface_infilled[surface == 0] = surface_nn[surface == 0]
    else:
        surface_infilled = z_surface.copy()

    if surface_infilled[np.isnan(surface_infilled)].shape[0] > 0:
        count = surface_infilled[np.isnan(surface_infilled)].shape[0]
        print(
            f"\n\n\n   ...inside infill_surface: there are some NaN values in the surface. count = {count}"
        )

    return surface_infilled


def mute_above_seafloor(surface, xyz):
    """
    Mute data above seafloor
    ------------------------

    Mute a cube above a surface that contains
    indices for the 3rd axis of the cube.

    Parameters
    ----------
    surface : np.ndarray
        Mute above this (2D) surface
    xyz : np.ndarray
        Cube to which muting is applied.

    Returns
    -------
    xyz_muted : np.ndarray
        Muted (3D array)
    """

    krange = np.arange(xyz.shape[2])

    # broadcast indices of vertical axis to 3D
    krange = krange.reshape((1, 1, len(krange)))

    if np.isnan(np.min(surface)) or 0 in surface and not np.all(surface == 0):
        # If surface contains nan's or at least one 0, this will return True, therefore...
        # fill empty picks with nearest-neighbor infilling
        # If all 0's, skip it
        surface_infilled = infill_surface(surface)
    else:
        surface_infilled = surface

    # broadcast surface to 3D
    surface_3d = surface_infilled.reshape((surface.shape[0], surface.shape[1], 1))

    # apply muting to copy of xyz
    xyz_muted = xyz.copy()
    xyz_muted[krange < surface_3d] *= 0.0

    return xyz_muted


def is_it_in_hull(hull, p) -> np.ndarray[bool]:
    """
    Is it in hull?
    --------------
    Checks if point is in hull

    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    Parameters
    ----------
    hull : np.ndarray
        The hull object.
    p : np.ndarray
        Point(s) to check.

    Returns
    -------
    np.ndarray[bool]
        An array containing the booleans where the object is hull.


    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


# Utility Functions
def even_odd(number):
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"


def next_odd(number):
    if even_odd(number) == "Even":
        odd_number = number + 1
    else:
        odd_number = number
    return odd_number


def find_index_next_bigger(mylist, mynumber):
    # return the index of the next number in mylist that is bigger (or equal to) mynumber
    # example:   mynumber=46;  mylist=[1,5,13,25,36,48,55,67,73,85,90]
    # example:  find_index_next_bigger(mylist,mynumber) returns 5
    return next((mylist.index(n) for n in mylist if n >= mynumber), len(mylist))


def linearinfill(x, y, newx):
    # ***************************************************************************************
    #
    #   Function to return data spaced at locations specified by input variable 'newx' after
    #   fitting linear function. Input data specified by 'x' and 'y' so data
    #   can be irregularly sampled.
    #
    # ***************************************************************************************
    s = np.interp(newx, x, y)
    return s


def check_resident_memory():
    import os
    import sys

    if "linux" in sys.platform:
        _proc_status = "/proc/%d/status" % os.getpid()
        _scale = {
            "kB": 1024.0,
            "mB": 1024.0 * 1024.0,
            "KB": 1024.0,
            "MB": 1024.0 * 1024.0,
        }

        t = open(_proc_status)
        v = t.read()
        t.close()

        i = v.index("VmRSS:")
        v = v[i:].split(None, 3)

        return (float(v[1]) * _scale[v[2]]) / 1024**3
    else:
        return 0.0


def import_matplotlib():
    """
    Function to perform matplotlib imports with non-interactive backend when running in background
    DPG 17/2/19
    """
    import sys

    if bool(getattr(sys, "ps1", sys.flags.interactive)):
        # this is an interactive session
        from matplotlib import pyplot as plt

        plt.ion()
    else:
        # import matplotlib with non-interactive backend ('Agg')
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        plt.ioff()
    return plt


def write_data_to_hdf(n, d, h5file):
    import h5py

    with h5py.File(h5file, "a") as hf:
        hf.create_dataset(name=n, data=d, compression="lzf")


# Useful functions, not directly used for data generation
def qc_folders(directory):
    """Count how many model_parameter.txt files contain elapse time

    Use this to remove models that failed to complete for whatever reason
    """
    import glob
    import os

    valid_models = list()
    model_closures = dict()
    # Grep for elapse in model_parameters.txt files in geocrawler subdirectories
    par = glob.glob(os.path.join(directory, "geoc*/model_parameters*.txt"))
    # Elapse time
    for p in par:
        with open(p, "r") as f:
            for line in f.readlines():
                if "elapse" in line:
                    valid_models.append(os.path.dirname(p))
                if "Number of Closures" in line:
                    num_closures = line.split()[-1]
                    model_closures[os.path.dirname(p)] = int(num_closures)

    # Invalid folders
    invalid_models = list(
        set(glob.glob("{}/geoc*".format(directory))) - set(valid_models)
    )
    # Remove tar.gz files from invalid_models
    invalid_final = [x for x in invalid_models if not x.endswith("tar.gz")]

    # Also sort closures by number of closures
    sorted_closures = [
        (k, model_closures[k])
        for k in sorted(model_closures, key=model_closures.get, reverse=True)
    ]

    return invalid_final, valid_models, sorted_closures


def hanflat(inarray, pctflat):
    # ***************************************************************************************
    #
    #   Function applies a Hanning taper to ends of "inarray".
    #   Center "pctflat" of samples remain unchanged.
    #
    #   Parameters:
    #   array :       array of values to have ends tapered
    #   pctflat :     this percent of  samples to remain unchanged (e.g. 0.80)
    #
    #   Returns weights (not weights applied to input array)
    #
    # ***************************************************************************************

    numsamples = len(inarray)
    lowflatindex = int(round(numsamples * (1.0 - pctflat) / 2.0))
    hiflatindex = numsamples - lowflatindex

    # get hanning for numsamples*(1.0-pctflat)
    hanwgt = np.hanning(len(inarray) - (hiflatindex - lowflatindex))

    # piece together hanning weights at ends and weights=1.0 in flat portion
    outarray = np.ones(len(inarray), dtype=float)
    outarray[:lowflatindex] = hanwgt[:lowflatindex]
    outarray[hiflatindex:] = hanwgt[numsamples - hiflatindex :]

    return outarray


def hanflat2d(inarray2d, pctflat):
    hanwgt1 = hanflat(np.ones((inarray2d.shape[0])), pctflat)
    hanwgt2 = hanflat(np.ones((inarray2d.shape[1])), pctflat)
    return np.sqrt(np.outer(hanwgt1, hanwgt2))


def hanflat3d(inarray3d, pctflat):
    hanwgt1 = hanflat(np.ones((inarray3d.shape[0])), pctflat)
    hanwgt2 = hanflat(np.ones((inarray3d.shape[1])), pctflat)
    hanwgt3 = hanflat(np.ones((inarray3d.shape[2])), pctflat)
    _a = np.outer(hanwgt1, hanwgt2)
    _b = np.multiply.outer(_a.ravel(), hanwgt3.ravel())
    return _b.reshape(inarray3d.shape) ** (1.0 / 3.0)


def create_3D_qc_plot(
    cfg, seismic, brine, oil, gas, fault_segments, title_text, camera_steps=20
) -> None:
    """
    Creates a 3D QC plot
    --------------------

    Generates a 3D QC plot for aiding in QC.

    Parameters
    ----------
    cfg : _type_
        _description_
    seismic : _type_
        _description_
    brine : _type_
        _description_
    oil : _type_
        _description_
    gas : _type_
        _description_
    fault_segments : _type_
        _description_
    title_text : _type_
        _description_
    camera_steps : int, optional
        _description_, by default 20

    Returns
    -------
    None
    """

    from scipy.ndimage import gaussian_filter1d, gaussian_filter
    from vtkplotter import addons, show, Text2D, Volume, Plotter
    from vtkplotter.pyplot import cornerHistogram
    from vtkplotter.vtkio import screenshot

    # custom lighting for meshes
    ambient = 0.4
    diffuse = 0.1
    specular = 0.5
    specular_power = 30

    # Brine
    # rescale the vertical axis to have similar height to x & y axes width
    v_exagg = 1.0 / (cfg.cube_shape[-1] // cfg.cube_shape[0])
    vol_br = Volume(brine[:, :, ::-1], spacing=(1, 1, v_exagg))
    vol_br = Volume(vol_br.GetMapper().GetInput())
    print(
        f" ... brine_closure amp range = {brine.min()}, {brine.mean()}, {brine.max()}"
    )
    brine_surface = (
        vol_br.isosurface(threshold=0.99 * brine.max())
        .c("blue")
        .lighting(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular + 0.2,
            specularPower=specular_power,
        )
    )
    # Oil
    vol_oil = Volume(oil[:, :, ::-1], spacing=(1, 1, v_exagg))
    vol_oil = Volume(vol_oil.GetMapper().GetInput())
    print(f" ... oil_closure amp range = {oil.min()}, {oil.mean()}, {oil.max()}")
    oil_surface = (
        vol_oil.isosurface(threshold=0.99 * brine.max())
        .c("green")
        .lighting(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specularPower=specular_power,
        )
    )
    # Gas
    vol_gas = Volume(gas[:, :, ::-1], spacing=(1, 1, v_exagg))
    vol_gas = Volume(vol_gas.GetMapper().GetInput())
    print(f" ... gas amp range = {gas.min()}, {gas.mean()}, {gas.max()}")
    gas_surface = (
        vol_gas.isosurface(threshold=0.99 * brine.max())
        .c("red")
        .lighting(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specularPower=specular_power,
        )
    )
    # Faults
    try:
        sigma = 0.35
        threshold = 0.0759
        fault_segments = gaussian_filter(fault_segments, sigma=(sigma, sigma, 1))
        fault_segments = gaussian_filter1d(fault_segments, sigma=sigma, axis=2)
        vol_fault = Volume(fault_segments[:, :, ::-1], spacing=(1, 1, v_exagg))
        vol_fault = Volume(vol_fault.GetMapper().GetInput())
        fault_surface = (
            vol_fault.isosurface(threshold=threshold)
            .c((0.5, 0.5, 0.5))
            .alpha(0.05)
            .lighting(
                ambient=ambient,
                diffuse=diffuse,
                specular=specular * 1.5,
                specularPower=specular_power * 2,
            )
        )
        include_faults = True
    except:
        include_faults = False

    # Seismic
    # re-order z so that big indices are deep
    vol = Volume(seismic[:, :, ::-1], spacing=(1, 1, v_exagg))
    vol.addScalarBar3D()
    vol = Volume(vol.GetMapper().GetInput())

    # Create the 3D scene and save images
    la, ld = 0.9, 0.9  # ambient, diffuse
    cmaps = ["bone_r", "gist_ncar_r", "jet", "Spectral_r", "hot_r", "gist_earth_r"]

    box = vol.box().wireframe().alpha(0)  # make an invisible box

    vp = Plotter(
        axes=1,
        bg="white",
        bg2="lightblue",
        size=(900, 600),
        title=title_text,
        interactive=False,
        offscreen=True,
    )
    vp.show(box, newPlotter=True, interactive=False)
    # vp.showInset(vol, c=cmaps[0], pos=(1, 1), size=0.3, draggable=False)

    # inits
    visibles = [None, None, None]
    cmap = cmaps[0]
    dims = vol.dimensions()
    i_init = 0
    msh = vol.xSlice(i_init).pointColors(cmap=cmap).lighting("", la, ld, 0)
    msh.addScalarBar(pos=(0.04, 0.0), horizontal=True, titleFontSize=0)
    vp.renderer.AddActor(msh)
    visibles[0] = msh
    j_init = 0
    msh2 = vol.ySlice(j_init).pointColors(cmap=cmap).lighting("", la, ld, 0)
    msh2.addScalarBar(pos=(0.04, 0.0), horizontal=True, titleFontSize=0)
    vp.renderer.AddActor(msh)
    visibles[1] = msh2
    k_init = 0
    msh3 = vol.zSlice(k_init).pointColors(cmap=cmap).lighting("", la, ld, 0, 0)
    msh3.addScalarBar(pos=(0.04, 0.0), horizontal=True, titleFontSize=0)
    vp.renderer.AddActor(msh3)
    visibles[2] = msh3

    # the colormap button
    def buttonfunc():
        global cmap
        bu.switch()
        cmap = bu.status()
        for mesh in visibles:
            if mesh:
                mesh.pointColors(cmap=cmap)
        vp.renderer.RemoveActor(mesh.scalarbar)
        mesh.scalarbar = addons.addScalarBar(
            mesh, pos=(0.04, 0.0), horizontal=True, titleFontSize=0
        )
        vp.renderer.AddActor(mesh.scalarbar)

    bu = vp.addButton(
        buttonfunc,
        pos=(0.27, 0.005),
        states=cmaps,
        c=["db"] * len(cmaps),
        bc=["lb"] * len(cmaps),
        size=14,
        bold=True,
    )

    hist = cornerHistogram(
        seismic,
        s=0.2,
        bins=75,
        logscale=0,
        pos=(0.02, 0.02),
        c=((0.0, 0.0, 0.75)),
        bg=((0.1, 0.1, 0.1)),
        alpha=0.7,
    )

    # Show the meshes and start interacting
    viewpoint1 = [-200.0, -200.0, 500.0]
    focalpoint1 = [300, 300, 150]
    vp.addLight(pos=viewpoint1, focalPoint=focalpoint1)

    viewpoint2 = [-500.0, -500.0, 500.0]
    focalpoint2 = [150, 150, 150]
    vp.addLight(pos=viewpoint2, focalPoint=focalpoint2)

    viewpoint3 = [-400.0, -400.0, 600.0]
    focalpoint3 = [150, 150, 150]
    vp.addLight(pos=viewpoint3, focalPoint=focalpoint3)

    # Can now add any other object to the Plotter scene:
    text0 = Text2D(title_text, s=1.0, font="arial")

    # set camera position
    start_pos = [825.0, 350.0, 600.0]
    start_vu = [-0.4, -0.1, 0.92]
    vp.camera.SetPosition(start_pos)
    vp.camera.SetFocalPoint([150.0, 150.0, 150.0])
    vp.camera.SetViewUp(start_vu)
    vp.camera.SetDistance(1065.168)
    vp.camera.SetClippingRange([517.354, 1757.322])

    if include_faults:
        vp.show(
            msh,
            msh2,
            msh3,
            hist,
            brine_surface,
            oil_surface,
            gas_surface,
            fault_surface,
            text0,
            N=1,
            zoom=1.5,
        )
    else:
        vp.show(
            msh,
            msh2,
            msh3,
            hist,
            brine_surface,
            oil_surface,
            gas_surface,
            text0,
            N=1,
            zoom=1.5,
        )

    # capture the scene from several vantage points
    delay = str(600 // camera_steps)

    # define end camera position
    end_pos = [450.0, 275.0, 1150.0]
    end_vu = [-0.9, -0.3, 0.3]

    n_images1 = 0
    camera_steps_1 = camera_steps // 3
    for icam in range(camera_steps):
        _scalar = float(icam) / (camera_steps - 1)
        _pos = np.array(start_pos) + (np.array(end_pos) - np.array(start_pos)) * _scalar
        _vu = np.array(start_vu) + (np.array(end_vu) - np.array(start_vu)) * _scalar
        vp.camera.SetPosition(_pos)
        vp.camera.SetViewUp(_vu)
        screenshot(
            filename=f"{cfg.temp_folder}/screenshot_{cfg.date_stamp}_{n_images1:03d}.png",
            scale=None,
            returnNumpy=False,
        )
        if cfg.verbose:
            print("\n" + str(n_images1), str(_pos), str(_vu))
        n_images1 += 1

    start_pos = [450.0, 275.0, 1150.0]
    start_vu = [-0.9, -0.3, 0.3]
    end_pos = [275.0, 450.0, 1150.0]
    end_vu = [-0.3, -0.9, 0.3]

    n_images2 = n_images1
    camera_steps_2 = camera_steps // 3
    for icam in range(camera_steps // 3):
        _scalar = float(icam) / (camera_steps - 1)
        _pos = np.array(start_pos) + (np.array(end_pos) - np.array(start_pos)) * _scalar
        _vu = np.array(start_vu) + (np.array(end_vu) - np.array(start_vu)) * _scalar
        vp.camera.SetPosition(_pos)
        vp.camera.SetViewUp(_vu)
        screenshot(
            filename=f"{cfg.temp_folder}/screenshot_{cfg.date_stamp}_{n_images2:03d}.png",
            scale=None,
            returnNumpy=False,
        )
        if cfg.verbose:
            print("\n" + str(n_images2), str(_pos), str(_vu))
        n_images2 += 1

    start_pos = [275.0, 450.0, 1150.0]
    start_vu = [-0.3, -0.9, 0.3]
    end_vu = [-0.1, -0.4, 0.92]
    end_pos = [350.0, 825.0, 600.0]

    n_images3 = n_images2
    camera_steps_3 = camera_steps // 3
    for icam in range(camera_steps // 3):
        _scalar = float(icam) / (camera_steps - 1)
        _pos = np.array(start_pos) + (np.array(end_pos) - np.array(start_pos)) * _scalar
        _vu = np.array(start_vu) + (np.array(end_vu) - np.array(start_vu)) * _scalar
        vp.camera.SetPosition(_pos)
        vp.camera.SetViewUp(_vu)
        screenshot(
            filename=f"{cfg.temp_folder}/screenshot_{cfg.date_stamp}_{n_images3:03d}.png",
            scale=None,
            returnNumpy=False,
        )
        if cfg.verbose:
            print("\n" + str(n_images3), str(_pos), str(_vu))
        n_images3 += 1

    n_images4 = n_images3
    for icam in range(camera_steps // 6):
        fig_num = n_images3 + icam
        os.system(
            f"cp {cfg.temp_folder}/screenshot_{cfg.date_stamp}_{n_images3 - 1:03d}.png "
            + f"{cfg.temp_folder}/screenshot_{cfg.date_stamp}_{fig_num:03d}.png"
        )
        n_images4 += 1

    # create animated gif from screenshots, convert requires ImageTools bin folder to be on $PATH
    output_imagename = os.path.join(
        cfg.work_subfolder, f"qc_image_3d_{cfg.date_stamp}.gif"
    )
    os.system(
        f"convert -delay {delay} {cfg.temp_folder}/screenshot_{cfg.date_stamp}_*.png -loop 0 \
          -dispose previous {output_imagename}"
    )

    vp.close()
    vp.closeWindow()
