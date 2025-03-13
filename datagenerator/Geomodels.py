import os
import numpy as np
from datagenerator.Parameters import Parameters
from datagenerator.util import next_odd
from scipy.ndimage import maximum_filter


class Geomodel:
    """
    Geomodel
    --------------------------
    The class of the Geomodel object.

    This class contains all the items that make up the Geologic model.

    Parameters
    ----------
    parameters : datagenerator.Parameters
        Parameter object storing all model parameters.
    depth_maps : np.ndarray
        A numpy array containing the depth maps.
    onlap_horizon_list : list
        A list of the onlap horizons.
    facies : np.ndarray
        The generated facies.

    Returns
    -------
    None
    """

    def __init__(
        self,
        parameters: Parameters,
        depth_maps: np.ndarray,
        onlap_horizon_list: list,
        facies: np.ndarray,
    ) -> None:
        """__init__

        Initializer for the Geomodel class.

        Parameters
        ----------
        parameters : datagenerator.Parameters
            Parameter object storing all model parameters.
        depth_maps : np.ndarray
            A numpy array containing the depth maps.
        onlap_horizon_list : list
            A list of the onlap horizons.
        facies : np.ndarray
            The generated facies.
        """
        self.cfg = parameters
        self.depth_maps = depth_maps
        self.onlap_horizon_list = onlap_horizon_list
        self.facies = facies

        # Initialise geomodels
        cube_shape = (
            self.cfg.cube_shape[0],
            self.cfg.cube_shape[1],
            self.cfg.cube_shape[2] + self.cfg.pad_samples,
        )
        self.geologic_age = self.cfg.hdf_init("geologic_age_prefault", shape=cube_shape)
        self.onlap_segments = self.cfg.hdf_init(
            "onlap_segments_prefault", shape=cube_shape
        )
        self.faulted_lithology = self.cfg.hdf_init(
            "lithology_prefault", shape=cube_shape
        )
        self.geomodel_ng = self.cfg.hdf_init("net_to_gross_prefault", shape=cube_shape)
        self.faulted_depth = self.cfg.hdf_init("depth_prefault", shape=cube_shape)
        self.faulted_depth_randomised = self.cfg.hdf_init(
            "depth_randomised_prefault", shape=cube_shape
        )

        # Channel volumes
        if self.cfg.include_channels:
            self.floodplain_shale = None
            self.channel_fill = None
            self.shale_channel_drape = None
            self.levee = None
            self.crevasse = None
            self.channel_segments = None

    def build_unfaulted_geomodels(self):
        """
        Build unfaulted geomodels.
        --------------------------
        A method that builds unfaulted geomodels.

        This method does the following:

        * Build geologic age cube from depth horizons
        * Build onlap segmentation cube
        * If channels are turned on, use fluvsim fortran code to build:
        - floodplain shale cube
        - channel fill cube
        - shale channel drape
        - levee cube
        - crevasse cube

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.geologic_age[:] = self.create_geologic_age_3d_from_infilled_horizons(
            self.depth_maps
        )
        self.onlap_segments[:] = self.insert_onlap_surfaces()
        # if self.cfg.include_channels:
        #     floodplain_shale, channel_fill, shale_channel_drape, levee, crevasse = self.build_channel_cubes()
        #     self.floodplain_shale = self.vertical_anti_alias_filter_simple(floodplain_shale)
        #     self.channel_fill = self.vertical_anti_alias_filter_simple(channel_fill)
        #     self.shale_channel_drape = self.vertical_anti_alias_filter_simple(shale_channel_drape)
        #     self.levee = self.vertical_anti_alias_filter_simple(levee)
        #     self.crevasse = self.vertical_anti_alias_filter_simple(crevasse)

    def infilled_cube_shape(self):
        cube_shape = (
            self.cfg.cube_shape[0],
            self.cfg.cube_shape[1],
            (self.cfg.pad_samples + self.cfg.cube_shape[2]) * self.cfg.infill_factor,
        )
        return cube_shape

    def build_meshgrid(self):
        """
        Build Meshgrid
        --------------------------
        Creates a meshgrid using the data in the Geomodel.

        Parameters
        ----------
        None

        Returns
        -------
        meshgrid : np.darray
            A meshgrid of the data in the Geomodel.
        """
        return np.meshgrid(
            range(self.cfg.cube_shape[0]),
            range(self.cfg.cube_shape[1]),
            sparse=False,
            indexing="ij",
        )

    def create_geologic_age_3d_from_infilled_horizons(self, depth_maps, verbose=False):
        """
        Create geologic age 3d model from infilled horizons.
        --------------------------
        Creates cube containing a geologic age model from a horizon stack.

        - depth_maps have units like ft or ms
        - geologic age has arbitrary units where 'age' is same as horizon index

        Parameters
        ----------
        depth_maps : np.ndarray
            THe depth maps to use to generate the geologic age model.
        verbose : bool
            The level of verbosity in the logs

        Returns
        -------
        returns : age : np.ndarray
            The geologic age model.
        """
        if self.cfg.verbose:
            print("\nCreating Geologic Age volume from unfaulted depth maps")
        cube_shape = self.infilled_cube_shape()
        # ensure that first depth_map has zeros at all X,Y locations
        if not np.all(depth_maps[:, :, 0] == 0.0):
            depth_maps_temp = np.dstack((np.zeros(cube_shape[:2], "float"), depth_maps))
        else:
            depth_maps_temp = depth_maps.copy()

        if verbose:
            print("\n\n   ... inside create_geologic_age_3D_from_infilled_horizons ")
            print(
                "    ... depth_maps min/mean/max, cube_shape = {} {} {} {}".format(
                    depth_maps[:, :, :-1].min(),
                    depth_maps[:, :, :-1].mean(),
                    depth_maps[:, :, :-1].max(),
                    cube_shape,
                )
            )

        # create geologic age cube
        age_range = np.linspace(0.0, float(cube_shape[2] - 1), cube_shape[2])
        age = np.zeros(cube_shape, "float")
        for i in range(cube_shape[0]):
            for j in range(cube_shape[1]):
                index_max_geo_age = np.argmax(
                    depth_maps_temp[i, j, :].clip(0.0, float(cube_shape[2] - 1))
                )
                age[i, j, :] = np.interp(
                    age_range,
                    depth_maps_temp[i, j, : int(index_max_geo_age)],
                    np.arange(index_max_geo_age),
                )

        if self.cfg.verbose:
            print(f"    ... age.shape = {age.shape}")
            print(
                f"    ... age min/mean/max = {age[:].min()}, {age[:].mean():.1f}, {age[:].max()}"
            )
            print(
                "    ... finished create_geologic_age_3D_from_infilled_horizons ...\n"
            )

            from datagenerator.util import plot_xsection
            from datagenerator.util import plot_voxels_not_in_regular_layers

            plot_xsection(
                age[:],
                depth_maps,
                cfg=self.cfg,
                line_num=int(cube_shape[0] / 2),
                title=f"relative geologic age interpolated from horizons\nprior to faulting\nInline:"
                f" {int(cube_shape[0] / 2)}",
                png_name="QC_plot__unfaulted_relative_geologic_age.png",
            )

            # Analyse voxels not in regular layers
            plot_voxels_not_in_regular_layers(
                volume=age[:],
                threshold=0.0,
                cfg=self.cfg,
                title="Example Trav through 3D model\nhistogram of raw layer values",
                png_name="QC_plot__histogram_raw_Layers_1.png",
            )

        return self.vertical_anti_alias_filter_simple(age)

    def vertical_anti_alias_filter_simple(self, cube) -> np.ndarray:
        """
        Simple vertical anti alias filter
        --------------------------
        Applies a simple version of the vertical anti-alias filter.

        Parameters
        ----------
        cube : np.ndarray
            The cube to apply the filter to.
        Returns
        -------
        filtered_cube : np.ndarray
            The anti-aliased filtered cube.
        """
        if self.cfg.verbose:
            print("Applying simple vertical anti-alias filter")
        filtered_cube = cube[..., :: self.cfg.infill_factor]
        return filtered_cube

    def vertical_anti_alias_filter(self, cube):
        """
        Vertical anti alias filter
        --------------------------
        Applies a vertical anti-alias filter.

        Parameters
        ----------
        cube : np.ndarray
            The cube to apply the filter to.
        Returns
        -------
        filtered_cube : np.ndarray
            The anti-aliased filtered cube.
        """
        infill_factor_o2_odd = next_odd(int(self.cfg.infill_factor + 1) / 2)
        max_filt = maximum_filter(cube, size=(1, 1, infill_factor_o2_odd + 2))
        filtered_cube = self.vertical_anti_alias_filter_simple(max_filt)
        return filtered_cube

    def plot_channels(self, channel_segments):
        """
        Plot Channels
        -------------

        Plot the channel segments.

        Parameters
        ----------
        channel_segments : np.ndarray
            An array of the channel segments.

        Returns
        -------
        None
        """
        from datagenerator.util import plot_voxels_not_in_regular_layers
        from datagenerator.util import find_line_with_most_voxels
        from datagenerator.util import import_matplotlib

        plt = import_matplotlib()

        # find inline with the most channel voxels
        inline_index = find_line_with_most_voxels(
            channel_segments, (1000, 2000), self.cfg
        )
        plt.close(1)
        plt.figure(1, figsize=(10, 7))
        plt.clf()
        plt.title(
            "Example Trav through 3D model\nLayers filled with Channels Facies Codes"
        )
        plt.imshow(
            np.fliplr(
                np.rot90(
                    channel_segments[
                        inline_index,
                        :,
                        : self.cfg.cube_shape[2] * self.cfg.infill_factor - 1,
                    ],
                    3,
                )
            ),
            aspect="auto",
            cmap="jet",
        )
        plt.colorbar()
        plt.ylim((self.geologic_age.shape[-1], 0))
        for i in range(0, self.depth_maps.shape[-1], 5):
            plt.plot(
                range(self.cfg.cube_shape[0]),
                self.depth_maps[inline_index, :, i],
                "k-",
                lw=0.3,
            )
        image_path = os.path.join(
            self.cfg.work_subfolder, "QC_plot__LayersFilledWith_ChannelsFacies.png"
        )
        plt.savefig(image_path, format="png")
        plt.close()

        # analyze voxel values not in regular layers
        title = "Example Trav through 3D model\nhistogram of voxels related to channel facies / before faulting"
        pname = "QC_plot__Channels__histogram_beforeFaulting.png"
        plot_voxels_not_in_regular_layers(
            channel_segments, 200.0, title, pname, self.cfg
        )

        title = "Example Trav through 3D model\nhistogram of raw layer values\nShould NOT see channel facies"
        pname = "QC_plot__histogram_raw_Layers_2.png"
        plot_voxels_not_in_regular_layers(
            self.geologic_age, 0.0, title, pname, self.cfg
        )

    def insert_onlap_surfaces(self):
        """
        Insert onlap surfaces
        -------------

        Insert onlap surfaces into the geomodel.

        Parameters
        ----------
        None

        Returns
        -------
        onlap_segments : np.ndarray
            An array of the onlap segments.
        """
        if self.cfg.verbose:
            print(
                "\n\n ... create 3D (pre-faulting) labels for tilting episodes"
                "\n  ... reminder: tilting events were added at horizons {}\n".format(
                    self.onlap_horizon_list
                )
            )

        # get work_cube_shape from cfg parameters
        cube_shape = self.infilled_cube_shape()

        # make sure
        if self.cfg.verbose:
            print("\n\n   ... inside insertOnlap3Dsurface_prefault ")
            print(
                "    ... depth_maps min/mean/max, cube_shape = {} {} {} {}".format(
                    self.depth_maps[:].min(),
                    self.depth_maps[:].mean(),
                    self.depth_maps[:].max(),
                    cube_shape,
                )
            )

        # create 3D cube to hold segmentation results
        onlap_segments = np.zeros(cube_shape, "float32")

        # create grids with grid indices
        ii, jj = np.meshgrid(
            range(cube_shape[0]), range(cube_shape[1]), sparse=False, indexing="ij"
        )

        # loop through horizons in 'onlaps_horizon_list'
        if len(self.onlap_horizon_list) == 0:
            return self.vertical_anti_alias_filter_simple(onlap_segments)

        sorted_onlaps_horizon_list = self.onlap_horizon_list
        sorted_onlaps_horizon_list.sort()

        voxel_count = 0
        for ihorizon in sorted_onlaps_horizon_list:
            # insert onlap label in voxels around horizon
            shallow_map = self.depth_maps[:, :, ihorizon].copy()
            shallow_depth_map_integer = (shallow_map + 0.5).astype("int")

            for k in range(
                -int(self.cfg.infill_factor * 1.5),
                int(self.cfg.infill_factor * 1.5) + 1,
            ):
                sublayer_ii = ii
                sublayer_jj = jj
                sublayer_depth_map = k + shallow_depth_map_integer

                sublayer_depth_map = np.clip(sublayer_depth_map, 0, cube_shape[2] - 1)
                sublayer = onlap_segments[sublayer_ii, sublayer_jj, sublayer_depth_map]
                voxel_count += sublayer.flatten().shape[0]

                del sublayer

                onlap_segments[sublayer_ii, sublayer_jj, sublayer_depth_map] += 1.0
                if self.cfg.verbose and ihorizon % 1 == 0:
                    print(
                        "\t... k: {}, voxel_count: {}, sublayer_current_depth_map.mean: {:.2f} ".format(
                            k, voxel_count, sublayer_depth_map.mean()
                        )
                    )

        # reset invalid values
        onlap_segments[onlap_segments < 0.0] = 0.0

        non_zero_pixels = onlap_segments[onlap_segments != 0.0].shape[0]
        pct_non_zero = float(non_zero_pixels) / (
            cube_shape[0] * cube_shape[1] * cube_shape[2]
        )
        if self.cfg.verbose:
            print(
                "\t...onlap_segments min: {}, mean: {}, max: {}, % non-zero: {} ".format(
                    onlap_segments.min(),
                    onlap_segments.mean(),
                    onlap_segments.max(),
                    pct_non_zero,
                )
            )

        if self.cfg.verbose:
            print("    ... finished putting onlap surface in onlap_segments ...\n")

        # Print non zero voxel count details
        if self.cfg.verbose:
            voxel_count_non_zero = onlap_segments[onlap_segments != 0].shape[0]
            print(
                f"\n   ...(pre-faulting) onlap segments created. min: {onlap_segments.min()},"
                f" mean: {onlap_segments.mean():.5f}, max: {onlap_segments.max()},"
                f" voxel count: {voxel_count_non_zero}\n"
                f"\n   ...(pre-faulting) onlap segments created.shape = {onlap_segments.shape}"
            )

        return self.vertical_anti_alias_filter_simple(onlap_segments)

    def write_cube_to_disk(self, data: np.ndarray, fname: str):
        """
        Writes cube to disk.
        -------------
        Writes a cube to disk.

        Parameters
        ----------
        data : np.ndarray
            The data to be written to disk.
        fname : str
            The name to be influded in the file name.

        Returns
        -------
        None

        It generates a `.npy` file on disk.
        """
        """Write 3D array to npy format."""
        fname = os.path.join(
            self.cfg.work_subfolder, f"{fname}_{self.cfg.date_stamp}.npy"
        )
        np.save(fname, data)
