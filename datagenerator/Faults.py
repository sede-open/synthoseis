import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import maximum_filter, binary_dilation
from datagenerator.Horizons import Horizons
from datagenerator.Geomodels import Geomodel
from datagenerator.Parameters import Parameters
from datagenerator.util import write_data_to_hdf, plot_3D_faults_plot
from skimage import measure


class Faults(Horizons, Geomodel):
    """
    Faults Class
    ------------
    Describes the class for faulting the model.

    Parameters
    ----------
    Horizons : datagenerator.Horizons
        The Horizons class used to build the faults.
    Geomodel : data_generator.Geomodels
        The Geomodel class used to build the faults.
    
    Returns
    -------
    None
    """
    def __init__(
        self,
        parameters: Parameters,
        unfaulted_depth_maps: np.ndarray,
        onlap_horizon_list: np.ndarray,
        geomodels: Geomodel,
        fan_horizon_list: np.ndarray,
        fan_thickness: np.ndarray
    ):
        """__init__

        Initializes the Faults class..

        Parameters
        ----------
        parameters : Parameters
            The parameters class.
        unfaulted_depth_maps : np.ndarray
            The depth maps to be faulted.
        onlap_horizon_list : np.ndarray
            The onlap horizon list.
        geomodels : Geomodel
            The geomodels class.
        fan_horizon_list : np.ndarray
            The fan horizon list.
        fan_thickness : np.ndarray
            The fan thickness list.
        """
        self.cfg = parameters
        # Horizons
        self.unfaulted_depth_maps = unfaulted_depth_maps
        self.onlap_horizon_list = onlap_horizon_list
        self.fan_horizon_list = fan_horizon_list
        self.fan_thickness = fan_thickness
        self.faulted_depth_maps = self.cfg.hdf_init(
            "faulted_depth_maps", shape=unfaulted_depth_maps.shape
        )
        self.faulted_depth_maps_gaps = self.cfg.hdf_init(
            "faulted_depth_maps_gaps", shape=unfaulted_depth_maps.shape
        )
        # Volumes
        cube_shape = geomodels.geologic_age[:].shape
        self.vols = geomodels
        self.faulted_age_volume = self.cfg.hdf_init(
            "faulted_age_volume", shape=cube_shape
        )
        self.faulted_net_to_gross = self.cfg.hdf_init(
            "faulted_net_to_gross", shape=cube_shape
        )
        self.faulted_lithology = self.cfg.hdf_init(
            "faulted_lithology", shape=cube_shape
        )
        self.reservoir = self.cfg.hdf_init("reservoir", shape=cube_shape)
        self.faulted_depth = self.cfg.hdf_init("faulted_depth", shape=cube_shape)
        self.faulted_onlap_segments = self.cfg.hdf_init(
            "faulted_onlap_segments", shape=cube_shape
        )
        self.fault_planes = self.cfg.hdf_init("fault_planes", shape=cube_shape)
        self.displacement_vectors = self.cfg.hdf_init(
            "displacement_vectors", shape=cube_shape
        )
        self.sum_map_displacements = self.cfg.hdf_init(
            "sum_map_displacements", shape=cube_shape
        )
        self.fault_intersections = self.cfg.hdf_init(
            "fault_intersections", shape=cube_shape
        )
        self.fault_plane_throw = self.cfg.hdf_init(
            "fault_plane_throw", shape=cube_shape
        )
        self.max_fault_throw = self.cfg.hdf_init("max_fault_throw", shape=cube_shape)
        self.fault_plane_azimuth = self.cfg.hdf_init(
            "fault_plane_azimuth", shape=cube_shape
        )
        # Salt
        self.salt_model = None

    def apply_faulting_to_geomodels_and_depth_maps(self) -> None:
        """
        Apply faulting to horizons and cubes
        ------------------------------------
        Generates random faults and applies faulting to horizons and cubes.

        The method does the following:

        * Generate faults and sum the displacements
        * Apply faulting to horizons
        * Write faulted depth maps to disk
        * Write faulted depth maps with gaps at faults to disk
        * Write onlapping horizons to disk
        * Apply faulting to geomodels
        * Make segmentation results conform to binary values after
          faulting and interpolation.
        * Write cubes to file (if qc_volumes turned on in config.json)

        (If segmentation is not reset to binary, the multiple 
        interpolations for successive faults destroys the crisp
        localization of the labels. Subjective observation suggests
        that slightly different thresholds for different 
        features provide superior results)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Make a dictionary of zero-thickness onlapping layers before faulting
        onlap_clip_dict = find_zero_thickness_onlapping_layers(
            self.unfaulted_depth_maps, self.onlap_horizon_list
        )
        _ = self.generate_faults()

        # Apply faulting to age model, net_to_gross cube & onlap segments
        self.faulted_age_volume[:] = self.apply_xyz_displacement(
            self.vols.geologic_age[:]
        ).astype("float")
        self.faulted_onlap_segments[:] = self.apply_xyz_displacement(
            self.vols.onlap_segments[:]
        )

        # Improve the depth maps post faulting by
        # re-interpolating across faulted age model
        (
            self.faulted_depth_maps[:],
            self.faulted_depth_maps_gaps[:],
        ) = self.improve_depth_maps_post_faulting(
            self.vols.geologic_age[:], self.faulted_age_volume[:], onlap_clip_dict
        )

        if self.cfg.include_salt:
            from datagenerator.Salt import SaltModel

            self.salt_model = SaltModel(self.cfg)
            self.salt_model.compute_salt_body_segmentation()
            (
                self.faulted_depth_maps[:],
                self.faulted_depth_maps_gaps[:],
            ) = self.salt_model.update_depth_maps_with_salt_segments_drag()

        # # Write the faulted maps to disk
        self.write_maps_to_disk(
            self.faulted_depth_maps[:] * self.cfg.digi, "depth_maps"
        )
        self.write_maps_to_disk(
            self.faulted_depth_maps_gaps[:] * self.cfg.digi, "depth_maps_gaps"
        )
        self.write_onlap_episodes(
            self.onlap_horizon_list[:],
            self.faulted_depth_maps_gaps[:],
            self.faulted_depth_maps[:],
        )
        if np.any(self.fan_horizon_list):
            self.write_fan_horizons(
                self.fan_horizon_list, self.faulted_depth_maps[:] * 4.0
            )

        if self.cfg.hdf_store:
            # Write faulted maps to hdf
            for n, d in zip(
                ["depth_maps", "depth_maps_gaps"],
                [
                    self.faulted_depth_maps[:] * self.cfg.digi,
                    self.faulted_depth_maps_gaps[:] * self.cfg.digi,
                ],
            ):
                write_data_to_hdf(n, d, self.cfg.hdf_master)

        # Create faulted binary segmentation volumes
        _fault_planes = self.fault_planes[:]
        self.fault_planes[:] = self.create_binary_segmentations_post_faulting(
            _fault_planes, 0.45
        )
        del _fault_planes
        _fault_intersections = self.fault_intersections[:]
        self.fault_intersections[:] = self.create_binary_segmentations_post_faulting(
            _fault_intersections, 0.45
        )
        del _fault_intersections
        _faulted_onlap_segments = self.faulted_onlap_segments[:]
        self.faulted_onlap_segments[:] = self.create_binary_segmentations_post_faulting(
            _faulted_onlap_segments, 0.45
        )
        del _faulted_onlap_segments
        if self.cfg.include_channels:
            self.vols.floodplain_shale = self.apply_xyz_displacement(
                self.vols.floodplain_shale
            )
            self.vols.channel_fill = self.apply_xyz_displacement(self.vols.channel_fill)
            self.vols.shale_channel_drape = self.apply_xyz_displacement(
                self.vols.shale_channel_drape
            )
            self.vols.levee = self.apply_xyz_displacement(self.vols.levee)
            self.vols.crevasse = self.apply_xyz_displacement(self.vols.crevasse)

            (
                self.vols.channel_segments,
                self.vols.geologic_age,
            ) = self.reassign_channel_segment_encoding(
                self.vols.geologic_age,
                self.vols.floodplain_shale,
                self.vols.channel_fill,
                self.vols.shale_channel_drape,
                self.vols.levee,
                self.vols.crevasse,
                self.maps.channels,
            )
            if self.cfg.model_qc_volumes:
                self.vols.write_cube_to_disk(
                    self.vols.channel_segments, "channel_segments"
                )

        if self.cfg.model_qc_volumes:
            # Output files if qc volumes required
            self.vols.write_cube_to_disk(self.faulted_age_volume[:], "geologic_age")
            self.vols.write_cube_to_disk(
                self.faulted_onlap_segments[:], "onlap_segments"
            )
            self.vols.write_cube_to_disk(self.fault_planes[:], "fault_segments")
            self.vols.write_cube_to_disk(
                self.fault_intersections[:], "fault_intersection_segments"
            )
            self.vols.write_cube_to_disk(
                self.fault_plane_throw[:], "fault_segments_throw"
            )
            self.vols.write_cube_to_disk(
                self.fault_plane_azimuth[:], "fault_segments_azimuth"
            )
        if self.cfg.hdf_store:
            # Write faulted age, onlap and fault segment cubes to hdf
            for n, d in zip(
                [
                    "geologic_age_faulted",
                    "onlap_segments",
                    "fault_segments",
                    "fault_intersection_segments",
                    "fault_segments_throw",
                    "fault_segments_azimuth",
                ],
                [
                    self.faulted_age_volume,
                    self.faulted_onlap_segments,
                    self.fault_planes,
                    self.fault_intersections,
                    self.fault_plane_throw,
                    self.fault_plane_azimuth,
                ],
            ):
                write_data_to_hdf(n, d, self.cfg.hdf_master)

        if self.cfg.qc_plots:
            self.create_qc_plots()
            try:
                # Create 3D qc plot
                plot_3D_faults_plot(self.cfg, self)
            except ValueError:
                self.cfg.write_to_logfile("3D Fault Plotting Failed")

    def build_faulted_property_geomodels(
        self,
        facies: np.ndarray
    ) -> None:
        """
        Build Faulted Property Geomodels
        ------------
        Generates faulted property geomodels.

        **The method does the following:**

        Use faulted geologic_age cube, depth_maps and facies
        to create geomodel properties (depth, lith)

        - lithology
        - net_to_gross (to create effective sand layers)
        - depth below mudline
        - randomised depth below mudline (to 
          randomise the rock properties per layer)

        Parameters
        ----------
        facies : np.ndarray
            The Horizons class used to build the faults.

        Returns
        -------
        None
        """
        work_cube_lith = (
            np.ones_like(self.faulted_age_volume) * -1
        )  # initialise lith cube to water
        work_cube_sealed = np.zeros_like(self.faulted_age_volume)
        work_cube_net_to_gross = np.zeros_like(self.faulted_age_volume)
        work_cube_depth = np.zeros_like(self.faulted_age_volume)
        # Also create a randomised depth cube for generating randomised rock properties
        # final dimension's shape is based on number of possible list types
        # currently  one of ['seawater', 'shale', 'sand']
        # n_lith = len(['shale', 'sand'])
        cube_shape = self.faulted_age_volume.shape
        # randomised_depth = np.zeros(cube_shape, 'float32')

        ii, jj = self.build_meshgrid()

        # Loop over layers in reverse order, start at base
        previous_depth_map = self.faulted_depth_maps[:, :, -1]
        if self.cfg.partial_voxels:
            # add .5 to consider partial voxels from half above and half below
            previous_depth_map += 0.5

        for i in range(self.faulted_depth_maps.shape[2] - 2, 0, -1):
            # Apply a random depth shift within the range as provided in config file
            # (provided in metres, so converted to samples here)

            current_depth_map = self.faulted_depth_maps[:, :, i]
            if self.cfg.partial_voxels:
                current_depth_map += 0.5

            # compute maps with indices of top map and base map to include partial voxels
            top_map_index = current_depth_map.copy().astype("int")
            base_map_index = (
                self.faulted_depth_maps[:, :, i + 1].copy().astype("int") + 1
            )

            # compute thickness over which to iterate
            thickness_map = base_map_index - top_map_index
            thickness_map_max = thickness_map.max()

            tvdml_map = previous_depth_map - self.faulted_depth_maps[:, :, 0]
            # Net to Gross Map for layer
            if not self.cfg.variable_shale_ng:
                ng_map = np.zeros(
                    shape=(
                        self.faulted_depth_maps.shape[0],
                        self.faulted_depth_maps.shape[1],
                    )
                )
            else:
                if (
                    facies[i] == 0.0
                ):  # if shale layer, make non-zero N/G map for layer using a low average net to gross
                    ng_map = self.create_random_net_over_gross_map(
                        avg=(0.0, 0.2), stdev=(0.001, 0.01), octave=3
                    )
            if facies[i] == 1.0:  # if sand layer, make non-zero N/G map for layer
                ng_map = self.create_random_net_over_gross_map()

            for k in range(thickness_map_max + 1):

                if self.cfg.partial_voxels:
                    # compute fraction of voxel containing layer
                    top_map = np.max(
                        np.dstack(
                            (current_depth_map, top_map_index.astype("float32") + k)
                        ),
                        axis=-1,
                    )
                    top_map = np.min(np.dstack((top_map, previous_depth_map)), axis=-1)
                    base_map = np.min(
                        np.dstack(
                            (
                                previous_depth_map,
                                top_map_index.astype("float32") + k + 1,
                            )
                        ),
                        axis=-1,
                    )
                    fraction_of_voxel = np.clip(base_map - top_map, 0.0, 1.0)
                    valid_k = np.where(
                        (fraction_of_voxel > 0.0)
                        & ((top_map_index + k).astype("int") < cube_shape[2]),
                        1,
                        0,
                    )

                    # put layer properties in the cube for each case
                    sublayer_ii = ii[valid_k == 1]
                    sublayer_jj = jj[valid_k == 1]
                else:
                    sublayer_ii = ii[thickness_map > k]
                    sublayer_jj = jj[thickness_map > k]

                if sublayer_ii.shape[0] > 0:
                    if self.cfg.partial_voxels:
                        sublayer_depth_map = (top_map_index + k).astype("int")[
                            valid_k == 1
                        ]
                        sublayer_depth_map_int = np.clip(sublayer_depth_map, 0, None)
                        sublayer_ng_map = ng_map[valid_k == 1]
                        sublayer_tvdml_map = tvdml_map[valid_k == 1]
                        sublayer_fraction = fraction_of_voxel[valid_k == 1]

                        # Lithology cube
                        input_cube = work_cube_lith[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ]
                        values = facies[i] * sublayer_fraction
                        input_cube[input_cube == -1.0] = (
                            values[input_cube == -1.0] * 1.0
                        )
                        input_cube[input_cube != -1.0] += values[input_cube != -1.0]
                        work_cube_lith[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] = (input_cube * 1.0)
                        del input_cube
                        del values

                        input_cube = work_cube_sealed[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ]
                        values = (1 - facies[i - 1]) * sublayer_fraction
                        input_cube[input_cube == -1.0] = (
                            values[input_cube == -1.0] * 1.0
                        )
                        input_cube[input_cube != -1.0] += values[input_cube != -1.0]
                        work_cube_sealed[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] = (input_cube * 1.0)
                        del input_cube
                        del values

                        # Depth cube
                        work_cube_depth[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] += (sublayer_tvdml_map * sublayer_fraction)
                        # Net to Gross cube
                        work_cube_net_to_gross[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] += (sublayer_ng_map * sublayer_fraction)

                        # Randomised Depth Cube
                        # randomised_depth[sublayer_ii, sublayer_jj, sublayer_depth_map_int] += \
                        #     (sublayer_tvdml_map + random_z_perturbation) * sublayer_fraction

                    else:
                        sublayer_depth_map_int = (
                            0.5
                            + np.clip(
                                previous_depth_map[thickness_map > k],
                                0,
                                self.vols.geologic_age.shape[2] - 1,
                            )
                        ).astype("int") - k
                        sublayer_tvdml_map = tvdml_map[thickness_map > k]
                        sublayer_ng_map = ng_map[thickness_map > k]

                        work_cube_lith[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] = facies[i]
                        work_cube_sealed[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] = (1 - facies[i - 1])

                        work_cube_depth[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] = sublayer_tvdml_map
                        work_cube_net_to_gross[
                            sublayer_ii, sublayer_jj, sublayer_depth_map_int
                        ] += sublayer_ng_map
                        # randomised_depth[sublayer_ii, sublayer_jj, sublayer_depth_map_int] += (sublayer_tvdml_map + random_z_perturbation)

            # replace previous depth map for next iteration
            previous_depth_map = current_depth_map.copy()

        if self.cfg.verbose:
            print("\n\n ... After infilling ...")
        self.write_cube_to_disk(work_cube_sealed.astype("uint8"), "sealed_label")

        # Clip cubes and convert from samples to units
        work_cube_lith = np.clip(work_cube_lith, -1.0, 1.0)  # clip lith to [-1, +1]
        work_cube_net_to_gross = np.clip(
            work_cube_net_to_gross, 0, 1.0
        )  # clip n/g to [0, 1]
        work_cube_depth = np.clip(work_cube_depth, a_min=0, a_max=None)
        work_cube_depth *= self.cfg.digi

        if self.cfg.include_salt:
            # Update age model after horizons have been modified by salt inclusion
            self.faulted_age_volume[
                :
            ] = self.create_geologic_age_3d_from_infilled_horizons(
                self.faulted_depth_maps[:] * 10.0
            )
            # Set lith code for salt
            work_cube_lith[self.salt_model.salt_segments[:] > 0.0] = 2.0
            # Fix deepest part of facies in case salt inclusion has shifted base horizon
            # This can leave default (water) facies codes at the base
            last_50_samples = self.cfg.cube_shape[-1] - 50
            work_cube_lith[..., last_50_samples:][
                work_cube_lith[..., last_50_samples:] == -1.0
            ] = 0.0

        if self.cfg.qc_plots:
            from datagenerator.util import plot_xsection
            import matplotlib as mpl

            line_number = int(
                work_cube_lith.shape[0] / 2
            )  # pick centre line for all plots

            if self.cfg.include_salt and np.max(work_cube_lith[line_number, ...]) > 1:
                lith_cmap = mpl.colors.ListedColormap(
                    ["blue", "saddlebrown", "gold", "grey"]
                )
            else:
                lith_cmap = mpl.colors.ListedColormap(["blue", "saddlebrown", "gold"])
            plot_xsection(
                work_cube_lith,
                self.faulted_depth_maps[:],
                line_num=line_number,
                title="Example Trav through 3D model\nLithology",
                png_name="QC_plot__AfterFaulting_lithology.png",
                cfg=self.cfg,
                cmap=lith_cmap,
            )
            plot_xsection(
                work_cube_depth,
                self.faulted_depth_maps,
                line_num=line_number,
                title="Example Trav through 3D model\nDepth Below Mudline",
                png_name="QC_plot__AfterFaulting_depth_bml.png",
                cfg=self.cfg,
                cmap="cubehelix_r",
            )
        self.faulted_lithology[:] = work_cube_lith
        self.faulted_net_to_gross[:] = work_cube_net_to_gross
        self.faulted_depth[:] = work_cube_depth
        # self.randomised_depth[:] = randomised_depth

        # Write the % sand in model to logfile
        sand_fraction = (
            work_cube_lith[work_cube_lith == 1].size
            / work_cube_lith[work_cube_lith >= 0].size
        )
        self.cfg.write_to_logfile(
            f"Sand voxel % in model {100 * sand_fraction:.1f}%",
            mainkey="model_parameters",
            subkey="sand_voxel_pct",
            val=100 * sand_fraction,
        )

        if self.cfg.hdf_store:
            for n, d in zip(
                ["lithology", "net_to_gross", "depth"],
                [
                    self.faulted_lithology[:],
                    self.faulted_net_to_gross[:],
                    self.faulted_depth[:],
                ],
            ):
                write_data_to_hdf(n, d, self.cfg.hdf_master)

        # Save out reservoir volume for XAI-NBDT
        reservoir = (work_cube_lith == 1) * 1.0
        reservoir_dilated = binary_dilation(reservoir)
        self.reservoir[:] = reservoir_dilated

        if self.cfg.model_qc_volumes:
            self.write_cube_to_disk(self.faulted_lithology[:], "faulted_lithology")
            self.write_cube_to_disk(
                self.faulted_net_to_gross[:], "faulted_net_to_gross"
            )
            self.write_cube_to_disk(self.faulted_depth[:], "faulted_depth")
            self.write_cube_to_disk(self.faulted_age_volume[:], "faulted_age")
            if self.cfg.include_salt:
                self.write_cube_to_disk(
                    self.salt_model.salt_segments[..., : self.cfg.cube_shape[2]].astype(
                        "uint8"
                    ),
                    "salt",
                )

    def create_qc_plots(self) -> None:
        """
        Create QC Plots
        ---------------
        Creates QC Plots of faulted models and histograms of
        voxels which are not in layers.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        from datagenerator.util import (
            find_line_with_most_voxels,
            plot_voxels_not_in_regular_layers,
            plot_xsection,
        )

        # analyze voxel values not in regular layers
        plot_voxels_not_in_regular_layers(
            volume=self.faulted_age_volume[:],
            threshold=0.0,
            cfg=self.cfg,
            title="Example Trav through 3D model\n"
            + "histogram of layers after faulting, before inserting channel facies",
            png_name="QC_plot__Channels__histogram_FaultedLayersNoChannels.png",
        )
        try:  # if channel_segments exists
            inline_index_channels = find_line_with_most_voxels(
                self.vols.channel_segments, 0.0, self.cfg
            )
            plot_xsection(
                self.vols.channel_segments,
                self.faulted_depth_maps,
                inline_index_channels,
                cfg=self.cfg,
                title="Example Trav through 3D model\nchannel_segments after faulting",
                png_name="QC_plot__AfterFaulting_channel_segments.png",
            )
            title = "Example Trav through 3D model\nLayers Filled With Layer Number / ChannelsAdded / Faulted"
            png_name = "QC_plot__LayersFilledWithLayerNumber_ChannelsAdded_Faulted.png"
        except (NameError, AttributeError):  # channel_segments does not exist
            inline_index_channels = int(self.faulted_age_volume.shape[0] / 2)
            title = "Example Trav through 3D model\nLayers Filled With Layer Number / Faulted"
            png_name = "QC_plot__LayersFilledWithLayerNumber_Faulted.png"
        plot_xsection(
            self.faulted_age_volume[:],
            self.faulted_depth_maps[:],
            inline_index_channels,
            title,
            png_name,
            self.cfg,
        )

    def generate_faults(self) -> np.ndarray:
        """
        Generate Faults
        ---------------
        Generates faults in the model.

        Parameters
        ----------
        None

        Returns
        -------
        displacements_classification : np.ndarray
            Array of fault displacement classifications
        """
        if self.cfg.verbose:
            print(f" ... create {self.cfg.number_faults} faults")
        fault_params = self.fault_parameters()

        # Write fault parameters to logfile
        self.cfg.write_to_logfile(
            f"Fault_mode: {self.cfg.fmode}",
            mainkey="model_parameters",
            subkey="fault_mode",
            val=self.cfg.fmode,
        )
        self.cfg.write_to_logfile(
            f"Noise_level: {self.cfg.fnoise}",
            mainkey="model_parameters",
            subkey="noise_level",
            val=self.cfg.fnoise,
        )

        # Build faults, and sum displacements
        displacements_classification, hockeys = self.build_faults(fault_params)
        if self.cfg.model_qc_volumes:
            # Write max fault throw cube to disk
            self.vols.write_cube_to_disk(self.max_fault_throw[:], "max_fault_throw")
        self.cfg.write_to_logfile(
            f"Hockey_Sticks generated: {sum(hockeys)}",
            mainkey="model_parameters",
            subkey="hockey_sticks_generated",
            val=sum(hockeys),
        )

        self.cfg.write_to_logfile(
            f"Fault Info: a, b, c, x0, y0, z0, throw/infill_factor, shear_zone_width,"
            " gouge_pctile, tilt_pct*100"
        )
        for i in range(self.cfg.number_faults):
            self.cfg.write_to_logfile(
                f"Fault_{i + 1}: {fault_params['a'][i]:.2f}, {fault_params['b'][i]:.2f}, {fault_params['c'][i]:.2f},"
                f" {fault_params['x0'][i]:>7.2f}, {fault_params['y0'][i]:>7.2f}, {fault_params['z0'][i]:>8.2f},"
                f" {fault_params['throw'][i] / self.cfg.infill_factor:>6.2f}, {fault_params['tilt_pct'][i] * 100:.2f}"
            )

            self.cfg.write_to_logfile(
                msg=None,
                mainkey=f"fault_{i + 1}",
                subkey="model_id",
                val=os.path.basename(self.cfg.work_subfolder),
            )
            for _subkey_name in [
                "a",
                "b",
                "c",
                "x0",
                "y0",
                "z0",
                "throw",
                "tilt_pct",
                "shear_zone_width",
                "gouge_pctile",
            ]:
                _val = fault_params[_subkey_name][i]
                self.cfg.write_to_logfile(
                    msg=None, mainkey=f"fault_{i + 1}", subkey=_subkey_name, val=_val
                )

        return displacements_classification

    def fault_parameters(self):
        """
        Get Fault Parameters
        ---------------
        Returns the fault parameters.

        Factory design pattern used to select fault parameters

        Parameters
        ----------
        None

        Returns
        -------
        fault_mode : dict
            Dictionary containing the fault parameters.
        """
        fault_mode = self._get_fault_mode()
        return fault_mode()

    def build_faults(self, fp: dict, verbose=False):
        """
        Build Faults
        ---------------
        Creates faults in the model.

        Parameters
        ----------
        fp : dict
            Dictionary containing the fault parameters.
        verbose : bool
            The level of verbosity to use.

        Returns
        -------
        dis_class : np.ndarray
            Array of fault displacement classifications
        hockey_sticks : list
            List of hockey sticks
        """
        def apply_faulting(traces, stretch_times, verbose=False):
            """
            Apply Faulting
            --------------
            Applies faulting to the traces.

            The method does the following:

            Apply stretching and squeezing previously applied to the input cube
            vertically to give all depths the same number of extrema.
            This is intended to be a proxy for making the
            dominant frequency the same everywhere.
            Variables:
            - traces - input, previously stretched/squeezed trace(s)
            - stretch_times - input, LUT for stretching/squeezing trace(s),
                              range is (0,number samples in last dimension of 'traces')
            - unstretch_traces - output, un-stretched/un-squeezed trace(s)

            Parameters
            ----------
            traces : np.ndarray
                Previously stretched/squeezed trace(s).
            stretch_times : np.ndarray
                A look up table for stretching and squeezing the traces.
            verbose : bool, optional
                The level of verbosity, by default False

            Returns
            -------
            np.ndarray
                The un-stretched/un-squeezed trace(s).
            """
            unstretch_traces = np.zeros_like(traces)
            origtime = np.arange(traces.shape[-1])

            if verbose:
                print("\t   ... Cube parameters going into interpolation")
                print(f"\t   ... Origtime shape  = {len(origtime)}")
                print(f"\t   ... stretch_times_effects shape  = {stretch_times.shape}")
                print(f"\t   ... unstretch_times shape  = {unstretch_traces.shape}")
                print(f"\t   ... traces shape  = {traces.shape}")

            for i in range(traces.shape[0]):
                for j in range(traces.shape[1]):
                    if traces[i, j, :].min() != traces[i, j, :].max():
                        unstretch_traces[i, j, :] = np.interp(
                            stretch_times[i, j, :], origtime, traces[i, j, :]
                        )
                    else:
                        unstretch_traces[i, j, :] = traces[i, j, :]
            return unstretch_traces

        print("\n\n . starting 'build_faults'.")
        print("   ... self.cfg.verbose = " + str(self.cfg.verbose))
        cube_shape = np.array(self.cfg.cube_shape)
        cube_shape[-1] += self.cfg.pad_samples
        samples_in_cube = self.vols.geologic_age[:].size
        wb = self.copy_and_divide_depth_maps_by_infill(
            self.unfaulted_depth_maps[..., 0]
        )

        sum_displacements = np.zeros_like(self.vols.geologic_age[:])
        displacements_class = np.zeros_like(self.vols.geologic_age[:])
        hockey_sticks = []
        fault_voxel_count_list = []
        number_fault_intersections = 0

        depth_maps_faulted_infilled = \
            self.copy_and_divide_depth_maps_by_infill(
                self.unfaulted_depth_maps[:]
            )
        depth_maps_gaps = self.copy_and_divide_depth_maps_by_infill(
            self.unfaulted_depth_maps[:]
        )

        # Create depth indices cube (moved from inside loop)
        faulted_depths = np.zeros_like(self.vols.geologic_age[:])
        for k in range(faulted_depths.shape[-1]):
            faulted_depths[:, :, k] = k
        unfaulted_depths = faulted_depths * 1.0
        _faulted_depths = (
            unfaulted_depths * 1.0
        )  # in case there are 0 faults, prepare _faulted_depths here

        for ifault in tqdm(range(self.cfg.number_faults)):
            semi_axes = [
                fp["a"][ifault],
                fp["b"][ifault],
                fp["c"][ifault] / self.cfg.infill_factor ** 2,
            ]
            origin = [
                fp["x0"][ifault],
                fp["y0"][ifault],
                fp["z0"][ifault] / self.cfg.infill_factor,
            ]
            throw = fp["throw"][ifault] / self.cfg.infill_factor
            tilt = fp["tilt_pct"][ifault]

            print(f"\n\n ... inserting fault {ifault} with throw {throw:.2f}")
            print(
                f"   ... fault ellipsoid semi-axes (a, b, c): {np.sqrt(semi_axes[0]):.2f}, "
                f"{np.sqrt(semi_axes[1]):.2f}, {np.sqrt(semi_axes[2]):.2f}"
            )
            print(
                f"   ... fault ellipsoid origin (x, y, z): {origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}"
            )
            print(f"   ... tilt_pct: {tilt * 100:.2f}")
            z_base = origin[2] * np.sqrt(semi_axes[2])
            print(
                f"   ...z for bottom of ellipsoid at depth (samples) = {np.around(z_base, 0)}"
            )
            print(f"   ...shape of output_cube = {self.vols.geologic_age.shape}")
            print(
                f"   ...infill_factor, pad_samples = {self.cfg.infill_factor}, {self.cfg.pad_samples}"
            )

            # add empty arrays for shear_zone_width and gouge_pctile to fault params dictionary
            fp["shear_zone_width"] = np.zeros(self.cfg.number_faults)
            fp["gouge_pctile"] = np.zeros(self.cfg.number_faults)
            (
                displacement,
                displacement_classification,
                interpolation,
                hockey_stick,
                fault_segm,
                ellipsoid,
                fp,
            ) = self.get_displacement_vector(
                semi_axes, origin, throw, tilt, wb, ifault, fp
            )

            if verbose:
                print("     ... hockey_stick = " + str(hockey_stick))
                print(
                    "     ... Is displacement the same as displacement_classification? "
                    + str(np.all(displacement == displacement_classification))
                )
                print(
                    "     ... Sample count where displacement differs from displacement_classification? "
                    + str(
                        displacement[displacement != displacement_classification].size
                    )
                )
                print(
                    "     ... percent of samples where displacement differs from displacement_classification = "
                    + format(
                        float(
                            displacement[
                                displacement != displacement_classification
                            ].size
                        )
                        / samples_in_cube,
                        "5.1%",
                    )
                )
                try:
                    print(
                        "     ... (displacement differs from displacement_classification).mean() "
                        + str(
                            displacement[
                                displacement != displacement_classification
                            ].mean()
                        )
                    )
                    print(
                        "     ... (displacement differs from displacement_classification).max() "
                        + str(
                            displacement[
                                displacement != displacement_classification
                            ].max()
                        )
                    )
                except:
                    pass

                print(
                    "   ...displacement_classification.min() = "
                    + ", "
                    + str(displacement_classification.min())
                )
                print(
                    "   ...displacement_classification.mean() = "
                    + ", "
                    + str(displacement_classification.mean())
                )
                print(
                    "   ...displacement_classification.max() = "
                    + ", "
                    + str(displacement_classification.max())
                )

                print("   ...displacement.min() = " + ", " + str(displacement.min()))
                print("   ...displacement.mean() = " + ", " + str(displacement.mean()))
                print("   ...displacement.max() = " + ", " + str(displacement.max()))
                if fault_segm[fault_segm > 0.0].size > 0:
                    print(
                        "   ...displacement[fault_segm >0].min() = "
                        + ", "
                        + str(displacement[fault_segm > 0].min())
                    )
                    print(
                        "   ...displacement[fault_segm >0] P10 = "
                        + ", "
                        + str(np.percentile(displacement[fault_segm > 0], 10))
                    )
                    print(
                        "   ...displacement[fault_segm >0] P25 = "
                        + ", "
                        + str(np.percentile(displacement[fault_segm > 0], 25))
                    )
                    print(
                        "   ...displacement[fault_segm >0].mean() = "
                        + ", "
                        + str(displacement[fault_segm > 0].mean())
                    )
                    print(
                        "   ...displacement[fault_segm >0] P75 = "
                        + ", "
                        + str(np.percentile(displacement[fault_segm > 0], 75))
                    )
                    print(
                        "   ...displacement[fault_segm >0] P90 = "
                        + ", "
                        + str(np.percentile(displacement[fault_segm > 0], 90))
                    )
                    print(
                        "   ...displacement[fault_segm >0].max() = "
                        + ", "
                        + str(displacement[fault_segm > 0].max())
                    )

            inline = self.cfg.cube_shape[0] // 2

            # limit labels to portions of fault plane with throw above threshold
            throw_threshold_samples = 1.0
            footprint = np.ones((3, 3, 1))
            fp_i, fp_j, fp_k = np.where(
                (fault_segm > 0.25)
                & (displacement_classification > throw_threshold_samples)
            )
            fault_segm = np.zeros_like(fault_segm)
            fault_plane_displacement = np.zeros_like(fault_segm)
            fault_segm[fp_i, fp_j, fp_k] = 1.0
            fault_plane_displacement[fp_i, fp_j, fp_k] = (
                displacement_classification[fp_i, fp_j, fp_k] * 1.0
            )

            fault_voxel_count_list.append(fault_segm[fault_segm > 0.5].size)

            # create blended version of displacement that accounts for simulation of fault drag
            drag_factor = 0.5
            displacement = (
                1.0 - drag_factor
            ) * displacement + drag_factor * displacement_classification

            # project fault depth on 2D map surface
            fault_plane_map = np.zeros_like(wb)
            depth_indices = np.arange(ellipsoid.shape[-1])
            for ii in range(fault_plane_map.shape[0]):
                for jj in range(fault_plane_map.shape[1]):
                    fault_plane_map[ii, jj] = np.interp(
                        1.0, ellipsoid[ii, jj, :], depth_indices
                    )
            fault_plane_map = fault_plane_map.clip(0, ellipsoid.shape[2] - 1)

            # compute fault azimuth (relative to i,j,k indices, not North)
            dx, dy = np.gradient(fault_plane_map)
            strike_angle = np.arctan2(dy, dx) * 180.0 / np.pi  # 2D
            del dx
            del dy
            strike_angle = np.zeros_like(fault_segm) + strike_angle.reshape(
                fault_plane_map.shape[0], fault_plane_map.shape[1], 1
            )
            strike_angle[fault_segm < 0.5] = 200.0

            # - compute faulted depth as max of fault plane or faulted depth
            fault_plane = np.zeros_like(displacement) + fault_plane_map.reshape(
                fault_plane_map.shape[0], fault_plane_map.shape[1], 1
            )

            if ifault == 0:
                print("      .... set _unfaulted_depths to array with all zeros...")
                _faulted_depths = unfaulted_depths * 1.0
                self.fault_plane_azimuth[:] = strike_angle * 1.0

            print("   ... interpolation = " + str(interpolation))

            if (
                interpolation
            ):  # i.e. if the fault should be considered, apply the displacements and append fault plane
                faulted_depths2 = np.zeros(
                    (ellipsoid.shape[0], ellipsoid.shape[1], ellipsoid.shape[2], 2),
                    "float",
                )
                faulted_depths2[:, :, :, 0] = displacement * 1.0
                faulted_depths2[:, :, :, 1] = fault_plane - faulted_depths
                faulted_depths2[:, :, :, 1][ellipsoid > 1] = 0.0
                map_displacement_vector = np.min(faulted_depths2, axis=-1).clip(
                    0.0, None
                )
                del faulted_depths2
                map_displacement_vector[ellipsoid > 1] = 0.0

                self.sum_map_displacements[:] += map_displacement_vector
                displacements_class += displacement_classification
                # Set displacements outside ellipsoid to 0
                displacement[ellipsoid > 1] = 0

                sum_displacements += displacement

                # apply fault to depth_cube
                if ifault == 0:
                    print("      .... set _unfaulted_depths to array with all zeros...")
                    _faulted_depths = unfaulted_depths * 1.0
                    adjusted_faulted_depths = (unfaulted_depths - displacement).clip(
                        0, ellipsoid.shape[-1] - 1
                    )
                    _faulted_depths = apply_faulting(
                        _faulted_depths, adjusted_faulted_depths
                    )
                    _mft = self.max_fault_throw[:]
                    _mft[ellipsoid <= 1.0] = throw
                    self.max_fault_throw[:] = _mft
                    del _mft
                    self.fault_planes[:] = fault_segm * 1.0
                    _fault_intersections = self.fault_intersections[:]
                    previous_intersection_voxel_count = _fault_intersections[
                        _fault_intersections > 1.1
                    ].size
                    _fault_intersections = self.fault_planes[:] * 1.0
                    if (
                        _fault_intersections[_fault_intersections > 1.1].size
                        > previous_intersection_voxel_count
                    ):
                        number_fault_intersections += 1
                    self.fault_intersections[:] = _fault_intersections
                    self.fault_plane_throw[:] = fault_plane_displacement * 1.0
                    self.fault_plane_azimuth[:] = strike_angle * 1.0
                else:
                    print(
                        "      .... update _unfaulted_depths array using faulted depths..."
                    )
                    try:
                        print(
                            "          .... (before) _faulted_depths.mean() = "
                            + str(_faulted_depths.mean())
                        )
                    except:
                        _faulted_depths = unfaulted_depths * 1.0
                    adjusted_faulted_depths = (unfaulted_depths - displacement).clip(
                        0, ellipsoid.shape[-1] - 1
                    )
                    _faulted_depths = apply_faulting(
                        _faulted_depths, adjusted_faulted_depths
                    )

                    _fault_planes = apply_faulting(
                        self.fault_planes[:], adjusted_faulted_depths
                    )
                    if (
                        _fault_planes[
                            np.logical_and(0.0 < _fault_planes, _fault_planes < 0.25)
                        ].size
                        > 0
                    ):
                        _fault_planes[
                            np.logical_and(0.0 < _fault_planes, _fault_planes < 0.25)
                        ] = 0.0
                    if (
                        _fault_planes[
                            np.logical_and(0.25 < _fault_planes, _fault_planes < 1.0)
                        ].size
                        > 0
                    ):
                        _fault_planes[
                            np.logical_and(0.25 < _fault_planes, _fault_planes < 1.0)
                        ] = 1.0
                    self.fault_planes[:] = _fault_planes
                    del _fault_planes

                    _fault_intersections = apply_faulting(
                        self.fault_intersections[:], adjusted_faulted_depths
                    )
                    if (
                        _fault_intersections[
                            np.logical_and(
                                1.25 < _fault_intersections, _fault_intersections < 2.0
                            )
                        ].size
                        > 0
                    ):
                        _fault_intersections[
                            np.logical_and(
                                1.25 < _fault_intersections, _fault_intersections < 2.0
                            )
                        ] = 2.0
                    self.fault_intersections[:] = _fault_intersections
                    del _fault_intersections

                    self.max_fault_throw[:] = apply_faulting(
                        self.max_fault_throw[:], adjusted_faulted_depths
                    )
                    self.fault_plane_throw[:] = apply_faulting(
                        self.fault_plane_throw[:], adjusted_faulted_depths
                    )
                    self.fault_plane_azimuth[:] = apply_faulting(
                        self.fault_plane_azimuth[:], adjusted_faulted_depths
                    )

                    self.fault_planes[:] += fault_segm
                    self.fault_intersections[:] += fault_segm
                    _mft = self.max_fault_throw[:]
                    _mft[ellipsoid <= 1.0] += throw
                    self.max_fault_throw[:] = _mft
                    del _mft

                    if verbose:
                        print(
                            "   ... fault_plane_displacement[fault_plane_displacement > 0.].size = "
                            + str(
                                fault_plane_displacement[
                                    fault_plane_displacement > 0.0
                                ].size
                            )
                        )
                    self.fault_plane_throw[
                        fault_plane_displacement > 0.0
                    ] = fault_plane_displacement[fault_plane_displacement > 0.0]
                    if verbose:
                        print(
                            "   ... self.fault_plane_throw[self.fault_plane_throw > 0.].size = "
                            + str(
                                self.fault_plane_throw[
                                    self.fault_plane_throw > 0.0
                                ].size
                            )
                        )
                    self.fault_plane_azimuth[fault_segm > 0.9] = (
                        strike_angle[fault_segm > 0.9] * 1.0
                    )

                    if verbose:
                        print(
                            "          .... (after) _faulted_depths.mean() = "
                            + str(_faulted_depths.mean())
                        )

                    # fix interpolated values in max_fault_throw
                    max_fault_throw_list, max_fault_throw_list_counts = np.unique(
                        self.max_fault_throw, return_counts=True
                    )
                    max_fault_throw_list = max_fault_throw_list[
                        max_fault_throw_list_counts > 500
                    ]
                    if verbose:
                        print(
                            "\n   ...max_fault_throw_list = "
                            + str(max_fault_throw_list)
                            + ", "
                            + str(max_fault_throw_list_counts)
                        )
                    mfts = self.max_fault_throw.shape
                    self.cfg.hdf_remove_node_list("max_fault_throw_4d_diff")
                    self.cfg.hdf_remove_node_list("max_fault_throw_4d")
                    max_fault_throw_4d_diff = self.cfg.hdf_init(
                        "max_fault_throw_4d_diff",
                        shape=(mfts[0], mfts[1], mfts[2], max_fault_throw_list.size),
                    )
                    max_fault_throw_4d = self.cfg.hdf_init(
                        "max_fault_throw_4d",
                        shape=(mfts[0], mfts[1], mfts[2], max_fault_throw_list.size),
                    )
                    if verbose:
                        print(
                            "\n   ...max_fault_throw_4d.shape = "
                            + ", "
                            + str(max_fault_throw_4d.shape)
                        )
                    _max_fault_throw_4d_diff = max_fault_throw_4d_diff[:]
                    _max_fault_throw_4d = max_fault_throw_4d[:]
                    for imft, mft in enumerate(max_fault_throw_list):
                        print(
                            "      ... imft, mft, max_fault_throw_4d_diff[:,:,:,imft].shape = "
                            + str(
                                (
                                    imft,
                                    mft,
                                    max_fault_throw_4d_diff[:, :, :, imft].shape,
                                )
                            )
                        )
                        _max_fault_throw_4d_diff[:, :, :, imft] = np.abs(
                            self.max_fault_throw[:, :, :] - mft
                        )
                        _max_fault_throw_4d[:, :, :, imft] = mft
                    max_fault_throw_4d[:] = _max_fault_throw_4d
                    max_fault_throw_4d_diff[:] = _max_fault_throw_4d_diff
                    if verbose:
                        print(
                            "   ...np.argmin(max_fault_throw_4d_diff, axis=-1).shape = "
                            + ", "
                            + str(np.argmin(max_fault_throw_4d_diff, axis=-1).shape)
                        )
                    indices_nearest_throw = np.argmin(max_fault_throw_4d_diff, axis=-1)
                    if verbose:
                        print(
                            "\n   ...indices_nearest_throw.shape = "
                            + ", "
                            + str(indices_nearest_throw.shape)
                        )
                    _max_fault_throw = self.max_fault_throw[:]
                    for imft, mft in enumerate(max_fault_throw_list):
                        _max_fault_throw[indices_nearest_throw == imft] = mft
                    self.max_fault_throw[:] = _max_fault_throw

                    del _max_fault_throw
                    del adjusted_faulted_depths

                if verbose:
                    print(
                        "   ...fault_segm[fault_segm>0.].size = "
                        + ", "
                        + str(fault_segm[fault_segm > 0.0].size)
                    )
                    print("   ...fault_segm.min() = " + ", " + str(fault_segm.min()))
                    print("   ...fault_segm.max() = " + ", " + str(fault_segm.max()))
                    print(
                        "   ...self.fault_planes.max() = "
                        + ", "
                        + str(self.fault_planes.max())
                    )
                    print(
                        "   ...self.fault_intersections.max() = "
                        + ", "
                        + str(self.fault_intersections.max())
                    )

                    print(
                        "   ...list of unique values in self.max_fault_throw = "
                        + ", "
                        + str(np.unique(self.max_fault_throw))
                    )

                # TODO: remove this block after qc/tests complete
                from datagenerator.util import import_matplotlib
                plt = import_matplotlib()

                plt.close(35)
                plt.figure(35, figsize=(15, 10))
                plt.clf()
                plt.imshow(_faulted_depths[inline, :, :].T, aspect="auto", cmap="prism")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.cfg.work_subfolder, f"faulted_depths_{ifault:02d}.png"
                    ),
                    format="png",
                )

                plt.close(35)
                plt.figure(36, figsize=(15, 10))
                plt.clf()
                plt.imshow(displacement[inline, :, :].T, aspect="auto", cmap="jet")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.cfg.work_subfolder, f"displacement_{ifault:02d}.png"
                    ),
                    format="png",
                )

                # TODO: remove this block after qc/tests complete
                _plotarray = self.fault_planes[inline, :, :].copy()
                _plotarray[_plotarray == 0.0] = np.nan
                plt.close(37)
                plt.figure(37, figsize=(15, 10))
                plt.clf()
                plt.imshow(_plotarray.T, aspect="auto", cmap="gist_ncar")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg.work_subfolder, f"fault_{ifault:02d}.png"),
                    format="png",
                )
                plt.close(37)
                plt.figure(36)
                plt.imshow(_plotarray.T, aspect="auto", cmap="gray", alpha=0.6)
                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        self.cfg.work_subfolder,
                        f"displacement_fault_overlay_{ifault:02d}.png",
                    ),
                    format="png",
                )
                plt.close(36)

                hockey_sticks.append(hockey_stick)
                print("   ...hockey_sticks = " + ", " + str(hockey_sticks))

        # print final count
        max_fault_throw_list, max_fault_throw_list_counts = np.unique(
            self.max_fault_throw, return_counts=True
        )
        max_fault_throw_list = max_fault_throw_list[max_fault_throw_list_counts > 100]
        if verbose:
            print(
                "\n   ... ** final ** max_fault_throw_list = "
                + str(max_fault_throw_list)
                + ", "
                + str(max_fault_throw_list_counts)
            )

        # create fault intersections
        _fault_intersections = self.fault_planes[:] * 1.0
        # self.fault_intersections = self.fault_planes * 1.
        if verbose:
            print("  ... line 592:")
            print(
                "   ...self.fault_intersections.max() = "
                + ", "
                + str(_fault_intersections.max())
            )
            print(
                "   ...self.fault_intersections[self.fault_intersections>0.].size = "
                + ", "
                + str(_fault_intersections[_fault_intersections > 0.0].size)
            )

        _fault_intersections[_fault_intersections <= 1.0] = 0.0
        _fault_intersections[_fault_intersections > 1.1] = 1.0
        if verbose:
            print("  ... line 597:")
            print(
                "   ...self.fault_intersections.max() = "
                + ", "
                + str(_fault_intersections.max())
            )
            print(
                "   ...self.fault_intersections[self.fault_intersections>0.].size = "
                + ", "
                + str(_fault_intersections[_fault_intersections > 0.0].size)
            )

        # make 2nd count of number of intersections between faults. write result to logfile.
        from datetime import datetime

        start_time = datetime.now()
        number_fault_intersections = max(
            number_fault_intersections,
            measure.label(_fault_intersections, background=0).max(),
        )
        print(
            f"   ... elapsed time for skimage.label = {(datetime.now() - start_time)}"
        )
        print("   ... number_fault_intersections = " + str(number_fault_intersections))

        # dilate intersection values
        # - window size of (5,5,15) is arbitrary. Should be based on isolating sections
        #   of fault planes on real seismic
        _fault_intersections = maximum_filter(_fault_intersections, size=(7, 7, 17))

        # Fault intersection segments > 1 at intersecting faults. Clip to 1
        _fault_intersections[_fault_intersections > 0.05] = 1.0
        _fault_intersections[_fault_intersections != 1.0] = 0.0
        if verbose:
            print("  ... line 607:")
            print(
                "   ...self.fault_intersections.max() = "
                + ", "
                + str(_fault_intersections.max())
            )
            print(
                "   ...self.fault_intersections[self.fault_intersections>0.].size = "
                + ", "
                + str(_fault_intersections[_fault_intersections > 0.0].size)
            )

        # Fault segments = 1 at fault > 1 at intersections. Clip intersecting fault voxels to 1
        _fault_planes = self.fault_planes[:]
        _fault_planes[_fault_planes > 0.05] = 1.0
        _fault_planes[_fault_planes != 1.0] = 0.0

        # make fault azi 3D and only retain voxels in fault plane

        for k, v in zip(
            [
                "n_voxels_faults",
                "n_voxels_fault_intersections",
                "number_fault_intersections",
                "fault_voxel_count_list",
                "hockey_sticks",
            ],
            [
                _fault_planes[_fault_planes > 0.5].size,
                _fault_intersections[_fault_intersections > 0.5].size,
                number_fault_intersections,
                fault_voxel_count_list,
                hockey_sticks,
            ],
        ):
            self.cfg.write_to_logfile(f"{k}: {v}")

            self.cfg.write_to_logfile(
                msg=None, mainkey="model_parameters", subkey=k, val=v
            )

        dis_class = _faulted_depths * 1
        self.fault_intersections[:] = _fault_intersections
        del _fault_intersections
        self.fault_planes[:] = _fault_planes
        del _fault_planes
        self.displacement_vectors[:] = _faulted_depths * 1.0

        # TODO: check if next line of code modifies 'displacement' properly
        self.sum_map_displacements[:] = _faulted_depths * 1.0

        # Save faulted maps
        self.faulted_depth_maps[:] = depth_maps_faulted_infilled
        self.faulted_depth_maps_gaps[:] = depth_maps_gaps

        # perform faulting for fault_segments and fault_intersections
        return dis_class, hockey_sticks

    def _qc_plot_check_faulted_horizons_match_fault_segments(
        self, faulted_depth_maps, faulted_geologic_age
    ):
        """_qc_plot_check_faulted_horizons_match_fault_segments _summary_

        Parameters
        ----------
        faulted_depth_maps : np.ndarray
            The faluted depth maps
        faulted_geologic_age : _type_
            _description_
        """
        import os
        from datagenerator.util import import_matplotlib

        plt = import_matplotlib()

        plt.figure(1, figsize=(15, 10))
        plt.clf()
        voxel_count_max = 0
        inline = self.cfg.cube_shape[0] // 2
        for i in range(0, self.fault_planes.shape[1], 10):
            voxel_count = self.fault_planes[:, i, :]
            voxel_count = voxel_count[voxel_count > 0.5].size
            if voxel_count > voxel_count_max:
                inline = i
                voxel_count_max = voxel_count
        # inline=50
        plotdata = (faulted_geologic_age[:, inline, :]).T
        plotdata_overlay = (self.fault_planes[:, inline, :]).T
        plotdata[plotdata_overlay > 0.9] = plotdata.max()
        plt.imshow(plotdata, cmap="prism", aspect="auto")
        plt.title(
            "QC plot: faulted layers with horizons and fault_segments overlain"
            + "\nInline "
            + str(inline)
        )
        plt.colorbar()
        for i in range(0, faulted_depth_maps.shape[2], 10):
            plt.plot(
                range(self.cfg.cube_shape[0]),
                faulted_depth_maps[:, inline],
                "k-",
                lw=0.25,
            )
        plot_name = os.path.join(
            self.cfg.work_subfolder, "QC_horizons_from_geologic_age_isovalues.png"
        )
        plt.savefig(plot_name, format="png")
        plt.close()
        #
        plt.figure(1, figsize=(15, 10))
        plt.clf()
        voxel_count_max = 0
        iz = faulted_geologic_age.shape[-1] // 2
        for i in range(0, self.fault_planes.shape[-1], 50):
            voxel_count = self.fault_planes[:, :, i]
            voxel_count = voxel_count[voxel_count > 0.5].size
            if voxel_count > voxel_count_max:
                iz = i
                voxel_count_max = voxel_count

        plotdata = faulted_geologic_age[:, :, iz].copy()
        plotdata2 = self.fault_planes[:, :, iz].copy()
        plotdata[plotdata2 > 0.05] = 0.0
        plt.subplot(1, 2, 1)
        plt.title(str(iz))
        plt.imshow(plotdata.T, cmap="prism", aspect="auto")
        plt.subplot(1, 2, 2)
        plt.imshow(plotdata2.T, cmap="jet", aspect="auto")
        plt.colorbar()
        plot_name = os.path.join(
            self.cfg.work_subfolder, "QC_fault_segments_on_geologic_age_timeslice.png"
        )
        plt.savefig(plot_name, format="png")
        plt.close()

    def improve_depth_maps_post_faulting(
        self,
        unfaulted_geologic_age: np.ndarray,
        faulted_geologic_age: np.ndarray,
        onlap_clips: np.ndarray
    ):
        """
        Re-interpolates the depth maps using the faulted geologic age cube

        Parameters
        ----------
        unfaulted_geologic_age : np.ndarray
            The unfaulted geologic age cube
        faulted_geologic_age : np.ndarray
            The faulted geologic age cube
        onlap_clips : np.ndarray
            The onlap clips
        
        Returns
        -------
        depth_maps : np.ndarray
            The improved depth maps
        depth_maps_gaps : np.ndarray
            The improved depth maps with gaps
        """
        faulted_depth_maps = np.zeros_like(self.faulted_depth_maps)
        origtime = np.arange(self.faulted_depth_maps.shape[-1])
        for i in range(self.faulted_depth_maps.shape[0]):
            for j in range(self.faulted_depth_maps.shape[1]):
                if (
                    faulted_geologic_age[i, j, :].min()
                    != faulted_geologic_age[i, j, :].max()
                ):
                    faulted_depth_maps[i, j, :] = np.interp(
                        origtime,
                        faulted_geologic_age[i, j, :],
                        np.arange(faulted_geologic_age.shape[-1]).astype("float"),
                    )
                else:
                    faulted_depth_maps[i, j, :] = unfaulted_geologic_age[i, j, :]
        # Waterbottom horizon has been set to 0. Re-insert this from the original depth_maps array
        if np.count_nonzero(faulted_depth_maps[:, :, 0]) == 0:
            faulted_depth_maps[:, :, 0] = self.faulted_depth_maps[:, :, 0] * 1.0

        # Shift re-interpolated horizons to replace first horizon (of 0's) with the second, etc
        zmaps = np.zeros_like(faulted_depth_maps)
        zmaps[..., :-1] = faulted_depth_maps[..., 1:]
        # Fix the deepest re-interpolated horizon by adding a constant thickness to the shallower horizon
        zmaps[..., -1] = self.faulted_depth_maps[..., -1] + 10
        # Clip this last horizon to the one above
        thickness_map = zmaps[..., -1] - zmaps[..., -2]
        zmaps[..., -1][np.where(thickness_map <= 0.0)] = zmaps[..., -2][
            np.where(thickness_map <= 0.0)
        ]
        faulted_depth_maps = zmaps.copy()

        if self.cfg.qc_plots:
            self._qc_plot_check_faulted_horizons_match_fault_segments(
                faulted_depth_maps, faulted_geologic_age
            )

        # Re-apply old gaps to improved depth_maps
        zmaps_imp = faulted_depth_maps.copy()
        merged = zmaps_imp.copy()
        _depth_maps_gaps_improved = merged.copy()
        _depth_maps_gaps_improved[np.isnan(self.faulted_depth_maps_gaps)] = np.nan
        depth_maps_gaps = _depth_maps_gaps_improved.copy()

        # for zero-thickness layers, set depth_maps_gaps to nan
        for i in range(depth_maps_gaps.shape[-1] - 1):
            thickness_map = depth_maps_gaps[:, :, i + 1] - depth_maps_gaps[:, :, i]
            # set thicknesses < zero to NaN. Use NaNs in thickness_map to 0 to avoid runtime warning when indexing
            depth_maps_gaps[:, :, i][np.nan_to_num(thickness_map) <= 0.0] = np.nan

        # restore zero thickness from faulted horizons to improved (interpolated) depth maps
        ii, jj = np.meshgrid(
            range(self.cfg.cube_shape[0]),
            range(self.cfg.cube_shape[1]),
            sparse=False,
            indexing="ij",
        )
        merged = zmaps_imp.copy()

        # create temporary copy of fault_segments with dilation
        from scipy.ndimage.morphology import grey_dilation

        _dilated_fault_planes = grey_dilation(self.fault_planes, size=(3, 3, 1))

        _onlap_segments = self.vols.onlap_segments[:]
        for ihor in range(depth_maps_gaps.shape[-1] - 1, 2, -1):
            # filter upper horizon being used for thickness if shallower events onlap it, except at faults
            improved_zmap_thickness = merged[:, :, ihor] - merged[:, :, ihor - 1]
            depth_map_int = ((merged[:, :, ihor]).astype(int)).clip(
                0, _onlap_segments.shape[-1] - 1
            )
            improved_map_onlap_segments = _onlap_segments[ii, jj, depth_map_int] + 0.0
            improved_map_fault_segments = (
                _dilated_fault_planes[ii, jj, depth_map_int] + 0.0
            )
            # remember that self.maps.depth_maps has horizons with direct fault application
            faulted_infilled_map_thickness = (
                self.faulted_depth_maps[:, :, ihor]
                - self.faulted_depth_maps[:, :, ihor - 1]
            )
            improved_zmap_thickness[
                np.where(
                    (faulted_infilled_map_thickness <= 0.0)
                    & (improved_map_onlap_segments > 0.0)
                    & (improved_map_fault_segments == 0.0)
                )
            ] = 0.0
            print(
                " ... ihor, improved_map_onlap_segments[improved_map_onlap_segments>0.].shape,"
                " improved_zmap_thickness[improved_zmap_thickness==0].shape = ",
                ihor,
                improved_map_onlap_segments[improved_map_onlap_segments > 0.0].shape,
                improved_zmap_thickness[improved_zmap_thickness == 0].shape,
            )
            merged[:, :, ihor - 1] = merged[:, :, ihor] - improved_zmap_thickness

        if np.any(self.fan_horizon_list):
            # Clip fans downwards when thickness map is zero
            for count, layer in enumerate(self.fan_horizon_list):
                merged = fix_zero_thickness_fan_layers(
                    merged, layer, self.fan_thickness[count]
                )

        # Re-apply clipping to onlapping layers post faulting
        merged = fix_zero_thickness_onlap_layers(merged, onlap_clips)

        del _dilated_fault_planes

        # Re-apply gaps to improved depth_maps
        _depth_maps_gaps_improved = merged.copy()
        _depth_maps_gaps_improved[np.isnan(self.faulted_depth_maps_gaps)] = np.nan
        depth_maps_gaps = _depth_maps_gaps_improved.copy()

        # for zero-thickness layers, set depth_maps_gaps to nan
        for i in range(depth_maps_gaps.shape[-1] - 1):
            thickness_map = depth_maps_gaps[:, :, i + 1] - depth_maps_gaps[:, :, i]
            # set nans in thickness_map to 0 to avoid runtime warning
            depth_maps_gaps[:, :, i][np.nan_to_num(thickness_map) <= 0.0] = np.nan

        return merged, depth_maps_gaps

    @staticmethod
    def partial_faulting(
        depth_map_gaps_faulted,
        fault_plane_classification,
        faulted_depth_map,
        ii,
        jj,
        max_throw,
        origtime_cube,
        unfaulted_depth_map,
    ):
        """
        Partial faulting
        ----------------

        Executes partial faluting.

        The docstring of this function is a work in progress.

        Parameters
        ----------
        depth_map_gaps_faulted : np.ndarray
            The depth map.
        fault_plane_classification : np.ndarray
            Fault plane classifications.
        faulted_depth_map : _type_
            The faulted depth map.
        ii : int
            The i position
        jj : int
            The j position
        max_throw : float
            The maximum amount of throw for the faults.
        origtime_cube : np.ndarray
            Original time cube.
        unfaulted_depth_map : np.ndarray
            Unfaulted depth map.

        Returns
        -------
        faulted_depth_map : np.ndarray
            The faulted depth map
        depth_map_gaps_faulted : np.ndarray
            Depth map with gaps filled
        """
        for ithrow in range(1, int(max_throw) + 1):
            # infilled
            partial_faulting_map = unfaulted_depth_map + ithrow
            partial_faulting_map_ii_jj = (
                partial_faulting_map[ii, jj]
                .astype("int")
                .clip(0, origtime_cube.shape[2] - 1)
            )
            partial_faulting_on_horizon = fault_plane_classification[
                ii, jj, partial_faulting_map_ii_jj
            ][..., 0]
            origtime_on_horizon = origtime_cube[ii, jj, partial_faulting_map_ii_jj][
                ..., 0
            ]
            faulted_depth_map[partial_faulting_on_horizon == 1] = np.dstack(
                (origtime_on_horizon, partial_faulting_map)
            ).min(axis=-1)[partial_faulting_on_horizon == 1]
            # gaps
            depth_map_gaps_faulted[partial_faulting_on_horizon == 1] = np.nan
        return faulted_depth_map, depth_map_gaps_faulted

    def get_displacement_vector(
        self,
        semi_axes: tuple,
        origin: tuple,
        throw: float,
        tilt,
        wb,
        index,
        fp
    ):
        """
        Gets a displacement vector.

        Parameters
        ----------
        semi_axes : tuple
            The semi axes.
        origin : tuple
            The origin.
        throw : float
            The throw of th fault to use.
        tilt : float
            The tilt of the fault.
        
        Returns
        -------
        stretch_times : np.ndarray
            The stretch times.
        stretch_times_classification : np.ndarray
            Stretch times classification.
        interpolation : bool
            Whether or not to interpolate.
        hockey_stick : int
            The hockey stick.
        fault_segments : np.ndarray

        ellipsoid : 
        fp : 
        """
        a, b, c = semi_axes
        x0, y0, z0 = origin

        random_shear_zone_width = (
            np.around(np.random.uniform(low=0.75, high=1.5) * 200, -2) / 200
        )
        if random_shear_zone_width == 0:
            random_gouge_pctile = 100
        else:
            # clip amplitudes inside shear_zone with this percentile of total (100 implies doing nothing)
            random_gouge_pctile = np.random.triangular(left=10, mode=50, right=100)
        # Store the random values
        fp["shear_zone_width"][index] = random_shear_zone_width
        fp["gouge_pctile"][index] = random_gouge_pctile

        if self.cfg.verbose:
            print(f"   ...shear_zone_width (samples) = {random_shear_zone_width}")
            print(f"   ...gouge_pctile (percent*100) = {random_gouge_pctile}")
            print(f"   .... output_cube.shape = {self.vols.geologic_age.shape}")
            _p = (
                np.arange(self.vols.geologic_age.shape[2]) * self.cfg.infill_factor
            ).shape
            print(
                f"   .... (np.arange(output_cube.shape[2])*infill_factor).shape = {_p}"
            )

        ellipsoid = self.rotate_3d_ellipsoid(x0, y0, z0, a, b, c, tilt)
        fault_segments = self.get_fault_plane_sobel(ellipsoid)
        z_idx = self.get_fault_centre(ellipsoid, wb, fault_segments, index)

        # Initialise return objects in case z_idx size == 0
        interpolation = False
        hockey_stick = 0
        # displacement_cube = None
        if np.size(z_idx) != 0:
            print("    ... Computing fault depth at max displacement")
            print("    ... depth at max displacement  = {}".format(z_idx[2]))
            down = float(ellipsoid[ellipsoid < 1.0].size) / np.prod(
                self.vols.geologic_age[:].shape
            )
            """
            down = np.int16(len(np.where(ellipsoid < 1.)[0]) / 1.0 * (self.vols.geologic_age.shape[2] *
                                                                      self.vols.geologic_age.shape[1] *
                                                                      self.vols.geologic_age.shape[0]))
            print("    ... This fault has {!s} %% of downthrown samples".format(down))
            """
            print(
                "    ... This fault has "
                + format(down, "5.1%")
                + " of downthrown samples"
            )

            (
                stretch_times,
                stretch_times_classification,
                interpolation,
                hockey_stick,
            ) = self.xyz_dis(z_idx, throw, fault_segments, ellipsoid, wb, index)
        else:
            print("  ... Ellipsoid larger than cube no fault inserted")
            stretch_times = np.ones_like(ellipsoid)
            stretch_times_classification = np.ones_like(self.vols.geologic_age[:])

        max_fault_throw = self.max_fault_throw[:]
        max_fault_throw[ellipsoid < 1.0] += int(throw)
        self.max_fault_throw[:] = max_fault_throw

        return (
            stretch_times,
            stretch_times_classification,
            interpolation,
            hockey_stick,
            fault_segments,
            ellipsoid,
            fp,
        )

    def apply_xyz_displacement(self, traces) -> np.ndarray:
        """
        Applies XYZ Displacement.
        
        Apply stretching and squeezing previously applied to the input cube
        vertically to give all depths the same number of extrema.

        This is intended to be a proxy for making the
        dominant frequency the same everywhere.

        Parameters
        ----------
        traces : np.ndarray
            Previously stretched/squeezed trace(s)

        Returns
        -------
        unstretch_traces: np.ndarray
            Un-stretched/un-squeezed trace(s)
        """
        unstretch_traces = np.zeros_like(traces)
        origtime = np.arange(traces.shape[-1])

        print("\t   ... Cube parameters going into interpolation")
        print(f"\t   ... Origtime shape  = {len(origtime)}")
        print(
            f"\t   ... stretch_times_effects shape  = {self.displacement_vectors.shape}"
        )
        print(f"\t   ... unstretch_times shape  = {unstretch_traces.shape}")
        print(f"\t   ... traces shape  = {traces.shape}")

        for i in range(traces.shape[0]):
            for j in range(traces.shape[1]):
                if traces[i, j, :].min() != traces[i, j, :].max():
                    unstretch_traces[i, j, :] = np.interp(
                        self.displacement_vectors[i, j, :], origtime, traces[i, j, :]
                    )
                else:
                    unstretch_traces[i, j, :] = traces[i, j, :]
        return unstretch_traces

    def copy_and_divide_depth_maps_by_infill(self, zmaps) -> np.ndarray:
        """
        Copy and divide depth maps by infill factor
        -------------------------------------------

        Copies and divides depth maps by infill factor.

        Parameters
        ----------
        zmaps : np.array
            The depth maps to copy and divide.

        Returns
        -------
        np.ndarray
            The result of the division
        """
        return zmaps / self.cfg.infill_factor

    def rotate_3d_ellipsoid(
        self, x0, y0, z0, a, b, c, fraction
    ) -> np.ndarray:
        """
        Rotate a 3D ellipsoid
        ---------------------

        Parameters
        ----------
        x0 : _type_
            _description_
        y0 : _type_
            _description_
        z0 : _type_
            _description_
        a : _type_
            _description_
        b : _type_
            _description_
        c : _type_
            _description_
        fraction : _type_
            _description_
        """
        def f(x1, y1, z1, x_0, y_0, z_0, a1, b1, c1):
            return (
                ((x1 - x_0) ** 2) / a1 + ((y1 - y_0) ** 2) / b1 + ((z1 - z_0) ** 2) / c1
            )

        x = np.arange(self.vols.geologic_age.shape[0]).astype("float")
        y = np.arange(self.vols.geologic_age.shape[1]).astype("float")
        z = np.arange(self.vols.geologic_age.shape[2]).astype("float")

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij", sparse=False)

        xyz = (
            np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
            .swapaxes(0, 1)
            .astype("float")
        )

        xyz_rotated = self.apply_3d_rotation(
            xyz, self.vols.geologic_age.shape, x0, y0, fraction
        )

        xx_rotated = xyz_rotated[:, 0].reshape(self.vols.geologic_age.shape)
        yy_rotated = xyz_rotated[:, 1].reshape(self.vols.geologic_age.shape)
        zz_rotated = xyz_rotated[:, 2].reshape(self.vols.geologic_age.shape)

        ellipsoid = f(xx_rotated, yy_rotated, zz_rotated, x0, y0, z0, a, b, c).reshape(
            self.vols.geologic_age.shape
        )

        return ellipsoid

    @staticmethod
    def apply_3d_rotation(inarray, array_shape, x0, y0, fraction):
        from math import sqrt, sin, cos, atan2
        from numpy import cross, eye

        # expm3 deprecated, changed to 'more robust' expm (TM)
        from scipy.linalg import expm, norm

        def m(axis, angle):
            return expm(cross(eye(3), axis / norm(axis) * angle))

        theta = atan2(
            fraction
            * sqrt((x0 - array_shape[0] / 2) ** 2 + (y0 - array_shape[1] / 2) ** 2),
            array_shape[2],
        )
        dip_angle = atan2(y0 - array_shape[1] / 2, x0 - array_shape[0] / 2)

        strike_unitvector = np.array(
            (sin(np.pi - dip_angle), cos(np.pi - dip_angle), 0.0)
        )
        m0 = m(strike_unitvector, theta)

        outarray = np.dot(m0, inarray.T).T

        return outarray

    @staticmethod
    def get_fault_plane_sobel(test_ellipsoid):
        from scipy.ndimage import sobel
        from scipy.ndimage import maximum_filter

        test_ellipsoid[test_ellipsoid <= 1.0] = 0.0
        inside = np.zeros_like(test_ellipsoid)
        inside[test_ellipsoid <= 1.0] = 1.0
        # method 2
        edge = (
            np.abs(sobel(inside, axis=0))
            + np.abs(sobel(inside, axis=1))
            + np.abs(sobel(inside, axis=-1))
        )
        edge_max = maximum_filter(edge, size=(5, 5, 5))
        edge_max[edge_max == 0.0] = 1e6
        fault_segments = edge / edge_max
        fault_segments[np.isnan(fault_segments)] = 0.0
        fault_segments[fault_segments < 0.5] = 0.0
        fault_segments[fault_segments > 0.5] = 1.0
        return fault_segments

    def get_fault_centre(self, ellipsoid, wb_time_map, z_on_ellipse, index):
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        def intersec(ell, thresh, x, y, z):
            abc = np.where(
                (ell[x, y, z] < np.float32(1 + thresh))
                & (ell[x, y, z] > np.float32(1 - thresh))
            )
            if np.size(abc[0]) != 0:
                direction = find_nearest(abc[0], 1)
            else:
                direction = 9999
            print(
                "   ... computing intersection points between ellipsoid and cube, raise error if none found"
            )
            xdir_min = intersec(ell, thresh, x, y[0], z[0])
            xdir_max = intersec(ell, thresh, x, y[-1], z[0])
            ydir_min = intersec(ell, thresh, x[0], y, z[0])
            ydir_max = intersec(ell, thresh, x[-1], y, z[0])

            print("    ... xdir_min coord = ", xdir_min, y[0], z[0])
            print("    ... xdir_max coord = ", xdir_max, y[-1], z[0])
            print("    ... ydir_min coord = ", y[0], ydir_min, z[0])
            print("    ... ydir_max coord = ", y[-1], ydir_max, z[0])
            return direction

        def get_middle_z(ellipse, wb_map, idx, verbose=False):
            # Retrieve indices and cube of point on the ellipsoid and under the sea bed

            random_idx = []
            do_it = True
            origtime = np.array(range(ellipse.shape[-1]))
            wb_time_cube = np.reshape(
                wb_map, (wb_map.shape[0], wb_map.shape[1], 1)
            ) * np.ones_like(origtime)
            abc = np.where((z_on_ellipse == 1) & ((wb_time_cube - origtime) <= 0))
            xyz = np.vstack(abc)
            if verbose:
                print("     ... xyz.shape  = ", xyz.shape)
            if np.size(abc[0]) != 0:
                xyz_xyz = np.array([])
                threshold_center = 5
                while xyz_xyz.size == 0:
                    if threshold_center < z_on_ellipse.shape[0]:
                        z_middle = np.where(
                            np.abs(xyz[2] - int((xyz[2].min() + xyz[2].max()) / 2))
                            < threshold_center
                        )
                        xyz_z = xyz[:, z_middle[0]].copy()
                        if verbose:
                            print("     ... xyz_z.shape  = ", xyz_z.shape)
                        if xyz_z.size != 0:
                            x_middle = np.where(
                                np.abs(
                                    xyz_z[0]
                                    - int((xyz_z[0].min() + xyz_z[0].max()) / 2)
                                )
                                < threshold_center
                            )
                            xyz_xz = xyz_z[:, x_middle[0]].copy()
                            if verbose:
                                print("     ... xyz_xz.shape  = ", xyz_xz.shape)
                            if xyz_xz.size != 0:
                                y_middle = np.where(
                                    np.abs(
                                        xyz_xz[1]
                                        - int((xyz_xz[1].min() + xyz_xz[1].max()) / 2)
                                    )
                                    < threshold_center
                                )
                                xyz_xyz = xyz_xz[:, y_middle[0]].copy()
                                if verbose:
                                    print("     ... xyz_xyz.shape  = ", xyz_xyz.shape)
                        if verbose:
                            print("     ... z for upper intersection  = ", xyz[2].min())
                            print("     ... z for lower intersection  = ", xyz[2].max())
                            print("     ... threshold_center used = ", threshold_center)
                        threshold_center += 5
                    else:
                        print("   ... Break the loop, could not find a suitable point")
                        random_idx = []
                        do_it = False
                        break
                if do_it:
                    # from scipy import random

                    random_idx = xyz_xyz[:, np.random.choice(xyz_xyz.shape[1])]
                    print(
                        "   ... Computing fault middle to hang max displacement function"
                    )
                    print("    ... x idx for max displacement  = ", random_idx[0])
                    print("    ... y idx for max displacement  = ", random_idx[1])
                    print("    ... z idx for max displacement  = ", random_idx[2])
                    print(
                        "    ... ellipsoid value  = ",
                        ellipse[random_idx[0], random_idx[1], random_idx[2]],
                    )
            else:
                print(
                    "    ... Empty intersection between fault and cube, assign d-max at cube lower corner"
                )

            return random_idx

        z_idx = get_middle_z(ellipsoid, wb_time_map, index)
        return z_idx

    def xyz_dis(self, z_idx, throw, z_on_ellipse, ellipsoid, wb, index):
        from scipy import interpolate, signal
        from math import atan2, degrees
        from scipy.stats import multivariate_normal
        from scipy.ndimage.interpolation import rotate

        cube_shape = self.vols.geologic_age.shape

        def u_gaussian(d_max, sig, shape, points):
            from scipy.signal.windows import general_gaussian

            return d_max * general_gaussian(points, shape, np.float32(sig))

        # Choose random values sigma, p and coef
        sigma = np.random.uniform(low=10 * throw - 50, high=300)
        p = np.random.uniform(low=1.5, high=5)
        coef = np.random.uniform(1.3, 1.5)

        # Fault plane
        infill_factor = 0.5
        origtime = np.arange(cube_shape[-1])
        z = np.arange(cube_shape[2]).astype("int")
        # Define Gaussian max throw and roll it to chosen z
        # Gaussian should be defined on at least 5 sigma on each side
        roll_int = int(10 * sigma)
        g = u_gaussian(throw, sigma, p, cube_shape[2] + 2 * roll_int)
        # Pad signal by 10*sigma before rolling
        g_padded_rolled = np.roll(g, np.int32(z_idx[2] + g.argmax() + roll_int))
        count = 0
        wb_x = np.where(z_on_ellipse == 1)[0]
        wb_y = np.where(z_on_ellipse == 1)[1]
        print("   ... Taper fault so it doesn't reach seabed")
        print(f"    ... Sea floor max = {wb[wb_x, wb_y].max()}")
        # Shift Z throw so that id doesn't cross sea bed
        while g_padded_rolled[int(roll_int + wb[wb_x, wb_y].max())] > 1:
            g_padded_rolled = np.roll(g_padded_rolled, 5)
            count += 5
            # Break loop if can't find spot
            if count > cube_shape[2] - wb[wb_x, wb_y].max():
                print("    ... Too many rolled sample, seafloor will not have 0 throw")
                break
        print(f"   ... Vertical throw shifted by {str(count)} samples")
        g_centered = g_padded_rolled[roll_int : cube_shape[2] + roll_int]

        ff = interpolate.interp1d(z, g_centered)
        z_shift = ff
        print("   ... Computing Gaussian distribution function")
        print(f"    ... Max displacement  = {int(throw)}")
        print(f"    ... Sigma  = {int(sigma)}")
        print(f"    ... P  = {int(p)}")

        low_fault_throw = 5
        high_fault_throw = 35
        # Parameters to set ratio of 1.4 seems to be optimal for a 1500x1500 grid
        mu_x = 0
        mu_y = 0

        # Use throw to get length
        throw_range = np.arange(low_fault_throw, high_fault_throw, 1)
        # Max throw vs length relationship
        fault_length = np.power(0.0013 * throw_range, 1.3258)
        # Max throw == 16000
        scale_factor = 16000 / fault_length[-1]
        # Coef random selection moved to top of function
        variance_x = scale_factor * fault_length[np.where(throw_range == int(throw))].item()
        variance_y = variance_x * coef
        # Do the same for the drag zone area
        fault_length_drag = fault_length / 10000
        variance_x_drag = (
            scale_factor * fault_length_drag[np.where(throw_range == int(throw))].item()
        )
        variance_y_drag = variance_x_drag * coef
        # Rotation from z_idx to center
        alpha = atan2(z_idx[1] - cube_shape[1] / 2, z_idx[0] - cube_shape[0] / 2)
        print(f"    ... Variance_x, Variance_y = {variance_x} {variance_y}")
        print(
            f"    ... Angle between max displacement point tangent plane and cube = {int(degrees(alpha))} Degrees"
        )
        print(f"    ... Max displacement point at x,y,z = {z_idx}")

        # Create grid and multivariate normal
        x = np.linspace(
            -int(cube_shape[0] + 1.5 * cube_shape[0]),
            int(cube_shape[0] + 1.5 * cube_shape[0]),
            2 * int(cube_shape[0] + 1.5 * cube_shape[0]),
        )
        y = np.linspace(
            -int(cube_shape[1] + 1.5 * cube_shape[1]),
            int(cube_shape[1] + 1.5 * cube_shape[1]),
            2 * int(cube_shape[1] + 1.5 * cube_shape[1]),
        )
        _x, _y = np.meshgrid(x, y)
        pos = np.empty(_x.shape + (2,))
        pos[:, :, 0] = _x
        pos[:, :, 1] = _y
        rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
        rv_drag = multivariate_normal(
            [mu_x, mu_y], [[variance_x_drag, 0], [0, variance_y_drag]]
        )
        # Scale up by mu order of magnitude and swap axes
        xy_dis = 10000 * (rv.pdf(pos))
        xy_dis_drag = 10000 * (rv_drag.pdf(pos))

        # Normalize
        xy_dis = xy_dis / np.amax(xy_dis.flatten())
        xy_dis_drag = (xy_dis_drag / np.amax(xy_dis_drag.flatten())) * 0.99

        # Rotate plane by alpha
        x = np.linspace(0, cube_shape[0], cube_shape[0])
        y = np.linspace(0, cube_shape[1], cube_shape[1])
        _x, _y = np.meshgrid(x, y)
        xy_dis_rotated = np.zeros_like(xy_dis)
        xy_dis_drag_rotated = np.zeros_like(xy_dis)
        rotate(
            xy_dis_drag,
            degrees(alpha),
            reshape=False,
            output=xy_dis_drag_rotated,
            mode="nearest",
        )
        rotate(
            xy_dis, degrees(alpha), reshape=False, output=xy_dis_rotated, mode="nearest"
        )
        print("   ...", xy_dis_rotated.shape, xy_dis.shape)
        xy_dis = xy_dis[
            int(cube_shape[0]) * 2 : cube_shape[0] + int(cube_shape[0]) * 2,
            int(cube_shape[1]) * 2 : cube_shape[1] + int(cube_shape[1]) * 2,
        ].copy()
        xy_dis_drag = xy_dis_drag[
            int(cube_shape[0]) * 2 : cube_shape[0] + int(cube_shape[0]) * 2,
            int(cube_shape[1]) * 2 : cube_shape[1] + int(cube_shape[1]) * 2,
        ].copy()

        # taper edges of xy_dis_drag in 2d to avoid edge artifacts in fft
        print("    ... xy_dis_drag.shape = " + str(xy_dis_drag.shape))
        print(
            "    ... xy_dis_drag[xy_dis_drag>0.].size = "
            + str(xy_dis_drag[xy_dis_drag > 0.0].size)
        )
        print(
            "    ... xy_dis_drag.shape min/mean/max= "
            + str((xy_dis_drag.min(), xy_dis_drag.mean(), xy_dis_drag.max()))
        )
        try:
            self.plot_counter += 1
        except:
            self.plot_counter = 0

        # xy_dis_rotated = rotate(xy_dis, degrees(alpha), mode='constant')
        print("   ...", xy_dis_rotated.shape, xy_dis.shape)

        # Normalize
        # xy_dis_rotated = rotate(xy_dis, degrees(alpha), mode='constant')
        print("   ...", xy_dis_rotated.shape, xy_dis.shape)
        # Normalize
        new_matrix = xy_dis_rotated
        xy_dis_norm = xy_dis
        x_center = np.where(new_matrix == new_matrix.max())[0][0]
        y_center = np.where(new_matrix == new_matrix.max())[1][0]

        # Pad event and move it to maximum depth location
        x_pad = 0
        y_pad = 0
        x_roll = z_idx[0] - x_center
        y_roll = z_idx[1] - y_center
        print(f"    ... padding x,y and roll x,y = {x_pad} {y_pad} {x_roll} {y_roll}")
        print(
            f"    ... Max displacement point before rotation and adding of padding x,y = {x_center} {y_center}"
        )
        new_matrix = np.lib.pad(
            new_matrix, ((abs(x_pad), abs(x_pad)), (abs(y_pad), abs(y_pad))), "edge"
        )
        new_matrix = np.roll(new_matrix, int(x_roll), axis=0)
        new_matrix = np.roll(new_matrix, int(y_roll), axis=1)
        new_matrix = new_matrix[
            abs(x_pad) : cube_shape[0] + abs(x_pad),
            abs(y_pad) : cube_shape[1] + abs(y_pad),
        ]
        # Check that fault center is at the right place
        x_center = np.where(new_matrix == new_matrix.max())[0]
        y_center = np.where(new_matrix == new_matrix.max())[1]
        print(
            f"    ... Max displacement point after rotation and removal of padding x,y = {x_center} {y_center}"
        )
        print(f"    ... z_idx = {z_idx[0]}, {z_idx[1]}")
        print(
            f"    ... Difference from origin z_idx = {x_center - z_idx[0]}, {y_center - z_idx[1]}"
        )

        # Build cube of lateral variable displacement
        xy_dis = new_matrix.reshape(cube_shape[0], cube_shape[1], 1)
        # Get enlarged fault plane
        bb = np.zeros_like(ellipsoid)
        for j in range(bb.shape[-1]):
            bb[:, :, j] = j
        stretch_times_effects_drag = bb - xy_dis * z_shift(range(cube_shape[2]))
        fault_plane_classification = np.where(
            (z_on_ellipse == 1)
            & ((origtime - stretch_times_effects_drag) > infill_factor),
            1,
            0,
        )
        hockey_stick = 0
        # Define as 0's, to be updated if necessary
        fault_plane_classification_drag = np.zeros_like(z_on_ellipse)
        if throw >= 0.85 * high_fault_throw:
            hockey_stick = 1
            # Check for non zero fault_plane_classification, to avoid division by 0
            if np.count_nonzero(fault_plane_classification) > 0:
                # Generate Hockey sticks by convolving small xy displacement with fault segment
                fault_plane_classification_drag = signal.fftconvolve(
                    fault_plane_classification,
                    np.reshape(xy_dis_drag, (cube_shape[0], cube_shape[1], 1)),
                    mode="same",
                )
                fault_plane_classification_drag = (
                    fault_plane_classification_drag
                    / np.amax(fault_plane_classification_drag.flatten())
                )

        xy_dis = xy_dis * np.ones_like(self.vols.geologic_age)
        xy_dis_classification = xy_dis.copy()
        interpolation = True

        xy_dis = xy_dis - fault_plane_classification_drag
        xy_dis = np.where(xy_dis < 0, 0, xy_dis)
        stretch_times = xy_dis * z_shift(range(cube_shape[2]))
        stretch_times_classification = xy_dis_classification * z_shift(
            range(cube_shape[2])
        )

        if self.cfg.qc_plots:
            self.fault_summary_plot(
                ff,
                z,
                throw,
                sigma,
                _x,
                _y,
                xy_dis_norm,
                ellipsoid,
                z_idx,
                xy_dis,
                index,
                alpha,
            )

        return stretch_times, stretch_times_classification, interpolation, hockey_stick

    def fault_summary_plot(
        self,
        ff,
        z,
        throw,
        sigma,
        x,
        y,
        xy_dis_norm,
        ellipsoid,
        z_idx,
        xy_dis,
        index,
        alpha,
    ) -> None:
        """
        Fault Summary Plot
        ------------------

        Generates a fault summary plot.

        Parameters
        ----------
        ff : _type_
            _description_
        z : _type_
            _description_
        throw : _type_
            _description_
        sigma : _type_
            _description_
        x : _type_
            _description_
        y : _type_
            _description_
        xy_dis_norm : _type_
            _description_
        ellipsoid : _type_
            _description_
        z_idx : _type_
            _description_
        xy_dis : _type_
            _description_
        index : _type_
            _description_
        alpha : _type_
            _description_
        
        Returns
        -------
        None
        """
        import os
        from math import degrees
        from datagenerator.util import import_matplotlib

        plt = import_matplotlib()
        # Import axes3d, required to create plot with projection='3d' below. DO NOT REMOVE!
        from mpl_toolkits.mplot3d import axes3d
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Make summary picture
        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax0, ax1, ax2, ax3 = axes.flatten()
        fig.set_size_inches(10, 8)
        # PLot Z displacement
        ax0.plot(ff(z))
        ax0.set_title("Fault with throw %s and sigma %s" % (throw, sigma))
        ax0.set_xlabel("Z axis")
        ax0.set_ylabel("Throw")
        # Plot 3D XY displacement
        ax1.axis("off")
        ax1 = fig.add_subplot(222, projection="3d")
        ax1.plot_surface(x, y, xy_dis_norm, cmap="Spectral", linewidth=0)
        ax1.set_xlabel("X axis")
        ax1.set_ylabel("Y axis")
        ax1.set_zlabel("Throw fraction")
        ax1.set_title("3D XY displacement")
        # Ellipsoid location
        weights = ellipsoid[:, :, z_idx[2]]
        # Plot un-rotated XY displacement
        cax2 = ax2.imshow(np.rot90(xy_dis_norm, 3))
        # Levels for imshow contour
        levels = np.arange(0, 1.1, 0.1)
        # Plot contour
        ax2.contour(np.rot90(xy_dis_norm, 3), levels, colors="k", linestyles="-")
        divider = make_axes_locatable(ax2)
        cax4 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(cax2, cax=cax4)
        ax2.set_xlabel("X axis")
        ax2.set_ylabel("Y axis")
        ax2.set_title("2D projection of XY displacement unrotated")
        ############################################################
        # Plot rotated displacement and contour
        cax3 = ax3.imshow(np.rot90(xy_dis[:, :, z_idx[2]], 3))
        ax3.contour(
            np.rot90(xy_dis[:, :, z_idx[2]], 3), levels, colors="k", linestyles="-"
        )
        # Add ellipsoid shape
        ax3.contour(np.rot90(weights, 3), levels=[1], colors="r", linestyles="-")
        divider = make_axes_locatable(ax3)
        cax5 = divider.append_axes("right", size="5%", pad=0.05)
        cbar3 = fig.colorbar(cax3, cax=cax5)
        ax3.set_xlabel("X axis")
        ax3.set_ylabel("Y axis")
        ax3.set_title(
            "2D projection of XY displacement rotated by %s degrees"
            % (int(degrees(alpha)))
        )
        plt.suptitle(
            "XYZ displacement parameters for fault Nr %s" % str(index),
            fontweight="bold",
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        image_path = os.path.join(
            self.cfg.work_subfolder, "QC_plot__Fault_%s.png" % str(index)
        )
        plt.savefig(image_path, format="png")
        plt.close(fig)

    @staticmethod
    def create_binary_segmentations_post_faulting(cube, segmentation_threshold):
        cube[cube >= segmentation_threshold] = 1.0
        return cube

    def reassign_channel_segment_encoding(
        self,
        faulted_age,
        floodplain_shale,
        channel_fill,
        shale_channel_drape,
        levee,
        crevasse,
        channel_flag_lut,
    ):
        # Generate list of horizons with channel episodes
        channel_segments = np.zeros_like(floodplain_shale)
        channel_horizons_list = list()
        for i in range(self.faulted_depth_maps.shape[2] - 7):
            if channel_flag_lut[i] == 1:
                channel_horizons_list.append(i)
        channel_horizons_list.append(999)

        # Re-assign correct channel segments encoding
        for i, iLayer in enumerate(channel_horizons_list[:-1]):
            if iLayer != 0 and iLayer < self.faulted_depth_maps.shape[-1]:
                # loop through channel facies
                #    --- 0  is floodplain shale
                #    --- 1  is channel fill (sand)
                #    --- 2  is shale channel drape
                #    --- 3  is levee (mid quality sand)
                #    --- 4  is crevasse (low quality sand)
                for j in range(1000, 4001, 1000):
                    print(
                        " ... re-assign channel segments after faulting: i, iLayer, j = ",
                        i,
                        iLayer,
                        j,
                    )
                    channel_facies_code = j + iLayer
                    layers_mask = np.where(
                        (
                            (faulted_age >= iLayer)
                            & (faulted_age < channel_horizons_list[i + 1])
                        ),
                        1,
                        0,
                    )
                    if j == 1000:
                        channel_segments[
                            np.logical_and(layers_mask == 1, channel_fill == 1)
                        ] = channel_facies_code
                        faulted_age[
                            np.logical_and(layers_mask == 1, channel_fill == 1)
                        ] = channel_facies_code
                    elif j == 2000:
                        channel_segments[
                            np.logical_and(layers_mask == 1, shale_channel_drape == 1)
                        ] = channel_facies_code
                        faulted_age[
                            np.logical_and(layers_mask == 1, shale_channel_drape == 1)
                        ] = channel_facies_code
                    elif j == 3000:
                        channel_segments[
                            np.logical_and(layers_mask == 1, levee == 1)
                        ] = channel_facies_code
                        faulted_age[
                            np.logical_and(layers_mask == 1, levee == 1)
                        ] = channel_facies_code
                    elif j == 4000:
                        channel_segments[
                            np.logical_and(layers_mask == 1, crevasse == 1)
                        ] = channel_facies_code
                        faulted_age[
                            np.logical_and(layers_mask == 1, crevasse == 1)
                        ] = channel_facies_code
        print(" ... finished Re-assign correct channel segments")
        return channel_segments, faulted_age

    def _get_fault_mode(self):
        if self.cfg.mode == 0:
            return self._fault_params_random
        elif self.cfg.mode == 1.0:
            if self.cfg.clustering == 0:
                return self._fault_params_self_branching
            elif self.cfg.clustering == 1:
                return self._fault_params_stairs
            elif self.cfg.clustering == 2:
                return self._fault_params_relay_ramps
            else:
                raise ValueError(self.cfg.clustering)

        elif self.cfg.mode == 2.0:
            return self._fault_params_horst_graben
        else:
            raise ValueError(self.cfg.mode)

    def _fault_params_random(self):
        print(f" ... {self.cfg.number_faults} faults will be inserted randomly")

        # Fault origin
        x0_min = int(self.cfg.cube_shape[0] / 4.0)
        x0_max = int(self.cfg.cube_shape[0] / 2.0)
        y0_min = int(self.cfg.cube_shape[1] / 4.0)
        y0_max = int(self.cfg.cube_shape[1] / 2.0)

        # Principal semi-axes location of fault ellipsoid
        a = np.random.uniform(100, 600, self.cfg.number_faults) ** 2
        b = np.random.uniform(100, 600, self.cfg.number_faults) ** 2
        x0 = np.random.uniform(
            x0_min - np.sqrt(a), np.sqrt(a) + x0_max, self.cfg.number_faults
        )
        y0 = np.random.uniform(
            y0_min - np.sqrt(b), np.sqrt(b) + y0_max, self.cfg.number_faults
        )
        z0 = np.random.uniform(
            -self.cfg.cube_shape[2] * 2.0,
            -self.cfg.cube_shape[2] * 6.0,
            self.cfg.number_faults,
        )
        _c0 = self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0 - z0
        _c1 = _c0 + self.cfg.cube_shape[2] * self.cfg.infill_factor / 4.0
        c = np.random.uniform(_c0, _c1, self.cfg.number_faults) ** 2
        tilt_pct = np.random.uniform(0.1, 0.75, self.cfg.number_faults)
        throw_lut = np.random.uniform(
            low=self.cfg.low_fault_throw,
            high=self.cfg.high_fault_throw,
            size=self.cfg.number_faults,
        )

        fault_param_dict = {
            "a": a,
            "b": b,
            "c": c,
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "tilt_pct": tilt_pct,
            "throw": throw_lut,
        }

        return fault_param_dict

    def _fault_params_self_branching(self):
        print(
            f" ... {self.cfg.number_faults} faults will be inserted in clustered mode with"
        )
        print(" ... Self branching")
        x0_min = 0
        x0_max = int(self.cfg.cube_shape[0])
        y0_min = 0
        y0_max = int(self.cfg.cube_shape[1])
        # Self Branching mode means that faults are inserted side by side with no separation.
        number_of_branches = int(self.cfg.number_faults / 3) + int(
            self.cfg.number_faults % 3 > 0
        )

        for i in range(number_of_branches):
            # Initialize first fault center, offset center between each branch
            print(" ... Computing branch number ", i)
            b_ini = np.array(np.random.uniform(100, 600) ** 2)
            a_ini = np.array(np.random.uniform(100, 600) ** 2)
            if a_ini > b_ini:
                while np.sqrt(b_ini) < self.cfg.cube_shape[1]:
                    print(" ... Recomputing b_ini for better branching")
                    b_ini = np.array(np.random.uniform(100, 600) ** 2)
                x0_ini = np.array(np.random.uniform(x0_min, x0_max))
                range_1 = list(range(int(y0_min - np.sqrt(b_ini)), 0))
                range_2 = list(range(int(self.cfg.cube_shape[1])))
                y0_ini = np.array(
                    np.random.choice(range_1 + range_2, int(np.sqrt(b_ini) + y0_max))
                )
                side = "x"
            else:
                while np.sqrt(a_ini) < self.cfg.cube_shape[0]:
                    print(" ... Recomputing a_ini for better branching")
                    a_ini = np.array(np.random.uniform(100, 600) ** 2)
                range_1 = list(range(int(x0_min - np.sqrt(a_ini)), 0))
                range_2 = list(range(int(self.cfg.cube_shape[0])))
                x0_ini = np.array(
                    np.array(
                        np.random.choice(
                            range_1 + range_2, int(np.sqrt(a_ini) + x0_max)
                        )
                    )
                )
                y0_ini = np.array(np.random.uniform(y0_min, y0_max))
                side = "y"
            # Compute the rest of the initial parameters normally
            z0_ini = np.array(
                np.random.uniform(
                    -self.cfg.cube_shape[2] * 2.0, -self.cfg.cube_shape[2] * 6.0
                )
            )

            _c0_ini = self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0 - z0_ini
            _c1_ini = _c0_ini + self.cfg.cube_shape[2] * self.cfg.infill_factor / 4.0
            c_ini = np.array(np.random.uniform(_c0_ini, _c1_ini) ** 2)
            tilt_pct_ini = np.array(np.random.uniform(0.1, 0.75))
            # direction = np.random.choice([-1, 1])
            throw_lut_ini = np.array(
                np.random.uniform(
                    low=self.cfg.low_fault_throw, high=self.cfg.high_fault_throw
                )
            )
            if i == 0:
                # Initialize return parameters
                a = a_ini.copy()
                b = b_ini.copy()
                c = c_ini.copy()
                x0 = x0_ini.copy()
                y0 = y0_ini.copy()
                z0 = z0_ini.copy()
                tilt_pct = tilt_pct_ini.copy()
                throw_lut = throw_lut_ini.copy()
            else:
                a = np.append(a, a_ini)
                b = np.append(b, b_ini)
                c = np.append(c, c_ini)
                x0 = np.append(x0, x0_ini)
                y0 = np.append(y0, y0_ini)
                z0 = np.append(z0, z0_ini)
                tilt_pct = np.append(tilt_pct, tilt_pct_ini)
                throw_lut = np.append(throw_lut, throw_lut_ini)
            # Construct fault in branch
            if i < number_of_branches - 1 or self.cfg.number_faults % 3 == 0:
                fault_in_branch = 2
            else:
                fault_in_branch = self.cfg.number_faults % 3
            print(" ... Branch along axis ", side)

            direction = 1
            if side == "x":
                if np.all(y0_ini > np.abs(self.cfg.cube_shape[1] - y0_ini)):
                    print("     ... Building from right to left")
                    direction = -1
                else:
                    print("     ... Building from left to right")
            else:
                if np.all(x0_ini > np.abs(self.cfg.cube_shape[0] - x0_ini)):
                    print("     ... Building from left to right")
                    direction = -1
                else:
                    print("     ... Building from right to left")
            move = 1
            for j in range(
                i * number_of_branches + 1, fault_in_branch + i * number_of_branches + 1
            ):
                print("     ... Computing fault number ", j)
                # Allow 20% deviation from initial fault parameter
                a_ramp = np.random.uniform(0.8, 1.2) * a_ini.copy()
                b_ramp = np.random.uniform(0.8, 1.2) * b_ini.copy()
                c_ramp = c_ini.copy()
                if side == "x":
                    x0_ramp = x0_ini + direction * move * int(
                        self.cfg.cube_shape[0] / 3.0
                    )
                    y0_ramp = np.random.uniform(0.9, 1.1) * y0_ini
                else:
                    y0_ramp = y0_ini + direction * move * int(
                        self.cfg.cube_shape[0] / 3.0
                    )
                    x0_ramp = np.random.uniform(0.9, 1.1) * x0_ini
                z0_ramp = z0_ini.copy()
                tilt_pct_ramp = tilt_pct_ini * np.random.uniform(0.85, 1.15)
                # Add to existing
                throw_lut_ramp = np.random.uniform(
                    low=self.cfg.low_fault_throw, high=self.cfg.high_fault_throw
                )
                throw_lut = np.append(throw_lut, throw_lut_ramp)
                a = np.append(a, a_ramp)
                b = np.append(b, b_ramp)
                c = np.append(c, c_ramp)
                x0 = np.append(x0, x0_ramp)
                y0 = np.append(y0, y0_ramp)
                z0 = np.append(z0, z0_ramp)
                tilt_pct = np.append(tilt_pct, tilt_pct_ramp)
                move += 1

        fault_param_dict = {
            "a": a,
            "b": b,
            "c": c,
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "tilt_pct": tilt_pct,
            "throw": throw_lut,
        }
        return fault_param_dict

    def _fault_params_stairs(self):
        print(" ... Stairs like feature")
        # Initialize value for first fault
        x0_min = int(self.cfg.cube_shape[0] / 4.0)
        x0_max = int(self.cfg.cube_shape[0] / 2.0)
        y0_min = int(self.cfg.cube_shape[1] / 4.0)
        y0_max = int(self.cfg.cube_shape[1] / 2.0)

        b_ini = np.array(np.random.uniform(100, 600) ** 2)
        a_ini = np.array(np.random.uniform(100, 600) ** 2)
        x0_ini = np.array(
            np.random.uniform(x0_min - np.sqrt(a_ini), np.sqrt(a_ini) + x0_max)
        )
        y0_ini = np.array(
            np.random.uniform(y0_min - np.sqrt(b_ini), np.sqrt(b_ini) + y0_max)
        )
        z0_ini = np.array(
            np.random.uniform(
                -self.cfg.cube_shape[2] * 2.0, -self.cfg.cube_shape[2] * 6.0
            )
        )

        _c0_ini = self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0 - z0_ini
        _c1_ini = _c0_ini + self.cfg.cube_shape[2] * self.cfg.infill_factor / 4.0
        c_ini = np.array(np.random.uniform(_c0_ini, _c1_ini) ** 2)

        tilt_pct_ini = np.array(np.random.uniform(0.1, 0.75))
        throw_lut = np.random.uniform(
            self.cfg.low_fault_throw, self.cfg.high_fault_throw, self.cfg.number_faults
        )
        direction = np.random.choice([-1, 1])
        separation_x = np.random.randint(1, 4)
        separation_y = np.random.randint(1, 4)
        # Initialize return parameters
        a = a_ini.copy()
        b = b_ini.copy()
        c = c_ini.copy()
        x0 = x0_ini.copy()
        y0 = y0_ini.copy()
        z0 = z0_ini.copy()
        x0_prec = x0_ini
        y0_prec = y0_ini
        tilt_pct = tilt_pct_ini.copy()
        for i in range(self.cfg.number_faults - 1):
            a_ramp = a_ini.copy() * np.random.uniform(0.8, 1.2)
            b_ramp = b_ini.copy() * np.random.uniform(0.8, 1.2)
            c_ramp = c_ini.copy() * np.random.uniform(0.8, 1.2)
            direction = np.random.choice([-1, 1])
            x0_ramp = (
                x0_prec + separation_x * direction * x0_ini / self.cfg.number_faults
            )
            y0_ramp = (
                y0_prec + separation_y * direction * y0_ini / self.cfg.number_faults
            )
            z0_ramp = z0_ini.copy()
            tilt_pct_ramp = tilt_pct_ini * np.random.uniform(0.85, 1.15)
            x0_prec = x0_ramp
            y0_prec = y0_ramp
            # Add to existing
            a = np.append(a, a_ramp)
            b = np.append(b, b_ramp)
            c = np.append(c, c_ramp)
            x0 = np.append(x0, x0_ramp)
            y0 = np.append(y0, y0_ramp)
            z0 = np.append(z0, z0_ramp)
            tilt_pct = np.append(tilt_pct, tilt_pct_ramp)

        fault_param_dict = {
            "a": a,
            "b": b,
            "c": c,
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "tilt_pct": tilt_pct,
            "throw": throw_lut,
        }
        return fault_param_dict

    def _fault_params_relay_ramps(self):
        print(" ... relay ramps")
        # Initialize value for first fault
        # Maximum of 3 fault per ramp
        x0_min = int(self.cfg.cube_shape[0] / 4.0)
        x0_max = int(self.cfg.cube_shape[0] / 2.0)
        y0_min = int(self.cfg.cube_shape[1] / 4.0)
        y0_max = int(self.cfg.cube_shape[1] / 2.0)
        number_of_branches = int(self.cfg.number_faults / 3) + int(
            self.cfg.number_faults % 3 > 0
        )

        for i in range(number_of_branches):
            # Initialize first fault center, offset center between each branch
            print(" ... Computing branch number ", i)
            b_ini = np.array(np.random.uniform(100, 600) ** 2)
            a_ini = np.array(np.random.uniform(100, 600) ** 2)
            x0_ini = np.array(
                np.random.uniform(x0_min - np.sqrt(a_ini) / 2, np.sqrt(a_ini) + x0_max)
                / 2
            )
            y0_ini = np.array(
                np.random.uniform(y0_min - np.sqrt(b_ini) / 2, np.sqrt(b_ini) + y0_max)
                / 2
            )
            # Compute the rest of the initial parameters normally
            z0_ini = np.array(
                np.random.uniform(
                    -self.cfg.cube_shape[2] * 2.0, -self.cfg.cube_shape[2] * 6.0
                )
            )
            c_ini = np.array(
                np.random.uniform(
                    self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0 - z0_ini,
                    self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0
                    - z0_ini
                    + self.cfg.cube_shape[2] * self.cfg.infill_factor / 4.0,
                )
                ** 2
            )
            tilt_pct_ini = np.array(np.random.uniform(0.1, 0.75))
            throw_lut = np.random.uniform(
                self.cfg.low_fault_throw,
                self.cfg.high_fault_throw,
                self.cfg.number_faults,
            )
            # Initialize return parameters
            if i == 0:
                a = a_ini.copy()
                b = b_ini.copy()
                c = c_ini.copy()
                x0 = x0_ini.copy()
                y0 = y0_ini.copy()
                z0 = z0_ini.copy()
                tilt_pct = tilt_pct_ini.copy()
            else:
                a = np.append(a, a_ini)
                b = np.append(b, b_ini)
                c = np.append(c, c_ini)
                x0 = np.append(x0, x0_ini)
                y0 = np.append(y0, y0_ini)
                z0 = np.append(z0, z0_ini)
                tilt_pct = np.append(tilt_pct, tilt_pct_ini)
                # Construct fault in branch
            if i < number_of_branches - 1 or self.cfg.number_faults % 3 == 0:
                fault_in_branche = 2
            else:
                fault_in_branche = self.cfg.number_faults % 3

            direction = np.random.choice([-1, 1])
            move = 1
            x0_prec = x0_ini
            y0_prec = y0_ini
            for j in range(
                i * number_of_branches + 1,
                fault_in_branche + i * number_of_branches + 1,
            ):
                print("     ... Computing fault number ", j)
                # Allow 20% deviation from initial fault parameter
                a_ramp = np.random.uniform(0.8, 1.2) * a_ini.copy()
                b_ramp = np.random.uniform(0.8, 1.2) * b_ini.copy()
                c_ramp = c_ini.copy()
                direction = np.random.choice([-1, 1])
                x0_ramp = x0_prec * np.random.uniform(0.8, 1.2) + direction * x0_ini / (
                    fault_in_branche
                )
                y0_ramp = y0_prec * np.random.uniform(0.8, 1.2) + direction * y0_ini / (
                    fault_in_branche
                )
                z0_ramp = z0_ini.copy()
                tilt_pct_ramp = tilt_pct_ini * np.random.uniform(0.85, 1.15)
                # Add to existing
                throw_LUT_ramp = np.random.uniform(
                    low=self.cfg.low_fault_throw, high=self.cfg.high_fault_throw
                )
                throw_lut = np.append(throw_lut, throw_LUT_ramp)
                a = np.append(a, a_ramp)
                b = np.append(b, b_ramp)
                c = np.append(c, c_ramp)
                x0 = np.append(x0, x0_ramp)
                y0 = np.append(y0, y0_ramp)
                z0 = np.append(z0, z0_ramp)
                tilt_pct = np.append(tilt_pct, tilt_pct_ramp)
                x0_prec = x0_ramp
                y0_prec = y0_ramp
                move += 1

        fault_param_dict = {
            "a": a,
            "b": b,
            "c": c,
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "tilt_pct": tilt_pct,
            "throw": throw_lut,
        }
        return fault_param_dict

    def _fault_params_horst_graben(self):
        print(
            "   ... %s faults will be inserted as Horst and Graben"
            % (self.cfg.number_faults)
        )
        # Initialize value for first fault
        x0_min = int(self.cfg.cube_shape[0] / 2.0)
        x0_max = int(self.cfg.cube_shape[0])
        y0_min = int(self.cfg.cube_shape[1] / 2.0)
        y0_max = int(self.cfg.cube_shape[1])

        b_ini = np.array(np.random.uniform(100, 600) ** 2)
        a_ini = np.array(np.random.uniform(100, 600) ** 2)
        if a_ini > b_ini:
            side = "x"
            while np.sqrt(b_ini) < self.cfg.cube_shape[1]:
                print("   ... Recomputing b_ini for better branching")
                b_ini = np.array(np.random.uniform(100, 600) ** 2)
            x0_ini = np.array(np.random.uniform(0, self.cfg.cube_shape[0]))
            # Compute so that first is near center
            range_1 = list(
                range(int(y0_min - np.sqrt(b_ini)), int(y0_max - np.sqrt(b_ini)))
            )
            range_2 = list(
                range(int(y0_min + np.sqrt(b_ini)), int(y0_max + np.sqrt(b_ini)))
            )
            y0_ini = np.array(np.random.choice(range_1 + range_2))
        else:
            side = "y"
            while np.sqrt(a_ini) < self.cfg.cube_shape[0]:
                print(" ... Recomputing a_ini for better branching")
                a_ini = np.array(np.random.uniform(100, 600) ** 2)
            # compute so that first is near center
            range_1 = list(
                range(int(x0_min - np.sqrt(a_ini)), int(x0_max - np.sqrt(a_ini)))
            )
            range_2 = list(
                range(int(x0_min + np.sqrt(a_ini)), int(x0_max + np.sqrt(a_ini)))
            )
            x0_ini = np.array(np.random.choice(range_1 + range_2))
            y0_ini = np.array(np.random.uniform(0, self.cfg.cube_shape[1]))
        z0_ini = np.array(
            np.random.uniform(
                -self.cfg.cube_shape[2] * 2.0, -self.cfg.cube_shape[2] * 6.0
            )
        )
        c_ini = np.array(
            np.random.uniform(
                self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0 - z0_ini,
                self.cfg.cube_shape[2] * self.cfg.infill_factor * 4.0
                - z0_ini
                + self.cfg.cube_shape[2] * self.cfg.infill_factor / 2.0,
            )
            ** 2
        )
        tilt_pct_ini = np.array(np.random.uniform(0.1, 0.75))
        throw_lut = np.random.uniform(
            self.cfg.low_fault_throw, self.cfg.high_fault_throw, self.cfg.number_faults
        )
        # Initialize return parameters
        a = a_ini.copy()
        b = b_ini.copy()
        c = c_ini.copy()
        x0 = x0_ini.copy()
        y0 = y0_ini.copy()
        z0 = z0_ini.copy()

        tilt_pct = tilt_pct_ini.copy()
        direction = "odd"
        mod = ["new"]

        x0_even = x0_ini
        y0_even = y0_ini
        if side == "x":
            # X only moves marginally
            y0_odd = int(self.cfg.cube_shape[1]) - y0_even
            x0_odd = x0_even
            direction_sign = np.sign(y0_ini)
            direction_sidex = 0
            direction_sidey = 1
        else:
            # Y only moves marginally
            x0_odd = int(self.cfg.cube_shape[0]) - x0_even
            y0_odd = y0_even
            direction_sign = np.sign(x0_ini)
            direction_sidex = 1
            direction_sidey = 0

        for i in range(self.cfg.number_faults - 1):
            if direction == "odd":
                # Put the next point as a mirror shifted by maximal shift and go backward
                a_ramp = np.random.uniform(0.8, 1.2) * a_ini.copy()
                b_ramp = np.random.uniform(0.8, 1.2) * b_ini.copy()
                c_ramp = c_ini.copy()
                x0_ramp = (
                    -1
                    * direction_sign
                    * direction_sidex
                    * int(self.cfg.cube_shape[0])
                    * (i - 1)
                    / self.cfg.number_faults
                ) + x0_odd * np.random.uniform(0.8, 1.2)
                y0_ramp = (
                    -1
                    * direction_sign
                    * direction_sidey
                    * int(self.cfg.cube_shape[0])
                    * (i - 1)
                    / self.cfg.number_faults
                ) + y0_odd * np.random.uniform(0.8, 1.2)
                z0_ramp = z0_ini.copy()
                tilt_pct_ramp = tilt_pct_ini * np.random.uniform(0.85, 1.15)
                x0_prec = x0_ramp
                y0_prec = y0_ramp
                # Add to existing
                a = np.append(a, a_ramp)
                b = np.append(b, b_ramp)
                c = np.append(c, c_ramp)
                x0 = np.append(x0, x0_ramp)
                y0 = np.append(y0, y0_ramp)
                z0 = np.append(z0, z0_ramp)
                tilt_pct = np.append(tilt_pct, tilt_pct_ramp)
                direction = "even"
                mod.append("old")
            elif direction == "even":
                # Put next to ini
                a_ramp = np.random.uniform(0.8, 1.2) * a_ini.copy()
                b_ramp = np.random.uniform(0.8, 1.2) * b_ini.copy()
                c_ramp = c_ini.copy()
                x0_ramp = direction_sign * direction_sidex * int(
                    self.cfg.cube_shape[0]
                ) * (i - 1) / self.cfg.number_faults + x0_even * np.random.uniform(
                    0.8, 1.2
                )
                y0_ramp = direction_sign * direction_sidey * int(
                    self.cfg.cube_shape[0]
                ) * (i - 1) / self.cfg.number_faults + y0_even * np.random.uniform(
                    0.8, 1.2
                )
                z0_ramp = z0_ini.copy()
                tilt_pct_ramp = tilt_pct_ini * np.random.uniform(0.85, 1.15)
                # Add to existing
                a = np.append(a, a_ramp)
                b = np.append(b, b_ramp)
                c = np.append(c, c_ramp)
                x0 = np.append(x0, x0_ramp)
                y0 = np.append(y0, y0_ramp)
                z0 = np.append(z0, z0_ramp)
                tilt_pct = np.append(tilt_pct, tilt_pct_ramp)
                direction = "odd"
                mod.append("new")

        fault_param_dict = {
            "a": a,
            "b": b,
            "c": c,
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "tilt_pct": tilt_pct,
            "throw": throw_lut,
        }
        return fault_param_dict


def find_zero_thickness_onlapping_layers(z, onlap_list):
    onlap_zero_z = dict()
    for layer in onlap_list:
        for x in range(layer, 1, -1):
            thickness = z[..., layer] - z[..., x - 1]
            zeros = np.where(thickness == 0.0)
            if zeros[0].size > 0:
                onlap_zero_z[f"{layer},{x-1}"] = zeros
    return onlap_zero_z


def fix_zero_thickness_fan_layers(z, layer_number, thickness):
    """
    Clip fan layers to horizon below the fan layer in areas where the fan thickness is zero

    Parameters
    ----------
    z : ndarray - depth maps
    layer_number : 1d array - horizon numbers which contain fans
    thickness : tuple of ndarrays - original thickness maps of the fans

    Returns
    -------
    zmaps : ndarray - depth maps with fan layers clipped to lower horizons where thicknesses is zero
    """
    zmaps = z.copy()
    zmaps[..., layer_number][np.where(thickness == 0.0)] = zmaps[..., layer_number + 1][
        np.where(thickness == 0.0)
    ]
    return zmaps


def fix_zero_thickness_onlap_layers(
    faulted_depth_maps: np.ndarray,
    onlap_dict: dict
) -> np.ndarray:
    """fix_zero_thickness_onlap_layers _summary_

    Parameters
    ----------
    faulted_depth_maps : np.ndarray
        The depth maps with faults
    onlap_dict : dict
        Onlaps dictionary

    Returns
    -------
    zmaps : np.ndarray
        Fixed depth maps
    """
    zmaps = faulted_depth_maps.copy()
    for layers, idx in onlap_dict.items():
        onlap_layer = int(str(layers).split(",")[0])
        layer_to_clip = int(str(layers).split(",")[1])
        zmaps[..., layer_to_clip][idx] = zmaps[..., onlap_layer][idx]

    return zmaps
