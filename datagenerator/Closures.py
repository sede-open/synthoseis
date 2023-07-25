import os

import numpy as np
from datagenerator.Horizons import Horizons
from datagenerator.Geomodels import Geomodel
from datagenerator.Parameters import Parameters
from skimage import morphology, measure
from scipy.ndimage import minimum_filter, maximum_filter


class Closures(Horizons, Geomodel, Parameters):
    def __init__(self, parameters, faults, facies, onlap_horizon_list):
        self.closure_dict = dict()
        self.cfg = parameters
        self.faults = faults
        self.facies = facies
        self.onlap_list = onlap_horizon_list
        self.top_lith_facies = None
        self.closure_vol_shape = self.faults.faulted_age_volume.shape
        self.closure_segments = self.cfg.hdf_init(
            "closure_segments", shape=self.closure_vol_shape
        )
        self.oil_closures = self.cfg.hdf_init(
            "oil_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.gas_closures = self.cfg.hdf_init(
            "gas_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.brine_closures = self.cfg.hdf_init(
            "brine_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.simple_closures = self.cfg.hdf_init(
            "simple_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.strat_closures = self.cfg.hdf_init(
            "strat_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.fault_closures = self.cfg.hdf_init(
            "fault_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.hc_labels = self.cfg.hdf_init(
            "hc_labels", shape=self.closure_vol_shape, dtype="uint8"
        )

        self.all_closure_segments = self.cfg.hdf_init(
            "all_closure_segments", shape=self.closure_vol_shape
        )

        # Class attributes added from Intersect3D
        self.wide_faults = self.cfg.hdf_init(
            "wide_faults", shape=self.closure_vol_shape
        )
        self.fat_faults = self.cfg.hdf_init("fat_faults", shape=self.closure_vol_shape)
        self.onlaps_upward = self.cfg.hdf_init(
            "onlaps_upward", shape=self.closure_vol_shape
        )
        self.onlaps_downward = self.cfg.hdf_init(
            "onlaps_downward", shape=self.closure_vol_shape
        )

        # Faulted closures
        self.faulted_closures_oil = self.cfg.hdf_init(
            "faulted_closures_oil", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.faulted_closures_gas = self.cfg.hdf_init(
            "faulted_closures_gas", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.faulted_closures_brine = self.cfg.hdf_init(
            "faulted_closures_brine", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.fault_closures_oil_segment_list = list()
        self.fault_closures_gas_segment_list = list()
        self.fault_closures_brine_segment_list = list()
        self.n_fault_closures_oil = 0
        self.n_fault_closures_gas = 0
        self.n_fault_closures_brine = 0

        self.faulted_all_closures = self.cfg.hdf_init(
            "faulted_all_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.fault_all_closures_segment_list = list()
        self.n_fault_all_closures = 0

        # Onlap closures
        self.onlap_closures_oil = self.cfg.hdf_init(
            "onlap_closures_oil", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.onlap_closures_gas = self.cfg.hdf_init(
            "onlap_closures_gas", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.onlap_closures_brine = self.cfg.hdf_init(
            "onlap_closures_brine", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.onlap_closures_oil_segment_list = list()
        self.onlap_closures_gas_segment_list = list()
        self.onlap_closures_brine_segment_list = list()
        self.n_onlap_closures_oil = 0
        self.n_onlap_closures_gas = 0
        self.n_onlap_closures_brine = 0

        self.onlap_all_closures = self.cfg.hdf_init(
            "onlap_all_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.onlap_all_closures_segment_list = list()
        self.n_onlap_all_closures_oil = 0

        # Simple closures
        self.simple_closures_oil = self.cfg.hdf_init(
            "simple_closures_oil", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.simple_closures_gas = self.cfg.hdf_init(
            "simple_closures_gas", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.simple_closures_brine = self.cfg.hdf_init(
            "simple_closures_brine", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.simple_closures_oil_segment_list = list()
        self.simple_closures_gas_segment_list = list()
        self.simple_closures_brine_segment_list = list()
        self.n_4way_closures_oil = 0
        self.n_4way_closures_gas = 0
        self.n_4way_closures_brine = 0

        self.simple_all_closures = self.cfg.hdf_init(
            "simple_all_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.simple_all_closures_segment_list = list()
        self.n_4way_all_closures = 0

        # False closures
        self.false_closures_oil = self.cfg.hdf_init(
            "false_closures_oil", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.false_closures_gas = self.cfg.hdf_init(
            "false_closures_gas", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.false_closures_brine = self.cfg.hdf_init(
            "false_closures_brine", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.n_false_closures_oil = 0
        self.n_false_closures_gas = 0
        self.n_false_closures_brine = 0

        self.false_all_closures = self.cfg.hdf_init(
            "false_all_closures", shape=self.closure_vol_shape, dtype="uint8"
        )
        self.n_false_all_closures = 0

        if self.cfg.include_salt:
            self.salt_closures = self.cfg.hdf_init(
                "salt_closures", shape=self.closure_vol_shape, dtype="uint8"
            )
            self.wide_salt = self.cfg.hdf_init(
                "wide_salt", shape=self.closure_vol_shape
            )
            self.salt_closures_oil = self.cfg.hdf_init(
                "salt_bounded_closures_oil", shape=self.closure_vol_shape, dtype="uint8"
            )
            self.salt_closures_gas = self.cfg.hdf_init(
                "salt_bounded_closures_gas", shape=self.closure_vol_shape, dtype="uint8"
            )
            self.salt_closures_brine = self.cfg.hdf_init(
                "salt_bounded_closures_brine",
                shape=self.closure_vol_shape,
                dtype="uint8",
            )
            self.salt_closures_oil_segment_list = list()
            self.salt_closures_gas_segment_list = list()
            self.salt_closures_brine_segment_list = list()
            self.n_salt_closures_oil = 0
            self.n_salt_closures_gas = 0
            self.n_salt_closures_brine = 0

            self.salt_all_closures = self.cfg.hdf_init(
                "salt_bounded_all_closures", shape=self.closure_vol_shape, dtype="uint8"
            )
            self.salt_all_closures_segment_list = list()
            self.n_salt_all_closures = 0

    def create_closure_labels_from_depth_maps(
        self, depth_maps, depth_maps_infilled, max_col_height
    ):
        if self.cfg.verbose:
            print("\n\t... inside insertClosureLabels3D ")
            print(
                f"\t... depth_maps min {depth_maps.min():.2f}, mean {depth_maps.mean():.2f},"
                f" max {depth_maps.max():.2f}, cube_shape {self.cfg.cube_shape}"
            )

        # create 3D cube to hold segmentation results
        closure_segments = np.zeros(self.faults.faulted_lithology.shape, "float32")

        # create grids with grid indices
        ii, jj = self.build_meshgrid()

        # loop through horizons in 'depth_maps'
        voxel_change_count = np.zeros(self.cfg.cube_shape, dtype=np.uint8)
        layers_with_closure = 0

        avg_sand_thickness = list()
        avg_shale_thickness = list()
        avg_unit_thickness = list()
        for ihorizon in range(depth_maps.shape[2] - 1):
            avg_unit_thickness.append(
                np.mean(
                    depth_maps_infilled[..., ihorizon + 1]
                    - depth_maps_infilled[..., ihorizon]
                )
            )

            if self.top_lith_facies[ihorizon] > 0:
                # If facies is not shale, calculate a closure map for the layer
                if self.cfg.verbose:
                    print(
                        f"\n...closure voxels computation for layer {ihorizon} in horizon list."
                    )
                avg_sand_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )
                # compute a closure map
                # - identical to top structure map when not in closure, 'max flooding' depth when in closure
                # - use thicknesses converted to samples instead of ft or ms
                # - assumes that fault intersections are inserted in input map with value of 0.
                # - assumes that input map values represent depth (i.e., bigger values are deeper)
                top_structure_depth_map = depth_maps[:, :, ihorizon].copy()
                top_structure_depth_map[
                    np.isnan(top_structure_depth_map)
                ] = 0.0  # replace nans with 0.
                top_structure_depth_map /= float(self.cfg.digi)
                if self.cfg.partial_voxels:
                    top_structure_depth_map -= (
                        1.0  # account for voxels partially in layer
                    )
                base_structure_depth_map = depth_maps_infilled[
                    :, :, ihorizon + 1
                ].copy()
                base_structure_depth_map[
                    np.isnan(top_structure_depth_map)
                ] = 0.0  # replace nans with 0.
                base_structure_depth_map /= float(self.cfg.digi)
                print(
                    " ...inside create_closure_labels_from_depth_maps... ihorizon, self.top_lith_facies[ihorizon] = ",
                    ihorizon,
                    self.top_lith_facies[ihorizon],
                )
                # if there is non-zero thickness between top/base closure
                if top_structure_depth_map.min() != top_structure_depth_map.max():
                    max_column = max_col_height[ihorizon] / self.cfg.digi
                    if self.cfg.verbose:
                        print(
                            f"   ...avg depth for layer {ihorizon}.",
                            top_structure_depth_map.mean(),
                        )
                    if self.cfg.verbose:
                        print(
                            f"   ...maximum column height for layer {ihorizon}.",
                            max_column,
                        )

                    if ihorizon == 27000 or ihorizon == 1000:
                        closure_depth_map = _flood_fill(
                            top_structure_depth_map,
                            max_column_height=max_column,
                            verbose=True,
                            debug=True,
                        )
                    else:
                        closure_depth_map = _flood_fill(
                            top_structure_depth_map, max_column_height=max_column
                        )
                    closure_depth_map[closure_depth_map == 0] = top_structure_depth_map[
                        closure_depth_map == 0
                    ]
                    closure_depth_map[closure_depth_map == 1] = top_structure_depth_map[
                        closure_depth_map == 1
                    ]
                    closure_depth_map[
                        closure_depth_map == 1e5
                    ] = top_structure_depth_map[closure_depth_map == 1e5]
                    # Select the maximum value between the top sand map and the flood-filled closure map
                    closure_depth_map = np.max(
                        np.dstack((closure_depth_map, top_structure_depth_map)), axis=-1
                    )
                    closure_depth_map = np.min(
                        np.dstack((closure_depth_map, base_structure_depth_map)),
                        axis=-1,
                    )
                    if self.cfg.verbose:
                        print(
                            f"\n    ... layer {ihorizon},"
                            f"\n\ttop structure map min, max {top_structure_depth_map.min():.2f},"
                            f" {top_structure_depth_map.max():.2f}\n\tclosure_depth_map min, max"
                            f" {closure_depth_map.min():.2f} {closure_depth_map.max()}"
                        )
                    closure_thickness = closure_depth_map - top_structure_depth_map
                    closure_thickness_no_nan = closure_thickness[
                        ~np.isnan(closure_thickness)
                    ]
                    max_closure = int(np.around(closure_thickness_no_nan.max(), 0))
                    if self.cfg.verbose:
                        print(f"    ... layer {ihorizon}, max_closure {max_closure}")

                    # locate 3D zone in closure after checking that closures exist for this horizon
                    # if False in (top_structure_depth_map == closure_depth_map):
                    if max_closure > 0:
                        # locate voxels anywhere in layer where top_structure_depth_map < closure_depth_map
                        # put label in cube between top_structure_depth_map and closure_depth_map
                        top_structure_depth_map_integer = top_structure_depth_map
                        closure_depth_map_integer = closure_depth_map

                        if self.cfg.verbose:
                            closure_map_min = closure_depth_map_integer[
                                closure_depth_map_integer > 0.1
                            ].min()
                            closure_map_max = closure_depth_map_integer[
                                closure_depth_map_integer > 0.1
                            ].max()
                            print(
                                f"\t... (2) layer: {ihorizon}, max_closure; {max_closure}, top structure map min, "
                                f"max: {top_structure_depth_map.min()}, {top_structure_depth_map_integer.max()},"
                                f" closure map min, max: {closure_map_min}, {closure_map_max}"
                            )

                        slices_with_substitution = 0
                        print("    ... max_closure: {}".format(max_closure))
                        for k in range(
                            max_closure + 1
                        ):  # add one more sample than seemingly needed for round-off
                            # Subtract 2 from the closure cube shape since adding one later
                            horizon_slice = (k + top_structure_depth_map).clip(
                                0, closure_segments.shape[2] - 2
                            )
                            sublayer_kk = horizon_slice[
                                horizon_slice < closure_depth_map.astype("int")
                            ]
                            sublayer_ii = ii[
                                horizon_slice < closure_depth_map.astype("int")
                            ]
                            sublayer_jj = jj[
                                horizon_slice < closure_depth_map.astype("int")
                            ]

                            if sublayer_ii.size > 0:
                                slices_with_substitution += 1

                                i_indices = sublayer_ii
                                j_indices = sublayer_jj
                                k_indices = sublayer_kk + 1

                                try:
                                    closure_segments[
                                        i_indices, j_indices, k_indices.astype("int")
                                    ] += 1.0
                                    voxel_change_count[
                                        i_indices, j_indices, k_indices.astype("int")
                                    ] += 1
                                except IndexError:
                                    print("\nIndex is out of bounds.")
                                    print(f"\tclosure_segments: {closure_segments}")
                                    print(f"\tvoxel_change_count: {voxel_change_count}")
                                    print(f"\ti_indices: {i_indices}")
                                    print(f"\tj_indices: {j_indices}")
                                    print(f"\tk_indices: {k_indices.astype('int')}")
                                    pass

                        if slices_with_substitution > 0:
                            layers_with_closure += 1

                        if self.cfg.verbose:
                            print(
                                "    ... finished putting closures in closures_segments for layer ...",
                                ihorizon,
                            )

                    else:
                        continue
            else:
                # Calculate shale unit thicknesses
                avg_shale_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )

        if len(avg_sand_thickness) == 0:
            avg_sand_thickness = 0
        self.cfg.write_to_logfile(
            f"Sand Unit Thickness (m): mean: {np.mean(avg_sand_thickness):.2f}, "
            f"std: {np.std(avg_sand_thickness):.2f}, min: {np.nanmin(avg_sand_thickness):.2f}, "
            f"max: {np.max(avg_sand_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Shale Unit Thickness (m): mean: {np.mean(avg_shale_thickness):.2f}, "
            f"std: {np.std(avg_shale_thickness):.2f}, min: {np.min(avg_shale_thickness):.2f}, "
            f"max: {np.max(avg_shale_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Overall Unit Thickness (m): mean: {np.mean(avg_unit_thickness):.2f}, "
            f"std: {np.std(avg_unit_thickness):.2f}, min: {np.min(avg_unit_thickness):.2f}, "
            f"max: {np.max(avg_unit_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_mean",
            val=np.mean(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_std",
            val=np.std(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_min",
            val=np.min(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_max",
            val=np.max(avg_sand_thickness),
        )
        #
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_mean",
            val=np.mean(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_std",
            val=np.std(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_min",
            val=np.min(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_max",
            val=np.max(avg_shale_thickness),
        )

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_mean",
            val=np.mean(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_std",
            val=np.std(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_min",
            val=np.min(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_max",
            val=np.max(avg_unit_thickness),
        )

        non_zero_pixels = closure_segments[closure_segments != 0.0].shape[0]
        pct_non_zero = float(non_zero_pixels) / (
            closure_segments.shape[0]
            * closure_segments.shape[1]
            * closure_segments.shape[2]
        )
        if self.cfg.verbose:
            print(
                "    ...closure_segments min {}, mean {}, max {}, % non-zero {}".format(
                    closure_segments.min(),
                    closure_segments.mean(),
                    closure_segments.max(),
                    pct_non_zero,
                )
            )

        print(f"\t... layers_with_closure {layers_with_closure}")
        print("\t... finished putting closures in closure_segments ...\n")

        if self.cfg.verbose:
            print(
                f"\n   ...closure segments created. min: {closure_segments.min()}, "
                f"mean: {closure_segments.mean():.2f}, max: {closure_segments.max()}"
                f" voxel count: {closure_segments[closure_segments != 0].shape}"
            )

        return closure_segments

    def create_closure_labels_from_all_depth_maps(
        self, depth_maps, depth_maps_infilled, max_col_height
    ):
        if self.cfg.verbose:
            print("\n\t... inside insertClosureLabels3D ")
            print(
                f"\t... depth_maps min {depth_maps.min():.2f}, mean {depth_maps.mean():.2f},"
                f" max {depth_maps.max():.2f}, cube_shape {self.cfg.cube_shape}"
            )

        # create 3D cube to hold segmentation results
        closure_segments = np.zeros(self.faults.faulted_lithology.shape, "float32")

        # create grids with grid indices
        ii, jj = self.build_meshgrid()

        # loop through horizons in 'depth_maps'
        voxel_change_count = np.zeros(self.cfg.cube_shape, dtype=np.uint8)
        layers_with_closure = 0

        avg_sand_thickness = list()
        avg_shale_thickness = list()
        avg_unit_thickness = list()
        for ihorizon in range(depth_maps.shape[2] - 1):
            avg_unit_thickness.append(
                np.mean(
                    depth_maps_infilled[..., ihorizon + 1]
                    - depth_maps_infilled[..., ihorizon]
                )
            )
            # calculate a closure map for the layer
            if self.cfg.verbose:
                print(
                    f"\n...closure voxels computation for layer {ihorizon} in horizon list."
                )

            # compute a closure map
            # - identical to top structure map when not in closure, 'max flooding' depth when in closure
            # - use thicknesses converted to samples instead of ft or ms
            # - assumes that fault intersections are inserted in input map with value of 0.
            # - assumes that input map values represent depth (i.e., bigger values are deeper)
            top_structure_depth_map = depth_maps[:, :, ihorizon].copy()
            top_structure_depth_map[
                np.isnan(top_structure_depth_map)
            ] = 0.0  # replace nans with 0.
            top_structure_depth_map /= float(self.cfg.digi)
            if self.cfg.partial_voxels:
                top_structure_depth_map -= 1.0  # account for voxels partially in layer
            base_structure_depth_map = depth_maps_infilled[:, :, ihorizon + 1].copy()
            base_structure_depth_map[
                np.isnan(top_structure_depth_map)
            ] = 0.0  # replace nans with 0.
            base_structure_depth_map /= float(self.cfg.digi)
            print(
                " ...inside create_closure_labels_from_depth_maps... ihorizon = ",
                ihorizon,
            )
            # if there is non-zero thickness between top/base closure
            if top_structure_depth_map.min() != top_structure_depth_map.max():
                max_column = max_col_height[ihorizon] / self.cfg.digi
                if self.cfg.verbose:
                    print(
                        f"   ...avg depth for layer {ihorizon}.",
                        top_structure_depth_map.mean(),
                    )
                if self.cfg.verbose:
                    print(
                        f"   ...maximum column height for layer {ihorizon}.", max_column
                    )

                if ihorizon == 27000 or ihorizon == 1000:
                    closure_depth_map = _flood_fill(
                        top_structure_depth_map,
                        max_column_height=max_column,
                        verbose=True,
                        debug=True,
                    )
                else:
                    closure_depth_map = _flood_fill(
                        top_structure_depth_map, max_column_height=max_column
                    )
                closure_depth_map[closure_depth_map == 0] = top_structure_depth_map[
                    closure_depth_map == 0
                ]
                closure_depth_map[closure_depth_map == 1] = top_structure_depth_map[
                    closure_depth_map == 1
                ]
                closure_depth_map[closure_depth_map == 1e5] = top_structure_depth_map[
                    closure_depth_map == 1e5
                ]
                # Select the maximum value between the top sand map and the flood-filled closure map
                closure_depth_map = np.max(
                    np.dstack((closure_depth_map, top_structure_depth_map)), axis=-1
                )
                closure_depth_map = np.min(
                    np.dstack((closure_depth_map, base_structure_depth_map)), axis=-1
                )
                if self.cfg.verbose:
                    print(
                        f"\n    ... layer {ihorizon},"
                        f"\n\ttop structure map min, max {top_structure_depth_map.min():.2f},"
                        f" {top_structure_depth_map.max():.2f}\n\tclosure_depth_map min, max"
                        f" {closure_depth_map.min():.2f} {closure_depth_map.max()}"
                    )
                closure_thickness = closure_depth_map - top_structure_depth_map
                closure_thickness_no_nan = closure_thickness[
                    ~np.isnan(closure_thickness)
                ]
                max_closure = int(np.around(closure_thickness_no_nan.max(), 0))
                if self.cfg.verbose:
                    print(f"    ... layer {ihorizon}, max_closure {max_closure}")

                # locate 3D zone in closure after checking that closures exist for this horizon
                # if False in (top_structure_depth_map == closure_depth_map):
                if max_closure > 0:
                    # locate voxels anywhere in layer where top_structure_depth_map < closure_depth_map
                    # put label in cube between top_structure_depth_map and closure_depth_map
                    top_structure_depth_map_integer = top_structure_depth_map
                    closure_depth_map_integer = closure_depth_map

                    if self.cfg.verbose:
                        closure_map_min = closure_depth_map_integer[
                            closure_depth_map_integer > 0.1
                        ].min()
                        closure_map_max = closure_depth_map_integer[
                            closure_depth_map_integer > 0.1
                        ].max()
                        print(
                            f"\t... (2) layer: {ihorizon}, max_closure; {max_closure}, top structure map min, "
                            f"max: {top_structure_depth_map.min()}, {top_structure_depth_map_integer.max()},"
                            f" closure map min, max: {closure_map_min}, {closure_map_max}"
                        )

                    slices_with_substitution = 0
                    print("    ... max_closure: {}".format(max_closure))
                    for k in range(
                        max_closure + 1
                    ):  # add one more sample than seemingly needed for round-off
                        # Subtract 2 from the closure cube shape since adding one later
                        horizon_slice = (k + top_structure_depth_map).clip(
                            0, closure_segments.shape[2] - 2
                        )
                        sublayer_kk = horizon_slice[
                            horizon_slice < closure_depth_map.astype("int")
                        ]
                        sublayer_ii = ii[
                            horizon_slice < closure_depth_map.astype("int")
                        ]
                        sublayer_jj = jj[
                            horizon_slice < closure_depth_map.astype("int")
                        ]

                        if sublayer_ii.size > 0:
                            slices_with_substitution += 1

                            i_indices = sublayer_ii
                            j_indices = sublayer_jj
                            k_indices = sublayer_kk + 1

                            try:
                                closure_segments[
                                    i_indices, j_indices, k_indices.astype("int")
                                ] += 1.0
                                voxel_change_count[
                                    i_indices, j_indices, k_indices.astype("int")
                                ] += 1
                            except IndexError:
                                print("\nIndex is out of bounds.")
                                print(f"\tclosure_segments: {closure_segments}")
                                print(f"\tvoxel_change_count: {voxel_change_count}")
                                print(f"\ti_indices: {i_indices}")
                                print(f"\tj_indices: {j_indices}")
                                print(f"\tk_indices: {k_indices.astype('int')}")
                                pass

                    if slices_with_substitution > 0:
                        layers_with_closure += 1

                    if self.cfg.verbose:
                        print(
                            "    ... finished putting closures in closures_segments for layer ...",
                            ihorizon,
                        )

                else:
                    continue

            if self.facies[ihorizon] == 1:
                avg_sand_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )
            elif self.facies[ihorizon] == 0:
                # Calculate shale unit thicknesses
                avg_shale_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )

        # TODO  handle case where avg_sand_thickness is zero-size array
        try:
            self.cfg.write_to_logfile(
                f"Sand Unit Thickness (m): mean: {np.mean(avg_sand_thickness):.2f}, "
                f"std: {np.std(avg_sand_thickness):.2f}, min: {np.nanmin(avg_sand_thickness):.2f}, "
                f"max: {np.max(avg_sand_thickness):.2f}"
            )
        except:
            print("No sands in model")
        self.cfg.write_to_logfile(
            f"Shale Unit Thickness (m): mean: {np.mean(avg_shale_thickness):.2f}, "
            f"std: {np.std(avg_shale_thickness):.2f}, min: {np.min(avg_shale_thickness):.2f}, "
            f"max: {np.max(avg_shale_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Overall Unit Thickness (m): mean: {np.mean(avg_unit_thickness):.2f}, "
            f"std: {np.std(avg_unit_thickness):.2f}, min: {np.min(avg_unit_thickness):.2f}, "
            f"max: {np.max(avg_unit_thickness):.2f}"
        )

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_mean",
            val=np.mean(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_std",
            val=np.std(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_min",
            val=np.min(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_max",
            val=np.max(avg_sand_thickness),
        )
        #
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_mean",
            val=np.mean(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_std",
            val=np.std(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_min",
            val=np.min(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_max",
            val=np.max(avg_shale_thickness),
        )

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_mean",
            val=np.mean(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_std",
            val=np.std(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_min",
            val=np.min(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_max",
            val=np.max(avg_unit_thickness),
        )

        non_zero_pixels = closure_segments[closure_segments != 0.0].shape[0]
        pct_non_zero = float(non_zero_pixels) / (
            closure_segments.shape[0]
            * closure_segments.shape[1]
            * closure_segments.shape[2]
        )
        if self.cfg.verbose:
            print(
                "    ...closure_segments min {}, mean {}, max {}, % non-zero {}".format(
                    closure_segments.min(),
                    closure_segments.mean(),
                    closure_segments.max(),
                    pct_non_zero,
                )
            )

        print(f"\t... layers_with_closure {layers_with_closure}")
        print("\t... finished putting closures in closure_segments ...\n")

        if self.cfg.verbose:
            print(
                f"\n   ...closure segments created. min: {closure_segments.min()}, "
                f"mean: {closure_segments.mean():.2f}, max: {closure_segments.max()}"
                f" voxel count: {closure_segments[closure_segments != 0].shape}"
            )

        return closure_segments

    def find_top_lith_horizons(self):
        """
        Find horizons which are the top of layers where the lithology changes

        Combine layers of the same lithology and retain the top of these new layers for closure calculations.
        """
        top_lith_indices = list(np.array(self.onlap_list) - 1)
        for i, _ in enumerate(self.facies[:-1]):
            if i == 0:
                continue
            print(
                f"i: {i}, sand_layer_label[i-1]: {self.facies[i - 1]},"
                f" sand_layer_label[i]: {self.facies[i]}"
            )
            if self.facies[i] != self.facies[i - 1]:
                top_lith_indices.append(i)
                if self.cfg.verbose:
                    print(
                        "  ... layer lith different than layer above it. i = {}".format(
                            i
                        )
                    )
        top_lith_indices.sort()
        if self.cfg.verbose:
            print(
                "\n   ...layers selected for closure computations...\n",
                top_lith_indices,
            )
        self.top_lith_indices = np.array(top_lith_indices)
        self.top_lith_facies = self.facies[top_lith_indices]

        return top_lith_indices

    def create_closures(self):
        if self.cfg.verbose:
            print("\n\n ... create 3D labels for closure")

        # Convert nan to 0's
        old_depth_maps = np.nan_to_num(self.faults.faulted_depth_maps[:], copy=True)
        old_depth_maps_gaps = np.nan_to_num(
            self.faults.faulted_depth_maps_gaps[:], copy=True
        )

        # Convert from samples to units
        old_depth_maps_gaps = self.convert_map_from_samples_to_units(
            old_depth_maps_gaps
        )
        old_depth_maps = self.convert_map_from_samples_to_units(old_depth_maps)

        # keep only horizons corresponding to top of layers where lithology changes
        self.find_top_lith_horizons()
        all_lith_indices = np.arange(old_depth_maps.shape[-1])
        import sys

        print("All lith indices (last, then all):", self.facies[-1], all_lith_indices)
        sys.stdout.flush()

        depth_maps_gaps_top_lith = old_depth_maps_gaps[
            :, :, self.top_lith_indices
        ].copy()
        depth_maps_gaps_all_lith = old_depth_maps_gaps[:, :, all_lith_indices].copy()
        depth_maps_top_lith = old_depth_maps[:, :, self.top_lith_indices].copy()
        depth_maps_all_lith = old_depth_maps[:, :, all_lith_indices].copy()
        max_column_heights = variable_max_column_height(
            self.top_lith_indices,
            self.faults.faulted_depth_maps_gaps.shape[-1],
            self.cfg.max_column_height[0],
            self.cfg.max_column_height[1],
        )
        all_max_column_heights = variable_max_column_height(
            all_lith_indices,
            self.faults.faulted_depth_maps_gaps.shape[-1],
            self.cfg.max_column_height[0],
            self.cfg.max_column_height[1],
        )

        if self.cfg.verbose:
            print("\n   ...facies for closure computations...\n", self.top_lith_facies)
            print(
                "\n   ...max column heights for closure computations...\n",
                max_column_heights,
            )

        self.closure_segments[:] = self.create_closure_labels_from_depth_maps(
            depth_maps_gaps_top_lith, depth_maps_top_lith, max_column_heights
        )

        self.all_closure_segments[:] = self.create_closure_labels_from_all_depth_maps(
            depth_maps_gaps_all_lith, depth_maps_all_lith, all_max_column_heights
        )

        if self.cfg.verbose:
            print(
                "     ...+++... number of nan's in depth_maps_gaps before insertClosureLabels3D ...+++... {}".format(
                    old_depth_maps_gaps[np.isnan(old_depth_maps_gaps)].shape
                )
            )
            print(
                "     ...+++... number of nan's in depth_maps_gaps after insertClosureLabels3D ...+++... {}".format(
                    self.faults.faulted_depth_maps_gaps[
                        np.isnan(self.faults.faulted_depth_maps_gaps)
                    ].shape
                )
            )
            print(
                "     ...+++... number of nan's in depth_maps after insertClosureLabels3D ...+++... {}".format(
                    self.faults.faulted_depth_maps[
                        np.isnan(self.faults.faulted_depth_maps)
                    ].shape
                )
            )
            _closure_segments = self.closure_segments[:]
            print(
                "     ...+++... number of closure voxels in self.closure_segments ...+++... {}".format(
                    _closure_segments[_closure_segments > 0.0].shape
                )
            )
            del _closure_segments

        labels_clean, self.closure_segments[:] = self.segment_closures(
            self.closure_segments[:], remove_shale=True
        )
        label_values, labels_clean = self.parse_label_values_and_counts(labels_clean)

        labels_clean_all, self.all_closure_segments[:] = self.segment_closures(
            self.all_closure_segments[:], remove_shale=False
        )
        label_values_all, labels_clean_all = self.parse_label_values_and_counts(
            labels_clean_all
        )
        self.write_cube_to_disk(self.all_closure_segments[:], "all_closure_segments")

        # Assign fluid types
        (
            self.oil_closures[:],
            self.gas_closures[:],
            self.brine_closures[:],
        ) = self.assign_fluid_types(label_values, labels_clean)
        all_closures_final = (labels_clean_all != 0).astype("uint8")

        # Identify closures by type (simple, faulted, onlap or salt bounded)
        self.find_faulted_closures(label_values, labels_clean)
        self.find_onlap_closures(label_values, labels_clean)
        self.find_simple_closures(label_values, labels_clean)
        self.find_false_closures(label_values, labels_clean)

        self.find_faulted_all_closures(label_values_all, labels_clean_all)
        self.find_onlap_all_closures(label_values_all, labels_clean_all)
        self.find_simple_all_closures(label_values_all, labels_clean_all)
        self.find_false_all_closures(label_values_all, labels_clean_all)

        if self.cfg.include_salt:
            self.find_salt_bounded_closures(label_values, labels_clean)
            self.find_salt_bounded_all_closures(label_values_all, labels_clean_all)

        # Remove false closures from oil & gas closure cubes
        if self.n_false_closures_oil > 0:
            print(f"Removing {self.n_false_closures_oil} false oil closures")
            self.oil_closures[self.false_closures_oil == 1] = 0.0
        if self.n_false_closures_gas > 0:
            print(f"Removing {self.n_false_closures_gas} false gas closures")
            self.gas_closures[self.false_closures_gas == 1] = 0.0

        # Remove false closures from allclosure cube
        if self.n_false_all_closures > 0:
            print(f"Removing {self.n_false_all_closures} false all closures")
            self.all_closure_segments[self.false_all_closures == 1] = 0.0

        # Create a closure cube with voxel count as labels, and include closure type in decimal
        # e.g. simple closure of size 5000 = 5000.1
        #      faulted closure of size 5000 = 5000.2
        #      onlap closure of size 5000 = 5000.3
        #      salt-bounded closure of size 5000 = 5000.4
        hc_closure_codes = np.zeros_like(self.gas_closures, dtype="float32")

        # AZ: COULD RUN THESE CLOSURE SIZE FILTERS ON ALL_CLOSURES, IF DESIRED

        if "simple" in self.cfg.closure_types:
            print("Filtering 4 Way Closures")
            (
                self.simple_closures_oil[:],
                self.n_4way_closures_oil,
            ) = self.closure_size_filter(
                self.simple_closures_oil[:],
                self.cfg.closure_min_voxels_simple,
                self.n_4way_closures_oil,
            )
            (
                self.simple_closures_gas[:],
                self.n_4way_closures_gas,
            ) = self.closure_size_filter(
                self.simple_closures_gas[:],
                self.cfg.closure_min_voxels_simple,
                self.n_4way_closures_gas,
            )

            # Add simple closures to closure code cube
            hc_closures = (
                self.simple_closures_oil[:] + self.simple_closures_gas[:]
            ).astype("float32")
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.1
            )
        else:  # if closure type not in config, set HC closures to 0
            self.simple_closures_oil[:] *= 0
            self.simple_closures_gas[:] *= 0
            self.simple_all_closures[:] *= 0

        self.oil_closures[self.simple_closures_oil[:] > 0.0] = 1.0
        self.oil_closures[self.simple_closures_oil[:] < 0.0] = 0.0
        self.gas_closures[self.simple_closures_gas[:] > 0.0] = 1.0
        self.gas_closures[self.simple_closures_gas[:] < 0.0] = 0.0

        all_closures_final[self.simple_all_closures[:] > 0.0] = 1.0
        all_closures_final[self.simple_all_closures[:] < 0.0] = 0.0

        if "faulted" in self.cfg.closure_types:
            print("Filtering 4 Way Closures")
            # Grow the faulted closures to the fault planes
            self.faulted_closures_oil[:] = self.grow_to_fault2(
                self.faulted_closures_oil[:]
            )
            self.faulted_closures_gas[:] = self.grow_to_fault2(
                self.faulted_closures_gas[:]
            )

            (
                self.faulted_closures_oil[:],
                self.n_fault_closures_oil,
            ) = self.closure_size_filter(
                self.faulted_closures_oil[:],
                self.cfg.closure_min_voxels_faulted,
                self.n_fault_closures_oil,
            )
            (
                self.faulted_closures_gas[:],
                self.n_fault_closures_gas,
            ) = self.closure_size_filter(
                self.faulted_closures_gas[:],
                self.cfg.closure_min_voxels_faulted,
                self.n_fault_closures_gas,
            )

            self.faulted_all_closures[:] = self.grow_to_fault2(
                self.faulted_all_closures[:],
                grow_only_sand_closures=False,
                remove_small_closures=False,
            )

            # Add faulted closures to closure code cube
            hc_closures = self.faulted_closures_oil[:] + self.faulted_closures_gas[:]
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.2
            )
        else:  # if closure type not in config, set HC closures to 0
            self.faulted_closures_oil[:] *= 0
            self.faulted_closures_gas[:] *= 0
            self.faulted_all_closures[:] *= 0

        self.oil_closures[self.faulted_closures_oil[:] > 0.0] = 1.0
        self.oil_closures[self.faulted_closures_oil[:] < 0.0] = 0.0
        self.gas_closures[self.faulted_closures_gas[:] > 0.0] = 1.0
        self.gas_closures[self.faulted_closures_gas[:] < 0.0] = 0.0

        all_closures_final[self.faulted_all_closures[:] > 0.0] = 1.0
        all_closures_final[self.faulted_all_closures[:] < 0.0] = 0.0

        if "onlap" in self.cfg.closure_types:
            print("Filtering Onlap Closures")
            (
                self.onlap_closures_oil[:],
                self.n_onlap_closures_oil,
            ) = self.closure_size_filter(
                self.onlap_closures_oil[:],
                self.cfg.closure_min_voxels_onlap,
                self.n_onlap_closures_oil,
            )
            (
                self.onlap_closures_gas[:],
                self.n_onlap_closures_gas,
            ) = self.closure_size_filter(
                self.onlap_closures_gas[:],
                self.cfg.closure_min_voxels_onlap,
                self.n_onlap_closures_gas,
            )

            # Add faulted closures to closure code cube
            hc_closures = self.onlap_closures_oil[:] + self.onlap_closures_gas[:]
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.3
            )
            # labels = labels.astype('float32')
            # if num > 0:
            #     for x in range(1, num + 1):
            #         y = 0.3 + labels[labels == x].size
            #         labels[labels == x] = y
            #     hc_closure_codes += labels
        else:  # if closure type not in config, set HC closures to 0
            self.onlap_closures_oil[:] *= 0
            self.onlap_closures_gas[:] *= 0
            self.onlap_all_closures[:] *= 0

        self.oil_closures[self.onlap_closures_oil[:] > 0.0] = 1.0
        self.oil_closures[self.onlap_closures_oil[:] < 0.0] = 0.0
        self.gas_closures[self.onlap_closures_gas[:] > 0.0] = 1.0
        self.gas_closures[self.onlap_closures_gas[:] < 0.0] = 0.0
        all_closures_final[self.onlap_all_closures[:] > 0.0] = 1.0
        all_closures_final[self.onlap_all_closures[:] < 0.0] = 0.0

        if self.cfg.include_salt:
            # Grow the salt-bounded closures to the salt body
            salt_closures_oil_grown = np.zeros_like(self.salt_closures_oil[:])
            salt_closures_gas_grown = np.zeros_like(self.salt_closures_gas[:])

            if np.max(self.salt_closures_oil[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_oil[:], "salt_closures_oil_initial"
                )
                print(
                    f"Salt-bounded Oil Closure voxel count: {self.salt_closures_oil[:][self.salt_closures_oil[:] > 0].size}"
                )
                salt_closures_oil_grown = self.grow_to_salt(self.salt_closures_oil[:])
                self.salt_closures_oil[:] = salt_closures_oil_grown
                print(
                    f"Salt-bounded Oil Closure voxel count: {self.salt_closures_oil[:][self.salt_closures_oil[:] > 0].size}"
                )
            if np.max(self.salt_closures_gas[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_gas[:], "salt_closures_gas_initial"
                )
                print(
                    f"Salt-bounded Gas Closure voxel count: {self.salt_closures_gas[:][self.salt_closures_gas[:] > 0].size}"
                )
                salt_closures_gas_grown = self.grow_to_salt(self.salt_closures_gas[:])
                self.salt_closures_gas[:] = salt_closures_gas_grown
                print(
                    f"Salt-bounded Gas Closure voxel count: {self.salt_closures_gas[:][self.salt_closures_gas[:] > 1].size}"
                )
            if np.max(self.salt_all_closures[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_all_closures[:], "salt_all_closures_initial"
                )  # maybe remove later
                print(
                    f"Salt-bounded All Closure voxel count: {self.salt_all_closures[:][self.salt_all_closures[:] > 0].size}"
                )
                salt_all_closures_grown = self.grow_to_salt(self.salt_all_closures[:])
                self.salt_all_closures[:] = salt_all_closures_grown
                print(
                    f"Salt-bounded All Closure voxel count: {self.salt_all_closures[:][self.salt_all_closures[:] > 1].size}"
                )
            else:
                salt_all_closures_grown = np.zeros_like(self.salt_all_closures)

            if np.max(self.salt_closures_oil[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_oil[:], "salt_closures_oil_grown"
                )
            if np.max(self.salt_closures_gas[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_gas[:], "salt_closures_gas_grown"
                )
            if np.max(self.salt_all_closures[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_all_closures[:], "salt_all_closures_grown"
                )  # maybe remove later

            (
                self.salt_closures_oil[:],
                self.n_salt_closures_oil,
            ) = self.closure_size_filter(
                self.salt_closures_oil[:],
                self.cfg.closure_min_voxels,
                self.n_salt_closures_oil,
            )
            (
                self.salt_closures_gas[:],
                self.n_salt_closures_gas,
            ) = self.closure_size_filter(
                self.salt_closures_gas[:],
                self.cfg.closure_min_voxels,
                self.n_salt_closures_gas,
            )

            # Append salt-bounded closures to main closure cubes for oil and gas
            if np.max(salt_closures_oil_grown) > 0.0:
                self.oil_closures[salt_closures_oil_grown > 0.0] = 1.0
                self.oil_closures[salt_closures_oil_grown < 0.0] = 0.0
            if np.max(salt_closures_gas_grown) > 0.0:
                self.gas_closures[salt_closures_gas_grown > 0.0] = 1.0
                self.gas_closures[salt_closures_gas_grown < 0.0] = 0.0
            if np.max(salt_all_closures_grown) > 0.0:
                all_closures_final[salt_all_closures_grown > 0.0] = 1.0
                all_closures_final[salt_all_closures_grown < 0.0] = 0.0

            # Add faulted closures to closure code cube
            hc_closures = self.salt_closures_oil[:] + self.salt_closures_gas[:]
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.4
            )

        # Write hc_closure_codes to disk
        self.write_cube_to_disk(hc_closure_codes, "closure_segments_hc_voxelcount")

        # Create closure volumes by type
        if self.simple_closures[:] is None:
            self.simple_closures[:] = self.simple_closures_oil[:].astype("uint8")
        else:
            self.simple_closures[:] += self.simple_closures_oil[:].astype("uint8")
        self.simple_closures[:] += self.simple_closures_gas[:].astype("uint8")
        self.simple_closures[:] += self.simple_closures_brine[:].astype("uint8")
        # Onlap closures
        if self.strat_closures is None:
            self.strat_closures[:] = self.onlap_closures_oil[:].astype("uint8")
        else:
            self.strat_closures[:] += self.onlap_closures_oil[:].astype("uint8")
        self.strat_closures[:] += self.onlap_closures_gas[:].astype("uint8")
        self.strat_closures[:] += self.onlap_closures_brine[:].astype("uint8")
        # Fault closures
        if self.fault_closures is None:
            self.fault_closures[:] = self.faulted_closures_oil[:].astype("uint8")
        else:
            self.fault_closures[:] += self.faulted_closures_oil[:].astype("uint8")
        self.fault_closures[:] += self.faulted_closures_gas[:].astype("uint8")
        self.fault_closures[:] += self.faulted_closures_brine[:].astype("uint8")

        # Salt-bounded closures
        if self.cfg.include_salt:
            if self.salt_closures is None:
                self.salt_closures[:] = self.salt_closures_oil[:].astype("uint8")
            else:
                self.salt_closures[:] += self.salt_closures_oil[:].astype("uint8")
            self.salt_closures[:] += self.salt_closures_gas[:].astype("uint8")

        # Convert closure cubes from int16 to uint8 for writing to disk
        self.closure_segments[:] = self.closure_segments[:].astype("uint8")

        # add any oil/gas/brine closures into all_closures_final in case missed
        all_closures_final[:][self.oil_closures[:] > 0] = 1
        all_closures_final[:][self.gas_closures[:] > 0] = 1
        all_closures_final[:][self.gas_closures[:] > 0] = 1
        # Write all_closures_final to disk
        self.write_cube_to_disk(all_closures_final.astype("uint8"), "trap_label")

        # add any oil/gas/brine closures into reservoir in case missed
        self.faults.reservoir[:][self.oil_closures[:] > 0] = 1
        self.faults.reservoir[:][self.gas_closures[:] > 0] = 1
        self.faults.reservoir[:][self.brine_closures[:] > 0] = 1
        # write reservoir_label to disk
        self.write_cube_to_disk(
            self.faults.reservoir[:].astype("uint8"), "reservoir_label"
        )

        if self.cfg.qc_plots:
            from datagenerator.util import plot_xsection
            from datagenerator.util import find_line_with_most_voxels

            # visualize closures QC
            inline_index_cl = find_line_with_most_voxels(
                self.closure_segments, 0.5, self.cfg
            )
            plot_xsection(
                volume=labels_clean,
                maps=self.faults.faulted_depth_maps_gaps,
                line_num=inline_index_cl,
                title="Example Trav through 3D model\nclosures after faulting",
                png_name="QC_plot__AfterFaulting_closure_segments.png",
                cmap="gist_ncar_r",
                cfg=self.cfg,
            )

    def closure_size_filter(self, closure_type, threshold, count):
        labels, num = measure.label(
            closure_type, connectivity=2, background=0, return_num=True
        )
        if (
            num > 0
        ):  # TODO add whether smallest closure is below threshold constraint too
            s = [labels[labels == x].size for x in range(1, 1 + np.max(labels))]
            labels = morphology.remove_small_objects(labels, threshold, connectivity=2)
            t = [labels[labels == x].size for x in range(1, 1 + np.max(labels))]
            print(
                f"Closure sizes before filter: {s}\nThreshold: {threshold}\n"
                f"Closure sizes after filter: {t}"
            )
            count = len(t)
        return labels, count

    def closure_type_info_for_log(self):
        fluid_types = ["oil", "gas", "brine"]
        if "faulted" in self.cfg.closure_types:
            # Faulted closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.faulted_closures_oil[:],
                    self.faulted_closures_gas[:],
                    self.faulted_closures_brine[:],
                ],
                [
                    self.n_fault_closures_oil,
                    self.n_fault_closures_gas,
                    self.n_fault_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_fault_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_fault_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_fault_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_fault_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"Faulted {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

        if "onlap" in self.cfg.closure_types:
            # Onlap Closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.onlap_closures_oil[:],
                    self.onlap_closures_gas[:],
                    self.onlap_closures_brine[:],
                ],
                [
                    self.n_onlap_closures_oil,
                    self.n_onlap_closures_gas,
                    self.n_onlap_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_onlap_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_onlap_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_onlap_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_onlap_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"Onlap {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

        if "simple" in self.cfg.closure_types:
            # Simple Closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.simple_closures_oil[:],
                    self.simple_closures_gas[:],
                    self.simple_closures_brine[:],
                ],
                [
                    self.n_4way_closures_oil,
                    self.n_4way_closures_gas,
                    self.n_4way_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_4way_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_4way_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_4way_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_4way_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"4-Way {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

        if self.cfg.include_salt:
            # Salt-Bounded Closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.salt_closures_oil[:],
                    self.salt_closures_gas[:],
                    self.salt_closures_brine[:],
                ],
                [
                    self.n_salt_closures_oil,
                    self.n_salt_closures_gas,
                    self.n_salt_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_salt_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_salt_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_salt_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_salt_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"Salt {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

    def get_voxel_counts(self, closures):
        next_label = 0
        label_values = [0]
        label_counts = [closures[closures == 0].size]
        for i in range(closures.max() + 1):
            try:
                next_label = closures[closures > next_label].min()
            except (TypeError, ValueError):
                break
            label_values.append(next_label)
            label_counts.append(closures[closures == next_label].size)
            print(
                f"Label: {i}, label_values: {label_values[-1]}, label_counts: {label_counts[-1]}"
            )

        print(
            f'{72 * "*"}\n\tNum Closures: {len(label_counts) - 1}\n\tVoxel counts\n{label_counts[1:]}\n{72 * "*"}'
        )
        for vox_count in label_counts:
            if vox_count < self.cfg.closure_min_voxels:
                print(f"voxel_count: {vox_count}")

    def populate_closure_dict(self, labels, fluid, seismic_nmf=None):
        clist = []
        max_num = np.max(labels)
        if seismic_nmf is not None:
            # calculate ai_gi
            ai, gi = compute_ai_gi(self.cfg, seismic_nmf)
        for i in range(1, max_num + 1):
            _c = np.where(labels == i)
            cl = dict()
            cl["model_id"] = os.path.basename(self.cfg.work_subfolder)
            cl["fluid"] = fluid
            cl["n_voxels"] = len(_c[0])
            # np.min() or x.min() returns type numpy.int64 which SQLITE cannot handle. Convert to int
            cl["x_min"] = int(np.min(_c[0]))
            cl["x_max"] = int(np.max(_c[0]))
            cl["y_min"] = int(np.min(_c[1]))
            cl["y_max"] = int(np.max(_c[1]))
            cl["z_min"] = int(np.min(_c[2]))
            cl["z_max"] = int(np.max(_c[2]))
            cl["zbml_min"] = np.min(self.faults.faulted_depth[_c])
            cl["zbml_max"] = np.max(self.faults.faulted_depth[_c])
            cl["zbml_avg"] = np.mean(self.faults.faulted_depth[_c])
            cl["zbml_std"] = np.std(self.faults.faulted_depth[_c])
            cl["zbml_25pct"] = np.percentile(self.faults.faulted_depth[_c], 25)
            cl["zbml_median"] = np.percentile(self.faults.faulted_depth[_c], 50)
            cl["zbml_75pct"] = np.percentile(self.faults.faulted_depth[_c], 75)
            cl["ng_min"] = np.min(self.faults.faulted_net_to_gross[_c])
            cl["ng_max"] = np.max(self.faults.faulted_net_to_gross[_c])
            cl["ng_avg"] = np.mean(self.faults.faulted_net_to_gross[_c])
            cl["ng_std"] = np.std(self.faults.faulted_net_to_gross[_c])
            cl["ng_25pct"] = np.percentile(self.faults.faulted_net_to_gross[_c], 25)
            cl["ng_median"] = np.median(self.faults.faulted_net_to_gross[_c])
            cl["ng_75pct"] = np.percentile(self.faults.faulted_net_to_gross[_c], 75)
            # Check for intersections with faults, salt and onlaps for closure type
            cl["intersects_fault"] = False
            cl["intersects_onlap"] = False
            cl["intersects_salt"] = False
            if np.max(self.wide_faults[_c] > 0):
                cl["intersects_fault"] = True
            if np.max(self.onlaps_upward[_c] > 0):
                cl["intersects_onlap"] = True
            if self.cfg.include_salt and np.max(self.wide_salt[_c] > 0):
                cl["intersects_salt"] = True

            if seismic_nmf is not None:
                # Using only the top of the closure, calculate seismic properties
                labels_copy = labels.copy()
                labels_copy[labels_copy != i] = 0
                top_closure = get_top_of_closure(labels_copy)
                near = seismic_nmf[0, ...][np.where(top_closure == 1)]
                cl["near_min"] = np.min(near)
                cl["near_max"] = np.max(near)
                cl["near_avg"] = np.mean(near)
                cl["near_std"] = np.std(near)
                cl["near_25pct"] = np.percentile(near, 25)
                cl["near_median"] = np.percentile(near, 50)
                cl["near_75pct"] = np.percentile(near, 75)
                mid = seismic_nmf[1, ...][np.where(top_closure == 1)]
                cl["mid_min"] = np.min(mid)
                cl["mid_max"] = np.max(mid)
                cl["mid_avg"] = np.mean(mid)
                cl["mid_std"] = np.std(mid)
                cl["mid_25pct"] = np.percentile(mid, 25)
                cl["mid_median"] = np.percentile(mid, 50)
                cl["mid_75pct"] = np.percentile(mid, 75)
                far = seismic_nmf[2, ...][np.where(top_closure == 1)]
                cl["far_min"] = np.min(far)
                cl["far_max"] = np.max(far)
                cl["far_avg"] = np.mean(far)
                cl["far_std"] = np.std(far)
                cl["far_25pct"] = np.percentile(far, 25)
                cl["far_median"] = np.percentile(far, 50)
                cl["far_75pct"] = np.percentile(far, 75)
                intercept = ai[np.where(top_closure == 1)]
                cl["intercept_min"] = np.min(intercept)
                cl["intercept_max"] = np.max(intercept)
                cl["intercept_avg"] = np.mean(intercept)
                cl["intercept_std"] = np.std(intercept)
                cl["intercept_25pct"] = np.percentile(intercept, 25)
                cl["intercept_median"] = np.percentile(intercept, 50)
                cl["intercept_75pct"] = np.percentile(intercept, 75)
                gradient = gi[np.where(top_closure == 1)]
                cl["gradient_min"] = np.min(gradient)
                cl["gradient_max"] = np.max(gradient)
                cl["gradient_avg"] = np.mean(gradient)
                cl["gradient_std"] = np.std(gradient)
                cl["gradient_25pct"] = np.percentile(gradient, 25)
                cl["gradient_median"] = np.percentile(gradient, 50)
                cl["gradient_75pct"] = np.percentile(gradient, 75)

            clist.append(cl)

        return clist

    def write_closure_info_to_log(self, seismic_nmf=None):
        """store info about closure in log file"""
        top_sand_layers = [x for x in self.top_lith_indices if self.facies[x] == 1.0]
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="top_sand_layers",
            val=top_sand_layers,
        )
        o = measure.label(self.oil_closures[:], connectivity=2, background=0)
        g = measure.label(self.gas_closures[:], connectivity=2, background=0)
        b = measure.label(self.brine_closures[:], connectivity=2, background=0)
        oil_closures = self.populate_closure_dict(o, "oil", seismic_nmf)
        gas_closures = self.populate_closure_dict(g, "gas", seismic_nmf)
        brine_closures = self.populate_closure_dict(b, "brine", seismic_nmf)
        all_closures = oil_closures + gas_closures + brine_closures
        for i, c in enumerate(all_closures):
            self.cfg.sqldict[f"closure_{i+1}"] = c
        num_labels = np.max(o) + np.max(g)
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="number_hc_closures",
            val=num_labels,
        )
        # Add total number of closure voxels, with ratio of closure voxels given as a percentage
        closure_voxel_count = o[o > 0].size + g[g > 0].size
        closure_voxel_pct = closure_voxel_count / o.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count",
            val=closure_voxel_count,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct",
            val=closure_voxel_pct * 100,
        )
        # Same for Brine
        _brine_voxels = b[b == 1].size
        _brine_voxels_pct = _brine_voxels / b.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count_brine",
            val=_brine_voxels,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct_brine",
            val=_brine_voxels_pct * 100,
        )
        # Same for Oil
        _oil_voxels = o[o == 1].size
        _oil_voxels_pct = _oil_voxels / o.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count_oil",
            val=_oil_voxels,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct_oil",
            val=_oil_voxels_pct * 100,
        )
        # Same for Gas
        _gas_voxels = g[g == 1].size
        _gas_voxels_pct = _gas_voxels / g.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count_gas",
            val=_gas_voxels,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct_gas",
            val=_gas_voxels_pct,
        )
        # Write old logfile as well as the sql dict
        msg = f"layers for closure computation: {str(self.top_lith_indices)}\n"
        msg += f"Number of HC Closures : {num_labels}\n"
        msg += (
            f"Closure voxel count: {closure_voxel_count} - "
            f"{closure_voxel_pct:5.2%}\n"
        )
        msg += (
            f"Closure voxel count: (brine) {_brine_voxels} - {_brine_voxels_pct:5.2%}\n"
        )
        msg += f"Closure voxel count: (oil) {_oil_voxels} - {_oil_voxels_pct:5.2%}\n"
        msg += f"Closure voxel count: (gas) {_gas_voxels} - {_gas_voxels_pct:5.2%}\n"
        print(msg)
        for i in range(self.facies.shape[0]):
            if self.facies[i] == 1:
                msg += f"  layers for closure computation:   {i}, sand\n"
            else:
                msg += f"  layers for closure computation:   {i}, shale\n"
        self.cfg.write_to_logfile(msg)

    def parse_label_values_and_counts(self, labels_clean):
        """parse label values and counts"""
        if self.cfg.verbose:
            print(" Inside parse_label_values_and_counts")
        next_label = 0
        label_values = [0]
        label_counts = [labels_clean[labels_clean == 0].size]
        for i in range(1, labels_clean.max() + 1):
            try:
                next_label = labels_clean[labels_clean > next_label].min()
            except (TypeError, ValueError):
                break
            label_values.append(next_label)
            label_counts.append(labels_clean[labels_clean == next_label].size)
            print(
                f"Label: {i}, label_values: {label_values[-1]}, label_counts: {label_counts[-1]}"
            )
        # force labels to use consecutive integer values
        for i, ilabel in enumerate(label_values):
            labels_clean[labels_clean == ilabel] = i
            label_values[i] = i
        # labels_clean = self.remove_small_objects(labels_clean)  # already applied to labels_clean
        # Remove label_value 0
        label_values.remove(0)
        return label_values, labels_clean

    def assign_fluid_types(self, label_values, labels_clean):
        """randomly assign oil or gas to closure"""
        print(
            " labels_clean.min(), labels_clean.max() = ",
            labels_clean.min(),
            labels_clean.max(),
        )
        _brine_closures = (labels_clean * 0.0).astype("uint8")
        _oil_closures = (labels_clean * 0.0).astype("uint8")
        _gas_closures = (labels_clean * 0.0).astype("uint8")

        fluid_type_code = np.random.randint(3, size=labels_clean.max() + 1)

        _closure_segments = self.closure_segments[:]
        for i in range(1, labels_clean.max() + 1):
            voxel_count = labels_clean[labels_clean == i].size
            if voxel_count > 0:
                print(f"Voxel Count: {voxel_count}\tFluid type: {fluid_type_code[i]}")
            # not in closure = 0
            # closure with brine filled reservoir fluid_type_code = 1
            # closure with oil filled reservoir fluid_type_code = 2
            # closure with gas filled reservoir fluid_type_code = 3
            if i in label_values:
                if fluid_type_code[i] == 0:
                    # brine: change labels_clean contents to fluid_type_code = 1 (same as background)
                    _brine_closures[
                        np.logical_and(labels_clean == i, _closure_segments > 0)
                    ] = 1
                elif fluid_type_code[i] == 1:
                    # oil: change labels_clean contents to fluid_type_code = 2
                    _oil_closures[labels_clean == i] = 1
                elif fluid_type_code[i] == 2:
                    # gas: change labels_clean contents to fluid_type_code = 3
                    _gas_closures[labels_clean == i] = 1
        return _oil_closures, _gas_closures, _brine_closures

    def remove_small_objects(self, labels, min_filter=True):
        try:
            # Use the global minimum voxel size initially, before closure types are identified
            labels_clean = morphology.remove_small_objects(
                labels, self.cfg.closure_min_voxels
            )
            if self.cfg.verbose:
                print("labels_clean succeeded.")
                print(
                    " labels.min:{}, labels.max: {}".format(labels.min(), labels.max())
                )
                print(
                    " labels_clean min:{}, labels_clean max: {}".format(
                        labels_clean.min(), labels_clean.max()
                    )
                )
        except Exception as e:
            print(
                f"Closures/create_closures: labels_clean (remove_small_objects) did not succeed: {e}"
            )
            if min_filter:
                labels_clean = minimum_filter(labels, size=(3, 3, 3))
                if self.cfg.verbose:
                    print(
                        " labels.min:{}, labels.max: {}".format(
                            labels.min(), labels.max()
                        )
                    )
                    print(
                        " labels_clean min:{}, labels_clean max: {}".format(
                            labels_clean.min(), labels_clean.max()
                        )
                    )
        return labels_clean

    def segment_closures(self, _closure_segments, remove_shale=True):
        """Segment the closures so that they can be randomly filled with hydrocarbons"""

        _closure_segments = np.clip(_closure_segments, 0.0, 1.0)
        # remove tiny clusters
        _closure_segments = minimum_filter(
            _closure_segments.astype("int16"), size=(3, 3, 1)
        )
        _closure_segments = maximum_filter(_closure_segments, size=(3, 3, 1))

        if remove_shale:
            # restrict closures to sand (non-shale) voxels
            if self.faults.faulted_lithology.shape[2] == _closure_segments.shape[2]:
                sand_shale = self.faults.faulted_lithology[:].copy()
            else:
                sand_shale = self.faults.faulted_lithology[
                    :, :, :: self.cfg.infill_factor
                ].copy()
            _closure_segments[sand_shale <= 0.0] = 0
            del sand_shale
        labels = measure.label(_closure_segments, connectivity=2, background=0)

        labels_clean = self.remove_small_objects(labels)
        return labels_clean, _closure_segments

    def write_closure_volumes_to_disk(self):
        # Create files for closure volumes
        self.write_cube_to_disk(self.brine_closures[:], "closure_segments_brine")
        self.write_cube_to_disk(self.oil_closures[:], "closure_segments_oil")
        self.write_cube_to_disk(self.gas_closures[:], "closure_segments_gas")
        # Create combined HC cube by adding oil and gas closures
        self.hc_labels[:] = (self.oil_closures[:] + self.gas_closures[:]).astype(
            "uint8"
        )
        self.write_cube_to_disk(self.hc_labels[:], "closure_segments_hc")

        if self.cfg.model_qc_volumes:
            self.write_cube_to_disk(self.closure_segments, "closure_segments_raw_all")
            self.write_cube_to_disk(self.simple_closures, "closure_segments_simple")
            self.write_cube_to_disk(self.strat_closures, "closure_segments_strat")
            self.write_cube_to_disk(self.fault_closures, "closure_segments_fault")

        # Triple check that no small closures exist in the final closure files
        for i, c in enumerate(
            [
                self.oil_closures,
                self.gas_closures,
                self.simple_closures,
                self.strat_closures,
                self.fault_closures,
            ]
        ):
            _t = measure.label(c, connectivity=2, background=0)
            counts = [_t[_t == x].size for x in range(np.max(_t))]
            print(f"Final closure volume voxels sizes: {counts}")
            for n, x in enumerate(counts):
                if x < self.cfg.closure_min_voxels:
                    print(f"Voxel count: {x}\t Count:{i}, index: {n}")

        # Return the hydrocarbon closure labels so that augmentation can be applied to the data & labels
        # return self.oil_closures + self.gas_closures

    def calculate_closure_statistics(self, in_array, closure_type):
        """
        Calculate the size and location of isolated features in an array

        :param in_array: ndarray. Input array to be labelled, where non-zero values are counted as features
        :param closure_type: string. Closure type label
        :param digi: int or float. To convert depth values from samples to units
        :return: string. Concatenated string of closure statistics to be written to log
        """
        labelled_array, max_labels = measure.label(
            in_array, connectivity=2, return_num=True
        )
        msg = ""
        for i in range(1, max_labels + 1):  # start at 1 to avoid counting 0's
            trap = np.where(labelled_array == i)
            ranges = [([np.min(trap[x]), np.max(trap[x])]) for x, _ in enumerate(trap)]
            sizes = [x[1] - x[0] for x in ranges]
            n_voxels = labelled_array[labelled_array == i].size
            if sum(sizes) > 0:
                msg += (
                    f"{closure_type}\t"
                    f"Num X,Y,Z Samples: {str(sizes).ljust(15)}\t"
                    f"Num Voxels: {str(n_voxels).ljust(5)}\t"
                    f"Track: {2000 + ranges[0][0]}-{2000 + ranges[0][1]}\t"
                    f"Bin: {1000 + ranges[1][0]}-{1000 + ranges[1][1]}\t"
                    f"Depth: {ranges[2][0] * self.cfg.digi}-{ranges[2][1] * self.cfg.digi + self.cfg.digi / 2}\n"
                )
        return msg

    def find_faulted_closures(self, closure_segment_list, closure_segments):
        self._dilate_faults()
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            if faults_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    # Faulted oil closure
                    self.faulted_closures_oil[i, j, k] = 1.0
                    self.n_fault_closures_oil += 1
                    self.fault_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    # Faulted gas closure
                    self.faulted_closures_gas[i, j, k] = 1.0
                    self.n_fault_closures_gas += 1
                    self.fault_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    # Faulted brine closure
                    self.faulted_closures_brine[i, j, k] = 1.0
                    self.n_fault_closures_brine += 1
                    self.fault_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is faulted but does not have oil, gas or brine assigned"
                    )

    def find_onlap_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            onlaps_within_closure = self.onlaps_upward[i, j, k]
            if onlaps_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    self.onlap_closures_oil[i, j, k] = 1.0
                    self.n_onlap_closures_oil += 1
                    self.onlap_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    self.onlap_closures_gas[i, j, k] = 1.0
                    self.n_onlap_closures_gas += 1
                    self.onlap_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    self.onlap_closures_brine[i, j, k] = 1.0
                    self.n_onlap_closures_brine += 1
                    self.onlap_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is onlap but does not have oil, gas or brine assigned"
                    )

    def find_simple_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            onlaps = self._threshold_volumes(self.faults.faulted_onlap_segments[:])
            onlaps_within_closure = onlaps[i, j, k]
            oil_within_closure = self.oil_closures[i, j, k]
            gas_within_closure = self.gas_closures[i, j, k]
            brine_within_closure = self.brine_closures[i, j, k]
            if faults_within_closure.max() == 0 and onlaps_within_closure.max() == 0:
                if oil_within_closure.max() > 0:
                    self.simple_closures_oil[i, j, k] = 1.0
                    self.n_4way_closures_oil += 1
                elif gas_within_closure.max() > 0:
                    self.simple_closures_gas[i, j, k] = 1.0
                    self.n_4way_closures_gas += 1
                elif brine_within_closure.max() > 0:
                    self.simple_closures_brine[i, j, k] = 1.0
                    self.n_4way_closures_brine += 1
                else:
                    print(
                        "Closure is not faulted or onlap but does not have oil, gas or brine assigned"
                    )

    def find_false_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.fat_faults[i, j, k]
            onlaps_within_closure = self.onlaps_downward[i, j, k]
            for fluid, false, num in zip(
                [self.oil_closures, self.gas_closures, self.brine_closures],
                [
                    self.false_closures_oil,
                    self.false_closures_gas,
                    self.false_closures_brine,
                ],
                [
                    self.n_false_closures_oil,
                    self.n_false_closures_gas,
                    self.n_false_closures_brine,
                ],
            ):
                fluid_within_closure = fluid[i, j, k]
                if fluid_within_closure.max() > 0:
                    if onlaps_within_closure.max() > 0:
                        _faulted_closure_threshold = float(
                            faults_within_closure[faults_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        _onlap_closure_threshold = float(
                            onlaps_within_closure[onlaps_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        if (
                            _faulted_closure_threshold > 0.65
                            and _onlap_closure_threshold > 0.65
                        ):
                            false[i, j, k] = 1
                            num += 1

    def find_salt_bounded_closures(self, closure_segment_list, closure_segments):
        self._dilate_salt()
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            salt_within_closure = self.wide_salt[i, j, k]
            if salt_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    # salt bounded oil closure
                    self.salt_closures_oil[i, j, k] = 1.0
                    self.n_salt_closures_oil += 1
                    self.salt_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    # salt bounded gas closure
                    self.salt_closures_gas[i, j, k] = 1.0
                    self.n_salt_closures_gas += 1
                    self.salt_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    # salt bounded brine closure
                    self.salt_closures_brine[i, j, k] = 1.0
                    self.n_salt_closures_brine += 1
                    self.salt_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is salt bounded but does not have oil, gas or brine assigned"
                    )

    def find_faulted_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            if faults_within_closure.max() > 0:
                self.faulted_all_closures[i, j, k] = 1.0
                self.n_fault_all_closures += 1
                self.fault_all_closures_segment_list.append(iclosure)

    def find_onlap_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            onlaps_within_closure = self.onlaps_upward[i, j, k]
            if onlaps_within_closure.max() > 0:
                self.onlap_all_closures[i, j, k] = 1.0
                self.n_onlap_all_closures += 1
                self.onlap_all_closures_segment_list.append(iclosure)

    def find_simple_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            onlaps = self._threshold_volumes(self.faults.faulted_onlap_segments[:])
            onlaps_within_closure = onlaps[i, j, k]
            if faults_within_closure.max() == 0 and onlaps_within_closure.max() == 0:
                self.simple_all_closures[i, j, k] = 1.0
                self.n_4way_all_closures += 1

    def find_false_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.fat_faults[i, j, k]
            onlaps_within_closure = self.onlaps_downward[i, j, k]
            if onlaps_within_closure.max() > 0:
                _faulted_closure_threshold = float(
                    faults_within_closure[faults_within_closure > 0].size / i.size
                )
                _onlap_closure_threshold = float(
                    onlaps_within_closure[onlaps_within_closure > 0].size / i.size
                )
                if (
                    _faulted_closure_threshold > 0.65
                    and _onlap_closure_threshold > 0.65
                ):
                    self.false_all_closures[i, j, k] = 1
                    self.n_false_all_closures += 1

    def find_salt_bounded_all_closures(self, closure_segment_list, closure_segments):
        self._dilate_salt()
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            salt_within_closure = self.wide_salt[i, j, k]
            if salt_within_closure.max() > 0:
                self.salt_all_closures[i, j, k] = 1.0
                self.n_salt_all_closures += 1
                self.salt_all_closures_segment_list.append(iclosure)

    def _dilate_faults(self):
        thresholded_faults = self._threshold_volumes(self.faults.fault_planes[:])
        self.wide_faults[:] = self.grow_lateral(
            thresholded_faults, iterations=9, dist=1
        )
        self.fat_faults[:] = self.grow_lateral(
            thresholded_faults, iterations=21, dist=1
        )
        if self.cfg.include_salt:
            # Treat the salt body as a fault to grow closures to boundary
            thresholded_salt = self._threshold_volumes(
                self.faults.salt_model.salt_segments[:]
            )
            wide_salt = self.grow_lateral(thresholded_salt, iterations=9, dist=1)
            self.wide_salt[:] = wide_salt
            # Add salt to faults to cehck if growing the closure works
            self.wide_faults[:] += wide_salt

    def _dilate_salt(self):
        thresholded_salt = self._threshold_volumes(
            self.faults.salt_model.salt_segments[:]
        )
        wide_salt = self.grow_lateral(thresholded_salt, iterations=9, dist=1)
        self.wide_salt[:] = wide_salt

    def _dilate_onlaps(self):
        onlaps = self._threshold_volumes(self.faults.faulted_onlap_segments[:])
        mask = np.zeros((1, 1, 3))
        mask[0, 0, :2] = 1
        self.onlaps_upward[:] = morphology.binary_dilation(onlaps, mask)
        mask = np.zeros((1, 1, 3))
        mask[0, 0, 1:] = 1
        self.onlaps_downward[:] = onlaps.copy()
        for _ in range(30):
            try:
                self.onlaps_downward[:] = morphology.binary_dilation(
                    self.onlaps_downward[:], mask
                )
            except:
                break

    def grow_to_fault2(
        self, closures, grow_only_sand_closures=True, remove_small_closures=True
    ):
        # - grow closures laterally and up within layer and within fault block
        print(
            "\n\n ... grow_to_fault2: grow closures laterally and up within layer and within fault block ..."
        )
        self.cfg.write_to_logfile("growing closures to fault plane: grow_to_fault2")

        # dilated_fault_closures = closures.copy()
        # n_faulted_closures = dilated_fault_closures.max()
        labels_clean = self.closure_segments[:].copy()
        labels_clean[closures == 0] = 0
        labels_clean_list = list(set(labels_clean.flatten()))
        labels_clean_list.remove(0)
        initial_closures = labels_clean.copy()
        print("\n    ... grow_to_fault2: n_faulted_closures = ", len(labels_clean_list))
        print("    ... grow_to_fault2: faulted_closures = ", labels_clean_list)

        # TODO remove this once small closures are found and fixed
        voxel_sizes = [
            self.closure_segments[self.closure_segments[:] == i].size
            for i in labels_clean_list
        ]
        for _v in voxel_sizes:
            print(f"Voxel_Sizes: {_v}")
            if _v < self.cfg.closure_min_voxels:
                print(_v)

        depth_cube = np.zeros(self.faults.faulted_age_volume.shape, float)
        _depths = np.arange(self.faults.faulted_age_volume.shape[2])
        depth_cube += _depths.reshape(1, 1, self.faults.faulted_age_volume.shape[2])
        _ng = self.faults.faulted_net_to_gross[:].copy()
        # Cannot solely use NG anymore since shales may have variable net to gross
        _lith = self.faults.faulted_lithology[:].copy()
        _age = self.faults.faulted_age_volume[:].copy()
        fault_throw = self.faults.max_fault_throw[:]

        for il, i in enumerate(labels_clean_list):
            fault_blocks_list = list(set(fault_throw[labels_clean == i].flatten()))
            print("    ... grow_to_fault2: fault_blocks_list = ", fault_blocks_list)
            for jl, j in enumerate(fault_blocks_list):
                print(
                    "\n\n    ... label, throw = ",
                    i,
                    j,
                    list(set(fault_throw[labels_clean == i].flatten())),
                    labels_clean[labels_clean == i].size,
                    fault_throw[fault_throw == j].size,
                    fault_throw[
                        np.where((labels_clean == i) & (fault_throw[:] == j))
                    ].size,
                )
                single_closure = labels_clean * 0.0
                size = single_closure[
                    np.where((labels_clean == i) & (np.abs(fault_throw - j) < 0.25))
                ].size
                if size >= self.cfg.closure_min_voxels:
                    print(f"Label: {i}, fault_block: {j}, Voxel_Count: {size}")
                    single_closure[
                        np.where((labels_clean == i) & (np.abs(fault_throw - j) < 0.25))
                    ] = 1
                if single_closure[single_closure > 0].size == 0:
                    # labels_clean[np.where((labels_clean == i) & (np.abs(self.fault_throw - j) < .25))] = 0
                    labels_clean[np.where(labels_clean == i)] = 0
                    continue
                avg_ng = _ng[single_closure == 1].mean()
                _geo_age_voxels = (_age[single_closure == 1] + 0.5).astype("int")
                _ng_voxels = _ng[single_closure == 1]
                _geo_age_voxels = _geo_age_voxels[_ng_voxels >= avg_ng / 2.0]
                min_geo_age = _geo_age_voxels.min() - 0.5
                avg_geo_age = int(_geo_age_voxels.mean())
                max_geo_age = _geo_age_voxels.max() + 0.5
                _depth_geobody_voxels = depth_cube[single_closure == 1]
                min_depth = _depth_geobody_voxels.min()
                max_depth = _depth_geobody_voxels.max()
                avg_throw = np.median(fault_throw[single_closure == 1])

                closure_boundary_cube = closures * 0.0
                if grow_only_sand_closures:
                    lith_flag = _lith > 0.0
                else:
                    lith_flag = _lith >= 0.0
                closure_boundary_cube[
                    np.where(
                        lith_flag
                        & (_age > min_geo_age)
                        & (_age < max_geo_age)
                        & (fault_throw == avg_throw)
                        & (depth_cube <= max_depth)
                    )
                ] = 1.0
                print(
                    "\n    ... grow_to_fault2: closure_boundary_cube voxels = ",
                    closure_boundary_cube[closure_boundary_cube == 1].size,
                )

                n_voxel = single_closure[single_closure == 1].size

                original_voxels = n_voxel + 0
                print(
                    "\n    ... closure label number, avg_throw, geobody shape, geo_age min/mean/max, depth min/max, avg_ng = ",
                    i,
                    j,
                    n_voxel,
                    (min_geo_age, avg_geo_age, max_geo_age),
                    (min_depth, max_depth),
                    avg_ng,
                    il,
                    " / ",
                    len(labels_clean_list),
                )

                grown_closure = single_closure.copy()
                grown_closure[depth_cube >= max_depth] = 0
                delta_voxel = 0
                previous_delta_voxel = 1e9
                converged = False
                for ii in range(15):
                    grown_closure = self.grow_lateral(grown_closure, 1, dist=2)
                    grown_closure = self.grow_upward(grown_closure, 1, dist=1)
                    # stay within layer, within age, within fault block, above HCWC
                    grown_closure[closure_boundary_cube == 0.0] = 0.0
                    single_closure = single_closure + grown_closure
                    single_closure[single_closure > 0] = i
                    new_n_voxel = single_closure[single_closure > 0].size
                    previous_delta_voxel = delta_voxel + 0
                    delta_voxel = new_n_voxel - n_voxel
                    print(
                        "    ... i, ii, closure label number, geobody shape, delta_voxel, previous_delta_voxel,"
                        " delta_voxel>previous_delta_voxel = ",
                        i,
                        ii,
                        new_n_voxel,
                        delta_voxel,
                        previous_delta_voxel,
                        delta_voxel > previous_delta_voxel,
                    )
                    if n_voxel == new_n_voxel:
                        # finish bottom voxel layer near "HCWC"
                        grown_closure = self.grow_downward(
                            grown_closure, 1, dist=1, verbose=False
                        )
                        # stay within layer, within age, within fault block, above HCWC
                        grown_closure[closure_boundary_cube == 0.0] = 0.0
                        single_closure = single_closure + grown_closure
                        single_closure[single_closure > 0] = i
                        converged = True
                        break
                    else:
                        n_voxel = new_n_voxel
                    previous_delta_voxel = delta_voxel + 0
                if converged is True:
                    labels_clean[single_closure > 0] = i
                    msg_postscript = " converged"
                else:
                    labels_clean[labels_clean == i] = -i
                    msg_postscript = " NOT converged"
                msg = (
                    f"closure_id: {int(i):04d}, fault_id: {int(j + .5):04d}, "
                    + f"original_voxels: {original_voxels:11.0f}, new_n_voxel: {new_n_voxel:11.0f}, "
                    + f"percent_growth: {float(new_n_voxel / original_voxels):6.2f}"
                )
                print(msg + msg_postscript)
                self.cfg.write_to_logfile(msg + msg_postscript)

        # Set small closures to 0 after growth
        # _grown_labels = measure.label(labels_clean, connectivity=2, background=0)
        # for x in np.unique(_grown_labels):
        #     size = _grown_labels[_grown_labels == x].size
        #     print(f'Size before editing: {size}')
        #     if size < self.cfg.closure_min_voxels:
        #         labels_clean[_grown_labels == x] = 0.
        # for x in np.unique(labels_clean):
        #     size = labels_clean[labels_clean == x].size
        #     print(f'Size after editing: {size}')

        if remove_small_closures:
            _initial_labels = measure.label(
                initial_closures, connectivity=2, background=0
            )
            _grown_labels = measure.label(labels_clean, connectivity=2, background=0)
            for x in np.unique(_grown_labels):
                size_initial = _initial_labels[_initial_labels == x].size
                size_grown = _grown_labels[_grown_labels == x].size
                print(f"Size before editing: {size_initial}")
                print(f"Size after editing: {size_grown}")
                if size_grown < self.cfg.closure_min_voxels:
                    print(
                        f"Closure below threshold of {self.cfg.closure_min_voxels} and will be removed"
                    )
                    labels_clean[_grown_labels == x] = 0.0
        return labels_clean

    def grow_to_salt(self, closures):
        # - grow closures laterally and up within layer up to salt body
        print("\n\n ... grow_to_salt: grow closures laterally and up within layer ...")
        self.cfg.write_to_logfile("growing closures to salt body: grow_to_salt")

        labels_clean = measure.label(
            self.closure_segments[:], connectivity=2, background=0
        )
        labels_clean[closures == 0] = 0
        # labels_clean = self.closure_segments[:].copy()
        # labels_clean[closures == 0] = 0
        labels_clean_list = list(set(labels_clean.flatten()))
        labels_clean_list.remove(0)
        initial_closures = labels_clean.copy()
        print("\n    ... grow_to_salt: n_salt_closures = ", len(labels_clean_list))
        print("    ... grow_to_salt: salt_closures = ", labels_clean_list)

        depth_cube = np.zeros(self.faults.faulted_age_volume.shape, float)
        _depths = np.arange(self.faults.faulted_age_volume.shape[2])
        depth_cube += _depths.reshape(1, 1, self.faults.faulted_age_volume.shape[2])
        _ng = self.faults.faulted_net_to_gross[:].copy()
        _age = self.faults.faulted_age_volume[:].copy()
        salt = self.faults.salt_model.salt_segments[:]

        for il, i in enumerate(labels_clean_list):
            salt_list = list(set(salt[labels_clean == i].flatten()))
            print("    ... grow_to_fault2: salt_list = ", salt_list)
            single_closure = labels_clean * 0.0
            size = single_closure[np.where(labels_clean == i)].size
            if size >= self.cfg.closure_min_voxels:
                print(f"Label: {i}, Voxel_Count: {size}")
                single_closure[np.where(labels_clean == i)] = 1
            if single_closure[single_closure > 0].size == 0:
                labels_clean[np.where(labels_clean == i)] = 0
                continue
            avg_ng = _ng[single_closure == 1].mean()
            _geo_age_voxels = (_age[single_closure == 1] + 0.5).astype("int")
            _ng_voxels = _ng[single_closure == 1]
            _geo_age_voxels = _geo_age_voxels[_ng_voxels >= avg_ng / 2.0]
            min_geo_age = _geo_age_voxels.min() - 0.5
            avg_geo_age = int(_geo_age_voxels.mean())
            max_geo_age = _geo_age_voxels.max() + 0.5
            _depth_geobody_voxels = depth_cube[single_closure == 1]
            min_depth = _depth_geobody_voxels.min()
            max_depth = _depth_geobody_voxels.max()
            # Define AOI where salt has been dilated
            # close_to_salt = np.zeros_like(salt)
            # close_to_salt[self.wide_salt[:] == 1] = 1.0
            # close_to_salt[salt == 1] = 0.0

            closure_boundary_cube = closures * 0.0
            closure_boundary_cube[
                np.where(
                    (_ng > 0.3)
                    & (_age > min_geo_age)  # account for partial voxels
                    & (_age < max_geo_age)
                    & (salt == 0.0)
                    & (depth_cube <= max_depth)
                )
            ] = 1.0
            print(
                "\n    ... grow_to_fault2: closure_boundary_cube voxels = ",
                closure_boundary_cube[closure_boundary_cube == 1].size,
            )

            n_voxel = single_closure[single_closure == 1].size

            original_voxels = n_voxel + 0
            print(
                "\n    ... closure label number, avg_throw, geobody shape, geo_age min/mean/max, depth min/max, avg_ng = ",
                i,
                n_voxel,
                (min_geo_age, avg_geo_age, max_geo_age),
                (min_depth, max_depth),
                avg_ng,
                il,
                " / ",
                len(labels_clean_list),
            )

            grown_closure = single_closure.copy()
            grown_closure[depth_cube >= max_depth] = 0
            delta_voxel = 0
            previous_delta_voxel = 1e9
            converged = False
            for ii in range(99):
                grown_closure = self.grow_lateral(grown_closure, 1, dist=2)
                grown_closure = self.grow_upward(grown_closure, 1, dist=1)
                # stay within layer, within age, close to salt and above HCWC
                grown_closure[closure_boundary_cube == 0.0] = 0.0
                single_closure = single_closure + grown_closure
                single_closure[single_closure > 0] = i
                new_n_voxel = single_closure[single_closure > 0].size
                previous_delta_voxel = delta_voxel + 0
                delta_voxel = new_n_voxel - n_voxel
                print(
                    "    ... i, ii, closure label number, geobody shape, delta_voxel, previous_delta_voxel,"
                    " delta_voxel>previous_delta_voxel = ",
                    i,
                    ii,
                    new_n_voxel,
                    delta_voxel,
                    previous_delta_voxel,
                    delta_voxel > previous_delta_voxel,
                )

                # If grown voxel is touching the egde of survey, stop and remove closure
                _a, _b, _ = np.where(single_closure > 0)
                max_boundary_i = self.cfg.cube_shape[0] - 1
                max_boundary_j = self.cfg.cube_shape[1] - 1
                if (
                    np.min(_a) == 0
                    or np.max(_a) == max_boundary_i
                    or np.min(_b) == 0
                    or np.max(_b) == max_boundary_j
                ):
                    print("Boundary reached, removing closure")
                    converged = False
                    break

                if n_voxel == new_n_voxel:
                    # finish bottom voxel layer near HCWC
                    grown_closure = self.grow_downward(
                        grown_closure, 1, dist=1, verbose=False
                    )
                    # stay within layer, within age, within fault block, above HCWC
                    grown_closure[closure_boundary_cube == 0.0] = 0.0
                    single_closure = single_closure + grown_closure
                    single_closure[single_closure > 0] = i
                    converged = True
                    break
                else:
                    n_voxel = new_n_voxel
                previous_delta_voxel = delta_voxel + 0
            if converged is True:
                labels_clean[single_closure > 0] = i
                msg_postscript = " converged"
            else:
                labels_clean[labels_clean == i] = -i
                msg_postscript = " NOT converged"
            msg = (
                f"closure_id: {int(i):04d}, "
                + f"original_voxels: {original_voxels:11.0f}, new_n_voxel: {new_n_voxel:11.0f}, "
                + f"percent_growth: {float(new_n_voxel / original_voxels):6.2f}"
            )
            print(msg + msg_postscript)
            self.cfg.write_to_logfile(msg + msg_postscript)

        # Set small closures to 0 after growth
        _initial_labels = measure.label(initial_closures, connectivity=2, background=0)
        _grown_labels = measure.label(labels_clean, connectivity=2, background=0)
        for x in np.unique(_grown_labels)[
            1:
        ]:  # ignore the first label of 0 (closures only)
            size_initial = _initial_labels[_initial_labels == x].size
            size_grown = _grown_labels[_grown_labels == x].size
            print(f"Size before editing: {size_initial}")
            print(f"Size after editing: {size_grown}")
            if size_grown < self.cfg.closure_min_voxels:
                print(
                    f"Closure below threshold of {self.cfg.closure_min_voxels} and will be removed"
                )
                labels_clean[_grown_labels == x] = 0.0

        return labels_clean

    @staticmethod
    def grow_lateral(geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((dist_size, dist_size, 1))
        mask[:, :, :] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_lateral: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_lateral: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    @staticmethod
    def grow_upward(geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((1, 1, dist_size))
        mask[0, 0, : dist + 1] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_upward: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_upward: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    @staticmethod
    def grow_downward(geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((1, 1, dist_size))
        mask[0, 0, dist:] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_downward: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_downward: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    @staticmethod
    def _threshold_volumes(volume, threshold=0.5):
        volume[volume >= threshold] = 1.0
        volume[volume < threshold] = 0.0
        return volume

    def parse_closure_codes(self, hc_closure_codes, labels, num, code=0.1):
        labels = labels.astype("float32")
        if num > 0:
            for x in range(1, num + 1):
                y = code + labels[labels == x].size
                labels[labels == x] = y
            hc_closure_codes += labels
        return hc_closure_codes


class Intersect3D(Closures):
    def __init__(
        self,
        faults,
        onlaps,
        oil_closures,
        gas_closures,
        brine_closures,
        closure_segment_list,
        closure_segments,
        parameters,
    ):
        self.closure_segment_list = closure_segment_list
        self.closure_segments = closure_segments
        self.cfg = parameters

        self.fault_throw = faults.max_fault_throw
        self.geologic_age = faults.faulted_age_volume
        self.geomodel_ng = faults.faulted_net_to_gross
        self.faults = self._threshold_volumes(faults.fault_planes.copy())
        self.onlaps = self._threshold_volumes(onlaps.copy())
        self.oil_closures = self._threshold_volumes(oil_closures.copy())
        self.gas_closures = self._threshold_volumes(gas_closures.copy())
        self.brine_closures = self._threshold_volumes(brine_closures.copy())

        self.wide_faults = None
        self.fat_faults = None
        self.onlaps_upward = None
        self.onlaps_downward = None
        self._dilate_faults_and_onlaps()

        # Outputs
        self.faulted_closures_oil = np.zeros_like(self.oil_closures)
        self.faulted_closures_gas = np.zeros_like(self.gas_closures)
        self.faulted_closures_brine = np.zeros_like(self.brine_closures)
        self.fault_closures_oil_segment_list = list()
        self.fault_closures_gas_segment_list = list()
        self.fault_closures_brine_segment_list = list()
        self.n_fault_closures_oil = 0
        self.n_fault_closures_gas = 0
        self.n_fault_closures_brine = 0

        self.onlap_closures_oil = np.zeros_like(self.oil_closures)
        self.onlap_closures_gas = np.zeros_like(self.gas_closures)
        self.onlap_closures_brine = np.zeros_like(self.brine_closures)
        self.onlap_closures_oil_segment_list = list()
        self.onlap_closures_gas_segment_list = list()
        self.onlap_closures_brine_segment_list = list()
        self.n_onlap_closures_oil = 0
        self.n_onlap_closures_gas = 0
        self.n_onlap_closures_brine = 0

        self.simple_closures_oil = np.zeros_like(self.oil_closures)
        self.simple_closures_gas = np.zeros_like(self.gas_closures)
        self.simple_closures_brine = np.zeros_like(self.brine_closures)
        self.n_4way_closures_oil = 0
        self.n_4way_closures_gas = 0
        self.n_4way_closures_brine = 0

        self.false_closures_oil = np.zeros_like(self.oil_closures)
        self.false_closures_gas = np.zeros_like(self.gas_closures)
        self.false_closures_brine = np.zeros_like(self.brine_closures)
        self.n_false_closures_oil = 0
        self.n_false_closures_gas = 0
        self.n_false_closures_brine = 0

    def find_faulted_closures(self):
        for iclosure in self.closure_segment_list:
            i, j, k = np.where(self.closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            if faults_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    # Faulted oil closure
                    self.faulted_closures_oil[i, j, k] = 1.0
                    self.n_fault_closures_oil += 1
                    self.fault_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    # Faulted gas closure
                    self.faulted_closures_gas[i, j, k] = 1.0
                    self.n_fault_closures_gas += 1
                    self.fault_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    # Faulted brine closure
                    self.faulted_closures_brine[i, j, k] = 1.0
                    self.n_fault_closures_brine += 1
                    self.fault_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is faulted but does not have oil, gas or brine assigned"
                    )

    def find_onlap_closures(self):
        for iclosure in self.closure_segment_list:
            i, j, k = np.where(self.closure_segments == iclosure)
            onlaps_within_closure = self.onlaps_upward[i, j, k]
            if onlaps_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    self.onlap_closures_oil[i, j, k] = 1.0
                    self.n_onlap_closures_oil += 1
                    self.onlap_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    self.onlap_closures_gas[i, j, k] = 1.0
                    self.n_onlap_closures_gas += 1
                    self.onlap_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    self.onlap_closures_brine[i, j, k] = 1.0
                    self.n_onlap_closures_brine += 1
                    self.onlap_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is onlap but does not have oil, gas or brine assigned"
                    )

    def find_simple_closures(self):
        for iclosure in self.closure_segment_list:
            i, j, k = np.where(self.closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            onlaps_within_closure = self.onlaps[i, j, k]
            oil_within_closure = self.oil_closures[i, j, k]
            gas_within_closure = self.gas_closures[i, j, k]
            brine_within_closure = self.brine_closures[i, j, k]
            if faults_within_closure.max() == 0 and onlaps_within_closure.max() == 0:
                if oil_within_closure.max() > 0:
                    self.simple_closures_oil[i, j, k] = 1.0
                    self.n_4way_closures_oil += 1
                elif gas_within_closure.max() > 0:
                    self.simple_closures_gas[i, j, k] = 1.0
                    self.n_4way_closures_gas += 1
                elif brine_within_closure.max() > 0:
                    self.simple_closures_brine[i, j, k] = 1.0
                    self.n_4way_closures_brine += 1
                else:
                    print(
                        "Closure is not faulted or onlap but does not have oil, gas or brine assigned"
                    )

    def find_false_closures(self):
        for iclosure in self.closure_segment_list:
            i, j, k = np.where(self.closure_segments == iclosure)
            faults_within_closure = self.fat_faults[i, j, k]
            onlaps_within_closure = self.onlaps_downward[i, j, k]
            for fluid, false, num in zip(
                [self.oil_closures, self.gas_closures, self.brine_closures],
                [
                    self.false_closures_oil,
                    self.false_closures_gas,
                    self.false_closures_brine,
                ],
                [
                    self.n_false_closures_oil,
                    self.n_false_closures_gas,
                    self.n_false_closures_brine,
                ],
            ):
                fluid_within_closure = fluid[i, j, k]
                if fluid_within_closure.max() > 0:
                    if onlaps_within_closure.max() > 0:
                        _faulted_closure_threshold = float(
                            faults_within_closure[faults_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        _onlap_closure_threshold = float(
                            onlaps_within_closure[onlaps_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        if (
                            _faulted_closure_threshold > 0.65
                            and _onlap_closure_threshold > 0.65
                        ):
                            false[i, j, k] = 1
                            num += 1

    def grow_to_fault2(self, closures):
        # - grow closures laterally and up within layer and within fault block
        print(
            "\n\n ... grow_to_fault2: grow closures laterally and up within layer and within fault block ..."
        )
        self.cfg.write_to_logfile("growing closures to fault plane: grow_to_fault2")

        dilated_fault_closures = closures.copy()
        n_faulted_closures = dilated_fault_closures.max()
        labels_clean = self.closure_segments.copy()
        labels_clean[closures == 0] = 0
        labels_clean_list = list(set(labels_clean.flatten()))
        labels_clean_list.remove(0)
        print("\n    ... grow_to_fault2: n_faulted_closures = ", len(labels_clean_list))
        print("    ... grow_to_fault2: faulted_closures = ", labels_clean_list)

        # fixme remove this once small closures are found and fixed
        voxel_sizes = [
            self.closure_segments[self.closure_segments == i].size
            for i in labels_clean_list
        ]
        for _v in voxel_sizes:
            print(f"Voxel_Sizes: {_v}")
            if _v < self.cfg.closure_min_voxels:
                print(_v)

        depth_cube = np.zeros(self.geologic_age.shape, float)
        _depths = np.arange(self.geologic_age.shape[2])
        depth_cube += _depths.reshape(1, 1, self.geologic_age.shape[2])
        _ng = self.geomodel_ng.copy()
        _age = self.geologic_age.copy()

        for il, i in enumerate(labels_clean_list):
            fault_blocks_list = list(set(self.fault_throw[labels_clean == i].flatten()))
            print("    ... grow_to_fault2: fault_blocks_list = ", fault_blocks_list)
            for jl, j in enumerate(fault_blocks_list):
                print(
                    "\n\n    ... label, throw = ",
                    i,
                    j,
                    list(set(self.fault_throw[labels_clean == i].flatten())),
                    labels_clean[labels_clean == i].size,
                    self.fault_throw[self.fault_throw == j].size,
                    self.fault_throw[
                        np.where((labels_clean == i) & (self.fault_throw == j))
                    ].size,
                )
                single_closure = labels_clean * 0.0
                size = single_closure[
                    np.where(
                        (labels_clean == i) & (np.abs(self.fault_throw - j) < 0.25)
                    )
                ].size
                if size >= self.cfg.closure_min_voxels:
                    print(f"Label: {i}, fault_block: {j}, Voxel_Count: {size}")
                    single_closure[
                        np.where(
                            (labels_clean == i) & (np.abs(self.fault_throw - j) < 0.25)
                        )
                    ] = 1
                if single_closure[single_closure > 0].size == 0:
                    # labels_clean[np.where((labels_clean == i) & (np.abs(self.fault_throw - j) < .25))] = 0
                    labels_clean[np.where(labels_clean == i)] = 0
                    continue
                avg_ng = _ng[single_closure == 1].mean()
                _geo_age_voxels = (_age[single_closure == 1] + 0.5).astype("int")
                _ng_voxels = _ng[single_closure == 1]
                _geo_age_voxels = _geo_age_voxels[_ng_voxels >= avg_ng / 2.0]
                min_geo_age = _geo_age_voxels.min() - 0.5
                avg_geo_age = int(_geo_age_voxels.mean())
                max_geo_age = _geo_age_voxels.max() + 0.5
                _depth_geobody_voxels = depth_cube[single_closure == 1]
                min_depth = _depth_geobody_voxels.min()
                max_depth = _depth_geobody_voxels.max()
                avg_throw = np.median(self.fault_throw[single_closure == 1])

                closure_boundary_cube = closures * 0.0
                closure_boundary_cube[
                    np.where(
                        (_ng > 0.0)
                        & (_age > min_geo_age)
                        & (_age < max_geo_age)
                        & (self.fault_throw == avg_throw)
                        & (depth_cube <= max_depth)
                    )
                ] = 1.0
                print(
                    "\n    ... grow_to_fault2: closure_boundary_cube voxels = ",
                    closure_boundary_cube[closure_boundary_cube == 1].size,
                )

                n_voxel = single_closure[single_closure == 1].size

                original_voxels = n_voxel + 0
                print(
                    "\n    ... closure label number, avg_throw, geobody shape, geo_age min/mean/max, depth min/max, avg_ng = ",
                    i,
                    j,
                    n_voxel,
                    (min_geo_age, avg_geo_age, max_geo_age),
                    (min_depth, max_depth),
                    avg_ng,
                    il,
                    " / ",
                    len(labels_clean_list),
                )

                grown_closure = single_closure.copy()
                grown_closure[depth_cube >= max_depth] = 0
                delta_voxel = 0
                previous_delta_voxel = 1e9
                converged = False
                for ii in range(15):
                    grown_closure = self.grow_lateral(grown_closure, 1, dist=2)
                    grown_closure = self.grow_upward(grown_closure, 1, dist=1)
                    # stay within layer, within age, within fault block, above HCWC
                    grown_closure[closure_boundary_cube == 0.0] = 0.0
                    single_closure = single_closure + grown_closure
                    single_closure[single_closure > 0] = i
                    new_n_voxel = single_closure[single_closure > 0].size
                    previous_delta_voxel = delta_voxel + 0
                    delta_voxel = new_n_voxel - n_voxel
                    print(
                        "    ... i, ii, closure label number, geobody shape, delta_voxel, previous_delta_voxel,"
                        " delta_voxel>previous_delta_voxel = ",
                        i,
                        ii,
                        new_n_voxel,
                        delta_voxel,
                        previous_delta_voxel,
                        delta_voxel > previous_delta_voxel,
                    )
                    if n_voxel == new_n_voxel:
                        # finish bottom voxel layer near "HCWC"
                        grown_closure = self.grow_downward(
                            grown_closure, 1, dist=1, verbose=False
                        )
                        # stay within layer, within age, within fault block, above HCWC
                        grown_closure[closure_boundary_cube == 0.0] = 0.0
                        single_closure = single_closure + grown_closure
                        single_closure[single_closure > 0] = i
                        converged = True
                        break
                    else:
                        n_voxel = new_n_voxel
                    previous_delta_voxel = delta_voxel + 0
                if converged is True:
                    labels_clean[single_closure > 0] = i
                    msg_postscript = " converged"
                else:
                    labels_clean[labels_clean == i] = -i
                    msg_postscript = " NOT converged"
                msg = (
                    "closure_id: "
                    + format(i, "4d")
                    + ", fault_id: "
                    + format(int(j + 0.5), "4d")
                    + ", original_voxels: "
                    + format(original_voxels, "11,.0f")
                    + ", new_n_voxel: "
                    + format(new_n_voxel, "11,.0f")
                    + ", percent_growth: "
                    + format(float(new_n_voxel) / original_voxels, "6.2f")
                )
                print(msg + msg_postscript)
                self.cfg.write_to_logfile(msg + msg_postscript)

        # Set small closures to 0 after growth
        _grown_labels = measure.label(labels_clean, connectivity=2, background=0)
        for x in np.unique(_grown_labels):
            size = _grown_labels[_grown_labels == x].size
            print(f"Size before editing: {size}")
            if size < self.cfg.closure_min_voxels:
                labels_clean[_grown_labels == x] = 0.0
        for x in np.unique(labels_clean):
            size = labels_clean[labels_clean == x].size
            print(f"Size after editing: {size}")

        return labels_clean

    def _dilate_faults_and_onlaps(self):
        self.wide_faults = self.grow_lateral(self.faults, 9, dist=1, verbose=False)
        self.fat_faults = self.grow_lateral(self.faults, 21, dist=1, verbose=False)
        mask = np.zeros((1, 1, 3))
        mask[0, 0, :2] = 1
        self.onlaps_upward = morphology.binary_dilation(self.onlaps, mask)
        mask = np.zeros((1, 1, 3))
        mask[0, 0, 1:] = 1
        self.onlaps_downward = self.onlaps.copy()
        for k in range(30):
            try:
                self.onlaps_downward = morphology.binary_dilation(
                    self.onlaps_downward, mask
                )
            except:
                break

    @staticmethod
    def _threshold_volumes(volume, threshold=0.5):
        volume[volume >= threshold] = 1.0
        volume[volume < threshold] = 0.0
        return volume

    @staticmethod
    def grow_up_and_lateral(geobody, iterations, vdist=1, hdist=1, verbose=False):
        from scipy.ndimage import maximum_filter

        hdist_size = 2 * hdist + 1
        vdist_size = 2 * vdist + 1
        mask = np.zeros((hdist_size, hdist_size, vdist_size))
        mask[:, :, : vdist + 1] = 1
        _geobody = geobody.copy()
        if verbose:
            print(
                " ...grow_up_and_lateral: _geobody.shape = ",
                _geobody[_geobody > 0].shape,
            )
        for k in range(iterations):
            try:
                _geobody = maximum_filter(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_up_and_lateral: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    @staticmethod
    def grow_lateral(geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((dist_size, dist_size, 1))
        mask[:, :, :] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_lateral: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_lateral: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    @staticmethod
    def grow_upward(geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((1, 1, dist_size))
        mask[0, 0, : dist + 1] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_upward: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_upward: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    @staticmethod
    def grow_downward(geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((1, 1, dist_size))
        mask[0, 0, dist:] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_downward: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_downward: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody


def variable_max_column_height(top_lith_idx, num_horizons, hmin=25, hmax=200):
    """
    Create a 1-D array of maximum column heights using linear function in layer numbers
    Shallow closures will have small vertical closure heights
    Deep closures will have larger vertical closure heights

    Would be better to use a pressure profile to determine maximum column heights at given depths

    :param top_lith_idx: 1-D array of horizon numbers corresponding to top of layers where lithology changes
    :param num_horizons: Total number of horizons in model
    :param hmin: Minimum column height to use in linear function
    :param hmax: Maximum column height to use in linear function
    :return: 1-D array of column heights of closures
    """
    # Use a linear function to determine max column height based on layer number
    column_heights = np.linspace(hmin, hmax, num=num_horizons)
    max_col_heights = column_heights[top_lith_idx]
    return max_col_heights


# Horizon Spill Point functions
def fill_to_spill(test_array, array_flags, empty_value=1.0e22, quiet=True):
    if not quiet:
        print("   ... start fillToSpill ......")
    temp_array = test_array.copy()
    flags = array_flags.copy()
    test_array_max = 2.0 * (temp_array[~np.isnan(temp_array)]).max()
    temp_array[array_flags == 255] = -empty_value
    flood_filled = test_array_max - flood_fill_heap(
        test_array_max - temp_array, empty_value=empty_value
    )
    if not quiet:
        print("   ... finish fillToSpill ......")

    flood_filled[array_flags != 1] = 0
    flood_filled[flood_filled == 1.0e5] = 0
    flags[flood_filled == empty_value] = 0

    return flood_filled


def flood_fill_heap(test_array, empty_value=1.0e22, quiet=True):
    # from internet: http://arcgisandpython.blogspot.co.uk/2012/01/python-flood-fill-algorithm.html

    import heapq
    from scipy import ndimage

    input_array = np.copy(test_array)
    num_validPoints = (
        test_array.flatten().shape[0]
        - input_array[np.isnan(input_array)].shape[0]
        - input_array[input_array > empty_value / 2].shape[0]
    )
    if not quiet:
        print(
            "     ... flood_fill_heap ... number of valid input horizon picks = ",
            num_validPoints,
        )

    validPoints = input_array[~np.isnan(input_array)]
    validPoints = validPoints[validPoints < empty_value / 2]
    validPoints = validPoints[validPoints < 1.0e5]
    validPoints = validPoints[validPoints > np.percentile(validPoints, 2)]

    if len(validPoints) > 2:
        amin = validPoints.min()
        amax = validPoints.max()
    else:
        return test_array

    if not quiet:
        print(
            "    ... validPoints stats = ",
            validPoints.min(),
            np.median(validPoints),
            validPoints.mean(),
            validPoints.max(),
        )
        print(
            "    ... validPoints %tiles = ",
            np.percentile(validPoints, 0),
            np.percentile(validPoints, 1),
            np.percentile(validPoints, 5),
            np.percentile(validPoints, 10),
            np.percentile(validPoints, 25),
            np.percentile(validPoints, 50),
            np.percentile(validPoints, 75),
            np.percentile(validPoints, 90),
            np.percentile(validPoints, 95),
            np.percentile(validPoints, 99),
            np.percentile(validPoints, 100),
        )
        from datagenerator.util import import_matplotlib

        plt = import_matplotlib()
        plt.figure(5)
        plt.clf()
        plt.imshow(np.flipud(input_array), vmin=amin, vmax=amax, cmap="jet_r")
        plt.colorbar()
        plt.show()
        plt.savefig("flood_fill.png", format="png")
        plt.close()

        print("     ... min & max for surface = ", amin, amax)

    # set empty values and nan's to huge
    input_array[np.isnan(input_array)] = empty_value

    # Set h_max to a value larger than the array maximum to ensure that the while loop will terminate
    h_max = np.max(input_array * 2.0)

    # Build mask of cells with data not on the edge of the image
    # Use 3x3 square structuring element
    el = ndimage.generate_binary_structure(2, 2).astype(int)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    inside_mask[input_array == empty_value] = False
    edge_mask = np.invert(inside_mask)
    # Initialize output array as max value test_array except edges
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max

    if not quiet:
        plt.figure(6)
        plt.clf()
        plt.imshow(np.flipud(input_array), cmap="jet_r")
        plt.colorbar()
        plt.show()
        plt.savefig("flood_fill2.png", format="png")
        plt.close()

    # Build priority queue and place edge pixels into priority queue
    # Last value is flag to indicate if cell is an edge cell
    put = heapq.heappush
    get = heapq.heappop
    fill_heap = [
        (output_array[t_row, t_col], int(t_row), int(t_col), 1)
        for t_row, t_col in np.transpose(np.where(edge_mask))
    ]
    heapq.heapify(fill_heap)

    # Iterate until priority queue is empty
    while 1:
        try:
            h_crt, t_row, t_col, edge_flag = get(fill_heap)
        except IndexError:
            break
        for n_row, n_col in [
            ((t_row - 1), t_col),
            ((t_row + 1), t_col),
            (t_row, (t_col - 1)),
            (t_row, (t_col + 1)),
        ]:
            # Skip cell if outside array edges
            if edge_flag:
                try:
                    if not inside_mask[n_row, n_col]:
                        continue
                except IndexError:
                    continue
            if output_array[n_row, n_col] == h_max:
                output_array[n_row, n_col] = max(h_crt, input_array[n_row, n_col])
                put(fill_heap, (output_array[n_row, n_col], n_row, n_col, 0))
    output_array[output_array == empty_value] = np.nan
    return output_array


def _flood_fill(horizon, max_column_height=20.0, verbose=False, debug=False):
    """Locate areas on horizon that are in structural closure.

    # horizon: depth horizon as 2D numpy array.
    # - assume that fault intersections are inserted with value of 0.
    # - assume that values represent depth (i.e., bigger values are deeper)"""
    from scipy import ndimage

    # copy input array
    temp_event = horizon.copy()

    emptypicks = temp_event * 0.0
    emptypicks[temp_event < 1.0] = 1.0
    emptypicks_dilated = ndimage.grey_dilation(
        emptypicks, size=(3, 3), structure=np.ones((3, 3))
    )
    # dilation removes some of the event - turn this off to honour the input events exactly
    # Changed to avoid vertical closure-boundaries near faults
    # emptypicks_dilated = emptypicks
    if verbose:
        print(
            " emptypicks_dilated min,mean,max = ",
            emptypicks_dilated.min(),
            emptypicks_dilated.mean(),
            emptypicks_dilated.max(),
        )

    # create boundary around edges of 2D array
    temp_event[:, :3] = 0.0
    temp_event[:, -3:] = 0.0
    temp_event[:3, :] = 0.0
    temp_event[-3:, :] = 0.0

    # replace pixels with value=0 with vertical 'wall' that is max_column_height deeper than nearby pixels
    temp_event[
        np.logical_and(emptypicks_dilated == 2.0, temp_event != 0.0)
    ] += max_column_height
    temp_event[emptypicks == 1.0] += 0.0

    # put deep point at map origin to 'collect' flood-fill run-off
    temp_event[0, 0] = 1.0e5

    # create flags to indicate pick vs no-pick in 2D array
    flags = np.zeros((horizon.shape[0], horizon.shape[1]), "int")
    flags[temp_event > 0.0] = 1
    flags[0, 0] = 1

    flood_filled = -fill_to_spill(-temp_event, flags)

    # set pixels near fault gaps to empty
    flood_filled[np.logical_and(emptypicks_dilated == 2.0, temp_event != 0.0)] = 0.0

    # limit closure heights to max_column_height
    # - Note that this typically causes under-filling of shallow 4-way closures
    if debug:
        import pdb

        pdb.set_trace()
    ff = flood_filled.copy()
    diff = horizon - ff
    diff[flood_filled == 0.0] = 0.0
    diff[diff != 0.0] = 1.0

    from skimage import morphology
    from skimage import measure

    labels = measure.label(diff, connectivity=2, background=0)
    labels_clean = morphology.remove_small_objects(labels, 50)
    labels_clean_list = list(set(labels_clean.flatten()))
    labels_clean_list.sort()
    for i in labels_clean_list:
        if i == 0:
            continue
        trap_crest = -horizon[labels_clean == i].min()
        initial_size = horizon[labels_clean == i].size
        spill_depth_map = np.ones_like(horizon) * (trap_crest - max_column_height)
        spill_points = np.dstack((-flood_filled, spill_depth_map))
        spill_point_map = spill_points.max(axis=-1)
        spill_point_map[spill_point_map == -100000.0] = 0.0
        spill_point_map[spill_point_map > 0.0] = 0.0
        flood_filled[labels_clean == i] = -spill_point_map[labels_clean == i]
        flood_filled[flood_filled < horizon] = horizon[flood_filled < horizon]
        final_size = horizon[
            np.where((horizon - flood_filled != 0.0) & (labels_clean == i))
        ].size
        print(
            "  ...inside _flood_fill: i, initial_size, final_size = ",
            i,
            initial_size,
            final_size,
        )
        del spill_depth_map
        del spill_points
        del spill_point_map
    del ff
    del diff
    del labels
    del labels_clean

    return flood_filled


def get_top_of_closure(inarray, pad_up=0, pad_down=0):
    """Create a mask leaving only the top of a closure."""
    mask = inarray != 0
    t = np.where(mask.any(axis=-1), mask.argmax(axis=-1), -1)
    xy = np.argwhere(t > 0)
    z = t[t > 0]
    outarray = np.zeros_like(inarray)
    for (x, y), z in zip(xy, z):
        zmin = z - pad_up
        zmax = z + pad_down + 1
        outarray[x, y, zmin:zmax] = 1
    return outarray


def lsq(x, y, axis=-1):
    ###
    ### compute the slope and intercept for an array with points to be fit
    ### in the last dimension. can be in other axis using the 'axis' parmameter.
    ###
    ### returns:
    ### - intercept
    ### - slope
    ### - pearson r (normalized cross-correlation coefficient)
    ###
    ### output will have dimensions of input with one less axis
    ### - (specified by axis parameter)
    ###

    """
    # compute x and y with mean removed
    x_zeromean = x * 1.
    x_zeromean -= x.mean(axis=axis).reshape(x.shape[0],x.shape[1],1)
    y_zeromean = y * 1.
    y_zeromean -= y.mean(axis=axis).reshape(y.shape[0],y.shape[1],1)
    """

    # compute pearsonr
    r = np.sum(x * y, axis=axis) - np.sum(x) * np.sum(y, axis=axis) / y.shape[axis]
    r /= np.sqrt(
        (np.sum(x ** 2, axis=axis) - np.sum(x, axis=axis) ** 2 / y.shape[axis])
        * (np.sum(y ** 2, axis=axis) - np.sum(y, axis=axis) ** 2 / y.shape[axis])
    )

    # compute slope
    slope = r * y.std(axis=axis) / x.std(axis=axis)

    # compute intercept
    intercept = y.mean(axis=axis) - slope * x.mean(axis=axis)

    return intercept, slope, r


def compute_ai_gi(parameters, seismic_data):
    """[summary]

    Args:
        cfg (Parameter class object): Model Parameters
        seismic_data (np.array): Seismic data with shape n * x * y * z,
                                 where n is number of angle stacks
    """
    inc_angles = np.array(parameters.incident_angles)
    inc_angles = np.sin(inc_angles * np.pi / 180.0) ** 2
    inc_angles = inc_angles.reshape(len(inc_angles), 1, 1, 1)

    intercept, slope, _ = lsq(inc_angles, seismic_data, axis=0)

    intercept[np.isnan(intercept)] = 0.0
    slope[np.isnan(slope)] = 0.0
    intercept[np.isinf(intercept)] = 0.0
    slope[np.isinf(slope)] = 0.0
    return intercept, slope
