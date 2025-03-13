from math import cos, sin
import numpy as np
from datagenerator.util import is_it_in_hull


class SaltModel:
    """
    Salt Model Class
    ----------------

    This class creates a 3D salt body and inserts it into the input cube.
    """

    def __init__(self, parameters) -> None:
        """
        Initialization function
        -----------------------

        Initializes the SaltModel class.


        Parameters
        ----------
        parameters : datagenerator.Parameters
            The parameters of the project.

        Returns
        -------
        None
        """
        self.cfg = parameters
        cube_shape = (
            self.cfg.cube_shape[0],
            self.cfg.cube_shape[1],
            self.cfg.cube_shape[2] + self.cfg.pad_samples,
        )
        self.salt_segments = self.cfg.hdf_init("salt_segments", shape=cube_shape)
        self.points = []

    def compute_salt_body_segmentation(self) -> None:
        """
        Compute Salt Body Segmentation
        ------------------------------

        Creates a 3D salt body.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("\n\n...compute salt segments cube...")
        salt_radius = np.random.triangular(
            self.cfg.cube_shape[0] / 6,
            self.cfg.cube_shape[0] / 5,
            self.cfg.cube_shape[0] / 4,
        )
        shallowest_salt = self.cfg.h5file.root.ModelData.faulted_depth_maps[
            :, :, 1
        ].copy()
        # shallowest_salt -= pad_samples / infill_factor
        # shallowest_salt -= pad_samples
        # shallowest_salt /= infill_factor
        shallowest_salt = shallowest_salt[np.abs(shallowest_salt) < 1.0e10]
        shallowest_salt = np.percentile(shallowest_salt[~np.isnan(shallowest_salt)], 99)
        shallowest_salt += np.random.uniform(150, 300)
        # output_cube = np.zeros(self.cfg.cube_shape)
        print(
            f"   ...top of salt at {shallowest_salt}, with cube having shape {self.salt_segments.shape}"
        )
        self.salt_segments[:] = self.insertSalt3D(
            shallowest_salt,
            salt_radius=salt_radius,
        )

    @staticmethod
    def create_circular_pointcloud(cx, cy, cz, radius, verbose=False):
        """
        Create a circular pointcloud
        ----------------------------

        Creates a circle of points with random jitter in (x, y, z).


        Parameters
        ----------
        cx : float
            X-coordinate for centre of points
        cy : float
            Y-coordinate for centre of points
        cz : float
            Z-coordinate for centre of points
        radius : float
            Radius of circle
        verbose : bool, optional
            Print points while building circle. Defaults to False.

        Returns
        -------
        None
        """
        points = []
        for iangle in np.arange(0.0, 360.0, 10.0):
            xy_max_jitter = 0.1 * radius
            x = (
                cx
                + radius * cos(iangle * np.pi / 180.0)
                + np.random.uniform(-xy_max_jitter, xy_max_jitter)
            )
            y = (
                cy
                + radius * sin(iangle * np.pi / 180.0)
                + np.random.uniform(-xy_max_jitter, xy_max_jitter)
            )
            z = cz + np.random.uniform(-xy_max_jitter / 2.0, xy_max_jitter / 2.0)
            points.append([x, y, z])
            if verbose:
                if iangle == 0.0:
                    print("\n")
                print(f"   ...angle, points = {iangle}, ({x}, {y}, {z})")

        return points

    def salt_circle_points(
        self,
        centre_points: list,
        radii: list,
        cx: float,
        cy: float,
    ) -> None:
        """
        Salt Circle Points
        ------------------

        Creates clouds of points representing the top or base of a salt body.

        Parameters
        ----------
        centre_points : list
            Z-coordinates for deep, mid, shallow and tip locations
        radii : list
            Radius constants for deep, mid and shallow circles
        cx : float)
            X-coordinate for centre of point cloud
        cy : float
            Y-coordinate for centre of point cloud

        Returns
        -------
        None
        """
        c_deep, c_mid, c_shallow, c_tip = centre_points
        r_deep, r_mid, r_shallow = radii
        print(
            f"\n   ...Centre_tip, Centre_shallow, Centre_mid, Centre_deep = {c_tip:.2f}, {c_shallow:.2f}, {c_mid:.2f}, {c_deep:.2f}"
        )
        print(
            f"\n   ...radius_deep, radius_mid, radius_shallow = {r_deep:.2f}, {r_mid:.2f}, {r_shallow:.2f}"
        )

        self.points += self.create_circular_pointcloud(cx, cy, c_deep, r_deep)
        self.points += self.create_circular_pointcloud(cx, cy, c_mid, r_mid)
        self.points += self.create_circular_pointcloud(cx, cy, c_shallow, r_shallow)

        # Add points for the tip
        xy_max_jitter = 0.1 * r_shallow
        x = cx + np.random.uniform(-xy_max_jitter, xy_max_jitter)
        y = cy + np.random.uniform(-xy_max_jitter, xy_max_jitter)
        z = c_tip + np.random.uniform(-xy_max_jitter / 2.0, xy_max_jitter / 2.0)
        self.points.append([x, y, z])
        print(f"\n   ...tip (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f})")

    def insertSalt3D(self, top, salt_radius=100.0) -> np.ndarray:
        """
        Insert 3D Salt
        --------------

        Generates a random salt body and insert into input_cube.

        generate a random salt body and insert into input_cube
        salt is modeled using 2 (almost) vertically aligned spheres connected
        by a convex hull. Points inside the hull are considered salt.
        center (in x,y) of upper and lower spheres are randomly jittered
        so salt body is not always predominantly vertical
        salt body is built using a 'bag of points'. Each point is
        randomly jittered in x,y,z

        Parameters
        ----------
        top : float
            Z-coordinate for top of salt body
        salt_radius : float, optional
            Radius of salt body. Defaults to 100.0.

        Returns
        -------
        None
        """
        # Points built to model the crest of a salt body, with
        # gentle anitclinal structure
        # Z-coordinates for center of top salt
        C1_deep = (top + salt_radius) * 1.1
        C1_mid = C1_deep - salt_radius * np.random.uniform(0.35, 0.45)
        C1_shallow = C1_deep - salt_radius * np.random.uniform(0.87, 0.93)
        C1_tip = C1_deep - salt_radius * np.random.uniform(0.98, 1.02)
        top_centres = [C1_deep, C1_mid, C1_shallow, C1_tip]

        # Radius constants for top of salt
        R1_deep = salt_radius * np.random.uniform(0.9, 1.1)
        R1_mid = R1_deep * np.random.uniform(0.37, 0.43)
        R1_shallow = R1_deep * np.random.uniform(0.18, 0.25)
        top_radii = [R1_deep, R1_mid, R1_shallow]

        cube_shape = self.salt_segments.shape

        # x,y coords for top of salt
        center_x = cube_shape[0] / 2 + np.random.uniform(
            -cube_shape[0] * 0.4, cube_shape[0] * 0.4
        )
        center_y = cube_shape[1] / 2 + np.random.uniform(
            -cube_shape[0] * 0.4, cube_shape[0] * 0.4
        )
        self.salt_circle_points(top_centres, top_radii, center_x, center_y)

        # Build points to model the base of a salt body
        C2_deep = min(
            max(
                C1_deep + salt_radius * np.random.uniform(2.3, 12.5),
                cube_shape[2] - 2.0 * (salt_radius * np.random.uniform(-1.3, 3.5)),
            ),
            cube_shape[2] + 1.0 * (salt_radius),
        )
        C2_mid = C2_deep + salt_radius * np.random.uniform(0.35, 0.45) / 2.0
        C2_shallow = C2_deep + salt_radius * np.random.uniform(0.87, 0.93) / 2.0
        C2_tip = C2_deep + salt_radius * np.random.uniform(0.98, 1.02) / 2.0
        base_centres = [C2_deep, C2_mid, C2_shallow, C2_tip]

        # radius constants for lower circle
        R2_deep = R1_shallow / 2.0 * np.random.uniform(0.9, 1.1)
        R2_mid = R2_deep * np.random.uniform(0.37, 0.43)
        R2_shallow = R2_deep * np.random.uniform(0.18, 0.25)
        base_radii = [R2_deep, R2_mid, R2_shallow]

        # x,y coords for lower circle
        x_center = center_x * np.random.uniform(0.3, 1.7)
        y_center = center_y * np.random.uniform(0.3, 1.7)

        self.salt_circle_points(base_centres, base_radii, x_center, y_center)

        # build convex hull around all points for salt

        # create list of indices of entire cube
        xx, yy, zz = np.meshgrid(
            list(range(cube_shape[0])),
            list(range(cube_shape[1])),
            list(range(cube_shape[2])),
            sparse=False,
            indexing="ij",
        )
        xyz = (
            np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
            .swapaxes(0, 1)
            .astype("float")
        )

        # check points in cube for inside or outside hull
        in_hull = is_it_in_hull(self.points, xyz)
        in_hull = in_hull.reshape(cube_shape)

        salt_segments = np.zeros(cube_shape, "int16")
        salt_segments[in_hull == True] = 1.0

        return salt_segments

    def update_depth_maps_with_salt_segments(self) -> None:
        """
        Update Depth maps and DepthMapsGaps with salt segments
        ------------------------------------------------------

        Updates the depth maps with salt segments

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ii, jj = np.meshgrid(
            range(self.cfg.cube_shape[0]),
            range(self.cfg.cube_shape[1]),
            sparse=False,
            indexing="ij",
        )

        depth_maps = self.cfg.h5file.root.ModelData.faulted_depth_maps[:]
        depth_maps_gaps = self.cfg.h5file.root.ModelData.faulted_depth_maps_gaps[:]
        salt_segments = self.cfg.h5file.root.ModelData.salt_segments[:]

        depth_maps_gaps_salt = np.zeros_like(depth_maps_gaps)

        for ihorizon in range(0, depth_maps.shape[2] - 1):
            print("   ...inserting salt in horizons...")
            print(
                "      ...depth_maps min/mean/max = ",
                depth_maps[:, :, ihorizon].min(),
                depth_maps[:, :, ihorizon].mean(),
                depth_maps[:, :, ihorizon].max(),
            )
            depth_map = depth_maps[:, :, ihorizon].copy()
            depth_map_gaps = depth_maps_gaps[:, :, ihorizon].copy()

            # remove horizon picks inside salt (depth_maps_gaps)
            depth_maps_gaps_horizon = depth_map_gaps.copy()
            depth_maps_gaps_horizon[np.isnan(depth_map_gaps)] = 0.0

            _faulted_depth_map = depth_map.copy()
            faulted_depth_map_indices = _faulted_depth_map.astype("int")
            faulted_depth_map_indices = np.clip(
                faulted_depth_map_indices,
                0,
                self.cfg.h5file.root.ModelData.faulted_depth.shape[2] - 1,
            )

            _label = salt_segments[ii, jj, faulted_depth_map_indices]
            depth_maps_gaps_horizon[depth_maps_gaps_horizon == 0.0] = (
                np.nan
            )  # reset nan's
            depth_maps_gaps_horizon[_label > 0] = np.nan
            depth_maps_gaps_salt[..., ihorizon] = depth_maps_gaps_horizon
            try:
                number_muted_points = depth_maps_gaps_horizon[_label > 0].shape[0]
                if number_muted_points > 0:
                    print(
                        "    ... horizon {} will have {}  points muted inside salt".format(
                            ihorizon, number_muted_points
                        )
                    )
            except:
                pass

        self.cfg.h5file.root.ModelData.faulted_depth_maps_gaps[:] = depth_maps_gaps_salt

    def update_depth_maps_with_salt_segments_drag(self):
        """
        Update depth maps with salt segments frag
        -----------------------------------------

        Updates depth maps with salt segments and drag.

        Parameters
        ----------
        None

        Returns
        -------
        depth_maps_salt : np.ndarray
            The depth maps with salt.
        depth_maps_gaps_salt : np.ndarray
            The depth maps gaps with salt.
        """
        from scipy import ndimage

        ii, jj = np.meshgrid(
            range(self.cfg.cube_shape[0]),
            range(self.cfg.cube_shape[1]),
            sparse=False,
            indexing="ij",
        )

        depth_maps = self.cfg.h5file.root.ModelData.faulted_depth_maps[:]
        depth_maps_gaps = self.cfg.h5file.root.ModelData.faulted_depth_maps_gaps[:]
        salt_segments = self.cfg.h5file.root.ModelData.salt_segments[:]

        if self.cfg.model_qc_volumes:
            np.save(f"{self.cfg.work_subfolder}/depth_maps_presalt.npy", depth_maps)
            np.save(
                f"{self.cfg.work_subfolder}/depth_maps_gaps_presalt.npy",
                depth_maps_gaps,
            )
            np.save(f"{self.cfg.work_subfolder}/salt_segments.npy", salt_segments)

        depth_maps_salt = np.zeros_like(depth_maps_gaps)
        depth_maps_gaps_salt = np.zeros_like(depth_maps_gaps)

        relative_salt_depth = 0

        for ihorizon in range(0, depth_maps.shape[2]):
            print("   ...inserting salt in horizons...")
            print(
                "      ...depth_maps min/mean/max = ",
                depth_maps[:, :, ihorizon].min(),
                depth_maps[:, :, ihorizon].mean(),
                depth_maps[:, :, ihorizon].max(),
            )
            depth_map = depth_maps[:, :, ihorizon].copy()
            # depth_map_gaps = depth_maps_gaps[:, :, ihorizon].copy()

            # remove horizon picks inside salt (depth_maps_gaps)
            # depth_maps_gaps_horizon = depth_map_gaps.copy()
            # depth_maps_gaps_horizon[np.isnan(depth_map_gaps)] = 0.0

            _faulted_depth_map = depth_map.copy()
            faulted_depth_map_indices = _faulted_depth_map.astype("int")
            faulted_depth_map_indices = np.clip(
                faulted_depth_map_indices,
                0,
                self.cfg.h5file.root.ModelData.faulted_depth.shape[2] - 1,
            )

            _label = salt_segments[ii, jj, faulted_depth_map_indices]
            if np.max(_label) > 0:
                relative_salt_depth += 1

            # depth_maps_gaps_horizon[
            #    depth_maps_gaps_horizon == 0.0
            # ] = np.nan  # reset nan's

            # Apply vertical shift to the horizons where they are inside the salt body, relative to top of salt
            # depth_maps_gaps_horizon[_label > 0] -= 2 * relative_salt_depth
            # depth_maps_gaps_horizon = ndimage.gaussian_filter(depth_maps_gaps_horizon, 3)

            # Do the same shift to the non-gapped horizon to build the facies volume
            depth_map[_label > 0] -= 2 * relative_salt_depth
            depth_map = ndimage.gaussian_filter(depth_map, 3)

            # depth_maps_gaps_horizon[_label > 0] = np.nan
            depth_maps_salt[..., ihorizon] = depth_map

            # Re-apply blanking inside the salt to the gapped horizons
            # _faulted_depth_map = depth_map.copy()
            # faulted_depth_map_indices = _faulted_depth_map.astype("int")
            # faulted_depth_map_indices = np.clip(
            #    faulted_depth_map_indices,
            #    0,
            #    self.cfg.h5file.root.ModelData.faulted_depth.shape[2] - 1,
            # )
            # _label = salt_segments[ii, jj, faulted_depth_map_indices]

            # depth_maps_gaps_horizon[_label > 0] = np.nan
            # depth_maps_gaps_salt[..., ihorizon] = depth_maps_gaps_horizon

            try:
                number_muted_points = depth_maps[_label > 0].shape[0]
                if number_muted_points > 0:
                    print(
                        f"    ... horizon {ihorizon} will have {number_muted_points}  points muted inside salt"
                    )
            except:
                pass

        if self.cfg.model_qc_volumes:
            np.save(
                f"{self.cfg.work_subfolder}/depth_maps_salt_prepushdown.npy",
                depth_maps_salt,
            )

        depth_maps_salt = push_down_remove_negative_thickness(depth_maps_salt)

        for ihorizon in range(0, depth_maps.shape[2] - 1):
            depth_map_gaps_salt = depth_maps_salt[:, :, ihorizon].copy()
            faulted_depth_map_indices = depth_map_gaps_salt.astype("int")
            faulted_depth_map_indices = np.clip(
                faulted_depth_map_indices,
                0,
                salt_segments.shape[2] - 1,
            )
            _label = salt_segments[ii, jj, faulted_depth_map_indices]
            depth_map_gaps_salt[_label > 0] = np.nan
            depth_maps_gaps_salt[..., ihorizon] = depth_map_gaps_salt

        depth_maps_gaps_salt[np.isnan(depth_maps_gaps)] = np.nan

        # self.cfg.h5file.root.ModelData.faulted_depth_maps[:] = depth_maps
        # self.cfg.h5file.root.ModelData.faulted_depth_maps_gaps[:] = depth_maps_gaps_salt

        if self.cfg.model_qc_volumes:
            np.save(f"{self.cfg.work_subfolder}/depth_maps_salt.npy", depth_maps_salt)
            np.save(
                f"{self.cfg.work_subfolder}/depth_maps_gaps_salt.npy",
                depth_maps_gaps_salt,
            )
            np.save(f"{self.cfg.work_subfolder}/facies_label.npy", _label)

        return depth_maps_salt, depth_maps_gaps_salt


def push_down_remove_negative_thickness(depth_maps: np.ndarray) -> np.ndarray:
    """
    Remove negative thicknesses
    ---------------------------

    Removes negative thicknesses.

    Parameters
    ----------
    depth_maps : np.ndarray
        The depth maps.

    Returns
    -------
    depth_maps : np.ndarray
        The depth maps with the negative thicknesses removed.
    """
    for i in range(depth_maps.shape[-1] - 1, 1, -1):
        layer_thickness = depth_maps[..., i] - depth_maps[..., i - 1]
        if np.min(layer_thickness) < 0:
            np.clip(layer_thickness, 0, a_max=None, out=layer_thickness)
            depth_maps[..., i - 1] = depth_maps[..., i] - layer_thickness

    return depth_maps
