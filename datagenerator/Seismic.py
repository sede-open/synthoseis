import os
import itertools
from multiprocessing import Pool
import numpy as np

from tqdm import trange
from datagenerator.histogram_equalizer import normalize_seismic
from datagenerator.Geomodels import Geomodel
from datagenerator.util import write_data_to_hdf
from datagenerator.wavelets import generate_wavelet, plot_wavelets
from rockphysics.RockPropertyModels import select_rpm, RockProperties, EndMemberMixing


class SeismicVolume(Geomodel):
    """
    SeismicVolume Class
    -------------------

    This class initializes the seismic volume that contains everything.
    """
    def __init__(self, parameters, faults, closures) -> None:
        """
        __init__
        --------

        The initialization function.

        Parameters
        ----------
        parameters : datagenerator.Parameters
            The parameters object of the project.
        faults : np.ndarray
            The faults array.
        closures : np.ndarray
            The closures array.
        
        Returns
        -------
        None
        """
        self.cfg = parameters
        self.faults = faults
        self.traps = closures

        if self.cfg.model_qc_volumes:
            # Add 0 and 45 degree angles to the list of user-input angles
            self.angles = tuple(sorted(set((0,) + self.cfg.incident_angles + (45,))))
        else:
            self.angles = self.cfg.incident_angles
        if self.cfg.verbose:
            print(f"Angles: {self.angles}")

        self.first_random_lyr = 20  # do not randomise shallow layers

        self.rho = self.cfg.hdf_init(
            "rho", shape=self.cfg.h5file.root.ModelData.faulted_depth.shape
        )
        self.vp = self.cfg.hdf_init(
            "vp", shape=self.cfg.h5file.root.ModelData.faulted_depth.shape
        )
        self.vs = self.cfg.hdf_init(
            "vs", shape=self.cfg.h5file.root.ModelData.faulted_depth.shape
        )
        self.rho_ff = self.cfg.hdf_init(
            "rho_ff", shape=self.cfg.h5file.root.ModelData.faulted_depth.shape
        )
        self.vp_ff = self.cfg.hdf_init(
            "vp_ff", shape=self.cfg.h5file.root.ModelData.faulted_depth.shape
        )
        self.vs_ff = self.cfg.hdf_init(
            "vs_ff", shape=self.cfg.h5file.root.ModelData.faulted_depth.shape
        )

        seis_shape = (
            len(self.angles),
            *self.cfg.h5file.root.ModelData.faulted_depth.shape,
        )
        rfc_shape = (seis_shape[0], seis_shape[1], seis_shape[2], seis_shape[3] - 1)
        self.rfc_raw = self.cfg.hdf_init("rfc_raw", shape=rfc_shape)
        self.rfc_noise_added = self.cfg.hdf_init("rfc_noise_added", shape=rfc_shape)

    def build_elastic_properties(self, mixing_method="inv_vel"):
        """
        Build Elastic Properties
        ------------------------

        Creates Rho, Vp, Vs volumes using depth trends.

        Parameters
        ----------
        mixing_method : str, optional
            Method for mixing Velocities. Defaults to "inv_vel".

        """
        rpm = select_rpm(self.cfg)
        self.build_property_models_randomised_depth(
            rpm,
            mixing_method=mixing_method
        )

    def bandlimit_volumes_wavelets(self, n_wavelets=10):
        # Pre-prepared n, m, f filter_specs from npy file
        fs_nr = np.load(self.cfg.wavelets[0])
        fs_md = np.load(self.cfg.wavelets[1])
        fs_fr = np.load(self.cfg.wavelets[2])

        for wavelet_count in range(n_wavelets):
            low_freq_pctile = np.random.uniform(0, 100)
            mid_freq_pctile = np.random.uniform(0, 100)
            high_freq_pctile = np.random.uniform(0, 100)
            low_med_high_percentiles = (
                low_freq_pctile,
                mid_freq_pctile,
                high_freq_pctile,
            )
            print(f"low_med_high_percentiles: {low_med_high_percentiles}")
            f_nr, a_nr, w_nr = generate_wavelet(fs_nr, low_med_high_percentiles)
            f_md, a_md, w_md = generate_wavelet(fs_md, low_med_high_percentiles)
            f_fr, a_fr, w_fr = generate_wavelet(fs_fr, low_med_high_percentiles)
            _f = (f_nr, f_md, f_fr)
            _a = (a_nr, a_md, a_fr)
            _w = (w_nr, w_md, w_fr)
            labels = ["Near", "Mid", "Far"]
            pngname = os.path.join(
                self.cfg.work_subfolder, f"random_wavelets_{wavelet_count}.png"
            )
            plot_wavelets(_f, _a, _w, labels, pngname)

            rfc_bandlimited_wavelet = np.zeros_like(self.rfc_noise_added)
            if self.cfg.model_qc_volumes:
                wavelets = [
                    w_nr,
                    w_nr,
                    w_md,
                    w_fr,
                    w_fr,
                ]  # use near for 0 and far for 45 deg cubes
            else:
                wavelets = [w_nr, w_md, w_fr]

            for idx in range(rfc_bandlimited_wavelet.shape[0]):
                rfc_bandlimited_wavelet[idx, ...] = apply_wavelet(
                    self.rfc_noise_added[idx, ...], wavelets[idx]
                )
            if self.cfg.lateral_filter_size > 1:
                # Apply lateral smoothing filter
                rfc_bandlimited_wavelet = self.apply_lateral_filter(
                    rfc_bandlimited_wavelet
                )
            cumsum_wavelet_noise_added = self.apply_cumsum(rfc_bandlimited_wavelet)
            _ = self.write_final_cubes_to_disk(
                rfc_bandlimited_wavelet, f"seismicCubes_RFC_Wavelet_{wavelet_count}_"
            )
            _ = self.write_final_cubes_to_disk(
                cumsum_wavelet_noise_added,
                f"seismicCubes_cumsum_Wavelet_{wavelet_count}_",
            )

    def postprocess_rfc_cubes(
        self, rfc_data, filename_suffix="", bb=False, stack=False
    ):
        """Postprocess the RFC cubes."""
        if bb:
            bandlimited = self.apply_bandlimits(rfc_data, (4.0, 90.0))
        else:
            bandlimited = self.apply_bandlimits(rfc_data)
        if self.cfg.lateral_filter_size > 1:
            bandlimited = self.apply_lateral_filter(bandlimited)
        if filename_suffix == "":
            fname = "seismicCubes_RFC_"
        else:
            fname = f"seismicCubes_RFC_{filename_suffix}_"
        _ = self.write_final_cubes_to_disk(bandlimited, fname)
        if stack:
            rfc_fullstack = self.stack_substacks(
                [bandlimited[x, ...] for x, _ in enumerate(self.angles)]
            )
            rfc_fullstack_scaled = self._scale_seismic(rfc_fullstack)
            self.write_cube_to_disk(
                rfc_fullstack_scaled,
                f"{fname}fullstack",
            )

        cumsum = self.apply_cumsum(bandlimited)
        if filename_suffix == "":
            fname = "seismicCubes_cumsum_"
        else:
            fname = f"seismicCubes_cumsum_{filename_suffix}_"
        normalised_cumsum = self.write_final_cubes_to_disk(cumsum, fname)
        if stack:
            rai_fullstack = self.stack_substacks(
                [cumsum[x, ...] for x, _ in enumerate(self.angles)]
            )
            rai_fullstack_scaled = self._scale_seismic(rai_fullstack)
            self.write_cube_to_disk(
                rai_fullstack_scaled,
                f"{fname}fullstack",
            )
        return normalised_cumsum

    def apply_augmentations(self, data, name="cumsum"):
        seabed = self.faults.faulted_depth_maps[..., 0] / self.cfg.digi
        augmented_data, augmented_labels = self.augment_data_and_labels(data, seabed)
        for i, ang in enumerate(self.cfg.incident_angles):
            data = augmented_data[i, ...]
            fname = f"seismicCubes_{name}_{ang}_degrees_normalized_augmented"
            self.write_cube_to_disk(data, fname)
            self.write_cube_to_disk(augmented_labels, "hc_closures_augmented")

    def apply_rmo(self, data, name="cumsum_RMO"):
        """Apply RMO to the cumsum with noise. Note rmo application expects angle in last dimension"""
        rmo_indices = self.compute_randomRMO_1D(
            data.shape[-1], self.cfg.incident_angles, velocity_fraction=0.015
        )
        rmo_applied_data = np.zeros_like(np.moveaxis(data, 0, -1))
        for iline in range(self.cfg.cube_shape[0]):
            for xline in range(self.cfg.cube_shape[1]):
                rmo_applied_data[iline, xline, ...] = self.apply_rmo_augmentation(
                    np.moveaxis(data, 0, -1)[iline, xline, ...],
                    rmo_indices,
                    remove_mean_rmo=True,
                )
        rmo_applied_data = np.moveaxis(
            rmo_applied_data, -1, 0
        )  # Put the angles back into 1st dimension
        normalised_rmo_applied_data = self.write_final_cubes_to_disk(
            rmo_applied_data, f"seismicCubes_{name}_"
        )
        return normalised_rmo_applied_data

    def build_seismic_volumes(self):
        """Build Seismic volumes.

        * Use Zoeppritz to create RFC volumes for each required angle, add 0 & 45 degree stacks for QC
        * Add exponential weighted noise to the angle stacks
        * Apply bandpass filter
        * Apply lateral filter
        * Calculate cumsum
        * Create full stacks by stacking the user-input angles (i.e. not including 0, 45 degree stacks)
        * Write seismic volumes to disk
        """
        if self.cfg.verbose:
            print(
                f"Generating 3D Rock Properties Rho, Vp and Vs using depth trends for project {self.cfg.project}"
            )
        # Calculate Raw RFC from rho, vp, vs using Zoeppritz & write to HdF
        self.create_rfc_volumes()
        self.add_weighted_noise(self.faults.faulted_depth_maps)
        if hasattr(self.cfg, "wavelets"):
            self.bandlimit_volumes_wavelets(n_wavelets=1)
        normalised_cumsum = self.postprocess_rfc_cubes(
            self.rfc_noise_added[:], "", stack=True
        )
        self.apply_augmentations(normalised_cumsum, name="cumsum")
        if self.cfg.model_qc_volumes:
            _ = self.postprocess_rfc_cubes(self.rfc_raw[:], "noise_free", bb=False)
            if self.cfg.broadband_qc_volume:
                _ = self.postprocess_rfc_cubes(self.rfc_raw[:], "noise_free_bb", bb=True)
            normalised_cumsum_rmo = self.apply_rmo(normalised_cumsum)
            self.apply_augmentations(normalised_cumsum_rmo, name="cumsum_RMO")

    def _scale_seismic(self, data):
        """Apply scaling factor to final seismic data volumes"""
        if data.ndim == 4:
            avg_stdev = np.mean([data[x, ...].std() for x in range(data.shape[0])])
        elif data.ndim == 3:
            avg_stdev = data.std()
        else:
            print(
                f"Input Data has {data.ndim} dimensions, should have 3 or 4 dimensions"
            )
        scaled_data = data * 100 / avg_stdev
        return scaled_data

    def write_final_cubes_to_disk(self, dat, name):
        scaled_data = self._scale_seismic(dat)

        # Apply scaling factors to N, M, F cubes from csv file before determining clip limits
        factor_near = self.cfg.rpm_scaling_factors["nearfactor"]
        factor_mid = self.cfg.rpm_scaling_factors["midfactor"]
        factor_far = self.cfg.rpm_scaling_factors["farfactor"]
        cube_incr = 0
        if self.cfg.model_qc_volumes and dat.shape[0] > 3:
            # 0 degrees has been added at start, and 45 at end
            cube_incr = 1
        scaled_data[0 + cube_incr, ...] *= factor_near
        scaled_data[1 + cube_incr, ...] *= factor_mid
        scaled_data[2 + cube_incr, ...] *= factor_far

        for i, ang in enumerate(self.cfg.incident_angles):
            data = scaled_data[i, ...]
            fname = f"{name}_{ang}_degrees"
            self.write_cube_to_disk(data, fname)
        normed_data = normalize_seismic(scaled_data)
        for i, ang in enumerate(self.cfg.incident_angles):
            data = normed_data[i, ...]
            fname = f"{name}_{ang}_degrees_normalized"
            self.write_cube_to_disk(data, fname)
        return normed_data

    @staticmethod
    def random_z_rho_vp_vs(dmin=-7, dmax=7):
        delta_z_rho = int(np.random.uniform(dmin, dmax))
        delta_z_vp = int(np.random.uniform(dmin, dmax))
        delta_z_vs = int(np.random.uniform(dmin, dmax))
        return delta_z_rho, delta_z_vp, delta_z_vs

    def create_rfc_volumes(self):
        """
        Build a 3D array of PP-Reflectivity using Zoeppritz for each incident angle given.

        :return: 4D array of PP-Reflectivity. 4th dimension holds reflectivities of each incident angle provided
        """
        theta = np.asanyarray(self.angles).reshape(
            (-1, 1)
        )  # Convert angles into a column vector

        # Make empty array to store RPP via zoeppritz
        zoep = np.zeros(
            (self.rho.shape[0], self.rho.shape[1], self.rho.shape[2] - 1, theta.size),
            dtype="complex64",
        )

        _rho = self.rho[:]
        _vp = self.vp[:]
        _vs = self.vs[:]
        if self.cfg.verbose:
            print("Checking for null values inside create_rfc_volumes:\n")
            print(f"Rho: {np.min(_rho)}, {np.max(_rho)}")
            print(f"Vp: {np.min(_vp)}, {np.max(_vp)}")
            print(f"Vs: {np.min(_vs)}, {np.max(_vs)}")
        # Doing this for each trace & all (5) angles is actually quicker than doing entire cubes for each angle
        for i in trange(
            self.rho.shape[0],
            desc=f"Calculating Zoeppritz for {len(self.angles)} angles",
        ):
            for j in range(self.rho.shape[1]):
                rho1, rho2 = _rho[i, j, :-1], _rho[i, j, 1:]
                vp1, vp2 = _vp[i, j, :-1], _vp[i, j, 1:]
                vs1, vs2 = _vs[i, j, :-1], _vs[i, j, 1:]
                rfc = RFC(vp1, vs1, rho1, vp2, vs2, rho2, theta)
                zoep[i, j, :, :] = rfc.zoeppritz_reflectivity().T
        # Set any voxels with imaginary parts to 0, since all transmitted energy travels along the reflector
        # zoep[np.where(np.imag(zoep) != 0)] = 0
        # discard complex values and set dtype to float64
        zoep = np.real(zoep).astype("float64")
        del _rho, _vp, _vs

        # Move the angle from last dimension to first
        self.rfc_raw[:] = np.moveaxis(zoep, -1, 0)

        if self.cfg.hdf_store:
            for n, d in zip(
                [
                    "qc_volume_rfc_raw_{}_degrees".format(
                        str(self.angles[x]).replace(".", "_")
                    )
                    for x in range(self.rfc_raw.shape[0])
                ],
                [self.rfc_raw[x, ...] for x in range(self.rfc_raw.shape[0])],
            ):
                write_data_to_hdf(n, d, self.cfg.hdf_master)
        if self.cfg.model_qc_volumes:
            # Write raw RFC values to disk
            _ = [
                self.write_cube_to_disk(
                    self.rfc_raw[x, ...],
                    f"qc_volume_rfc_raw_{self.angles[x]}_degrees",
                )
                for x in range(self.rfc_raw.shape[0])
            ]
            max_amp = np.max(np.abs(self.rfc_raw[:]))
            _ = [
                self.write_cube_to_disk(
                    self.rfc_raw[x, ...],
                    f"qc_volume_rfc_raw_{self.angles[x]}_degrees",
                )
                for x in range(self.rfc_raw.shape[0])
            ]

    def noise_3d(self, cube_shape, verbose=False):
        if verbose:
            print("   ... inside noise3D")

        noise_seed = np.random.randint(1, high=(2 ** (32 - 1)))
        sign_seed = np.random.randint(1, high=(2 ** (32 - 1)))

        np.random.seed(noise_seed)
        noise3d = np.random.exponential(
            1.0 / 100.0, size=cube_shape[0] * cube_shape[1] * cube_shape[2]
        )
        np.random.seed(sign_seed)
        sign = np.random.binomial(
            1, 0.5, size=cube_shape[0] * cube_shape[1] * cube_shape[2]
        )

        sign[sign == 0] = -1
        noise3d *= sign
        noise3d = noise3d.reshape((cube_shape[0], cube_shape[1], cube_shape[2]))
        return noise3d

    def add_weighted_noise(self, depth_maps):
        """
        Apply 3D weighted random noise to ndarray.

        :return: Sum of rfc_volumes and 3D noise model, weighted by angle
        """

        from math import sin, cos
        from datagenerator.util import mute_above_seafloor
        from datagenerator.util import infill_surface

        # Calculate standard deviation ratio to use based on user-input sn_db parameter
        if self.cfg.verbose:
            print(f"\n...adding random noise to {self.rfc_raw.shape[0]} cubes...")
            print(f"S/N ratio = {self.cfg.sn_db:.4f}")
        std_ratio = np.sqrt(10 ** (self.cfg.sn_db / 10.0))

        # Compute standard deviation in bottom half of cube, and use this to scale the noise
        # if wb has holes (i.e. Nan or 0 values), fill holes using replacement with interpolated (NN) values
        wb_map = depth_maps[..., 0]
        # Add test for all 0's in wb_map in case z_shift has been applied
        if (
            np.isnan(np.min(wb_map)) or 0 in wb_map and not np.all(wb_map == 0)
        ):  # If wb contains nan's or at least one 0, this will return True
            wb_map = infill_surface(wb_map)
        wb_plus_15samples = wb_map / (self.cfg.digi + 15) * self.cfg.digi

        # Use the Middle angle cube for normalisation
        if self.cfg.model_qc_volumes:
            # 0 and 45 degree cubes have been added
            norm_cube = self.rfc_raw[2, ...]
        else:
            norm_cube = self.rfc_raw[1, ...]
        data_below_seafloor = norm_cube[
            mute_above_seafloor(wb_plus_15samples, np.ones(norm_cube.shape, "float"))
            != 0.0
        ]
        data_std = data_below_seafloor.std()

        # Create array to store the 3D noise model for each angle
        noise3d_cubes = np.zeros_like(self.rfc_raw[:])

        # Add weighted stack of noise using Hilterman equation to each angle substack
        noise_0deg = self.noise_3d(self.rfc_raw.shape[1:], verbose=False)
        noise_45deg = self.noise_3d(self.rfc_raw.shape[1:], verbose=False)
        # Store the noise models
        self.noise_0deg = self.cfg.hdf_init("noise_0deg", shape=noise_0deg.shape)
        self.noise_45deg = self.cfg.hdf_init("noise_45deg", shape=noise_45deg.shape)
        self.noise_0deg[:] = noise_0deg
        self.noise_45deg[:] = noise_45deg

        for x, ang in enumerate(self.angles):
            weighted_noise = noise_0deg * (cos(ang) ** 2) + noise_45deg * (
                sin(ang) ** 2
            )
            noise_std = weighted_noise.std()
            noise3d_cubes[x, ...] = weighted_noise * (data_std / noise_std) / std_ratio
            if self.cfg.verbose:
                print(
                    f"\t...Normalised noise3d for angle {ang}:\tMin: {noise3d_cubes[x, ...].min():.4f},"
                    f" mean: {noise3d_cubes[x, ...].mean():.4f}, max: {noise3d_cubes[x, ...].max():.4f},"
                    f" std: {noise3d_cubes[x, ...].std():.4f}"
                )
                print(
                    f"\tS/N ratio = {self.cfg.sn_db:3.1f} dB.\n\tstd_ratio = {std_ratio:.4f}"
                    f"\n\tdata_std = {data_std:.4f}\n\tnoise_std = {noise_std:.4f}"
                )

        self.rfc_noise_added[:] = noise3d_cubes + self.rfc_raw[:]

    def apply_bandlimits(self, data, frequencies=None):
        """
        Apply Butterworth Bandpass Filter to data
        :param data: 4D array of RFC values
        :param frequencies: Explicit frequency bounds (lower, upper)
        :return: 4D array of band-limited RFC
        """
        if self.cfg.verbose:
            print(f"Data Min: {np.min(data):.2f}, Data Max: {np.max(data):.2f}")
        dt = self.cfg.digi / 1000.0
        if (
            data.shape[-1] / self.cfg.infill_factor - self.cfg.pad_samples
            == self.cfg.cube_shape[-1]
        ):
            # Input data is infilled
            dt /= self.cfg.infill_factor
        if frequencies:  # if frequencies explicitly defined
            low = frequencies[0]
            high = frequencies[1]
        else:
            low = self.cfg.lowfreq
            high = self.cfg.highfreq
        b, a = derive_butterworth_bandpass(
            low, high, (dt * 1000.0), order=self.cfg.order
        )
        if self.cfg.verbose:
            print(f"\t... Low Frequency; {low:.2f} Hz, High Frequency: {high:.2f} Hz")
            print(
                f"\t... start_width: {1. / (dt * low):.4f}, end_width: {1. / (dt * high):.4f}"
            )

        # Loop over 3D angle stack arrays
        if self.cfg.verbose:
            print(" ... shape of data being bandlimited = ", data.shape, "\n")
        num = data.shape[0]
        if self.cfg.verbose:
            print(f"Applying Bandpass to {num} cubes")
        if self.cfg.multiprocess_bp:
            # Much faster to use multiprocessing and apply band-limitation to entire cube at once
            with Pool(processes=min(num, os.cpu_count() - 2)) as p:
                iterator = zip(
                    [data[x, ...] for x in range(num)],
                    itertools.repeat(b),
                    itertools.repeat(a),
                )
                out_cubes_mp = p.starmap(self._run_bandpass_on_cubes, iterator)
            out_cubes = np.asarray(out_cubes_mp)
        else:
            # multiprocessing can fail using Python version 3.6 and very large arrays
            out_cubes = np.zeros_like(data)
            for idx in range(num):
                out_cubes[idx, ...] = self._run_bandpass_on_cubes(data[idx, ...], b, a)

        return out_cubes

    @staticmethod
    def _run_bandpass_on_cubes(data, b, a):
        out_cube = apply_butterworth_bandpass(data, b, a)
        return out_cube

    def apply_lateral_filter(self, data):
        from scipy.ndimage import uniform_filter

        n_filt = self.cfg.lateral_filter_size
        return uniform_filter(data, size=(0, n_filt, n_filt, 0))

    def apply_cumsum(self, data):
        """Calculate cumulative sum on the Z-axis of input data
        Sipmap's cumsum also applies an Ormsby filter to the cumulatively summed data
        To replicate this, apply a butterworth filter to the data using 2, 100Hz frequency limits
        todo: make an ormsby function"""
        cumsum = data.cumsum(axis=-1)
        return self.apply_bandlimits(cumsum, (2.0, 100.0))

    @staticmethod
    def stack_substacks(cube_list):
        """Stack cubes by taking mean"""
        return np.mean(cube_list, axis=0)

    def augment_data_and_labels(self, normalised_seismic, seabed):
        from datagenerator.Augmentation import tz_stretch, uniform_stretch

        hc_labels = self.cfg.h5file.root.ModelData.hc_labels[:]
        data, labels = tz_stretch(
            normalised_seismic,
            hc_labels[..., : self.cfg.cube_shape[2] + self.cfg.pad_samples - 1],
            seabed,
        )
        seismic, hc = uniform_stretch(data, labels)
        return seismic, hc

    def compute_randomRMO_1D(
        self, nsamples, inc_angle_list, velocity_fraction=0.015, return_plot=False
    ):
        """
        compute 1D indices for to apply residual moveout via interpolation
        - nsamples is the array length in depth
        - inc_angle_list is a list of angle in degrees for angle stacks to which
        results will be applied
        - velocity_fraction is the maximum percent error in absolute value.
        for example, .03 would generate RMO based on velocity from .97 to 1.03 times correct velocity
        """
        from scipy.interpolate import UnivariateSpline

        input_indices = np.arange(nsamples)
        if self.cfg.qc_plots:
            from datagenerator.util import import_matplotlib

            plt = import_matplotlib()
            plt.close(1)
            plt.figure(1, figsize=(4.75, 6.0), dpi=150)
            plt.clf()
            plt.grid()
            plt.xlim((-10, 10))
            plt.ylim((nsamples, 0))
            plt.ylabel("depth (samples)")
            plt.xlabel("RMO (samples)")
            plt.savefig(
                os.path.join(self.cfg.work_subfolder, "rmo_1d.png"), format="png"
            )
        # generate 1 to 3 randomly located velocity fractions at random depth indices
        for _ in range(250):
            num_tie_points = np.random.randint(low=1, high=4)
            tie_fractions = np.random.uniform(
                low=-velocity_fraction, high=velocity_fraction, size=num_tie_points
            )
            if num_tie_points > 1:
                tie_indices = np.random.uniform(
                    low=0.25 * nsamples, high=0.75 * nsamples, size=num_tie_points
                ).astype("int")

            else:
                tie_indices = int(
                    np.random.uniform(low=0.25 * nsamples, high=0.75 * nsamples)
                )
            tie_fractions = np.hstack(([0.0, 0.0], tie_fractions, [0.0, 0.0]))
            tie_indices = np.hstack(([0, 1], tie_indices, [nsamples - 2, nsamples - 1]))
            tie_indices = np.sort(tie_indices)
            if np.diff(tie_indices).min() <= 0.0:
                continue
            s = UnivariateSpline(tie_indices, tie_fractions, s=0.0)
            tie_fraction_array = s(input_indices)
            output_indices = np.zeros((nsamples, len(inc_angle_list)))
            for i, iangle in enumerate(inc_angle_list):
                output_indices[:, i] = input_indices * (
                    1.0
                    + tie_fraction_array * np.tan(float(iangle) * np.pi / 180.0) ** 2
                )
            if np.abs(output_indices[:, i] - input_indices).max() < 10.0:
                break
        for i, iangle in enumerate(inc_angle_list):
            if self.cfg.qc_plots:
                plt.plot(
                    output_indices[:, i] - input_indices,
                    input_indices,
                    label=str(inc_angle_list[i]),
                )

                if i == len(inc_angle_list) - 1:
                    output_tie_indices = tie_indices * (
                        1.0
                        + tie_fraction_array[tie_indices]
                        * np.tan(float(iangle) * np.pi / 180.0) ** 2
                    )
                    plt.plot(output_tie_indices - tie_indices, tie_indices, "ko")
                    rmo_min = (output_indices[:, -1] - input_indices).min()
                    rmo_max = (output_indices[:, -1] - input_indices).max()
                    rmo_range = np.array((rmo_min, rmo_max)).round(1)
                    plt.title(
                        "simultated RMO arrival time offsets\n"
                        + "rmo range = "
                        + str(rmo_range),
                        fontsize=11,
                    )
                    plt.legend()
                    plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg.work_subfolder, "rmo_arrival_time.png"),
                    format="png",
                )

        if return_plot:
            return output_indices, plt
        else:
            return output_indices

    def apply_rmo_augmentation(self, angle_gather, rmo_indices, remove_mean_rmo=False):
        """
        apply residual moveout via interpolation
        applied to 'angle_gather' with results from compute_randomRMO_1D in rmo_indices
        angle_gather and rmo_indices need to be 2D arrays with the same shape (n_samples x n_angles)
        - angle_gather is the array to which RMO is applied
        - rmo_indices is the array used to modify the two-way-times of angle_gather
        - remove_mean_rmo can be used to (when True) to apply relative RMO that keeps
        event energy at roughly the same TWT. for example to avoid applying RMO
        to both data and labels
        """
        if remove_mean_rmo:
            # apply de-trending to ensure that rmo does not change TWT
            # after rmo compared to before rmo 'augmentation'
            residual_times = rmo_indices.mean(axis=-1) - np.arange(
                angle_gather.shape[0]
            )
            residual_rmo_indices = rmo_indices * 1.0
            residual_rmo_indices[:, 0] -= residual_times
            residual_rmo_indices[:, 1] -= residual_times
            residual_rmo_indices[:, 2] -= residual_times
            _rmo_indices = residual_rmo_indices
        else:
            _rmo_indices = rmo_indices

        rmo_angle_gather = np.zeros_like(angle_gather)
        for iangle in range(len(rmo_indices[-1])):
            rmo_angle_gather[:, iangle] = np.interp(
                range(angle_gather.shape[0]),
                _rmo_indices[:, iangle],
                angle_gather[:, iangle],
            )

        return rmo_angle_gather

    def build_property_models_randomised_depth(self, rpm, mixing_method="inv_vel"):
        """
        Build property models with randomised depth
        -------------------------------------------
        Builds property models with randomised depth.

        v2 but with Backus moduli mixing instead of inverse velocity mixing
        mixing_method one of ["InverseVelocity", "BackusModuli"]

        Parameters
        ----------
        rpm : str
            Rock properties model.
        mixing_method : str
            Mixing method for randomised depth.
        
        Returns
        -------
        None
        """
        layer_half_range = self.cfg.rpm_scaling_factors["layershiftsamples"]
        property_half_range = self.cfg.rpm_scaling_factors["RPshiftsamples"]

        depth = self.cfg.h5file.root.ModelData.faulted_depth[:]
        lith = self.faults.faulted_lithology[:]
        net_to_gross = self.faults.faulted_net_to_gross[:]

        oil_closures = self.cfg.h5file.root.ModelData.oil_closures[:]
        gas_closures = self.cfg.h5file.root.ModelData.gas_closures[:]

        integer_faulted_age = (
            self.cfg.h5file.root.ModelData.faulted_age_volume[:] + 0.00001
        ).astype(int)

        # Use empty class object to store all Rho, Vp, Vs volumes (randomised, fluid factor and non randomised)
        mixed_properties = ElasticProperties3D()
        mixed_properties.rho = np.zeros_like(lith)
        mixed_properties.vp = np.zeros_like(lith)
        mixed_properties.vs = np.zeros_like(lith)
        mixed_properties.sum_net = np.zeros_like(lith)
        cube_shape = lith.shape

        if self.cfg.verbose:
            mixed_properties.rho_not_random = np.zeros_like(lith)
            mixed_properties.vp_not_random = np.zeros_like(lith)
            mixed_properties.vs_not_random = np.zeros_like(lith)

        # water layer
        mixed_properties = self.water_properties(lith, mixed_properties)

        mixed_properties.rho_ff = np.copy(mixed_properties.rho)
        mixed_properties.vp_ff = np.copy(mixed_properties.vp)
        mixed_properties.vs_ff = np.copy(mixed_properties.vs)

        delta_z_lyrs = []
        delta_z_shales = []
        delta_z_brine_sands = []
        delta_z_oil_sands = []
        delta_z_gas_sands = []

        for z in range(1, integer_faulted_age.max()):
            __i, __j, __k = np.where(integer_faulted_age == z)

            if len(__k) > 0:
                if self.cfg.verbose:
                    print(
                        f"\n\n*********************\n ... layer number = {z}"
                        f"\n ... n2g (min,mean,max) = "
                        f"({np.min(net_to_gross[__i, __j, __k]).round(3)},"
                        f"{np.mean(net_to_gross[__i, __j, __k]).round(3)},"
                        f"{np.max(net_to_gross[__i, __j, __k]).round(3)},)"
                    )

                delta_z_layer = self.get_delta_z_layer(z, layer_half_range, __k)
                delta_z_lyrs.append(delta_z_layer)

            # shale required for all voxels
            i, j, k = np.where((net_to_gross >= 0.0) & (integer_faulted_age == z))

            shale_voxel_count = len(k)
            delta_z_rho, delta_z_vp, delta_z_vs = self.get_delta_z_properties(
                z, property_half_range
            )
            delta_z_shales.append((delta_z_rho, delta_z_vp, delta_z_vs))

            if len(k) > 0:
                _k_rho = list(
                    np.array(k + delta_z_layer + delta_z_rho).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vp = list(
                    np.array(k + delta_z_layer + delta_z_vp).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vs = list(
                    np.array(k + delta_z_layer + delta_z_vs).clip(
                        0, cube_shape[-1] - 10
                    )
                )

                if self.cfg.verbose:
                    print(
                        f" ... shale: (delta_z_rho, delta_z_vp, delta_z_vs) = {delta_z_rho, delta_z_vp, delta_z_vs}"
                    )
                    print(f" ... shale: i = {np.mean(i)}")
                    print(f" ... shale: k = {np.mean(k)}")
                    print(f" ... shale: _k = {np.mean(_k_rho)}")

                mixed_properties = self.calculate_shales(
                    xyz=(i, j, k),
                    shifts=(_k_rho, _k_vp, _k_vs),
                    props=mixed_properties,
                    rpm=rpm,
                    depth=depth,
                    z=z,
                )

            # brine sand or mixture of brine sand and shale in same voxel
            i, j, k = np.where(
                (net_to_gross > 0.0)
                & (oil_closures == 0.0)
                & (gas_closures == 0.0)
                & (integer_faulted_age == z)
            )
            brine_voxel_count = len(k)
            delta_z_rho, delta_z_vp, delta_z_vs = self.get_delta_z_properties(
                z, property_half_range
            )
            delta_z_brine_sands.append((delta_z_rho, delta_z_vp, delta_z_vs))

            if len(k) > 0:
                _k_rho = list(
                    np.array(k + delta_z_layer + delta_z_rho).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vp = list(
                    np.array(k + delta_z_layer + delta_z_vp).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vs = list(
                    np.array(k + delta_z_layer + delta_z_vs).clip(
                        0, cube_shape[-1] - 10
                    )
                )

                if self.cfg.verbose:
                    print(
                        f"\n ... brine: (delta_z_rho, delta_z_vp, delta_z_vs) = {delta_z_rho, delta_z_vp, delta_z_vs}"
                    )
                    print(f" ... brine: i = {np.mean(i)}")
                    print(f" ... brine: k = {np.mean(k)}")
                    print(f" ... brine: _k = {np.mean(_k_rho)}")

                mixed_properties = self.calculate_sands(
                    xyz=(i, j, k),
                    shifts=(_k_rho, _k_vp, _k_vs),
                    props=mixed_properties,
                    rpm=rpm,
                    depth=depth,
                    ng=net_to_gross,
                    z=z,
                    fluid="brine",
                    mix=mixing_method,
                )

            # oil sands
            i, j, k = np.where(
                (net_to_gross > 0.0)
                & (oil_closures == 1.0)
                & (gas_closures == 0.0)
                & (integer_faulted_age == z)
            )
            oil_voxel_count = len(k)
            delta_z_rho, delta_z_vp, delta_z_vs = self.get_delta_z_properties(
                z, property_half_range
            )
            delta_z_oil_sands.append((delta_z_rho, delta_z_vp, delta_z_vs))

            if len(k) > 0:
                _k_rho = list(
                    np.array(k + delta_z_layer + delta_z_rho).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vp = list(
                    np.array(k + delta_z_layer + delta_z_vp).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vs = list(
                    np.array(k + delta_z_layer + delta_z_vs).clip(
                        0, cube_shape[-1] - 10
                    )
                )

                if self.cfg.verbose:
                    print(
                        "\n\n ... Perform fluid substitution for oil sands. ",
                        "\n ... Number oil voxels in closure = ",
                        mixed_properties.rho[i, j, k].shape,
                    )
                    print(
                        " ... closures_oil min/mean/max =  ",
                        oil_closures.min(),
                        oil_closures.mean(),
                        oil_closures.max(),
                    )
                    print(
                        "\n\n ... np.all(oil_sand_rho[:,:,:] == brine_sand_rho[:,:,:]) ",
                        np.all(
                            rpm.calc_oil_sand_properties(
                                depth[:], depth[:], depth[:]
                            ).rho
                            == rpm.calc_brine_sand_properties(
                                depth[:], depth[:], depth[:]
                            ).rho
                        ),
                    )
                    print(
                        " ... oil: (delta_z_rho, delta_z_vp, delta_z_vs) = "
                        + str((delta_z_rho, delta_z_vp, delta_z_vs))
                    )

                if self.cfg.verbose:
                    print(" ... oil: i = " + str(np.mean(i)))
                    print(" ... oil: k = " + str(np.mean(k)))
                    print(" ... oil: _k = " + str(np.mean(_k_rho)))

                mixed_properties = self.calculate_sands(
                    xyz=(i, j, k),
                    shifts=(_k_rho, _k_vp, _k_vs),
                    props=mixed_properties,
                    rpm=rpm,
                    depth=depth,
                    ng=net_to_gross,
                    z=z,
                    fluid="oil",
                    mix=mixing_method,
                )

            # gas sands
            i, j, k = np.where(
                (net_to_gross > 0.0)
                & (oil_closures == 0.0)
                & (gas_closures == 1.0)
                & (integer_faulted_age == z)
            )
            gas_voxel_count = len(k)
            delta_z_rho, delta_z_vp, delta_z_vs = self.get_delta_z_properties(
                z, property_half_range
            )
            delta_z_gas_sands.append((delta_z_rho, delta_z_vp, delta_z_vs))

            if len(k) > 0:
                _k_rho = list(
                    np.array(k + delta_z_layer + delta_z_rho).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vp = list(
                    np.array(k + delta_z_layer + delta_z_vp).clip(
                        0, cube_shape[-1] - 10
                    )
                )
                _k_vs = list(
                    np.array(k + delta_z_layer + delta_z_vs).clip(
                        0, cube_shape[-1] - 10
                    )
                )

                if self.cfg.verbose:
                    print(
                        "\n ... Perform fluid substitution for gas sands. ",
                        "\n ... Number gas voxels in closure = ",
                        mixed_properties.rho[i, j, k].shape,
                    )
                    print(
                        " ... closures_gas min/mean/max =  ",
                        gas_closures.min(),
                        gas_closures.mean(),
                        gas_closures.max(),
                    )
                    print(
                        " ... gas: (delta_z_rho, delta_z_vp, delta_z_vs) = "
                        + str((delta_z_rho, delta_z_vp, delta_z_vs))
                    )

                if self.cfg.verbose:
                    print(" ... gas: i = " + str(np.mean(i)))
                    print(" ... gas: k = " + str(np.mean(k)))
                    print(" ... gas: _k = " + str(np.mean(_k_rho)))

                mixed_properties = self.calculate_sands(
                    xyz=(i, j, k),
                    shifts=(_k_rho, _k_vp, _k_vs),
                    props=mixed_properties,
                    rpm=rpm,
                    depth=depth,
                    ng=net_to_gross,
                    z=z,
                    fluid="gas",
                    mix=mixing_method,
                )

            # layer check
            __i, __j, __k = np.where(integer_faulted_age == z)

            if len(__k) > 0:
                if self.cfg.verbose:
                    print(
                        "\n ... layer number = "
                        + str(z)
                        + "\n ... sum_net (min,mean,max) = "
                        + str(
                            (
                                mixed_properties.sum_net[__i, __j, __k].min(),
                                mixed_properties.sum_net[__i, __j, __k].mean(),
                                mixed_properties.sum_net[__i, __j, __k].max(),
                            )
                        )
                    )
                    print(
                        "\n ... layer number = "
                        + str(z)
                        + "\n ... (shale, brine, oil, gas) voxel_counts = "
                        + str(
                            (
                                shale_voxel_count,
                                brine_voxel_count,
                                oil_voxel_count,
                                gas_voxel_count,
                            )
                        )
                        + "\n ... shale_count = brine+oil+gas? "
                        + str(
                            shale_voxel_count
                            == brine_voxel_count + oil_voxel_count + gas_voxel_count
                        )
                        + "\n*********************"
                    )

        # overwrite rho, vp, vs for salt if required
        if self.cfg.include_salt:
            # Salt. Set lith = 2.0
            i, j, k = np.where(lith == 2.0)
            mixed_properties.rho[i, j, k] = 2.17  # divide by 2 since lith = 2.0
            mixed_properties.vp[i, j, k] = 4500.0
            mixed_properties.vs[i, j, k] = 2250.0
            # Include fluidfactor rho, vp, vs inside salt body
            mixed_properties.rho_ff[i, j, k] = mixed_properties.rho[i, j, k]
            mixed_properties.vp_ff[i, j, k] = mixed_properties.vp[i, j, k]
            mixed_properties.vs_ff[i, j, k] = mixed_properties.vs[i, j, k]

        if self.cfg.verbose:
            print("Checking for null values inside build_randomised_properties:\n")
            print(
                f"Rho: {np.min(mixed_properties.rho)}, {np.max(mixed_properties.rho)}"
            )
            print(f"Vp: {np.min(mixed_properties.vp)}, {np.max(mixed_properties.vp)}")
            print(f"Vs: {np.min(mixed_properties.vs)}, {np.max(mixed_properties.vs)}")

        # Fix empty voxels at base of property volumes
        mixed_properties = self.fix_zero_values_at_base(mixed_properties)

        if self.cfg.verbose:
            print(
                "Checking for null values inside build_randomised_properties AFTER fix:\n"
            )
            print(
                f"Rho: {np.min(mixed_properties.rho)}, {np.max(mixed_properties.rho)}"
            )
            print(f"Vp: {np.min(mixed_properties.vp)}, {np.max(mixed_properties.vp)}")
            print(f"Vs: {np.min(mixed_properties.vs)}, {np.max(mixed_properties.vs)}")

        mixed_properties = self.apply_scaling_factors(
            mixed_properties, net_to_gross, lith
        )
        self.rho[:] = mixed_properties.rho
        self.vp[:] = mixed_properties.vp
        self.vs[:] = mixed_properties.vs

        self.rho_ff[:] = mixed_properties.rho_ff
        self.vp_ff[:] = mixed_properties.vp_ff
        self.vs_ff[:] = mixed_properties.vs_ff

        if self.cfg.qc_plots:
            print("\nCreating RPM qc plots")
            from rockphysics.RockPropertyModels import rpm_qc_plots

            rpm_qc_plots(self.cfg, rpm)

        if self.cfg.model_qc_volumes:
            self.write_property_volumes_to_disk()

    def water_properties(self, lith, properties):
        i, j, k = np.where(lith < 0.0)
        properties.rho[i, j, k] = 1.028
        properties.vp[i, j, k] = 1500.0
        properties.vs[i, j, k] = 1000.0
        return properties

    def get_delta_z_layer(self, z, half_range, z_cells):
        if z > self.first_random_lyr:
            delta_z_layer = int(np.random.uniform(-half_range, half_range))
        else:
            delta_z_layer = 0
        if self.cfg.verbose:
            print(f" .... Layer {z}: voxel_count = {len(z_cells)}")
            print(f" .... Layer {z}: delta_z_layer = {delta_z_layer}")
            print(
                f" .... Layer {z}: z-range (m): {np.min(z_cells) * self.cfg.digi}, "
                f"{np.max(z_cells) * self.cfg.digi}"
            )
        return delta_z_layer

    def get_delta_z_properties(self, z, half_range):
        if z > self.first_random_lyr:
            delta_z_rho, delta_z_vp, delta_z_vs = self.random_z_rho_vp_vs(
                dmin=-half_range, dmax=half_range
            )
        else:
            delta_z_rho, delta_z_vp, delta_z_vs = (0, 0, 0)
        return delta_z_rho, delta_z_vp, delta_z_vs

    def calculate_shales(self, xyz, shifts, props, rpm, depth, z):
        # Shales required for all voxels, other than water
        # Calculate the properties, select how to mix with other facies later
        i, j, k = xyz
        k_rho, k_vp, k_vs = shifts

        z_rho = depth[i, j, k_rho]
        z_vp = depth[i, j, k_vp]
        z_vs = depth[i, j, k_vs]
        shales = rpm.calc_shale_properties(z_rho, z_vp, z_vs)

        # Do not use net to gross here.
        # Since every voxel will have some shale in, apply the net to gross
        # when combining shales and sands
        # _ng = (1.0 - ng[i, j, k])

        props.rho[i, j, k] = shales.rho  # * _ng
        props.rho_ff[i, j, k] = shales.rho  # * _ng
        props.vp[i, j, k] = shales.vp  # * _ng
        props.vp_ff[i, j, k] = shales.vp  # * _ng
        props.vs[i, j, k] = shales.vs  # * _ng
        props.vs_ff[i, j, k] = shales.vs  # * _ng
        # props.sum_net[i, j, k] = _ng

        if self.cfg.verbose and z > self.first_random_lyr:
            # Calculate non-randomised properties and differences
            _z0 = depth[i, j, k]
            _shales0 = rpm.calc_shale_properties(_z0, _z0, _z0)
            props.rho_not_random[i, j, k] = _shales0.rho  # * _ng
            props.vp_not_random[i, j, k] = _shales0.vp  # * _ng
            props.vs_not_random[i, j, k] = _shales0.vs  # * _ng

            delta_rho = 1.0 - (props.rho[i, j, k] / props.rho_not_random[i, j, k])
            delta_vp = 1.0 - (props.vp[i, j, k] / props.vp_not_random[i, j, k])
            delta_vs = 1.0 - (props.vs[i, j, k] / props.vs_not_random[i, j, k])
            pct_change_rho = delta_rho.mean().round(3)
            pct_change_vp = delta_vp.mean().round(3)
            pct_change_vs = delta_vs.mean().round(3)
            print(
                f" ... shale: randomization percent change (rho,vp,vs) = "
                f"{pct_change_rho}, {pct_change_vp }, {pct_change_vs}"
            )
            print(
                f" ... shale: min/max pre-randomization (rho, vp, vs)  = "
                f"{np.min(props.rho_not_random[i, j, k]):.3f} - "
                f"{np.max(props.rho_not_random[i, j, k]):.3f}, "
                f"{np.min(props.vp_not_random[i, j, k]):.2f} - "
                f"{np.max(props.vp_not_random[i, j, k]):.2f}, "
                f"{np.min(props.vs_not_random[i, j, k]):.2f} - "
                f"{np.max(props.vs_not_random[i, j, k]):.2f}"
            )
            print(
                f" ... shale: min/max post-randomization (rho, vp, vs) = "
                f"{np.min(props.rho[i, j, k]):.3f} - "
                f"{np.max(props.rho[i, j, k]):.3f}, "
                f"{np.min(props.vp[i, j, k]):.2f} - "
                f"{np.max(props.vp[i, j, k]):.2f}, "
                f"{np.min(props.vs[i, j, k]):.2f} - "
                f"{np.max(props.vs[i, j, k]):.2f}"
            )

        return props

    def calculate_sands(
        self, xyz, shifts, props, rpm, depth, ng, z, fluid, mix="inv_vel"
    ):
        # brine sand or mixture of brine sand and shale in same voxel
        # - perform velocity sums in slowness or via backus moduli mixing
        i, j, k = xyz
        k_rho, k_vp, k_vs = shifts

        z_rho = depth[i, j, k_rho]
        z_vp = depth[i, j, k_vp]
        z_vs = depth[i, j, k_vs]
        if fluid == "brine":
            sands = rpm.calc_brine_sand_properties(z_rho, z_vp, z_vs)
        elif fluid == "oil":
            sands = rpm.calc_oil_sand_properties(z_rho, z_vp, z_vs)
        elif fluid == "gas":
            sands = rpm.calc_gas_sand_properties(z_rho, z_vp, z_vs)

        shales = RockProperties(
            rho=props.rho[i, j, k], vp=props.vp[i, j, k], vs=props.vs[i, j, k]
        )
        rpmix = EndMemberMixing(shales, sands, ng[i, j, k])

        if mix == "inv_vel":
            rpmix.inverse_velocity_mixing()
        else:
            rpmix.backus_moduli_mixing()

        props.sum_net[i, j, k] += ng[i, j, k]
        props.rho[i, j, k] = rpmix.rho
        props.vp[i, j, k] = rpmix.vp
        props.vs[i, j, k] = rpmix.vs

        if self.cfg.verbose and z > self.first_random_lyr:
            # Calculate non-randomised properties and differences
            _z0 = depth[i, j, k]
            if fluid == "brine":
                _sands0 = rpm.calc_brine_sand_properties(_z0, _z0, _z0)
            elif fluid == "oil":
                _sands0 = rpm.calc_oil_sand_properties(_z0, _z0, _z0)
            elif fluid == "gas":
                _sands0 = rpm.calc_gas_sand_properties(_z0, _z0, _z0)

            rpmix_0 = EndMemberMixing(shales, _sands0, ng[i, j, k])
            if mix == "inv_vel":
                # Apply Inverse Velocity mixing
                rpmix_0.inverse_velocity_mixing()
            else:
                # Apply Backus Moduli mixing
                rpmix_0.backus_moduli_mixing()
            props.rho_not_random[i, j, k] = rpmix_0.rho
            props.vp_not_random[i, j, k] = rpmix_0.vp
            props.vs_not_random[i, j, k] = rpmix_0.vs

            delta_rho = 1.0 - (props.rho[i, j, k] / props.rho_not_random[i, j, k])
            delta_vp = 1.0 - (props.vp[i, j, k] / props.vp_not_random[i, j, k])
            delta_vs = 1.0 - (props.vs[i, j, k] / props.vs_not_random[i, j, k])
            pct_change_rho = delta_rho.mean().round(3)
            pct_change_vp = delta_vp.mean().round(3)
            pct_change_vs = delta_vs.mean().round(3)
            print(
                f" ... {fluid}: randomization percent change (rho,vp,vs) = "
                f"{pct_change_rho}, {pct_change_vp }, {pct_change_vs}"
            )
            print(
                f" ... {fluid}: min/max pre-randomization (rho, vp, vs)  = "
                f"{np.min(props.rho_not_random[i, j, k]):.3f} - "
                f"{np.max(props.rho_not_random[i, j, k]):.3f}, "
                f"{np.min(props.vp_not_random[i, j, k]):.2f} - "
                f"{np.max(props.vp_not_random[i, j, k]):.2f}, "
                f"{np.min(props.vs_not_random[i, j, k]):.2f} - "
                f"{np.max(props.vs_not_random[i, j, k]):.2f}"
            )
            print(
                f" ... {fluid}: min/max post-randomization (rho, vp, vs) = "
                f"{np.min(props.rho[i, j, k]):.3f} - "
                f"{np.max(props.rho[i, j, k]):.3f}, "
                f"{np.min(props.vp[i, j, k]):.2f} - "
                f"{np.max(props.vp[i, j, k]):.2f}, "
                f"{np.min(props.vs[i, j, k]):.2f} - "
                f"{np.max(props.vs[i, j, k]):.2f}"
            )

        return props

    @staticmethod
    def fix_zero_values_at_base(props):
        """Check for zero values at base of property volumes and replace with
        shallower values if present

        Args:
            a ([type]): [description]
            b ([type]): [description]
            c ([type]): [description]
        """
        for vol in [
            props.rho,
            props.vp,
            props.vs,
            props.rho_ff,
            props.vp_ff,
            props.vs_ff,
        ]:
            for (x, y, z) in np.argwhere(vol == 0.0):
                vol[x, y, z] = vol[x, y, z - 1]
        return props

    def apply_scaling_factors(self, props, ng, lith):
        """Apply final random scaling factors to sands and shales"""
        rho_factor_shale = self.cfg.rpm_scaling_factors["shalerho_factor"]
        vp_factor_shale = self.cfg.rpm_scaling_factors["shalevp_factor"]
        vs_factor_shale = self.cfg.rpm_scaling_factors["shalevs_factor"]
        rho_factor_sand = self.cfg.rpm_scaling_factors["sandrho_factor"]
        vp_factor_sand = self.cfg.rpm_scaling_factors["sandvp_factor"]
        vs_factor_sand = self.cfg.rpm_scaling_factors["sandvs_factor"]

        if self.cfg.verbose:
            print(
                f"\n... Additional random scaling factors: "
                f"\n ... Shale Rho {rho_factor_shale:.3f}, Vp {vp_factor_shale:.3f}, Vs {vs_factor_shale:.3f}"
                f"\n ... Sand Rho {rho_factor_sand:.3f}, Vp {vp_factor_sand:.3f}, Vs {vs_factor_sand:.3f}"
            )

        # Apply final scaling factors to shale/sand properties
        props.rho[(ng <= 1.0e-2) & (lith < 2.0)] = (
            props.rho[(ng <= 1.0e-2) & (lith < 2.0)] * rho_factor_shale
        )
        props.vp[(ng <= 1.0e-2) & (lith < 2.0)] = (
            props.vp[(ng <= 1.0e-2) & (lith < 2.0)] * vp_factor_shale
        )
        props.vs[(ng <= 1.0e-2) & (lith < 2.0)] = (
            props.vs[(ng <= 1.0e-2) & (lith < 2.0)] * vs_factor_shale
        )
        props.rho[(ng > 1.0e-2) & (lith < 2.0)] = (
            props.rho[(ng > 1.0e-2) & (lith < 2.0)] * rho_factor_sand
        )
        props.vp[(ng > 1.0e-2) & (lith < 2.0)] = (
            props.vp[(ng > 1.0e-2) & (lith < 2.0)] * vp_factor_sand
        )
        props.vs[(ng > 1.0e-2) & (lith < 2.0)] = (
            props.vs[(ng > 1.0e-2) & (lith < 2.0)] * vs_factor_sand
        )
        # Apply same factors to the fluid-factor property cubes
        props.rho_ff[(ng <= 1.0e-2) & (lith < 2.0)] = (
            props.rho_ff[(ng <= 1.0e-2) & (lith < 2.0)] * rho_factor_shale
        )
        props.vp_ff[(ng <= 1.0e-2) & (lith < 2.0)] = (
            props.vp_ff[(ng <= 1.0e-2) & (lith < 2.0)] * vp_factor_shale
        )
        props.vs_ff[(ng <= 1.0e-2) & (lith < 2.0)] = (
            props.vs_ff[(ng <= 1.0e-2) & (lith < 2.0)] * vs_factor_shale
        )
        props.rho_ff[(ng > 1.0e-2) & (lith < 2.0)] = (
            props.rho_ff[(ng > 1.0e-2) & (lith < 2.0)] * rho_factor_sand
        )
        props.vp_ff[(ng > 1.0e-2) & (lith < 2.0)] = (
            props.vp_ff[(ng > 1.0e-2) & (lith < 2.0)] * vp_factor_sand
        )
        props.vs_ff[(ng > 1.0e-2) & (lith < 2.0)] = (
            props.vs_ff[(ng > 1.0e-2) & (lith < 2.0)] * vs_factor_sand
        )
        return props

    def write_property_volumes_to_disk(self):
        """Write Rho, Vp, Vs volumes to disk."""
        self.write_cube_to_disk(
            self.rho[:],
            "qc_volume_rho",
        )
        self.write_cube_to_disk(
            self.vp[:],
            "qc_volume_vp",
        )
        self.write_cube_to_disk(
            self.vs[:],
            "qc_volume_vs",
        )


class ElasticProperties3D:
    """Empty class to hold 3D property cubes Rho, Vp, Vs, Rho_ff, Vp_ff, Vs_ff and sum_net"""

    def __init__(self):
        self.rho = None
        self.vp = None
        self.vs = None
        self.rho_ff = None
        self.vp_ff = None
        self.vs_ff = None
        self.sum_net = None
        self.rho_not_random = None
        self.vp_not_random = None
        self.vs_not_random = None


class RFC:
    """
    Reflection Coefficient object
    Contains methods for calculating the reflection coefficient at an interface
    from input Vp, Vs, Rho and incident angle
    Angle should be given in degrees and is converted to radians internally
    """

    def __init__(
        self, vp_upper, vs_upper, rho_upper, vp_lower, vs_lower, rho_lower, theta
    ):
        self.theta = theta
        self.vp_upper = vp_upper
        self.vs_upper = vs_upper
        self.rho_upper = rho_upper
        self.vp_lower = vp_lower
        self.vs_lower = vs_lower
        self.rho_lower = rho_lower

    def normal_incidence_rfc(self):
        z0 = self.rho_upper * self.vp_upper
        z1 = self.rho_lower * self.vp_lower
        return (z1 - z0) / (z1 + z0)

    def shuey(self):
        """Use 3-term Shuey to calculate RFC."""
        radians = self._degrees_to_radians()  # Angle to radians
        sin_squared = np.sin(radians) ** 2
        tan_squared = np.tan(radians) ** 2

        # Delta Vp, Vs, Rho
        d_vp = self.vp_lower - self.vp_upper
        d_vs = self.vs_lower - self.vs_upper
        d_rho = self.rho_lower - self.rho_upper
        avg_vp = np.mean([self.vp_lower, self.vp_upper])
        avg_vs = np.mean([self.vs_lower, self.vs_upper])
        avg_rho = np.mean([self.rho_lower, self.rho_upper])

        # 3 Term Shuey
        r0 = 0.5 * (d_vp / avg_vp + d_rho / avg_rho)
        g = 0.5 * (d_vp / avg_vp) - 2.0 * avg_vs ** 2 / avg_vp ** 2 * (
            d_rho / avg_rho + 2.0 * d_vs / avg_vs
        )
        f = 0.5 * (d_vp / avg_vp)

        return r0 + g * sin_squared + f * (tan_squared - sin_squared)

    def zoeppritz_reflectivity(self):
        """Calculate PP reflectivity using Zoeppritz equation"""
        theta = self._degrees_to_radians(
            dtype="complex"
        )  # Convert angle to radians, as complex value
        p = np.sin(theta) / self.vp_upper  # ray param
        theta2 = np.arcsin(p * self.vp_lower)  # Transmission angle of P-wave
        phi1 = np.arcsin(p * self.vs_upper)  # Reflection angle of converted S-wave
        phi2 = np.arcsin(p * self.vs_lower)  # Transmission angle of converted S-wave

        a = self.rho_lower * (1 - 2 * np.sin(phi2) ** 2.0) - self.rho_upper * (
            1 - 2 * np.sin(phi1) ** 2.0
        )
        b = (
            self.rho_lower * (1 - 2 * np.sin(phi2) ** 2.0)
            + 2 * self.rho_upper * np.sin(phi1) ** 2.0
        )
        c = (
            self.rho_upper * (1 - 2 * np.sin(phi1) ** 2.0)
            + 2 * self.rho_lower * np.sin(phi2) ** 2.0
        )
        d = 2 * (
            self.rho_lower * self.vs_lower ** 2 - self.rho_upper * self.vs_upper ** 2
        )

        e = (b * np.cos(theta) / self.vp_upper) + (c * np.cos(theta2) / self.vp_lower)
        f = (b * np.cos(phi1) / self.vs_upper) + (c * np.cos(phi2) / self.vs_lower)
        g = a - d * np.cos(theta) / self.vp_upper * np.cos(phi2) / self.vs_lower
        h = a - d * np.cos(theta2) / self.vp_lower * np.cos(phi1) / self.vs_upper

        d = e * f + g * h * p ** 2

        zoep_pp = (
            f
            * (
                b * (np.cos(theta) / self.vp_upper)
                - c * (np.cos(theta2) / self.vp_lower)
            )
            - h
            * p ** 2
            * (a + d * (np.cos(theta) / self.vp_upper) * (np.cos(phi2) / self.vs_lower))
        ) * (1 / d)

        return np.squeeze(zoep_pp)

    def _degrees_to_radians(self, dtype="float"):
        """Convert angle in degrees to ange in radians. If dtype != 'float', return a complex dtype array"""
        if dtype == "float":
            return np.radians(self.theta)
        else:
            return np.radians(self.theta).astype(complex)


def derive_butterworth_bandpass(lowcut, highcut, digitisation, order=4):
    from scipy.signal import butter

    fs = 1.0 / (digitisation / 1000.0)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass", output="ba")
    return b, a


def apply_butterworth_bandpass(data, b, a):
    """Apply Butterworth bandpass to data"""
    from scipy.signal import tf2zpk, filtfilt

    # Use irlen to remove artefacts generated at base of cubes during bandlimitation
    _, p, _ = tf2zpk(b, a)
    eps = 1e-9
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    y = filtfilt(b, a, data, method="gust", irlen=approx_impulse_len)
    # y = filtfilt(b, a, data)
    return y


def trim_triangular(low, mid, high):
    ###
    ### use trimmed triangular distributions
    ###
    from numpy.random import triangular

    for _ in range(50):
        num = triangular(2 * low - mid, mid, 2 * high - mid)
        if low <= num <= high:
            break
    return num


def apply_wavelet(cube, wavelet):
    filtered_cube = np.zeros_like(cube)
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            filtered_cube[i, j, :] = np.convolve(cube[i, j, :], wavelet, mode="same")
    return filtered_cube
