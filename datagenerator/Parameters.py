from collections import defaultdict
import datetime
import json
import multiprocessing as mp
import os
import pathlib
import glob
import sqlite3
from subprocess import CalledProcessError
import numpy as np
import tables
import subprocess

dir_name = os.path.dirname(__file__)
CONFIG_PATH = os.path.abspath(os.path.join(dir_name, "../config/config_ht.json"))


class _Borg:
    # Any objects using Borg base class will have same shared_state values (Alex Martelli's Borg)
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Parameters(_Borg):
    """
    Parameter object storing all model parameters.

    Attributes
    ----------
    model_dir_name : str
        This is the name that the directory will be given, by default `seismic`
    parameter_file : str
        User parameters are read from the 'user_config' json file, and
        additional model parameters are set.
    test_mode : int
        If test_mode is set using an integer, the size of the model will
        be reduced e.g 100 makes a 100x100.It reduces the ammount of 
        time that the program takes to generate data usueful when testing
        a model.

        **Warning: If you put too small a number, the model may fail due to 
        not enough space to place faults etc...**

        Value should ideally be >= 50
    runid : str
        The string runid will be added to the final model directory.
    rpm_scaling_factors : dict
        These are user-defined parameter. You can use the defaults
        provided, but results might be unrealistic. 
        These might need to be tuned to get reaslistic synthetic
        data.
    sqldict : dict
        This is a dictionary structure that stores all the parameters
        of the model. This dictionary eventually gets written to a
        sqlite DB file.

    Methods
    -------
    setup_model(rpm_factors=None) -> None:
        Method to set up all the necesary parameters to start a new model.
    make_directories() -> None:
        Method that generates all necessary directory structures on disk
    write_key_file():
        Method to generate a key file that describes coordinate systems, 
        track, bin, digi (inline, xlines, )
    write_to_logfile(msg, mainkey=None, subkey=None, val=""):
        Method that writes to the logfile
    """

    def __init__(self, user_config: str = CONFIG_PATH, test_mode=None, runid=None):
        """
        Initialize the Parameters object.

        Parameters
        ----------
        user_config : `str`, optional
            This is the path on disk that points to a `.json` file
            that contains the configurations for each run, by default CONFIG_PATH
        test_mode : `int`, optional
            The parameter that sets the running mode, by default 0
        runid : `str`, optional
            This is the runid of the run, this comes in handy when you have many runs
            with various permutations of parameters, by default None
        """
        # reset the parameter dict in case we are building models within a loop, and the shared_state dict is not empty
        self._shared_state = {}
        super().__init__()
        self.model_dir_name: str = "seismic"
        self.parameter_file = user_config
        self.test_mode = test_mode
        self.runid = runid
        self.rpm_scaling_factors = dict()
        self.sqldict = defaultdict(dict)

    def __repr__(self):
        """
        Representation method

        Parameters
        ----------
        self : `Parameters`
            The instance of the Parameters object
        """
        # Make nice repr instead of a print method
        items = ("\t{} = {}".format(k, v) for k, v in self.__dict__.items())
        return "{}:\n{}".format(self.__class__.__name__, "\n".join(sorted(items)))

    def __getitem__(self, key: str):
        """__getitem__

        Enable retrieval of values as though the class instance is a dict

        Parameters
        ----------
        key : str
            The key desired to be accessed

        Returns
        -------
        any
            Value of the key
        """
        return self._shared_state[key]

    def setup_model(self, rpm_factors=None) -> None:
        """
        Setup Model
        -----------
        Sets up the creation of essential parameters and directories

        Parameters
        ----------
        rpm_factors : `dict`, optional
            The rock physics model factors for generating the synthetic cube.
            By default the rpm factors come from a default in the main.py file

        Returns
        -------
        None
        """
        # Set model parameters
        self._set_model_parameters(self.model_dir_name)
        self.make_directories()
        self.write_key_file()
        self._setup_rpm_scaling_factors(rpm_factors)

        # Write model parameters to logfile
        self._write_initial_model_parameters_to_logfile()

    def make_directories(self) -> None:
        """
        Make directories.
        -----------------

        Creates the necessary directories to run the model.

        This function creates the directories on disk
        necessary for the model to run.

        Parameters
        ----------
        self : `Parameters`

        Returns
        -------
        None
        """
        print(f"\nModel folder: {self.work_subfolder}")
        self.sqldict["model_id"] = pathlib.Path(self.work_subfolder).name
        for folder in [self.project_folder, self.work_subfolder, self.temp_folder]:
            try:
                os.stat(folder)
            except OSError:
                print(f"Creating directory: {folder}")
                # Try making folders (can fail if multiple models are being built simultaneously in a new dir)
                try:
                    os.mkdir(folder)
                except OSError:
                    pass
        try:
            os.system(f"chmod -R 777 {self.work_subfolder}")
        except OSError:
            print(f"Could not chmod {self.work_subfolder}. Continuing...")
            pass

    def write_key_file(self) -> None:
        """
        Write key file
        --------------

        Writes a file that ocntains important parameters about the cube.

        Method that writes important parameters about the synthetic cube
        such as coordinate transforms and sizes.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Set plausible key file values
        geom_expand = dict()
        geom_expand["3D_NAME"] = "synthetic data for training"
        geom_expand["COORD_METHOD"] = 1
        geom_expand["DATA_TYPE"] = "3D"
        geom_expand["DELTA_BIN_NUM"] = 1
        geom_expand["DELTA_TRACK_NUM"] = 1
        geom_expand["DIGITIZATION"] = 4
        geom_expand["EPSG_CRS"] = 32066
        geom_expand["FIRST_BIN"] = 1000
        geom_expand["FIRST_TRACK"] = 2000
        geom_expand["FORMAT"] = 1
        geom_expand["N_BIN"] = self.cube_shape[1]
        geom_expand["N_SAMP"] = self.cube_shape[2]
        geom_expand["N_TRACK"] = self.cube_shape[0]
        geom_expand["PROJECTION"] = 316
        geom_expand["REAL_DELTA_X"] = 100.0
        geom_expand["REAL_DELTA_Y"] = 100.0
        geom_expand["REAL_GEO_X"] = 1250000.0
        geom_expand["REAL_GEO_Y"] = 10500000.0
        geom_expand["SKEW_ANGLE"] = 0.0
        geom_expand["SUBPOINT_CODE"] = "TTTBBB"
        geom_expand["TIME_OR_DEPTH"] = "TIME"
        geom_expand["TRACK_DIR"] = "H"
        geom_expand["XFORM_TO_WGS84"] = 1241
        geom_expand["ZERO_TIME"] = 0

        # Write the keyfile
        outputkey = os.path.join(
            self.work_subfolder, "seismicCube_" + self.date_stamp + ".key"
        )
        with open(outputkey, "w") as key:
            key.write(
                "{}MESSAGE_FILE\n".format(20 * " ")
            )  # spaces are important here.. Require 20 of them
            key.write("3D_NAME C %s\n" % geom_expand["3D_NAME"])
            key.write("COORD_METHOD I %d\n" % int(geom_expand["COORD_METHOD"]))
            key.write("DATA_TYPE C %s\n" % geom_expand["DATA_TYPE"])
            key.write("DELTA_BIN_NUM I %d\n" % int(geom_expand["DELTA_BIN_NUM"]))
            key.write("DELTA_TRACK_NUM I %d\n" % int(geom_expand["DELTA_TRACK_NUM"]))
            key.write("DIGITIZATION I %d\n" % int(geom_expand["DIGITIZATION"]))
            key.write("EPSG_CRS I %d\n" % int(geom_expand["EPSG_CRS"]))
            key.write("FIRST_BIN I %d\n" % int(geom_expand["FIRST_BIN"]))
            key.write("FIRST_TRACK I %d\n" % int(geom_expand["FIRST_TRACK"]))
            key.write("FORMAT I %d\n" % int(geom_expand["FORMAT"]))
            key.write("N_BIN I %d\n" % int(geom_expand["N_BIN"]))
            key.write("N_SAMP I %d\n" % int(geom_expand["N_SAMP"]))
            key.write("N_TRACK I %d\n" % int(geom_expand["N_TRACK"]))
            key.write("PROJECTION I %d\n" % int(geom_expand["PROJECTION"]))
            key.write("REAL_DELTA_X R %f\n" % float(geom_expand["REAL_DELTA_X"]))
            key.write("REAL_DELTA_Y R %f\n" % float(geom_expand["REAL_DELTA_Y"]))
            key.write("REAL_GEO_X R %f\n" % float(geom_expand["REAL_GEO_X"]))
            key.write("REAL_GEO_Y R %f\n" % float(geom_expand["REAL_GEO_Y"]))
            key.write("SKEW_ANGLE R %f\n" % float(geom_expand["SKEW_ANGLE"]))
            key.write("SUBPOINT_CODE C %s\n" % geom_expand["SUBPOINT_CODE"])
            key.write("TIME_OR_DEPTH C %s\n" % geom_expand["TIME_OR_DEPTH"])
            key.write("TRACK_DIR C %s\n" % geom_expand["TRACK_DIR"])
            key.write("XFORM_TO_WGS84 I %d\n" % int(geom_expand["XFORM_TO_WGS84"]))
            key.write("ZERO_TIME I %d\n" % int(geom_expand["ZERO_TIME"]))
        print(f"\nKeyfile created at {outputkey}")

    def write_to_logfile(self, msg, mainkey=None, subkey=None, val="") -> None:
        """
        write_to_logfile

        Method to write msg to model_parameter file
        (includes newline)

        Parameters
        ----------
        msg : `string`
        Required string object that will be written tom model parameter file.
        mainkey : `string`
        String of the key to be written into de sql dictionary.
        subkey : `string`
        String of the subkey to be written into de sql dictionary.
        val : `string`
        String of the value that should be written into the sql dictionary.
        
        Returns
        -------
        None
        """
        if msg is not None:
            with open(self.logfile, "a") as f:
                f.write(f"{msg}\n")
        if mainkey is not None:
            self.sqldict[mainkey][subkey] = val
            # for k, v in self.sqldict.items():
            #     print(f"{k}: {v}")

    def write_sqldict_to_logfile(self, logfile=None) -> None:
        """
        write_sqldict_to_logfile

        Write the sql dictionary to the logfile

        Parameters
        ----------
        logfile : `string`
        The path to the log file. By default None
        
        Returns
        -------
        None
        """
        if logfile is None:
            logfile = self.logfile
        with open(logfile, "a") as f:
            for k, nested in self.sqldict.items():
                print(k, file=f)
                if k == "model_id":
                    print(f"\t{nested}", file=f)
                else:
                    for subkey, value in nested.items():
                        print(f"\t{subkey}: {value}", file=f)
                print(file=f)

    def write_sqldict_to_db(self) -> None:
        """
        write_sqldict_to_db

        Method to write the sqldict to database sqlite file

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        model_id = pathlib.Path(self.work_subfolder).name
        model_parameters = self.sqldict["model_parameters"]
        fault_keys = [k for k in self.sqldict.keys() if "fault" in k]
        closure_keys = [k for k in self.sqldict.keys() if "closure" in k]

        conn = sqlite3.connect(os.path.join(self.work_subfolder, "parameters.db"))
        # tables = ["model_parameters", "fault_parameters", "closure_parameters"]
        # create tables
        sql = f"CREATE TABLE model_parameters (model_id string primary key, {','.join(model_parameters.keys())})"
        conn.execute(sql)
        # insert model_parameters
        columns = "model_id, " + ", ".join(model_parameters.keys())
        placeholders = ", ".join("?" * (len(model_parameters) + 1))
        sql = f"INSERT INTO model_parameters ({columns}) VALUES ({placeholders})"
        values = tuple([model_id] + [str(x) for x in model_parameters.values()])
        conn.execute(sql, values)
        conn.commit()

        # fault parameters
        if len(fault_keys) > 0:
            f = tuple(self.sqldict[fault_keys[0]].keys())
            sql = f"CREATE TABLE fault_parameters ({','.join(f)})"
            conn.execute(sql)
            columns = ", ".join(self.sqldict[fault_keys[0]].keys())
            placeholders = ", ".join("?" * len(self.sqldict[fault_keys[0]].keys()))
            for f in fault_keys:
                sql = (
                    f"INSERT INTO fault_parameters ({columns}) VALUES ({placeholders})"
                )
                conn.execute(sql, tuple(self.sqldict[f].values()))
                conn.commit()

        if len(closure_keys) > 0:
            c = tuple(self.sqldict[closure_keys[0]].keys())
            sql = f"CREATE TABLE closure_parameters ({','.join(c)})"
            conn.execute(sql)
            columns = ", ".join(self.sqldict[closure_keys[0]].keys())
            placeholders = ", ".join("?" * len(self.sqldict[closure_keys[0]].keys()))
            for c in closure_keys:
                sql = f"INSERT INTO closure_parameters ({columns}) VALUES ({placeholders})"
                conn.execute(sql, tuple(self.sqldict[c].values()))
                conn.commit()

    def _setup_rpm_scaling_factors(self, rpm_factors: dict) -> None:
        """
        Setup Rock Physics Model scaling factors
        ----------------------------------------

        Method to initialize all the rock physics model
        scaling factors. Method also writes the values to
        the model_parameters log file.

        Parameters
        ----------
        TODO remove the default in the main.py or have a single source of truth
        rpm_factors : `dict`
        Dictionary containing the scaling factors for the RPM.
        If no RPM factors are provided, the default values are used.
        
        Returns
        -------
        None
        """
        if rpm_factors and not self.test_mode:
            self.rpm_scaling_factors = rpm_factors
        else:
            # Use defaults for RPM Z-shifts and scaling factors
            self.rpm_scaling_factors = dict()
            self.rpm_scaling_factors["layershiftsamples"] = int(
                np.random.triangular(35, 75, 125)
            )
            self.rpm_scaling_factors["RPshiftsamples"] = int(
                np.random.triangular(5, 11, 20)
            )
            self.rpm_scaling_factors["shalerho_factor"] = 1.0
            self.rpm_scaling_factors["shalevp_factor"] = 1.0
            self.rpm_scaling_factors["shalevs_factor"] = 1.0
            self.rpm_scaling_factors["sandrho_factor"] = 1.0
            self.rpm_scaling_factors["sandvp_factor"] = 1.0
            self.rpm_scaling_factors["sandvs_factor"] = 1.0
            self.rpm_scaling_factors["nearfactor"] = 1.0
            self.rpm_scaling_factors["midfactor"] = 1.0
            self.rpm_scaling_factors["farfactor"] = 1.0
        # Write factors to logfile
        for k, v in self.rpm_scaling_factors.items():
            self.write_to_logfile(
                msg=f"{k}: {v}", mainkey="model_parameters", subkey=k, val=v
            )

    def _set_model_parameters(self, dname: str) -> None:
        """
        Set Model Parameters
        ----------------------------------------

        Method that sets model parameters from user-provided
        config.json file

        Parameters
        ----------
        dname : `str`
        Directory name specified in the configuration file,
        or the default is used
        
        Returns
        -------
        None
        """
        self.current_dir = os.getcwd()
        self.start_time = datetime.datetime.now()
        self.date_stamp = self.year_plus_fraction()

        # Read from input json
        self.parameters_json = self._read_json()
        self._read_user_params()

        # Directories
        model_dir = f"{dname}__{self.date_stamp}"
        temp_dir = f"temp_folder__{self.date_stamp}"
        self.work_subfolder = os.path.abspath(
            os.path.join(self.project_folder, model_dir)
        )
        self.temp_folder = os.path.abspath(
            os.path.join(self.work_folder, f"temp_folder__{self.date_stamp}")
        )
        if self.runid:
            self.work_subfolder = f"{self.work_subfolder}_{self.runid}"
            self.temp_folder = f"{self.temp_folder}_{self.runid}"

        # Various model parameters, not in config
        self.num_lyr_lut = self.cube_shape[2] * 2 * self.infill_factor
        # 2500 voxels = 25x25x4m voxels size, 25% porosity and closures > ~40,000 bbl
        # Use the minimum voxel count as initial closure size filter
        self.closure_min_voxels = min(
            self.closure_min_voxels_simple,
            self.closure_min_voxels_faulted,
            self.closure_min_voxels_onlap,
        )
        self.order = self.bandwidth_ord

        if self.test_mode:
            self._set_test_mode(self.test_mode, self.test_mode)

        # Random choices are separated into this method
        self._randomly_chosen_model_parameters()
        # Fault choices
        self._fault_settings()

        # Logfile
        self.logfile = os.path.join(
            self.work_subfolder, f"model_parameters_{self.date_stamp}.txt"
        )

        # HDF file to store various model data
        self.hdf_master = os.path.join(
            self.work_subfolder, f"seismicCube__{self.date_stamp}.hdf"
        )

    def _calculate_snr_after_lateral_filter(self, sn_db: float) -> float:
        """
        Calculate Signal:Noise Ratio after lateral filter
        ----------------------------------------

        Method that computes the signal to noise ratio after
        the lateral filter is applied.

        Parameters
        ----------
        sn_db : `float`
            Value of the signal to noise value from the database
        
        Returns
        -------
        pre_smear_snr : `float`
            Signal to noise ratio after the lateral filter is applied
        """
        snr_of_lateral_filter = 10 * np.log10(self.lateral_filter_size ** 2)
        pre_smear_snr = sn_db - snr_of_lateral_filter
        return pre_smear_snr

    def _randomly_chosen_model_parameters(self) -> None:
        """
        Randomly Chosen Model Parameters
        ----------------------------------------

        Method that sets all randomly chosen model parameters

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Initial layer standard deviation
        self.initial_layer_stdev = (
            np.random.uniform(self.lyr_stdev[0], high=self.lyr_stdev[1])
            * self.infill_factor
        )

        # lateral filter size, either 1x1, 3x3 or 5x5
        self.lateral_filter_size = int(np.random.uniform(0, 2) + 0.5) * 2 + 1

        # Signal to noise in decibels
        sn_db = triangle_distribution_fix(
            left=self.snr_db[0], mode=self.snr_db[1], right=self.snr_db[2]
        )
        # sn_db = np.random.triangular(left=self.snr_db[0], mode=self.snr_db[1], right=self.snr_db[2])
        # self.sn_db = self._calculate_snr_after_lateral_filter(sn_db)
        self.sn_db = sn_db

        # Percentage of layers that are sand
        self.sand_layer_pct = np.random.uniform(
            low=self.sand_layer_pct_min, high=self.sand_layer_pct_max
        )

        # Minimum shallowest depth of seabed
        if (
            len(self.seabed_min_depth) > 1
        ):  # if low/high value provided, select a value between these
            self.seabed_min_depth = np.random.randint(
                low=self.seabed_min_depth[0], high=self.seabed_min_depth[1]
            )

        # Low/High bandwidth to be used
        self.lowfreq = np.random.uniform(self.bandwidth_low[0], self.bandwidth_low[1])
        self.highfreq = np.random.uniform(
            self.bandwidth_high[0], self.bandwidth_high[1]
        )

        # Choose whether to add coherent noise
        self.add_noise = np.random.choice((0, 1))
        if self.add_noise == 1:
            self.smiley_or_frowny = np.random.choice((0, 1))
            if self.smiley_or_frowny == 1:
                self.fnoise = "random_coherent_frowns"
                print("Coherent frowns will be inserted")
            else:
                self.fnoise = "random_coherent_smiles"
                print("Coherent smiles will be inserted")
        else:
            self.fnoise = "random"
            print("No coherent noise will be inserted")

        # Salt inclusion
        # self.include_salt = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        self.noise_stretch_factor = np.random.uniform(1.15, 1.35)
        if self.include_salt:
            print(
                "Salt will be inserted. noise_stretch_factor = {}".format(
                    np.around(self.noise_stretch_factor, 2)
                )
            )
        else:
            print("Salt will be NOT be inserted.")

    def _read_json(self) -> dict:
        # TODO Move this to a separate function in utlis?
        """
        Read JSON file
        ----------------------------------------

        Reads a json file on disk and loads it as
        dictionary

        Parameters
        ----------
        None
        
        Returns
        -------
        config : `dict`
            Dictionary with the configuration options
        """
        with open(self.parameter_file) as f:
            config: dict = json.load(f)
        return config

    def _read_user_params(self) -> None:
        """
        Read User Params
        ----------------------------------------

        Takes the read in dictionary of JSON configuration
        and reads each parameter and inserts it into the
        attributes.

        In the end it prints a summary of the parameters
        to the console.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        d = self._read_json()
        self.project = d["project"]
        self.project_folder = d["project_folder"]
        wfolder = d["work_folder"]
        if not os.path.exists(wfolder):
            wfolder = "/tmp"  # In case work_folder does not exist, use /tmp
        self.work_folder = wfolder
        # read parameters into Parameter class attributes
        self.cube_shape = tuple(d["cube_shape"])
        self.incident_angles = tuple(d["incident_angles"])
        self.digi = d["digi"]
        self.infill_factor = d["infill_factor"]
        self.lyr_stdev = d["initial_layer_stdev"]
        self.thickness_min = d["thickness_min"]
        self.thickness_max = d["thickness_max"]
        self.seabed_min_depth = d["seabed_min_depth"]
        self.snr_db = d["signal_to_noise_ratio_db"]
        # self.random_depth_perturb = d['random_depth_perturb_range']
        self.bandwidth_low = d["bandwidth_low"]
        self.bandwidth_high = d["bandwidth_high"]
        self.bandwidth_ord = d["bandwidth_ord"]
        self.dip_factor_max = d["dip_factor_max"]
        self.min_number_faults = d["min_number_faults"]
        self.max_number_faults = d["max_number_faults"]
        self.basin_floor_fans = d["basin_floor_fans"]
        self.pad_samples = d["pad_samples"]
        self.qc_plots = d["extra_qc_plots"]
        self.verbose = d["verbose"]
        self.include_channels = d["include_channels"]
        self.include_salt = d["include_salt"]
        self.max_column_height = d["max_column_height"]
        self.closure_types = d["closure_types"]
        self.closure_min_voxels_simple = d["min_closure_voxels_simple"]
        self.closure_min_voxels_faulted = d["min_closure_voxels_faulted"]
        self.closure_min_voxels_onlap = d["min_closure_voxels_onlap"]
        self.partial_voxels = d["partial_voxels"]
        self.variable_shale_ng = d["variable_shale_ng"]
        self.sand_layer_thickness = d["sand_layer_thickness"]
        self.sand_layer_pct_min = d["sand_layer_fraction"]["min"]
        self.sand_layer_pct_max = d["sand_layer_fraction"]["max"]
        self.hdf_store = d["write_to_hdf"]
        self.broadband_qc_volume = d["broadband_qc_volume"]
        self.model_qc_volumes = d["model_qc_volumes"]
        self.multiprocess_bp = d["multiprocess_bp"]

        # print em
        self.__repr__()

    def _set_test_mode(self, size_x: int = 50, size_y: int = 50) -> None:
        """
        Set test mode
        -------------

        Sets whether the parameters for testinf mode. If no size integer
        is provided is defaults to 50.

        This value is a good minimum because it allows for the 3D model
        to be able to contain faults and other objects inside.

        Parameters
        ----------
        size_x : `int`
        The parameter that sets the size of the model in the x direction
        size_y : `int`
        The parameter that sets the size of the model in the y direction
        
        Returns
        -------
        None
        """

        # Set output model folder in work_folder location but with same directory name as project_folder
        normpath = (
            os.path.normpath(self.project_folder) + "_test_mode_"
        )  # strip trailing / if added
        new_project_folder = os.path.join(self.work_folder, os.path.basename(normpath))
        # Put all folders inside project folder for easy deleting
        self.work_folder = new_project_folder
        self.project_folder = new_project_folder
        self.work_subfolder = os.path.join(
            new_project_folder, os.path.basename(self.work_subfolder)
        )
        if self.runid:
            # Append runid if provided
            self.temp_folder = f"{self.temp_folder}_{self.runid}__{self.date_stamp}"
        else:
            self.temp_folder = os.path.abspath(
                os.path.join(self.work_folder, f"temp_folder__{self.date_stamp}")
            )
        # Set smaller sized model
        self.cube_shape = tuple([size_x, size_y, self.cube_shape[-1]])
        # Print message to user
        print(
            "{0}\nTesting Mode\nOutput Folder: {1}\nCube_Shape: {2}\n{0}".format(
                36 * "-", self.project_folder, self.cube_shape
            )
        )

    def _fault_settings(self) -> None:
        """
        Set Fault Settings
        -------------

        Sets the parameters that will be used to generate faults throughout
        the synthetic model.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Fault parameters
        self.low_fault_throw = 5.0 * self.infill_factor
        self.high_fault_throw = 35.0 * self.infill_factor

        # mode & clustering are randomly chosen
        self.mode = np.random.choice([0, 1, 2], 1)[
            0
        ]
        self.clustering = np.random.choice([0, 1, 2], 1)[0]

        if self.mode == 0:
            # As random as it can be
            self.number_faults = np.random.randint(
                self.min_number_faults, self.max_number_faults
            )
            self.fmode = "random"

        elif self.mode == 1:
            if self.clustering == 0:
                self.fmode = "self_branching"
                # Self Branching. avoid large fault
                self.number_faults = np.random.randint(
                    3, 9
                )
                self.low_fault_throw = 5.0 * self.infill_factor
                self.high_fault_throw = 15.0 * self.infill_factor
            if self.clustering == 1:
                # Stair case
                self.fmode = "stair_case"
                self.number_faults = np.random.randint(
                    5, self.max_number_faults
                )
            if self.clustering == 2:
                # Relay ramps
                self.fmode = "relay_ramp"
                self.number_faults = np.random.randint(
                    3, 9
                )
                self.low_fault_throw = 5.0 * self.infill_factor
                self.high_fault_throw = 15.0 * self.infill_factor
        elif self.mode == 2:
            # Horst and graben
            self.fmode = "horst_and_graben"
            self.number_faults = np.random.randint(
                3, 7
            )

        self.fault_param = [
            str(self.mode) + str(self.clustering),
            self.number_faults,
            self.low_fault_throw,
            self.high_fault_throw,
        ]

    def _get_commit_hash(self) -> str:
        """
        Get Commit Hash
        -------------

        Gets the commit hash of the current git repository.

        #TODO Explain what this is for exactly

        Parameters
        ----------
        None
        
        Returns
        -------
        sha : `str`
            The commit hash of the current git repository
        """

        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
        except CalledProcessError:
            sha = "cwd not a git repository"
        return sha

    def _write_initial_model_parameters_to_logfile(self) -> None:
        """
        Write Initial Model Parameters to Logfile
        ----------------------------------------

        Method that writes the initial parameters set for the model
        to the logfile.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        _sha = self._get_commit_hash()
        self.write_to_logfile(
            f"SHA: {_sha}", mainkey="model_parameters", subkey="sha", val=_sha
        )
        self.write_to_logfile(
            f"modeling start time: {self.start_time}",
            mainkey="model_parameters",
            subkey="start_time",
            val=self.start_time,
        )
        self.write_to_logfile(
            f"project_folder: {self.project_folder}",
            mainkey="model_parameters",
            subkey="project_folder",
            val=self.project_folder,
        )
        self.write_to_logfile(
            f"work_subfolder: {self.work_subfolder}",
            mainkey="model_parameters",
            subkey="work_subfolder",
            val=self.work_subfolder,
        )
        self.write_to_logfile(
            f"cube_shape: {self.cube_shape}",
            mainkey="model_parameters",
            subkey="cube_shape",
            val=self.cube_shape,
        )
        self.write_to_logfile(
            f"incident_angles: {self.incident_angles}",
            mainkey="model_parameters",
            subkey="incident_angles",
            val=self.incident_angles,
        )
        self.write_to_logfile(
            f"number_faults: {self.number_faults}",
            mainkey="model_parameters",
            subkey="number_faults",
            val=self.number_faults,
        )
        self.write_to_logfile(
            f"lateral_filter_size: {self.lateral_filter_size}",
            mainkey="model_parameters",
            subkey="lateral_filter_size",
            val=self.lateral_filter_size,
        )
        self.write_to_logfile(
            f"salt_inserted: {self.include_salt}",
            mainkey="model_parameters",
            subkey="salt_inserted",
            val=self.include_salt,
        )
        self.write_to_logfile(
            f"salt noise_stretch_factor: {self.noise_stretch_factor:.2f}",
            mainkey="model_parameters",
            subkey="salt_noise_stretch_factor",
            val=self.noise_stretch_factor,
        )
        self.write_to_logfile(
            f"bandpass_bandlimits: {self.lowfreq:.2f}, {self.highfreq:.2f}"
        )
        self.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="bandpass_bandlimit_low",
            val=self.lowfreq,
        )
        self.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="bandpass_bandlimit_high",
            val=self.highfreq,
        )
        self.write_to_logfile(
            f"sn_db: {self.sn_db:.2f}",
            mainkey="model_parameters",
            subkey="sn_db",
            val=self.sn_db,
        )
        self.write_to_logfile(
            f"initial layer depth stdev (flatness of layer): {self.initial_layer_stdev:.2f}",
            mainkey="model_parameters",
            subkey="initial_layer_stdev",
            val=self.initial_layer_stdev,
        )

    @staticmethod
    def year_plus_fraction() -> str:
        # TODO Move this to utils separate module
        """
        Year Plus Fraction
        ----------------------------------------

        Method generates a time stamp in the format of
        year + fraction of year.

        Parameters
        ----------
        None
        
        Returns
        -------
        fraction of the year : str
            The time stamp in the format of year + fraction of year

        """
        now = datetime.datetime.now()
        year = now.year
        secs_in_year = datetime.timedelta(days=365).total_seconds()
        fraction_of_year = (
            now - datetime.datetime(year, 1, 1, 0, 0)
        ).total_seconds() / secs_in_year
        return format(year + fraction_of_year, "14.8f").replace(" ", "")

    def hdf_setup(self, hdf_name: str) -> None:
        """
        Setup HDF files
        ---------------

        This method sets up the HDF structures

        Parameters
        ----------
        hdf_name : str
            The name of the HDF file to be created
        
        Returns
        -------
        None
        """
        num_threads = min(8, mp.cpu_count() - 1)
        tables.set_blosc_max_threads(num_threads)
        self.hdf_filename = os.path.join(self.temp_folder, hdf_name)
        self.filters = tables.Filters(
            complevel=5, complib="blosc"
        )  # compression with fast write speed
        self.h5file = tables.open_file(self.hdf_filename, "w")
        self.h5file.create_group("/", "ModelData")

    def hdf_init(self, dset_name, shape: tuple, dtype: str = "float64") -> tables.CArray:
        
        """
        HDF Initialize
        ----------------------------------------

        Method that initializes the HDF chunked
        array

        Parameters
        ----------
        dset_name : str
            The name of the dataset to be created
        shape : tuple

        
        Returns
        -------
        new_array: tables.CArray
        """
        if "float" in dtype:
            atom = tables.FloatAtom()
        elif "uint8" in dtype:
            atom = tables.UInt8Atom()
        else:
            atom = tables.IntAtom()
        group = self.h5file.root.ModelData
        new_array = self.h5file.create_carray(
            group, dset_name, atom, shape, filters=self.filters
        )
        return new_array

    def hdf_node_list(self):
        return [x.name for x in self.h5file.list_nodes("ModelData")]

    def hdf_remove_node_list(self, dset_name):
        group = self.h5file.root.ModelData
        try:
            self.h5file.remove_node(group, dset_name)
        except:
            pass
        self.hdf_node_list


def triangle_distribution_fix(left, mode, right, random_seed=None):
    """
    Triangle Distribution Fix
    -------------------------

    Draw samples from the triangular distribution over the interval [left, right] with modifications.

    Ensure some values are drawn at the left and right values by enlarging the interval to
    [left - (mode - left), right + (right - mode)]

    Parameters
    ----------
    left: `float`
        lower limit
    mode: `float`
        mode
    right: `float`
        upper limit
    random_seed: `int`
        seed to set numpy's random seed

    Returns
    -------
    sn_db: `float`
        Drawn samples from parameterised triangular distribution
    """
    sn_db = 0
    while sn_db < left or sn_db > right:
        if random_seed:
            np.random.seed(random_seed)
        sn_db = np.random.triangular(left - (mode - left), mode, right + (right - mode))

    return sn_db
