"""Entry point for building a syntehtic model."""

import argparse
import datetime
import os
import shutil
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from datagenerator.Closures import Closures
from datagenerator.Faults import Faults
from datagenerator.Geomodels import Geomodel
from datagenerator.Horizons import build_unfaulted_depth_maps, create_facies_array
from datagenerator.Parameters import Parameters
from datagenerator.Seismic import SeismicVolume
from datagenerator.log import setup_logging
from datagenerator.output_writer import write_volume_to_zarr
from datagenerator.util import plot_3D_closure_plot


def build_model(user_json: str, run_id, test_mode: int = None, rpm_factors=None, seed=None):
    """Build model from config file."""
    # Set up model parameters
    p = Parameters(user_json, runid=run_id, test_mode=test_mode)
    p.setup_model(rpm_factors=rpm_factors, seed=seed)

    p.setup_model_store()
    # Build un-faulted depth maps and facies array
    depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(p)
    facies = create_facies_array(p, depth_maps, onlap_list, fan_list)

    # Build un-faulted geological models
    geo_models = Geomodel(p, depth_maps, onlap_list, facies)
    geo_models.build_unfaulted_geomodels()

    # Build Faults
    f = Faults(p, depth_maps, onlap_list, geo_models, fan_list, fan_thicknesses)
    # Apply faults to depth maps and age volume, and output faulted files
    f.apply_faulting_to_geomodels_and_depth_maps()
    # Build faulted lithology, net_to_gross and depth and randomised models
    f.build_faulted_property_geomodels(facies)

    # Write final geology outputs as zarr stores
    geology_dir = os.path.join(p.work_subfolder, "geology")
    os.makedirs(geology_dir, exist_ok=True)
    write_volume_to_zarr(
        f.faulted_age_volume[:],
        os.path.join(geology_dir, "geologic_age.zarr"),
        name="age",
        dims=("inline", "crossline", "time"),
    )
    write_volume_to_zarr(
        f.faulted_lithology[:],
        os.path.join(geology_dir, "faulted_lithology.zarr"),
        name="lithology",
        dims=("inline", "crossline", "time"),
    )
    horizons_dir = os.path.join(p.work_subfolder, "horizons")
    os.makedirs(horizons_dir, exist_ok=True)
    write_volume_to_zarr(
        f.faulted_depth_maps,
        os.path.join(horizons_dir, "depth_maps.zarr"),
        name="depth",
        dims=("inline", "crossline", "horizon"),
    )

    # Create closures, remove false closures and and output closures
    closures = Closures(p, f, facies, onlap_list)
    closures.create_closures()
    closures.write_closure_volumes_to_zarr()

    # Create 3D qc plot
    if p.qc_plots:
        try:
            plot_3D_closure_plot(p, closures)
        except ValueError:
            p.write_to_logfile("3D Closure Plotting Failed")

    # Create Rho, Vp, Vs volumes, apply Zoeppritz and write seismic volumes to disk
    seismic = SeismicVolume(p, f, closures)
    seismic.build_elastic_properties("inv_vel")
    seismic.build_seismic_volumes()
    seismic.join_gather_write()

    n_incident = len(p.incident_angles)
    rfc_start = 1 if p.model_qc_volumes else 0
    closures.write_closure_info_to_log(seismic.rfc_raw[rfc_start : rfc_start + n_incident, ...])

    elapsed_time = datetime.datetime.now() - p.start_time
    print("\n\n\n...elapsed time is {}".format(elapsed_time))
    p.write_to_logfile(
        f"elapsed_time: {elapsed_time}\n",
        mainkey="model_parameters",
        subkey="elapsed_time",
        val=elapsed_time,
    )
    p.write_sqldict_to_logfile(f"{p.work_subfolder}/sql_log.txt")
    p.write_sqldict_to_db()

    # Cleanup
    if getattr(p, 'cleanup_intermediates', True):
        shutil.rmtree(p.temp_folder, ignore_errors=True)
    try:
        os.system(f"chmod -R 777 {p.work_subfolder}")
    except OSError:
        pass

    # Change back to original directory for next run
    os.chdir(p.current_dir)

    plt.close("all")

    return p.work_subfolder


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_mode",
        help="Run in testing mode, number will g",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-c", "--config_file", help="Provide model parameter file", required=True
    )
    parser.add_argument(
        "-n", "--num_runs", help="Number of models to create", default=1, type=int
    )
    parser.add_argument("-r", "--run_id", help="Run ID", default=None)
    parser.add_argument("-s", "--seed", help="Random seed for reproducibility", default=None, type=int)

    args = parser.parse_args()

    # Print args
    print("Input arguments used:")
    for arg in vars(args):
        print(f"\t* {arg}: {getattr(args, arg)}")

    for iRun in range(args.num_runs):
        # Seed the pre-run RPM factor draws from the same seed passed to build_model
        # so that --seed N always produces the same layershiftsamples / RPshiftsamples.
        _run_rng = default_rng(args.seed if args.seed is not None else None)
        # Apply randomisation to the rock properties
        factor_dict = dict()
        factor_dict["layershiftsamples"] = int(_run_rng.triangular(35, 75, 125))
        factor_dict["RPshiftsamples"] = int(_run_rng.triangular(5, 11, 20))
        factor_dict["shalerho_factor"] = 1.0
        factor_dict["shalevp_factor"] = 1.0
        factor_dict["shalevs_factor"] = 1.0
        factor_dict["sandrho_factor"] = 1.0
        factor_dict["sandvp_factor"] = 1.0
        factor_dict["sandvs_factor"] = 1.0
        # Amplitude scaling factors for n, m ,f volumes
        factor_dict["nearfactor"] = 1.0
        factor_dict["midfactor"] = 1.0
        factor_dict["farfactor"] = 1.0
        # Build model using the selected rpm factors
        build_model(
            args.config_file, args.run_id, args.test_mode, rpm_factors=factor_dict, seed=args.seed
        )
