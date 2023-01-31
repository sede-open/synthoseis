########################################################################
# Code to create numpy 3D array with output from channel simulations
# - runs fortran program fluvsim.for for a single channel
# -- creates fluvsim parameter file (fluvsim.par)
# -- reads parameter file
# -- runs fluvsim simulation
# -- returns 3D array
#    --- Codes for facies output:
#    --- 0  is floodplain shale
#    --- 1  is channel fill (sand)
#    --- 2  is shale channel drape
#    --- 3  is levee (mid quality sand)
#    --- 4  is crevasse (low quality sand)
########################################################################

import os
import numpy as np
from datetime import datetime
from datagenerator.util import import_matplotlib

plt = import_matplotlib()


def create_fluvsim_params(
    nx=100,
    ny=100,
    maxthickness=50,
    work_folder="/scratch"
) -> None:
    """
    Create Fluvsim parameter file
    -----------------------------

    Create fluvsim.par parameter file with 
    parameters for fluvsim simulation

    Parameters
    ----------
    nx : int, optional
        _description_, by default 100
    ny : int, optional
        _description_, by default 100
    maxthickness : int, optional
        Maximum thickness, by default 50
    work_folder : str, optional
        Default work folder, by default "/scratch"
    
    Returns
    -------
    None
    """

    # create fluvsim.par parameter file with randomly chosen values for some parameters
    seed = datetime.now().microsecond

    # number of channels
    # - compute number of channels (needs 10 to work properly)
    num_chans = int(np.random.triangular(left=3, mode=5, right=25) + 0.5)
    print("...num_chans = ", num_chans)

    # channel depth
    # - compute number that range from 1+ to 7+, centered at ~3
    # - re-scale for maxthickness/50.
    vert_scale_factor = maxthickness / 50.0
    # random_chan_thickness = np.random.gamma(shape=15.,scale=3.)/13.5       *  vert_scale_factor
    # random_chan_thickness = np.random.gamma(shape=5.,scale=7.)/13.5       *  vert_scale_factor
    random_chan_thickness = (
        np.random.gamma(shape=5.0, scale=5.65) / 13.5 * vert_scale_factor
    )
    random_chan_thickness = format(random_chan_thickness, "2.1f")
    print("...random_chan_thickness = ", random_chan_thickness)

    # channel lateral departure (meander width)
    # - re-scale
    lateral_scale_factor = nx / 100.0
    avg_departure = 175.0 * lateral_scale_factor
    avg_departure = format(avg_departure, "5.0f")

    # channel sinuosity (meander) wavelength
    # - re-scale for low/mid/high (L/M/H)
    wavelength_factor = nx / 100.0
    avg_length = 800.0 * wavelength_factor
    avg_length_l = format(avg_length * 0.7, "5.0f")
    avg_length_m = format(avg_length, "5.0f")
    avg_length_h = format(avg_length * 1.3, "5.0f")

    # levee width
    levee_onoff = np.random.binomial(1, 0.6)
    # levee width
    # - re-scale for low/mid/high (L/M/H)
    avg_levee = 240.0 * lateral_scale_factor
    avg_levee_l = format(avg_levee * 0.67, "5.0f")
    avg_levee_m = format(avg_levee, "5.0f")
    avg_levee_h = format(avg_levee * 2.0, "5.0f")
    print("...levee widths = ", avg_levee_l, avg_levee_m, avg_levee_h)

    param_file_text = (
        "                   Parameters for FLUVSIM"
        + "\n"
        + "                   **********************"
        + "\n"
        + ""
        + "\n"
        + "START OF PARAMETERS:"
        + "\n"
        + "data/well01.dat                -file with well conditioning data"
        + "\n"
        + "1  2  3  4  5                  -  columns for X, Y, Z, well #, facies"
        + "\n"
        + "-1.0       1.0e21              -  trimming limits"
        + "\n"
        + "1                              -debugging level: 0,1,2,3"
        + "\n"
        + "output/fluvsim.dbg             -file for debugging output"
        + "\n"
        + "output/fluvsim.geo             -file for geometric specification"
        + "\n"
        + "output/fluvsim.out             -file for simulation output"
        + "\n"
        + "output/fluvsim.vp              -file for vertical prop curve output"
        + "\n"
        + "output/fluvsim.ap              -file for areal prop map output"
        + "\n"
        + "output/fluvsim.wd              -file for well data output"
        + "\n"
        + "1                              -number of realizations to generate"
        + "\n"
        + format(nx, "<3d")
        + "   0.0    40.0             -nx,xmn,xsiz - geological coordinates"
        + "\n"
        + format(nx, "<3d")
        + "   0.0    40.0             -ny,ymn,ysiz - geological coordinates"
        + "\n"
        + format(int(maxthickness), "<3d")
        + "          50.0              -nz, average thickness in physical units"
        + "\n"
        + format(seed, "<6d")
        + "                         -random number seed"
        + "\n"
        + "1   0   0   1                  -1=on,0=off: global, vert, areal, wells"
        + "\n"
        + "1.  1.  1.  1.                 -weighting : global, vert, areal, wells"
        + "\n"
        + "1     10   0.05                -maximum iter, max no change, min. obj."
        + "\n"
        + "0.0   0.10   3  1  8           -annealing schedule: t0,redfac,ka,k,num"
        + "\n"
        + "0.1 0.1 0.1 1.0                -Pert prob: 1on+1off, 1on, 1off, fix well"
        + "\n"
        + "   1    1    "
        + format(levee_onoff, "1d")
        + "                 -Facies(on): channel, levee, crevasse"
        + "\n"
        + "0.30 0.15 0.03                 -Proportion: channel, levee, crevasse"
        + "\n"
        + "pcurve.dat                     -  vertical proportion curves"
        + "\n"
        + "0                              -     0=net-to-gross, 1=all facies"
        + "\n"
        + "1  7  8                        -     column numbers"
        + "\n"
        + "arealprop.dat                  -  areal proportion map"
        + "\n"
        + "1                              -     0=net-to-gross, 1=all facies"
        + "\n"
        + "2  3  4                        -     column numbers"
        + "\n"
        + format(num_chans, "<3d")
        + "                            -maximum number of channels"
        + "\n"
        + "-45.0    0.0    45.0           -channel:  orientation (degrees)"
        + "\n"
        + avg_departure
        + "  "
        + avg_departure
        + "   "
        + avg_departure
        + "           -channel:  sinuosity: average departure"
        + "\n"
        + avg_length_l
        + "  "
        + avg_length_m
        + "   "
        + avg_length_h
        + "           -channel:  sinuosity: length scale"
        + "\n"
        + "  "
        + random_chan_thickness
        + "    "
        + random_chan_thickness
        + "     "
        + random_chan_thickness
        + "           -channel:  thickness"
        + "\n"
        + "  1.0    1.0     1.0           -channel:  thickness undulation"
        + "\n"
        + "250.0  400.0   650.0           -channel:  thickness undul. length scale"
        + "\n"
        + " 50.0  150.0   300.0           -channel:  width/thickness ratio"
        + "\n"
        + "  1.0    1.0     1.0           -channel:  width: undulation"
        + "\n"
        + "250.0  400.0   550.0           -channel:  width: undulation length scale"
        + "\n"
        + "500.0 1500.0  3500.0           -levee:    average width"
        + "\n"
        + "  0.1    0.15    0.25          -levee:    average height"
        + "\n"
        + "  0.05   0.1     0.3           -levee:    depth below top"
        + "\n"
        + " 80.0  200.0   500.0           -crevasse: attachment length"
        + "\n"
        + "  0.25   0.5     0.75          -crevasse: relative thickness by channel"
        + "\n"
        + "200.0  500.0  4500.0           -crevasse: areal size (diameter)"
        + "\n"
    )

    # avg_levee_l+'  '+avg_levee_m+'   '+avg_levee_h+'           -levee:    average width'+'\n'+\

    # write param file to disk
    params_file = os.path.abspath(os.path.join(work_folder, "fluvsim.par"))
    with open(params_file, "w") as paramfile:
        paramfile.write(param_file_text)
    return


def read_fluvsim_output(nx, ny, nz, work_folder="/scratch"):
    # from scipy.ndimage.morphology import grey_closing

    output_file = os.path.abspath(os.path.join(work_folder, "output", "fluvsim.out"))
    f = open(output_file)
    a = f.read().split("\n")

    kk = 0
    facies = np.zeros((nx, ny, nz), "int")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                facies[i, j, k] = int(a[kk])
                kk = kk + 1

    return facies


def run_fluvsim(nx=100, ny=100, maxthickness=50, work_folder="/scratch", quiet=True):
    # set up to work on scratch folder
    current_dir = os.getcwd()
    code_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    code_file = os.path.join(code_dir, "fluvsim.f90")
    workfolder_code_file = os.path.join(work_folder, "fluvsim.f90")
    os.system("cp " + code_file + " " + workfolder_code_file)
    try:
        workfolder_output = os.path.abspath(os.path.join(work_folder, "output"))
        os.system("mkdir -p " + workfolder_output + "&>/scratch/outputfile")
    except OSError:
        pass
    os.chdir(work_folder)
    fortran_code_file = os.path.abspath(os.path.join(work_folder, "fluvsim.f90"))
    compiled_fortran_code_file = "./fluvsim.o"
    return_code_2 = os.system(
        "gfortran " + fortran_code_file + " -o " + compiled_fortran_code_file
    )
    if return_code_2 != 0:
        print("...retrieving fluvsim.o from code repository....")
        os.system(
            "cp -p "
            + os.path.join(code_dir, "fluvsim2.o")
            + " "
            + compiled_fortran_code_file
        )

    # create params file
    create_fluvsim_params(
        nx=nx, ny=ny, maxthickness=maxthickness, work_folder=work_folder
    )

    # run fluvsim
    os.chdir(work_folder)
    os.system(
        compiled_fortran_code_file
        + "&>"
        + os.path.join(work_folder, "fluvsim_output.txt")
    )

    # read fluvsim.out ascii file into numpy array
    facies = read_fluvsim_output(nx, ny, maxthickness, work_folder=work_folder)

    if quiet:  # TODO save plots to output dir?
        # plot
        xsection = 10
        ysection = 10
        zsection = 30

        # xsection
        plt.figure(1, figsize=(15, 9))
        plt.subplot(2, 1, 1)
        temp = facies[xsection, :, :]
        plt.imshow(np.rot90(temp))
        plt.title("xsection = " + str(xsection))

        # ysection
        plt.subplot(2, 1, 2)
        temp = facies[:, ysection, :]
        plt.imshow(np.rot90(temp))
        plt.title("ysection = " + str(ysection))

        # zsection
        plt.figure(2, figsize=(10, 10))
        temp = facies[:, :, zsection]
        plt.imshow(temp)
        plt.title("zsection = " + str(zsection))

        plt.show()

    os.chdir(current_dir)

    return facies


if __name__ == "__main__":

    # nx=300
    # ny=300
    # maxthickness=800
    # quiet=False
    model = run_fluvsim(nx=300, ny=300, maxthickness=800, quiet=False)
    print("model shape = ", model.shape)
