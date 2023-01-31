"""
Functions for applying historgram equalization to array
to fit standard normal distribution (including kurtosis)
"""

import sys
from math import sqrt

import numpy as np
from scipy.stats import kurtosis


def _derive_standard_normal(im, nbr_bins=256, verbose=False):
    ##############################################################
    ##    The function derives the equalization array to scale
    ##    the values in input array 'im' based on
    ##    a converson of its histogram into a uniform distribution.
    ##    When 'pcteq' is 100%, the output will have a uniform
    ##    distribution. When 'pcteq' is 0%, the function returns the
    ##    input array unchanged. 'pcteq' = 50% returns an array
    ##    with half the impact of conversion to a uniform distribution.
    ##
    ##    does not apply transform, just returns curve
    ##############################################################

    from numpy import histogram
    from scipy.interpolate import interp1d

    if verbose:
        print(" Deriving equalizing transform to standard normal shape")

    # decimate to about 10000 traces. Perform calculations on decimated subset
    decimate = max(1, int(sqrt(im.flatten().shape[0] / 10000)))
    amin, amax = im.min(), im.max()
    _data = im.flatten()[::decimate]
    _data = np.hstack((amin, _data, amax))
    _data = np.hstack((_data, -_data))

    # assume data is centered roughly at zero
    # find biggest abs value and make data range double that
    # (symmetrical about zero)
    # These are the bin boundaries
    histrange = _data.max() - _data.min()
    inputbins = np.arange(_data.min(), _data.max(), histrange / (nbr_bins - 1))
    if len(inputbins) == 0:
        return im
    imhist, bins = histogram(_data, bins=nbr_bins, density=True)

    imhist[0] = 0.0
    imhist[-1] = 0.0

    # calculate center of bins
    centerbins = np.linspace(_data.min(), _data.max(), nbr_bins)
    # centerbins  =np.empty(len(imhist),dtype=float)
    # for i in range(len(bins)-1):
    #    centerbins[i] = .5 * ( bins[i] + bins[i+1] )

    # rescale to outsides of first and last bin
    # centerbins *= histrange / 2. / centerbins[-1]

    # smooth input pdf using moving median filter to remove spike values
    imhistmean = np.empty(len(imhist), dtype=float)
    imhistmedian = np.empty(len(imhist), dtype=float)
    deltacdf = np.empty(len(imhist), dtype=float)

    for i in range(len(imhist)):
        indexmin = max(0, i - 2)
        indexmax = min(i + 2, len(imhist))
        imhistmedian[i] = np.median(imhist[indexmin:indexmax])

    if imhistmedian[imhistmedian != 0].shape[0] == 0:
        imhistmedian = imhist

    imhistmedian[0] = 0.0
    cdf = imhistmedian.cumsum()  # cumulative distribution function

    if cdf[-1] == 0:
        print(" ... in histeqDerive... bins = ", bins)
        print(" ... in histeqDerive... imhist = ", imhist)

    # normalize cdf
    cdf /= cdf[-1]

    ###
    ### create standard normal distribution to which to transform
    ###
    fit_vals = np.random.normal(loc=0.0, scale=1.0, size=_data.flatten().shape[0])
    fit_vals = np.hstack((fit_vals, -fit_vals))

    # find biggest abs value and make data range double that
    # (symmetrical about zero)
    # These are the bin boundaries
    fit_histrange = fit_vals.max() - fit_vals.min()
    fit_inputbins = np.arange(
        fit_vals.min(), fit_vals.max(), fit_histrange / (nbr_bins - 1)
    )
    fit_imhist, fit_bins = histogram(fit_vals, bins=nbr_bins, density=True)

    fit_imhist[0] = 0.0
    fit_imhist[-1] = 0.0

    # calculate center of bins
    fit_centerbins = np.linspace(fit_vals.min(), fit_vals.max(), nbr_bins)
    # fit_centerbins  = np.empty(len(fit_imhist), dtype=float)
    # for i in range(len(fit_bins)-1):
    #    fit_centerbins[i] = .5 * ( fit_bins[i] + fit_bins[i+1] )

    # normalized cdf
    fit_cdf = fit_imhist.cumsum()
    fit_cdf /= fit_cdf[-1]

    # compute deltacdf
    f = interp1d(fit_cdf, cdf)
    normed_centerbins = centerbins - centerbins.min()
    normed_centerbins /= normed_centerbins.max()
    deltacdf = f(normed_centerbins)
    equality_line = np.linspace(0.0, 1.0, cdf.shape[0])
    deltacdf = deltacdf - equality_line

    # calculate target amplitude
    # (force it to pass through zero and be symmetric)
    target_centerbins = fit_centerbins + deltacdf * (
        fit_centerbins.max() - fit_centerbins.min()
    )
    mirrored = -target_centerbins[::-1]
    target_centerbins = (target_centerbins + mirrored) / 2.0

    # compute output with value that approximately
    # fit standard-normal distribution (including kurtosis)
    f = interp1d(centerbins, target_centerbins)

    # ensure stdev is 1.0
    im_std = f(_data).std()

    return centerbins, target_centerbins / im_std


def _apply_standard_normal(im, centerbins, target_centerbins, verbose=False):
    ##############################################################
    ##    The function derives the equalization array to scale
    ##    the values in input array 'im' based on
    ##    a converson of its histogram into a uniform distribution.
    ##    When 'pcteq' is 100%, the output will have a uniform
    ##    distribution. When 'pcteq' is 0%, the function returns the
    ##    input array unchanged. 'pcteq' = 50% returns an array
    ##    with half the impact of conversion to a uniform distribution.
    ##
    ##    only applies transform curve from _derive_standard_normal
    ##############################################################

    from scipy.interpolate import interp1d

    if verbose:
        print(" Applying equalizing transform to standard normal shape")

    # compute output with value that approximately fit
    # standard-normal distribution (including kurtosis)
    f = interp1d(centerbins, target_centerbins)

    # output array
    output_array = f(im)

    return output_array


def load_centerbins(centerbin_data, include_mean=False):
    print("Loading centerbin data from: " + centerbin_data)
    sys.stdout.flush()
    data = np.load(centerbin_data)
    centerbins = data["centerbins"]
    target_centerbins = data["target_centerbins"]
    if include_mean == True:
        data_mean = data["mean"]
        print("Done.")
        sys.stdout.flush()
        return centerbins, target_centerbins, data_mean
    else:
        print("Done.")
        sys.stdout.flush()
        return centerbins, target_centerbins


def normalize_seismic(
    seismic,
    verbose=False,
    centerbin_data=None,
    save_file=None,
    load_mean_and_cbs=False,
    return_stats=False,
    stats_in=None,
):
    """
    Normalise a 4D volume numpy array
    """

    if stats_in != None:
        centerbins, target_centerbins, seismic_mean = stats_in
        seismic -= seismic_mean

    elif load_mean_and_cbs == True:
        centerbins, target_centerbins, seismic_mean = load_centerbins(
            centerbin_data, include_mean=True
        )
        seismic -= seismic_mean

    else:
        seismic_mean = seismic.mean()
        seismic -= seismic_mean
        if centerbin_data is None:
            centerbins, target_centerbins = _derive_standard_normal(seismic)
        else:
            centerbins, target_centerbins = load_centerbins(centerbin_data)

    if verbose:
        print(
            "mean/std/kurtosis -- before: ",
            seismic_mean,
            seismic.std(),
            kurtosis(seismic.flatten()),
        )
        sys.stdout.flush()

    if save_file is not None:
        print("Saving centerbin data to: " + save_file)
        sys.stdout.flush()

        np.savez(
            save_file,
            centerbins=centerbins,
            target_centerbins=target_centerbins,
            mean=seismic_mean,
        )
        print("Done.")
        sys.stdout.flush()

    seismic = _apply_standard_normal(seismic, centerbins, target_centerbins)
    if verbose:
        print(
            "mean/std/kurtosis -- after:  ",
            seismic.mean(),
            seismic.std(),
            kurtosis(seismic.flatten()),
        )
        sys.stdout.flush()
    if return_stats == True:
        return centerbins, target_centerbins, seismic_mean
    else:
        return seismic
