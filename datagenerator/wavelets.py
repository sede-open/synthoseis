"""Generate wavelets from filters derived from real data"""
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft
from scipy.interpolate import UnivariateSpline
from datagenerator.util import import_matplotlib


def fs_arr(fs):
    """Split input data into array of frequencies and array of amplitudes"""
    freq_values = fs[:, 0]
    db_levels = fs[:, 1]
    freq_list_arr = np.array(list(freq_values))
    db_list_arr = np.array(list(db_levels))
    for i in range(db_list_arr.shape[0]):
        db_list_arr[i, :] = 10.0 * np.log10(db_list_arr[i, :])
        db_list_arr[i, :] -= db_list_arr[i, :].max()
    return freq_list_arr, db_list_arr


def percentiles(freq_list_arr, db_list_arr, percentile_list):
    """Calculate percentiles of frequencies and amplitudes"""
    plot_freqs = []
    plot_dbs = []
    for ipctile in percentile_list:
        plot_freqs.append(np.mean(freq_list_arr, axis=0))
        dbs = np.percentile(db_list_arr, ipctile, axis=0)
        dbs -= dbs.max()
        plot_dbs.append(dbs)
    return plot_freqs, plot_dbs


def passband_freqs(plot_freqs_arr, plot_dbs_arr, verbose):
    """Get passband frequencies"""
    upper_passband_index = np.argmax(np.std(plot_dbs_arr, axis=0))
    mid_freq_index = np.argmin(
        np.std(plot_dbs_arr[:, : upper_passband_index + 1], axis=0)
    )
    low_bandlimit_index = np.argmax(
        np.std(plot_dbs_arr[:, : mid_freq_index + 1], axis=0)
    )
    hi_std = np.max(np.std(plot_dbs_arr[:, upper_passband_index:], axis=0))
    lo_std = np.min(np.std(plot_dbs_arr[:, upper_passband_index:], axis=0))
    clipped_std = np.std(plot_dbs_arr[:, upper_passband_index:], axis=0).clip(
        lo_std + 0.1 * (hi_std - lo_std), hi_std
    )
    hi_bandlimit_index = np.argmin(clipped_std) + upper_passband_index
    passband_low_freq = plot_freqs_arr[0, low_bandlimit_index]
    passband_mid_freq = plot_freqs_arr[0, mid_freq_index]
    passband_hi_freq = plot_freqs_arr[0, hi_bandlimit_index]

    if verbose:
        print(
            f"(low, mid, hi) frequencies = {passband_low_freq:.4f}, {passband_mid_freq:.4f}, {passband_hi_freq:.4f}"
        )
    return passband_low_freq, passband_mid_freq, passband_hi_freq


def hanflat(inarray, pctflat):
    """
    #   Function applies a Hanning taper to ends of "inarray".
    #   Center "pctflat" of samples remain unchanged.
    #
    #   Parameters:
    #   array :       array of values to have ends tapered
    #   pctflat :     this percent of  samples to remain unchanged (e.g. 0.80)
    """
    numsamples = len(inarray)
    lowflatindex = int(round(numsamples * (1.0 - pctflat) / 2.0))
    hiflatindex = numsamples - lowflatindex

    # print 'length of array, start, end ',len(inarray),lowflatindex,hiflatindex

    # get hanning for numsamples*(1.0-pctflat)
    hanwgt = np.hanning(len(inarray) - (hiflatindex - lowflatindex))

    # piece together hanning weights at ends and weights=1.0 in flat portion
    outarray = np.ones(len(inarray), dtype=float)
    outarray[:lowflatindex] = hanwgt[:lowflatindex]
    outarray[hiflatindex:] = hanwgt[numsamples - hiflatindex :]

    return outarray


def ricker(f, dt, convolutions=2):
    """
    input frequency and sampling interval (sec)
    """

    lenhalf = 1250.0 / f
    halfsmp = int(lenhalf / dt) + 1
    lenhalf = halfsmp * dt
    t = np.arange(-lenhalf, lenhalf + dt, dt) / 1000.0
    s = (1 - 2 * np.pi ** 2 * f ** 2 * t ** 2) * np.exp(-(np.pi ** 2) * f ** 2 * t ** 2)
    t *= 1000.0
    for _ in range(convolutions - 1):
        s = np.convolve(s, s, mode="full")
    # i_shorten1,i_shorten2 = shortenwavlet(s,.9999,method="strip")
    # print "sinc length = ", s[i_shorten1:i_shorten2].shape
    # print "sinc peak at ", np.argmax(s[i_shorten1:i_shorten2])
    # return t,s[i_shorten1:i_shorten2+1]
    hanningwindow = hanflat(s, 0.50)
    return t, s * hanningwindow


def default_fft(digi=4, convolutions=4):
    """Setup a default fft"""
    Nyquist = 0.5 * 1000.0 / digi
    _, s = ricker(0.12 * Nyquist, digi, convolutions=convolutions)
    return s, fftfreq(len(s), d=digi / 1000.0)


def splineinfill(x, y, newx, damp=0.0):
    """
    #   Function to return data spaced at locations specified by input variable 'newx' after
    #   fitting cubic spline function. Input data specified by 'x' and 'y' so data
    #   can be irregularly sampled.
    """
    s = UnivariateSpline(x, y, s=damp)
    return s(newx)


def calculate_wavelet_in_time_domain(ampls):
    """Generate wavelet in time domain from amplitudes and frequencies"""
    ampli_spectrum_linear = (10.0 ** (ampls / 10.0)).astype("complex")
    ampli_spectrum_linear = np.roll(
        ampli_spectrum_linear, ampli_spectrum_linear.shape[0] // 2
    )
    new_wavelet = np.real(ifft(ampli_spectrum_linear))
    new_wavelet = np.roll(new_wavelet, new_wavelet.size // 2)
    # apply hanning taper at beginning and end of time-domain wavelet
    new_wavelet = new_wavelet * hanflat(new_wavelet, 0.60)
    return ampli_spectrum_linear, new_wavelet


def generate_wavelet(fs, pcts, digi=4.0, convolutions=4, verbose=False):
    """Generate a wavelet using interpolated percentiles of the spectrum"""
    freq_list_arr, db_list_arr = fs_arr(fs)
    percentile_list = [5, 10, 25, 50, 75, 90, 95]
    plot_freqs, plot_dbs = percentiles(freq_list_arr, db_list_arr, percentile_list)
    plot_freqs_arr = np.array(plot_freqs)
    plot_dbs_arr = np.array(plot_dbs)

    passbands = passband_freqs(plot_freqs_arr, plot_dbs_arr, verbose=verbose)

    # set up default fft
    s, freq = default_fft(digi=digi, convolutions=convolutions)
    s /= np.sum(np.sqrt(s ** 2))  # normalize
    ampli_spectrum = abs(fft(s))
    ampli_spectrum = fft(s)
    freq = np.roll(freq, freq.shape[0] // 2)
    ampli_spectrum = np.roll(ampli_spectrum, ampli_spectrum.shape[0] // 2)

    __freqs = plot_freqs[0] * 1.0
    ___freqs = np.hstack((0.0, plot_freqs[0], (freq[-2] + freq[-1]) / 2.0, freq[-1]))
    ___freqs = np.hstack((-___freqs[1:][::-1], ___freqs))

    # low_freq_pctile = np.random.uniform(0, 100)
    # mid_freq_pctile = np.random.uniform(0, 100)
    # high_freq_pctile = np.random.uniform(0, 100)
    freq_pctiles = np.interp(__freqs, passbands, pcts)
    ampls = []
    for ifreq in range(__freqs.size):
        ampls.append(np.percentile(db_list_arr[:, ifreq], freq_pctiles[ifreq], axis=0))

    ampls = np.array(ampls)
    ampls -= ampls.max()
    ampls_with_edges = np.hstack(
        (-40.0, -40.0, ampls[::-1], -40.0, ampls, -40.0, -40.0)
    )

    # interpolate onto freqs in fft
    try:
        ____dbs = splineinfill(___freqs, ampls_with_edges, freq, damp=0.0)
    except:
        ____dbs = np.interp(freq, ___freqs, ampls_with_edges)

    # generate wavelet in time domain
    ampli_spectrum, new_wavelet = calculate_wavelet_in_time_domain(____dbs)
    freqs = [freq, __freqs, ___freqs]
    amps = [ampli_spectrum, ampls, ampls_with_edges]
    return freqs, amps, new_wavelet


def plot_wavelets(freqs, ampls, wavs, labels, savepng=None):
    """Plot wavelet and spectra"""
    plt = import_matplotlib()
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for f, a, w, l in zip(freqs, ampls, wavs, labels):
        axs[0].plot(w, label=l)
        axs[1].plot(
            f[0][f[0].size // 2 :],
            10.0 * np.log10(np.real(a[0])[f[0].size // 2 :])[::-1],
            label=l,
        )
    for a in axs.ravel():
        a.grid()
        a.legend(loc="upper right")
    if savepng is not None:
        fig.savefig(savepng)
