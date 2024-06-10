import os
import re
import ctypes
import shutil
import subprocess

from astropy.time import Time
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from matplotlib.colors import Normalize, LogNorm
from scipy.constants import speed_of_light
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time, TimeDelta
from scipy.ndimage import sobel, median_filter, gaussian_filter, minimum_filter, maximum_filter
from scipy.interpolate import interp1d, splrep, BSpline
from astropy.io import fits
import cv2
import numpy as np
from scipy.optimize import curve_fit
import astropy.units as u
import astropy.time as atime
from scipy.signal import argrelextrema

VIEW_DEBUG_PLOTS = False
USE_MARKOV = True


def splitname(name):
    allsplit = name.split("_")
    if len(allsplit) < 3:
        return allsplit
    return "_".join(allsplit[:-1]), allsplit[-1]


def load_spectrum(filename, preserve_below_zero=False, no_meta=False):
    """
    :param filename: Spectrum File location
    :return: Spectrum Wavelengths, Corresponding flux, time of observation, flux error (if available)
    """
    # Modify this to account for your specific needs!
    data = np.loadtxt(filename, comments="#", delimiter=" ")
    wavelength = data[:, 0]
    flux = data[:, 1]
    flux_std = data[:, 2]

    mask = flux > 0
    wavelength = wavelength[mask]
    flux_std = flux_std[mask]
    flux = flux[mask]
    if not no_meta:
        filename_prefix, nspec = splitname(filename)
        nspec = nspec.replace(".txt", "")
        nspec = int(nspec)
        try:
            t = atime.Time(np.loadtxt(splitname(filename)[0] + "_mjd.txt", comments="#",
                                      delimiter=" ")[nspec - 1], format="mjd")
        except IndexError:
            t = atime.Time(np.loadtxt(splitname(filename)[0] + "_mjd.txt", comments="#",
                                      delimiter=" "), format="mjd")

        return wavelength, flux, t, flux_std
    else:
        return wavelength, flux, flux_std


COADD_SIDS = []  # [2806984745409075328]
N_COADD = 2
SKYFLUXSEP = 150

BALMER_LINES = [6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397]

modes = {
    "SOAR_930": [3603.075643191824, 0.8497538883401163, -7.2492132499996624e-06, 2.007096874944881e-10],  # original
    "SOAR_2100_red": [6033.248234134633, 0.3035016078753955, -1.0030882233491887e-05, 0],  # red
    "SOAR_2100_blue": [4730.005531248959, 0.33842714010297165, -1.9395755345432072e-05, 1.0245482532434017e-08],  # blue
    # [3594.262791381754, 0.839510510739731, 6.181295356216964e-06, -4.739555115787775e-09]  # 9 12
    # [3622.911239892657, 0.8271359098015762, -5.86289033680784e-06, 0]  # 20.1.2024
    # [3597.075643191824, 0.8407538883401163, -7.2492132499996624e-06, 2.007096874944881e-10] # 11.2.2024
}


def detect_spectral_area(flats_image):
    # Edge detection
    minumum_truncation = 5

    image_data = flats_image.astype(np.float64)[minumum_truncation:-minumum_truncation,
                 minumum_truncation:-minumum_truncation]

    x_img = sobel(image_data, axis=0, mode="nearest")
    y_img = sobel(image_data, axis=1, mode="nearest")

    edge_detection = np.sqrt(x_img ** 2 + y_img ** 2)
    edge_detection *= 1 / np.max(edge_detection)
    edge_detection[edge_detection > 0.075] = 1
    edge_detection[edge_detection < 1] = 0

    edge_detection = (255 * edge_detection / edge_detection.max()).astype(np.uint8)

    lines = cv2.HoughLinesP(edge_detection, 1, np.pi / 180, 50, None, 500, 0)

    x_intercepts = []
    y_intercepts = []
    # Loop through the detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edge_detection, (x1, y1), (x2, y2), (255,), 1)

        if y2 == y1:
            y_intercepts.append(y1)
            continue

        if x2 == x1:
            x_intercepts.append(x1)
            continue

        m = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - m * x1
        x_intercept = -y_intercept / m if m != 0 else None

        if x_intercept is not None:
            if 0 < x_intercept < edge_detection.shape[0]:
                y_intercepts.append(x_intercept + minumum_truncation)

        if 0 < y_intercept < edge_detection.shape[1]:
            x_intercepts.append(y_intercept + minumum_truncation)

    x_intercepts = np.array(x_intercepts)
    y_intercepts = np.array(y_intercepts)

    u_x = x_intercepts[x_intercepts > edge_detection.shape[1] / 2]
    l_x = x_intercepts[x_intercepts < edge_detection.shape[1] / 2]

    u_y = y_intercepts[y_intercepts > edge_detection.shape[0] / 2]
    l_y = y_intercepts[y_intercepts < edge_detection.shape[0] / 2]

    if len(u_x) == 0:
        u_x = image_data.shape[1] - 5
    else:
        u_x = np.min(u_x) - 5
    if len(l_x) == 0:
        l_x = 5
    else:
        l_x = np.max(l_x) + 5
    if len(u_y) == 0:
        u_y = image_data.shape[0] - 5
    else:
        u_y = np.min(u_y) - 5
    if len(l_y) == 0:
        l_y = 5
    else:
        l_y = np.max(l_y) + 5

    return [(l_x, u_x), (l_y, u_y)]


def crop_image(image, xlim, ylim):
    return image[ylim[0]:ylim[1], xlim[0]:xlim[1]]


def get_central_wavelength(gra_angle, cam_angle, d_grating):
    return (np.sin(gra_angle * 2 * np.pi / 360) + np.sin((cam_angle - gra_angle) * 2 * np.pi / 360)) / (
                d_grating * 1.e-7)


def merge_dicts(*dicts):
    merged_dict = {}
    count_dict = {}

    for d in dicts:
        for key, value in d.items():
            if isinstance(value, (int, float)):
                merged_dict[key] = merged_dict.get(key, 0) + value
                count_dict[key] = count_dict.get(key, 0) + 1
            elif key not in merged_dict:
                merged_dict[key] = value

    for key in merged_dict:
        if isinstance(merged_dict[key], (int, float)):
            merged_dict[key] /= count_dict[key]

    return merged_dict


def create_master_image(image_list, hdu_id, bounds, master_bias=None, master_continuum=None, return_header=False):
    hdul = fits.open(image_list[0])
    image_data = crop_image(hdul[hdu_id].data, *bounds)
    headers = [dict(hdul[hdu_id].header)]

    master = np.zeros(image_data.shape, dtype=np.uint32)
    master += image_data
    for image in image_list[1:]:
        hdul = fits.open(image)
        image = crop_image(hdul[hdu_id].data, *bounds)
        headers.append(dict(hdul[hdu_id].header))
        if master_bias is not None:
            image[image < master_bias] = 0
            image[image >= master_bias] = (image - master_bias)[image >= master_bias]  #
        master += image

    master //= len(image_list)

    if master_continuum is not None:
        master = master.astype(np.float64)
        master /= master_continuum
        master /= master.max()
        master *= 65535

    master = master.astype(np.uint16)

    master_header = merge_dicts(*headers)

    if not return_header:
        return master, None
    else:
        return master, master_header


def create_master_flat(image_list, second_image_list, hdu_id, master_bias=None, bounds=None):
    if bounds is None:
        image_data = fits.open(image_list[0])[hdu_id].data
    else:
        image_data = crop_image(fits.open(image_list[0])[hdu_id].data, *bounds)
    master = np.zeros(image_data.shape, dtype=np.float64)
    master2 = np.copy(master)
    master += image_data
    for image in image_list[1:]:
        image = fits.open(image)[hdu_id].data
        if bounds is not None:
            image = crop_image(image, *bounds)
        if master_bias is not None:
            image[image < master_bias] = 0
            image[image >= master_bias] = (image - master_bias)[image >= master_bias]
        master += image

    for image in second_image_list:
        image = fits.open(image)[hdu_id].data
        if bounds is not None:
            image = crop_image(image, *bounds)
        if master_bias is not None:
            image[image < master_bias] = 0
            image[image >= master_bias] = (image - master_bias)[image >= master_bias]
        master2 += image

    # Create smooth master, divide by that to get rid of high frequency noise

    master *= 1 / master.max()
    master2 *= 1 / master2.max()

    if bounds is not None:
        # Get rid of littrow ghost
        center_diff = np.median(master) - np.median(master2)
        master2 += center_diff

    master = np.minimum(master, master2)
    # master *= 1 / master.max()

    smooth_master = median_filter(master, 25)

    if bounds is not None:
        master /= smooth_master
        smooth_master /= smooth_master.max()
        return master, smooth_master
    else:
        master /= master.max()
        return master


def gaussian(x, a, mean, std_dev, h):
    return a / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) + h


def gaussian_with_tilt(x, a, mean, std_dev, h, b):
    return a / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) + h + x * b


def lo(x, gamma, x_0):
    return 1 / np.pi * (gamma / 2) / ((x - x_0) ** 2 + (gamma / 2) ** 2)


log_two = np.log(2)


def ga(x, gamma, x_0):
    sigma = gamma / (2 * np.sqrt(2 * log_two))
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp((-(x - x_0) ** 2) / (2 * sigma ** 2))


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    g = ga(x, gamma, shift)
    l = lo(x, gamma, shift)
    return -scaling * (eta * g + (1 - eta) * l) + slope * x + height


def line(x, w, v, u, m, n):
    return w * x ** 4 + v * x ** 3 + u * x ** 2 + x * m + n
    # return v * x ** 3 + u * x ** 2 + x * m + n


def lowpoly(x, u, m, n):
    return u * x ** 2 + x * m + n


def get_flux(image, x_ind, y_ind, width):
    fluxsum = np.sum(image[int(np.ceil(y_ind - width)):int(np.floor(y_ind + width)), int(x_ind)])
    upperfraction = image[int(np.ceil(y_ind - width)) - 1, int(x_ind)] * (
        np.abs(np.ceil(y_ind - width) - (y_ind - width)))
    lowerfraction = image[int(np.floor(y_ind + width)) + 1, int(x_ind)] * (
        np.abs(np.floor(y_ind + width) - (y_ind + width)))
    fluxsum = upperfraction + lowerfraction + fluxsum
    return fluxsum


def get_fluxfraction(image, x_ind, y_ind, width):
    upperfraction = image[int(np.ceil(y_ind - width)) - 1, int(x_ind)] * (
        np.abs(np.ceil(y_ind - width) - (y_ind - width)))
    lowerfraction = image[int(np.floor(y_ind + width)) + 1, int(x_ind)] * (
        np.abs(np.floor(y_ind + width) - (y_ind + width)))

    return upperfraction, lowerfraction, len(
        image[int(np.ceil(y_ind - width)):int(np.floor(y_ind + width)), int(x_ind)])


class WavelenthPixelTransform():
    def __init__(self, wstart, dwdp=None, dwdp2=None, dwdp3=None, dwdp4=None, polyorder=3):
        self.wstart = wstart  # wavelength at pixel 0
        self.dwdp = dwdp  # d(wavelength)/d(pixel)
        self.dwdp2 = dwdp2  # d(wavelength)^2/d(pixel)^2
        self.dwdp3 = dwdp3  # d(wavelength)^3/d(pixel)^3
        self.dwdp4 = dwdp4  # d(wavelength)^4/d(pixel)^4
        self.polyorder = polyorder

    def wl_to_px(self, wl_arr):
        pxspace = np.linspace(0, 2500, 2500)
        f = interp1d(self.px_to_wl(pxspace), pxspace, bounds_error=False, fill_value="extrapolate")
        return f(wl_arr)

    def px_to_wl(self, px_arr):
        if self.polyorder == 4:
            return line(px_arr, self.dwdp4, self.dwdp3, self.dwdp2, self.dwdp, self.wstart)
        elif self.polyorder == 3:
            return self.wstart + self.dwdp * px_arr + self.dwdp2 * px_arr ** 2 + self.dwdp3 * px_arr ** 3


def wlshift(wl, vel_corr):
    # wl_shift = vel_corr/speed_of_light * wl
    # return wl+wl_shift
    return wl / (1 - (vel_corr / (speed_of_light / 1000)))


def fluxstatistics(wl, flux):
    med = median_filter(flux, 5)
    flux_norm = flux / med - 1
    std = pd.Series(flux_norm).rolling(min_periods=1, window=20, center=True).std().to_numpy()

    # plt.plot(flux_norm)
    # plt.plot(3*std)
    # plt.tight_layout()
    # plt.show()

    flux = flux[flux_norm < 3 * std]
    wl = wl[flux_norm < 3 * std]

    med = median_filter(flux, 5)
    flux_norm = flux / med - 1
    std = pd.Series(flux_norm).rolling(min_periods=1, window=20, center=True).std().to_numpy()

    # plt.plot(flux_norm)
    # plt.plot(3 * std)
    # plt.tight_layout()
    # plt.show()

    flx_std = flux * std

    # plt.plot(wl, flux)
    # plt.fill_between(wl, flux-flx_std, flux+flx_std, color="red", alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    return wl, flux, flx_std


def px_to_wl(px_arr, a, b, c, d):
    return a + b * px_arr + c * px_arr ** 2 + d * px_arr ** 3


def polynomial_three(px_arr, a, b, c):  # , d):
    return a + b * px_arr + c * px_arr ** 2  # + d * px_arr ** 3


def markov_gaussian(x, amp, mean, std):
    return amp * np.exp(-(x - mean) ** 2 / (2 * std ** 2))


def get_montecarlo_results():
    i = 0
    data = np.genfromtxt(f"mcmkc_output0.txt", delimiter=",")
    while os.path.isfile(f"mcmkc_output{i+1}.txt"):
        data_append = np.genfromtxt(f"mcmkc_output{i+1}.txt", delimiter=",")
        data = np.concatenate([data, data_append])
        i += 1

    threshold = np.percentile(data[:, -1], 1)
    data = data[data[:, -1] < threshold]

    params = []

    for i in range(4):
        hist, bin_edges = np.histogram(data[:, i], weights=1/data[:, -1], bins=int(np.sqrt(len(data[:, -1]))))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if i == 0:
            threshold = 0.05 * np.max(hist)
            valid_bins = hist >= threshold

            # Step 3: Determine the range of data corresponding to these bins
            if np.any(valid_bins):
                min_edge = bin_edges[np.where(valid_bins)[0][0]]
                max_edge = bin_edges[np.where(valid_bins)[0][-1] + 1]

                # Step 4: Filter the data array to be within this range
                data = data[(data[:, i] >= min_edge) & (data[:, i] <= max_edge)]

                print(int(np.sqrt(len(data[:, -1]))))
                # Step 5: Re-bin the filtered data
                hist, bin_edges = np.histogram(data[:, i], weights=1/data[:, -1], bins=int(np.sqrt(len(data[:, -1]))))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit the Gaussian to the histogram data
        popt, pcov = curve_fit(markov_gaussian, bin_centers, hist,
                               p0=[np.max(hist), bin_centers[np.argmax(hist)], np.std(data[:, i])], maxfev=1000000)

        # Extract the fitting parameters and their errors
        amp, mean, std = popt
        amp_err, mean_err, std_err = np.sqrt(np.diag(pcov))

        # Print the fitting parameters and their errors
        print(f"Parameter {i}")
        print(f"Amplitude: {amp} ± {amp_err}")
        print(f"Mean: {mean} ± {mean_err}")
        print(f"Standard Deviation: {std} ± {std_err}")

        params.append(mean)
        if VIEW_DEBUG_PLOTS:
            plt.hist(data[:, i], weights=1/data[:, -1], bins=int(np.sqrt(len(data[:, -1]))), alpha=0.6, label='Data')
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
            y_fit = markov_gaussian(x_fit, *popt)
            plt.plot(x_fit, y_fit, color='red', label='Gaussian fit')
            plt.xlabel('Data')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
    return params


def extract_spectrum(image_path, master_bias, master_flat, crop, master_comp, mjd, location, ra, dec, comp_header, n_samp,
                     accept_param, compparams=None, hglamp=False):

    arlamp = False
    if comp_header["LAMP_HGA"] == "TRUE":
        hglamp = True

    if comp_header["LAMP_AR"] == "TRUE":
        arlamp = True


    if "930" in comp_header["GRATING"]:
        d_grating = 930.
    elif "2100" in comp_header["GRATING"]:
        d_grating = 2100.
    else:
        d_grating = int(re.search(r'\d+', comp_header["GRATING"]).group())

    central_wl = get_central_wavelength(comp_header["GRT_ANG"], comp_header["CAM_ANG"], d_grating)

    if os.name == "nt":
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    image = fits.open(image_path)[0].data.astype(np.uint16)

    image = crop_image(image, *crop)

    image[image < master_bias] = 0
    image[image >= master_bias] = (image - master_bias)[image >= master_bias]

    image = np.floor_divide(image.astype(np.float64), master_flat)
    image *= 65535 / image.max()
    image = image.astype(np.uint16)

    # plt.imshow(image, cmap="Greys_r", zorder=1, norm=Normalize(vmin=0, vmax=750))
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    ycenters = []
    xcenters = []
    width = []

    # 0.82 to 0.86 Angströms per pixel is usual for the SOAR 930 grating

    for i in np.linspace(10, image.shape[1] - 10, 20):
        data = np.min(image[:, int(i - 5):int(i + 5)], axis=1)
        data = median_filter(data, 5)

        if np.all(data < 5):
            continue

        xarr = np.arange(len(data))

        params, _ = curve_fit(gaussian,
                              xarr,
                              data,
                              p0=[
                                  5 * np.max(data), len(data) / 2, 20, 0
                              ],
                              bounds=[
                                  [0, len(data) * 1 / 4, -np.inf, -np.inf],
                                  [np.inf, len(data) * 3 / 4, np.inf, np.inf]
                              ],
                              maxfev=100000)

        width.append(params[2])
        xcenters.append(int(i))
        ycenters.append(params[1])

        # plt.plot(xarr, data)
        # plt.plot(xarr, gaussian(xarr, *params))
        # # plt.plot(xarr, gaussian(xarr, *[5*np.max(data), len(data) / 2, 20, 0]))
        # plt.show()

    width = 2 * np.mean(width)
    params, _ = curve_fit(lowpoly,
                          xcenters,
                          ycenters,
                          p0=[0, np.mean(np.diff(ycenters) / np.diff(xcenters)), np.mean(ycenters)])

    xcenters = np.array(xcenters)
    ycenters = np.array(ycenters)

    resids = np.abs(ycenters - lowpoly(xcenters, *params))
    outsidestd = resids > 2 * np.std(resids)
    if np.sum(outsidestd.astype(int)) > 0 and not len(outsidestd) > 0.5 * len(xcenters):
        params, _ = curve_fit(lowpoly,
                              xcenters[~outsidestd],
                              ycenters[~outsidestd],
                              p0=[0, np.mean(np.diff(ycenters) / np.diff(xcenters)), np.mean(ycenters)])

    if VIEW_DEBUG_PLOTS:
        xspace = np.linspace(0, image.shape[1], 1000)
        fig, axs = plt.subplots(2, 1, figsize=(4.8 * 16 / 9, 4.8))
        axs[0].plot(xspace, lowpoly(xspace, *params), zorder=1)
        axs[0].scatter(xcenters, ycenters, color="red", marker="x", zorder=5)
        axs[1].imshow(image, cmap="Greys_r", norm=LogNorm(1, 1000))
        axs[1].plot(xspace, lowpoly(xspace, *params), color="lime", linewidth=0.5)
        axs[1].plot(xspace, lowpoly(xspace, *params) - SKYFLUXSEP, color="red", linestyle="--", linewidth=0.5)
        axs[1].plot(xspace, lowpoly(xspace, *params) + SKYFLUXSEP, color="red", linestyle="--", linewidth=0.5)
        axs[1].plot(xspace, lowpoly(xspace, *params) + width, color="lime", linestyle="--", linewidth=0.5)
        axs[1].plot(xspace, lowpoly(xspace, *params) - width, color="lime", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    image64 = image.astype(np.float64)
    master_comp64 = master_comp.astype(np.float64)

    pixel = np.arange(image.shape[1]).astype(np.float64)
    flux = np.array([get_flux(image64, p, lowpoly(p, *params), width) for p in pixel])
    compflux = np.array([get_flux(master_comp64, p, lowpoly(p, *params), width) for p in pixel])
    uskyflx = np.array([get_flux(image64, p, lowpoly(p, *params) + SKYFLUXSEP, width) for p in pixel])
    lskyflx = np.array([get_flux(image64, p, lowpoly(p, *params) - SKYFLUXSEP, width) for p in pixel])

    # testuskyflx = np.array([get_flux(image64, p, lowpoly(p, *params) + SKYFLUXSEP, 5*width) for p in pixel])
    # testlskyflx = np.array([get_flux(image64, p, lowpoly(p, *params) - SKYFLUXSEP, 5*width) for p in pixel])
    # fractions = np.array([get_fluxfraction(image64, p, line(p, *params), width) for p in pixel])
    # uf = fractions[:, 0]
    # lf = fractions[:, 1]
    # lens = fractions[:, 2]
    #
    # plt.plot(pixel, flux)
    # plt.show()

    skyflx = np.minimum(uskyflx, lskyflx)
    flux -= skyflx

    # testskyflux = (testlskyflx+testuskyflx)/2

    if not hglamp:
        realcflux = fits.open("compspec.fits")[0]
        zeropoint = realcflux.header["CRVAL1"]
        delta = realcflux.header["CDELT1"]
        realcflux = realcflux.data
        realcflux = realcflux[:int(len(realcflux) / 2.1)]
        realcflux *= compflux.max() / realcflux.max()
        realcwl = np.arange(len(realcflux)) * delta + zeropoint
    else:
        realdata = np.genfromtxt("compspec_HgAr.txt")
        realcwl = realdata[:, 0]
        realcflux = realdata[:, 1]
        realcflux *= compflux.max() / realcflux.max()

    compflux_cont = minimum_filter(compflux, 10)
    compflux -= compflux_cont

    # kill obvious cosmics
    flux[flux > 3 * np.median(flux)] = np.nan

    realcflux = gaussian_filter(realcflux, 3)
    if not hglamp and not arlamp:
        lines = np.genfromtxt("FeAr_lines.txt", delimiter="  ")[:, 0]
    elif hglamp:
        lines = np.genfromtxt("HgAr.txt", delimiter="  ")[:, 0]
    elif arlamp:
        lines = np.genfromtxt("Nelines.txt", delimiter="  ")[:, 0]
    else:
        lines = []

    if compparams is None:
        if not USE_MARKOV:
            def call_fitlines(compspec_x, compspec_y, center, extent, quadratic_ext, cubic_ext, c_size,
                              s_size, q_size, cub_size, c_cov, s_cov, q_cov, cub_cov, zoom_fac):

                print("Finding wavelength solution, this may take some time...")
                compspec_x = np.array(compspec_x, dtype=np.double)
                compspec_y = np.array(compspec_y, dtype=np.double)
                if not hglamp:
                    # compspec_y = gaussian_filter(compspec_y, 2)
                    compspec_y /= maximum_filter(compspec_y, 50)
                elif hglamp:
                    # compspec_y = gaussian_filter(compspec_y, 2)
                    compspec_y /= maximum_filter(compspec_y, 300)
                elif arlamp:
                    # compspec_y = gaussian_filter(compspec_y, 2)
                    compspec_y /= maximum_filter(compspec_y, 50)

                if not os.path.isdir("./temp"):
                    os.mkdir("temp")
                else:
                    shutil.rmtree("./temp")
                    os.mkdir("temp")

                np.savetxt("./temp/compspec_x.txt", compspec_x, fmt="%.9e")
                np.savetxt("./temp/compspec_y.txt", compspec_y, fmt="%.9e")
                np.savetxt("./temp/lines.txt", lines, fmt="%.9e")
                np.savetxt("./temp/arguments.txt", np.array([center, extent, quadratic_ext, cubic_ext, c_size,
                                                             s_size, q_size, cub_size, c_cov, s_cov, q_cov, cub_cov,
                                                             zoom_fac, n_refine]), fmt="%.9e")

                if os.name == "nt":
                    process = subprocess.Popen(
                        "linefit temp/compspec_x.txt temp/compspec_y.txt temp/lines.txt temp/arguments.txt 0", shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
                else:
                    process = subprocess.Popen(
                        "./linefit temp/compspec_x.txt temp/compspec_y.txt temp/lines.txt temp/arguments.txt 0", shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
                for line in process.stdout:
                    print(line, end='')  # end='' prevents adding extra newlines

                # Wait for the process to finish and capture any remaining output or errors
                stdout, stderr = process.communicate()
                print(stdout)
                print(stderr)

                result = np.genfromtxt("./temp/output.txt")
                # shutil.rmtree("./temp")

                return result

            # print("px", [p for p in pixel])
            # print("cmpfl", [p for p in compflux])
            # print("cwl", central_wl)

            extent = 1700

            if "2100" in comp_header["GRATING"]:
                extent = 630

            result = call_fitlines(pixel, compflux, central_wl, extent, -7e-6, -1.5e-10, 100, 50, 100,
                                   100, 100., 0.05, 2.e-5, 2.5e-10, 10., 3)

            compparams = [result[3], result[2] / len(compflux), result[1], result[0]]

        else:
            def call_fitlines_markov(compspec_x, compspec_y, center, extent, quadratic_ext, cubic_ext,
                                     wl_stepsize, spacing_stepsize, quad_stepsize, cub_stepsize,
                                     wl_cov, spacing_cov, quad_cov, cub_cov):

                print("Finding wavelength solution, this may take some time...")
                compspec_x = np.array(compspec_x, dtype=np.double)
                compspec_y = np.array(compspec_y, dtype=np.double)

                if not hglamp:
                    compspec_y = gaussian_filter(compspec_y, 2)
                    compspec_y /= maximum_filter(compspec_y, 25)
                elif hglamp:
                    compspec_y = gaussian_filter(compspec_y, 2)
                    compspec_y /= maximum_filter(compspec_y, 300)
                elif arlamp:
                    compspec_y = gaussian_filter(compspec_y, 2)
                    compspec_y /= maximum_filter(compspec_y, 25)

                if not os.path.isdir("./temp"):
                    os.mkdir("temp")
                else:
                    shutil.rmtree("./temp")
                    os.mkdir("temp")

                np.savetxt("./temp/compspec_x.txt", compspec_x, fmt="%.9e")
                np.savetxt("./temp/compspec_y.txt", compspec_y, fmt="%.9e")
                np.savetxt("./temp/lines.txt", lines, fmt="%.9e")
                np.savetxt("./temp/arguments.txt", np.array([len(lines), len(compspec_x), n_samp, center-extent/2,
                                                             extent/len(compspec_x), quadratic_ext, cubic_ext,
                                                             wl_stepsize, spacing_stepsize, quad_stepsize,
                                                             cub_stepsize, wl_cov, spacing_cov, quad_cov, cub_cov, accept_param]), fmt="%.9e")

                if os.name == "nt":
                    process = subprocess.Popen(
                        "linefit temp/compspec_x.txt temp/compspec_y.txt temp/lines.txt temp/arguments.txt 1", shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
                else:
                    process = subprocess.Popen(
                        "./linefit temp/compspec_x.txt temp/compspec_y.txt temp/lines.txt temp/arguments.txt 1", shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
                for line in process.stdout:
                    print(line, end='')  # end='' prevents adding extra newlines

                # Wait for the process to finish and capture any remaining output or errors
                stdout, stderr = process.communicate()

                result = get_montecarlo_results()
                # shutil.rmtree("./temp")

                return result
            extent = 1725

            if "2100" in comp_header["GRATING"]:
                extent = 630

            result = call_fitlines_markov(pixel, compflux, central_wl, extent, -7e-6, 0,
                                          1, 0.001, 5.e-7, 1.e-10,
                                          150., 0.1, 5.e-5, 3.e-8)

            # extremely good solver:
            # result = call_fitlines_markov(pixel, compflux, central_wl, extent, -7e-6, 0,
            #                               0.5, 0.001, 5.e-7, 1.e-10,
            #                               150., 0.05, 5.e-5, 2.e-8,
            #                               1000000)

            compparams = [result[0], result[1], result[2], result[3]]


    wpt = WavelenthPixelTransform(*compparams)

    # velrange = np.linspace(-150, 150, 200)
    #
    # lines = np.genfromtxt("/home/fabian/PycharmProjects/RVVD_plus_LAMOST/G_star_lines.txt")# np.array([6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3888.052, 3835.387, 4026.19, 4471.4802,
    #         #          4921.9313, 5015.678, 5875.6, 6678.15, 4541.59, 4685.70, 5411.52])
    #
    # velresid = []
    # basewl = wpt.px_to_wl(pixel)
    # for v in velrange:
    #     thiswl = wlshift(basewl, -v)
    #     interpolator = interp1d(thiswl, flux, bounds_error=False, fill_value=0)
    #     velresid.append(np.sum(interpolator(lines)))
    #
    # velresid = np.array(velresid)
    #
    # params, errs = curve_fit(gaussian,
    #                          velrange,
    #                          velresid,
    #                          [-np.ptp(velresid) * 100, velrange[np.argmin(velresid)], 100, np.max(velresid)])
    # errs = np.sqrt(np.diag(errs))
    # print(params[1], errs[1])
    # plt.plot(velrange, gaussian(velrange, *params))
    # plt.plot(velrange, velresid)
    # plt.show()


    final_wl_arr = wpt.px_to_wl(pixel)

    if VIEW_DEBUG_PLOTS:
        # initial_guess_trafo = WavelenthPixelTransform(*compparams)
        # initial_guess_array = initial_guess_trafo.px_to_wl(pixel)
        plt.plot(realcwl, realcflux)
        plt.plot(final_wl_arr, compflux.min() - np.nanmax(flux) + flux, linewidth=1, color="gray")
        plt.plot(final_wl_arr, compflux, color="darkred")
        for b in lines:
            plt.axvline(b, linestyle="--", color="darkgreen", zorder=-5)
        # plt.plot(initial_guess_array, compflux, color="lightblue", linestyle="--")
        # plt.plot(final_wl_arr, uf)
        # plt.plot(final_wl_arr, lf)
        # plt.plot(final_wl_arr, lens*100)
        plt.show()

    # np.savetxt("testoutput/"+image_path.split("/")[-1], np.stack([final_wl_arr, testskyflux], axis=-1))

    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    barycorr = sc.radial_velocity_correction(obstime=Time(mjd, format="mjd"), location=location)
    barycorr = barycorr.to(u.km / u.s)
    barycorr = barycorr.value

    final_wl_arr = wlshift(final_wl_arr, barycorr)

    final_wl_arr, flux, flx_std = fluxstatistics(final_wl_arr, flux)

    return final_wl_arr, flux, flx_std, compparams


def gather_sinfo_from_backup(sid):
    hdul = fits.open("backupcat.fits")
    tab = Table(hdul[1].data)

    row = tab[tab["source_id"] == sid]
    if len(row) != 1:
        return None
    name = row["name"][0]

    spec_class = row["spec_class"][0]

    if not isinstance(name, str):
        name = name.decode("utf-8")
    if not isinstance(spec_class, str):
        spec_class = spec_class.decode("utf-8")
    elif spec_class == "" or spec_class is None:
        spec_class = "unknown"

    return pd.DataFrame({
        "name": [name],
        "source_id": [sid],
        "ra": [row["ra"][0]],
        "dec": [row["dec"][0]],
        "SPEC_CLASS": [spec_class],
        "bp_rp": [row["BP_RP"][0]],
        "gmag": [row["M_G"][0]],
        "nspec": [1],
        "pmra": [row["pmRA"][0]],
        "pmra_error": [row["e_pmRA"][0]],
        "pmdec": [row["pmDE"][0]],
        "pmdec_error": [row["e_pmDE"][0]],
        "parallax": [row["Plx"][0]],
        "parallax_error": [row["e_Plx"][0]],
    }).iloc[0]


def get_star_info(file):
    rename_dict = {
        "Name": "name",
        "RA_ICRS": "ra",
        "DE_ICRS": "dec",
        "GaiaEDR3": "source_id",
        "SpClass": "SPEC_CLASS",
        "BP-RP": "bp_rp",
        "GGAIA": "gmag",
        "Gmag": "gmag",
        "pmRA": "pmra",
        "pmRAGAIA": "pmra",
        "pmDE": "pmdec",
        "pmDEGAIA": "pmdec",
        "e_pmRAGAIA": "pmra_error",
        "e_pmRA": "pmra_error",
        "e_pmDEGAIA": "pmdec_error",
        "e_pmDE": "pmdec_error",
        "Plx": "parallax",
        "e_Plx": "parallax_error",
    }

    header = dict(fits.open(file)[0].header)
    ra = header["RA"]
    dec = header["DEC"]
    sky_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
    # Define the VizieR catalog ID
    catalog_id = "J/A+A/662/A40"

    # Query the VizieR catalog
    vizier = Vizier(columns=['all'], row_limit=1)
    sinfo = vizier.query_region(sky_coord, radius=30 * u.arcsec, catalog=catalog_id)

    if len(sinfo) == 0:

        # Define the coordinates
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

        # Query Gaia DR3
        width = u.Quantity(10, u.arcsecond)
        result = Gaia.query_object(coordinate=coord, radius=width)
        star = result[0]
        sinfo = {}
        print(f"WARNING: Star from file {file} not found in hot subdwarf catalogues!")
        sinfo["name"] = f"Gaia DR3 {star['SOURCE_ID']}"
        sinfo["source_id"] = star['SOURCE_ID']
        sinfo["ra"] = star['ra']
        sinfo["dec"] = star['dec']

        # Fill in other values if they exist, otherwise set to "N/A"
        sinfo["file"] = file
        sinfo["SPEC_CLASS"] = "unknown"
        sinfo["bp_rp"] = star['bp_rp'] if 'bp_rp' in star.columns else "N/A"
        sinfo["gmag"] = star['phot_g_mean_mag'] if 'phot_g_mean_mag' in star.columns else "N/A"
        sinfo["nspec"] = 1
        sinfo["pmra"] = star['pmra'] if 'pmra' in star.columns else "N/A"
        sinfo["pmra_error"] = star['pmra_error'] if 'pmra_error' in star.columns else "N/A"
        sinfo["pmdec"] = star['pmdec'] if 'pmdec' in star.columns else "N/A"
        sinfo["pmdec_error"] = star['pmdec_error'] if 'pmdec_error' in star.columns else "N/A"
        sinfo["parallax"] = star['parallax'] if 'parallax' in star.columns else "N/A"
        sinfo["parallax_error"] = star['parallax_error'] if 'parallax_error' in star.columns else "N/A"

    elif len(sinfo) > 0:
        if len(sinfo) == 1:
            sinfo = sinfo[0].to_pandas().to_dict(orient='records')[0]
            sinfo["name"] = "-"
        else:
            sinfo = sinfo[1].to_pandas().to_dict(orient='records')[0]
            sinfo["name"] = "-"
            sinfo["bp_rp"] = sinfo["BPGAIA"] - sinfo["RPGAIA"]
    else:
        sinfo = sinfo[0].to_pandas().to_dict(orient='records')[0]
        sinfo["name"] = "-"

    for a, b in rename_dict.items():
        if a in sinfo.keys():
            sinfo[b] = sinfo.pop(a)

    sinfo["nspec"] = 1

    if os.name == "nt":
        sinfo["file"] = file.split("/")[-1]
    else:
        sinfo["file"] = file.split("/")[-1]

    time = Time(header["DATE-OBS"], format='isot', scale='utc')
    time += TimeDelta(header["EXPTIME"], format='sec')

    return sinfo, time.mjd


def save_to_ascii(wl, flx, flx_std, mjd, trow,
                  dir=r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR",
                  outtablefile=r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv",
                  addnoncoadd=False):
    if os.path.isfile(outtablefile):
        outtable = pd.read_csv(outtablefile)
        if not trow["file"] in outtable["file"].to_list():
            outtable = pd.concat([outtable, pd.DataFrame([trow])])
    else:
        outtable = pd.DataFrame([trow])

    if addnoncoadd:
        if not os.path.isdir(dir + "_noncoadd"):
            os.mkdir(dir + "_noncoadd")
        fname = trow["file"].replace(".fits", "_01.txt")
        fname = dir + "_noncoadd/" + fname
    else:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        fname = trow["file"].replace(".fits", "_01.txt")
        fname = dir + "/" + fname

    with open(fname.replace("_01.", "_mjd."), "w") as datefile:
        datefile.write(str(mjd))

    if addnoncoadd:
        outtable.to_csv(outtablefile.replace(".csv", "_noncoadd.csv"), index=False)
    else:
        outtable.to_csv(outtablefile, index=False)
    outdata = np.stack((wl, flx, flx_std), axis=-1)
    np.savetxt(fname, outdata, fmt='%1.4f')
    return fname


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def potentialfctn(x, points):
    return -np.sum(1 / (np.abs(x - points) ** 2 + 0.1))


def pointpotentials(sample_arrays, min_val, max_val):
    potentialsample = np.linspace(min_val, max_val, len(sample_arrays[0]) * 100)
    potential = np.zeros(len(potentialsample))
    for sample in sample_arrays:
        potential += np.array([potentialfctn(x, sample) for x in potentialsample])

    interpolated_potential = interp1d(potentialsample, potential, bounds_error=False, fill_value='extrapolate')
    return potentialsample, potential, interpolated_potential


def generate_bins(sample_arrays):
    min_length = np.min([len(s) for s in sample_arrays])
    min_point = np.min([np.min(s) for s in sample_arrays])
    max_point = np.max([np.max(s) for s in sample_arrays])

    sample_arrays = np.stack([s[:min_length] for s in sample_arrays])
    bin_centers = np.median(sample_arrays, axis=0)

    pot_x, pot_y, pot_fn = pointpotentials(sample_arrays, min_point, max_point)

    def fitwrapper(bin_centers, a, b, c, d):
        new_bin_centers = bin_centers + polynomial(bin_centers, a, b, c, d)
        return pot_fn(new_bin_centers)

    params, errs = curve_fit(fitwrapper, bin_centers, np.full(len(bin_centers), pot_y.min()), p0=[0, 0, 0, 0],
                             maxfev=1000000)

    bin_centers += polynomial(bin_centers, *params)

    return bin_centers, pot_x, pot_y


def truncate_and_align_arrays(wls, flxs, flx_stds):
    first_values = [s[0] for s in wls]
    lowest_value = np.max(first_values)
    arr_with_lowest_val = np.argmax(first_values)
    arr_with_lowest_val_stepsize = wls[arr_with_lowest_val][1] - wls[arr_with_lowest_val][0]

    new_wls = []
    new_flxs = []
    new_flx_stds = []
    for f, w, fstd in zip(flxs, wls, flx_stds):
        new_flxs.append(f[w > lowest_value - arr_with_lowest_val_stepsize / 2])
        new_wls.append(w[w > lowest_value - arr_with_lowest_val_stepsize / 2])
        new_flx_stds.append(fstd[w > lowest_value - arr_with_lowest_val_stepsize / 2])

    last_values = [s[-1] for s in wls]
    highest_value = np.min(last_values)
    arr_with_highest_val = np.argmin(last_values)
    arr_with_highest_val_stepsize = wls[arr_with_highest_val][-1] - wls[arr_with_highest_val][-2]

    out_wls = []
    out_flx = []
    out_flx_std = []
    for f, w, fstd in zip(new_flxs, new_wls, new_flx_stds):
        out_flx.append(f[w < highest_value - arr_with_highest_val_stepsize / 2])
        out_wls.append(w[w < highest_value - arr_with_highest_val_stepsize / 2])
        out_flx_std.append(fstd[w < highest_value - arr_with_highest_val_stepsize / 2])

    return out_wls, out_flx, out_flx_std


def polynomial(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


def coadd_spectrum(wls, flxs, flx_stds):
    print("Truncating...")
    wls, flxs, flx_stds = truncate_and_align_arrays(wls, flxs, flx_stds)
    print("Generating bins...")
    bins, px, py = generate_bins(wls)

    global_wls = np.concatenate([np.array([0]), bins[:-1] + np.diff(bins) / 2])

    # Initialize the global flux array
    global_flxs = np.zeros(len(bins))
    global_flx_stds = np.zeros(len(bins))

    n_in_bin = np.zeros(global_flxs.shape)
    wls = np.concatenate(wls)
    flxs = np.concatenate(flxs)
    flx_stds = np.concatenate(flx_stds)

    print("Binning...")
    for i, bin in enumerate(bins):
        if i == 0:
            global_flxs[0] += np.sum(flxs[wls < bins[0]])
            global_flx_stds[0] += np.sqrt(np.sum(flx_stds[wls < bins[0]] ** 2))
        elif i == len(bins) - 1:
            global_flxs[-1] += np.sum(flxs[wls > bins[-1]])
            global_flx_stds[-1] += np.sqrt(np.sum(flx_stds[wls < bins[-1]] ** 2))
        else:
            mask = np.logical_and(wls >= bin, wls < bins[i + 1])
            if np.sum(mask) == 0:
                continue
            flx_between = flxs[mask]
            flx_std_between = flx_stds[mask]
            wls_between = wls[mask]

            if np.sum(mask) > 1:
                np.delete(flx_between, flx_std_between.argmax())
                np.delete(wls_between, flx_std_between.argmax())
                np.delete(flx_std_between, flx_std_between.argmax())

            frac_to_next_bin = (wls_between - bin) / (bins[i + 1] - bin)
            global_flxs[i] += np.sum(flx_between * (1 - frac_to_next_bin)) / len(flx_between)
            global_flxs[i + 1] += np.sum(flx_between * frac_to_next_bin) / len(flx_between)
            global_flx_stds[i] += np.sqrt(np.sum((flx_std_between * (1 - frac_to_next_bin)) ** 2)) / len(
                flx_std_between)
            global_flx_stds[i + 1] += np.sqrt(np.sum((flx_std_between * frac_to_next_bin) ** 2)) / len(flx_std_between)
            n_in_bin[i] += len(flx_between)
            n_in_bin[i + 1] += len(flx_between)

    n_in_bin /= 2
    normal_n_count = float(np.argmax(np.bincount(n_in_bin.astype(int))))

    global_flxs = global_flxs[n_in_bin == normal_n_count]
    global_flx_stds = global_flx_stds[n_in_bin == normal_n_count]
    global_wls = global_wls[n_in_bin == normal_n_count]
    n_in_bin = n_in_bin[n_in_bin == normal_n_count]

    n_in_bin = n_in_bin[1:-1]
    global_flxs = global_flxs[1:-1]
    global_wls = global_wls[1:-1]
    global_flx_stds = global_flx_stds[1:-1]
    global_flxs /= n_in_bin
    global_flx_stds /= n_in_bin

    return global_wls, global_flxs, global_flx_stds


def data_reduction(flat_list, shifted_flat_list, bias_list, science_list, comp_list, output_csv_path, output_folder,
                   n_samp, accept_param, comp_divider=3, science_divider=3,
                   coadd_chunk=False, show_debug_plot=False, hglamp=False):
    global VIEW_DEBUG_PLOTS
    if show_debug_plot:
        VIEW_DEBUG_PLOTS = True
    compparams = None
    print("Starting data reduction...")
    if os.path.isfile("saved_solutions.csv"):
        previous_solutions = pd.read_csv("saved_solutions.csv")
    else:
        previous_solutions = pd.DataFrame({"file": [], "a": [], "b": [], "c": [], "d": []})

    print("Cropping images...")
    master_flat = create_master_flat(flat_list, shifted_flat_list, 0)
    crop = detect_spectral_area(master_flat)

    if VIEW_DEBUG_PLOTS:
        plt.imshow(master_flat, cmap="Greys_r", zorder=1)
        plt.axvline(crop[0][0], color="lime", zorder=5)
        plt.axvline(crop[0][1], color="lime", zorder=5)
        plt.axhline(crop[1][0], color="lime", zorder=5)
        plt.axhline(crop[1][1], color="lime", zorder=5)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print("Creating Master Bias...")
    master_bias, _ = create_master_image(bias_list, 0, crop)
    if VIEW_DEBUG_PLOTS:
        plt.imshow(master_bias, cmap="Greys_r", zorder=1)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print("Creating Master Flat...")
    master_flat, master_continuum = create_master_flat(flat_list, shifted_flat_list, 0, master_bias=master_bias,
                                                       bounds=crop)

    print("Extracting Spectra...")
    divided_comp_list = np.split(comp_list, np.arange(comp_divider, len(comp_list), comp_divider))

    mjds = []
    flocs = []
    trows = []
    prev_comp_ind = 0
    for ind, file in enumerate(science_list):
        comp_ind = int(np.floor(ind / science_divider))
        if prev_comp_ind != comp_ind:
            compparams = None
            prev_comp_ind = comp_ind

        this_solution = previous_solutions[previous_solutions["file"] == file]
        if len(this_solution) != 0:
            compparams = [this_solution["a"].iloc[0],
                          this_solution["b"].iloc[0],
                          this_solution["c"].iloc[0],
                          this_solution["d"].iloc[0]]

        compfiles = divided_comp_list[comp_ind]  # Complamp list for this file

        master_comp, master_comp_header = create_master_image(compfiles, 0, crop, master_bias, return_header=True)

        trow, mjd = get_star_info(
            file)  # You probably need to write your own function. Trow needs to be a dict with "ra" and "dec" keys. Mjd is self-explanatory
        print(f'Working on GAIA DR3 {trow["source_id"]}...')

        cerropachon = EarthLocation.of_site('Cerro Pachon')  # Location of SOAR
        wl, flx, flx_std, compparams = extract_spectrum(
            file,
            master_bias,
            master_flat,
            crop,
            master_comp,
            mjd,
            cerropachon,
            trow["ra"],
            trow["dec"],
            master_comp_header,
            n_samp,
            accept_param,
            compparams if compparams is not None else None,
            hglamp = hglamp)

        if len(this_solution) == 0:
            previous_solutions = pd.concat([previous_solutions, pd.DataFrame({"file": [file],
                                                                              "a": [compparams[0]],
                                                                              "b": [compparams[1]],
                                                                              "c": [compparams[2]],
                                                                              "d": [compparams[3]]})])

        ordered_cols = ["name", "source_id", "ra", "dec",
                        "file", "SPEC_CLASS", "bp_rp", "gmag", "nspec",
                        "pmra", "pmra_error", "pmdec", "pmdec_error", "parallax", "parallax_error"]
        trow_new = dict([(a, b) for a, b in trow.items() if a in ordered_cols])
        trow = dict(sorted(trow_new.items(), key=lambda pair: ordered_cols.index(pair[0])))

        previous_solutions.to_csv(path_or_buf="saved_solutions.csv", sep=",", index=False)

        if coadd_chunk:
            floc = save_to_ascii(wl, flx, flx_std, mjd, trow, output_folder, output_csv_path,
                                 addnoncoadd=True)

            flocs.append(floc)
            mjds.append(mjd)
            trows.append(trow)
        else:
            save_to_ascii(wl, flx, flx_std, mjd, trow, output_folder, output_csv_path)

    flocs = np.array(flocs)
    mjds = np.array(mjds)
    for i in range(len(flocs)):
        if i % science_divider == 0:
            indices = i + np.array(range(science_divider))
            wls = []
            flxs = []
            flx_stds = []
            for f in flocs[indices]:
                data = np.genfromtxt(f)
                wls.append(data[:, 0])
                flxs.append(data[:, 1])
                flx_stds.append(data[:, 2])

            np.mean(mjds[indices])
            wl, flx, flx_std = coadd_spectrum(wls, flxs, flx_stds)
            save_to_ascii(wl, flx, flx_std, mjd, trows[i], output_folder, output_csv_path)

    print("Finished!")


# You should only need to modify
if __name__ == "__main__":
    print("Starting data reduction...")
    catalogue = pd.read_csv("all_objects_withlamost.csv")
    allfiles = sorted(os.listdir(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR"))

    print("Searching files...")
    flat_list = []  # Flats
    shifted_flat_list = []  # Flats created with a small camera tilt to get rid of the Littrow ghost
    for file in allfiles:
        if "quartz" in file and "test" not in file and "bias" not in file and "shifted" not in file:
            flat_list.append(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)
        elif "quartz" in file and "test" not in file and "bias" not in file and "shifted" in file:
            shifted_flat_list.append(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)

    bias_list = []
    for file in allfiles:
        if "bias" in file and "test" not in file:
            bias_list.append(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)

    print("Cropping images...")
    master_flat = create_master_flat(flat_list, shifted_flat_list, 0)
    crop = detect_spectral_area(master_flat)

    # plt.imshow(master_flat, cmap="Greys_r", zorder=1)
    # plt.axvline(crop[0][0], color="lime", zorder=5)
    # plt.axvline(crop[0][1], color="lime", zorder=5)
    # plt.axhline(crop[1][0], color="lime", zorder=5)
    # plt.axhline(crop[1][1], color="lime", zorder=5)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    print("Creating Master Bias...")
    master_bias, _ = create_master_image(bias_list, 0, crop)
    # plt.imshow(master_bias, cmap="Greys_r", zorder=1)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    print("Creating Master Flat...")
    master_flat, master_continuum = create_master_flat(flat_list, shifted_flat_list, 0, master_bias=master_bias,
                                                       bounds=crop)

    soardf = pd.DataFrame({
        "name": [],
        "source_id": [],
        "ra": [],
        "dec": [],
        "file": [],
        "SPEC_CLASS": [],
        "bp_rp": [],
        "gmag": [],
        "nspec": [],
    })

    print("Extracting Spectra...")
    if os.path.isfile(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv"):
        os.remove(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv")
    for file in allfiles:
        file = r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file
        if "bias" not in file and "quartz" not in file and "test" not in file and "FeAr" not in file and ".txt" not in file and "RED" not in file:
            compfiles = []  # Complamp list for this file

            if os.name == "nt":
                int_file_index = int(file.split("\\")[-1][:4])
            else:
                int_file_index = int(file.split("/")[-1][:4])

            for i in range(6):
                i += 1
                searchind = int_file_index + i
                file_index = str(int_file_index)

                if len(file_index) == 3:
                    file_index = "0" + file_index
                if len(str(searchind)) == 3:
                    searchind = "0" + str(searchind)

                cfile = file.replace(file_index, str(searchind)).replace(".fits", "_FeAr.fits")
                if os.path.isfile(cfile):
                    compfiles.append(cfile)

            master_comp, _ = create_master_image(compfiles, 0, crop, master_bias)

            trow, mjd = get_star_info(
                file)  # You probably need to write your own function. Trow needs to be a dict with "ra" and "dec" keys. Mjd is self-explanatory
            print(f'Working on index {int_file_index}, GAIA DR3 {trow["source_id"]}...')
            soardf = pd.concat([soardf, trow])

            cerropachon = EarthLocation.of_site('Cerro Pachon')  # Location of SOAR
            wl, flx, flx_std = extract_spectrum(
                file,
                master_bias,
                master_flat,
                crop,
                master_comp,
                mjd,
                cerropachon,
                trow["ra"],
                trow["dec"])
            save_to_ascii(wl, flx, flx_std, mjd,
                          trow)  # You probably need to write your own function for saving the wl and flx

    # You can ignore everything below, this is only for Coadding spectra.
    if len(COADD_SIDS) > 0:
        directory = r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR"
        labeltable = pd.read_csv(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv")

        notincoaddtable = labeltable[~labeltable["source_id"].isin(COADD_SIDS)]
        notincoaddtable.to_csv(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv",
                               index=False)

        for sid in COADD_SIDS:
            thissidlist = labeltable[labeltable["source_id"] == sid]
            filelist = thissidlist["file"].to_numpy()
            for_coadd = split_given_size(filelist, N_COADD)
            for coadd_list in for_coadd:
                n_file = coadd_list[0].replace(".fits", "_01.txt")
                trow, _ = get_star_info(
                    r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR" + "/" + coadd_list[0])
                mjds = []
                for c in coadd_list:
                    with open(directory + "/" + c.replace(".fits", "_mjd.txt")) as mjdfile:
                        mjds.append(float(mjdfile.read()))  #
                mean_mjd = np.mean(mjds)

                allflx = []
                allwl = []
                all_flx_std = []
                for f in coadd_list:
                    wl, flx, t, flx_std = load_spectrum(directory + "/" + f.replace(".fits", "_01.txt"))
                    allwl.append(wl)
                    allflx.append(flx)
                    all_flx_std.append(flx_std)

                allwl = np.vstack(allwl)
                allflx = np.vstack(allflx)
                all_flx_std = np.vstack(all_flx_std)

                allwl = np.mean(allwl, axis=0)
                allflx = np.sum(allflx, axis=0) / len(allflx)
                all_flx_std = np.sum(all_flx_std, axis=0) / len(all_flx_std)

                save_to_ascii(allwl, allflx, all_flx_std, mean_mjd, trow)
