from data_reduction import *


def extract_spectrum(image_path, master_bias, master_flat, crop, master_comp, mjd, location, ra, dec, compparams=None, modechoice="SOAR_930"):
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

    xspace = np.linspace(0, image.shape[1], 1000)
    # fig, axs = plt.subplots(2, 1, figsize=(4.8 * 16 / 9, 4.8))
    # axs[0].plot(xspace, lowpoly(xspace, *params), zorder=1)
    # axs[0].scatter(xcenters, ycenters, color="red", marker="x", zorder=5)
    # axs[1].imshow(image, cmap="Greys_r", norm=LogNorm(1, 1000))
    # axs[1].plot(xspace, lowpoly(xspace, *params), color="lime", linewidth=0.5)
    # axs[1].plot(xspace, lowpoly(xspace, *params) - SKYFLUXSEP, color="red", linestyle="--", linewidth=0.5)
    # axs[1].plot(xspace, lowpoly(xspace, *params) + SKYFLUXSEP, color="red", linestyle="--", linewidth=0.5)
    # axs[1].plot(xspace, lowpoly(xspace, *params) + width, color="lime", linestyle="--", linewidth=0.5)
    # axs[1].plot(xspace, lowpoly(xspace, *params) - width, color="lime", linestyle="--", linewidth=0.5)
    # plt.tight_layout()
    # plt.show()

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

    realcflux = fits.open("compspec.fits")[0]
    zeropoint = realcflux.header["CRVAL1"]
    delta = realcflux.header["CDELT1"]
    realcflux = realcflux.data
    realcflux = realcflux[:int(len(realcflux) / 2.1)]
    realcflux *= compflux.max() / realcflux.max()
    realcwl = np.arange(len(realcflux)) * delta + zeropoint

    compflux_cont = minimum_filter(compflux, 10)
    compflux -= compflux_cont

    # kill obvious cosmics
    flux[flux > 3 * np.median(flux)] = np.nan

    realcflux = gaussian_filter(realcflux, 3)

    if compparams is None:
        initial_minimization = np.zeros(101)
        for i in range(101):
            backtrafo = WavelenthPixelTransform(modes[modechoice][0] + i - 50, *modes[modechoice][1:])
            p_lines = backtrafo.wl_to_px(np.array(BALMER_LINES))
            initial_minimization[i] = np.sum(interp1d(pixel, flux, bounds_error=False, fill_value=0)(p_lines))

        if VIEW_DEBUG_PLOTS:
            plt.plot(np.linspace(-50, 50, 101), initial_minimization)
            plt.axvline(np.nanargmin(initial_minimization) - 50)
            plt.show()
        backtrafo = WavelenthPixelTransform(modes[modechoice][0] + np.linspace(-50, 50, 101)[np.nanargmin(initial_minimization)], *modes[modechoice][1:])
        p_lines = backtrafo.wl_to_px(np.array(BALMER_LINES))

        balmer_p_vals = []
        linevals = []
        for i, l in enumerate(p_lines):
            if pixel.min() < l < pixel.max():
                if i <= 2:
                    px_section = pixel[int(l - 60):int(l + 60)]
                    flx_section = flux[int(l - 60):int(l + 60)]
                else:
                    px_section = pixel[int(l - 40):int(l + 40)]
                    flx_section = flux[int(l - 40):int(l + 40)]

                clean_px_section = px_section[flx_section - np.nanmedian(flx_section) < 2 * np.nanstd(flx_section)]
                clean_flx_section = flx_section[flx_section - np.nanmedian(flx_section) < 2 * np.nanstd(flx_section)]
                plt.scatter(px_section[flx_section - np.nanmedian(flx_section) > 2 * np.nanstd(flx_section)],
                            flx_section[flx_section - np.nanmedian(flx_section) > 2 * np.nanstd(flx_section)],
                            c="red")

                try:
                    params, errs = curve_fit(
                        pseudo_voigt,
                        clean_px_section,
                        clean_flx_section,
                        [np.abs(np.nanmedian(flux) - np.nanmin(flux)), 15, l, clean_flx_section[-1] - clean_flx_section[0], np.nanmedian(clean_flx_section) + (clean_flx_section[-1] - clean_flx_section[0]) / 200, 0.5],
                        bounds=[
                            [0, 5, l - 50, -np.inf, -np.inf, 0],
                            [np.inf, 50, l + 50, np.inf, np.inf, 1]
                        ],
                        maxfev=10000
                    )
                except RuntimeError:
                    continue

                if np.abs(pseudo_voigt(0, params[0], params[1], 0, 0, 0, params[-1])) > 2 * np.nanstd(clean_flx_section - pseudo_voigt(clean_px_section, *params)):
                    balmer_p_vals.append(params[2])
                    linevals.append(BALMER_LINES[i])
                    if VIEW_DEBUG_PLOTS:
                        plt.plot(clean_px_section, pseudo_voigt(clean_px_section, *params))
        if VIEW_DEBUG_PLOTS:
            for p in p_lines:
                plt.axvline(p, ls="--", color="grey")
        if VIEW_DEBUG_PLOTS:
            plt.plot(pixel, flux)
            plt.show()

        compparams, _ = curve_fit(polynomial_three,
                                  balmer_p_vals,
                                  linevals,
                                  maxfev=10000)

        compparams = np.concatenate([compparams, np.array([0])])

    wpt = WavelenthPixelTransform(*compparams)

    velrange = np.linspace(-1500, 1500, 200)

    lines = np.genfromtxt("FeAr_lines", delimiter="  ")[:, 0]

    velresid = []
    basewl = wpt.px_to_wl(pixel)
    for v in velrange:
        thiswl = wlshift(basewl, v)
        interpolator = interp1d(thiswl, compflux, bounds_error=False, fill_value=0)
        velresid.append(np.sum(interpolator(lines)))

    velresid = np.array(velresid)
    velresid[velresid < velresid.mean()] = np.min(velresid[velresid > velresid.mean()])

    params, errs = curve_fit(gaussian,
                             velrange,
                             velresid,
                             [np.ptp(velresid) * 100, velrange[np.argmax(velresid)], 100, np.min(velresid)])

    # plt.plot(velrange, gaussian(velrange, *params))
    # plt.plot(velrange, velresid)
    # plt.show()

    pxlines = wpt.wl_to_px(wlshift(lines, -params[1]))

    mask = np.logical_and(pxlines > pixel.min(), pxlines < pixel.max())
    fp = pxlines[mask]
    op = lines[mask]

    # plt.plot(pixel, compflux, color="lightblue", zorder=2)
    # plt.show()

    # plt.scatter(fp, op)

    if compparams is None:
        compparams, errs = curve_fit(px_to_wl,
                                     fp,
                                     op)  # ,
    # sigma=p_errs)

    # plt.plot(fp, px_to_wl(fp, *compparams))
    # plt.errorbar(fp, op, yerr=p_errs, capsize=3, linestyle='')
    # plt.show()

    # compparams = np.concatenate([compparams, np.array([0])])
    wpt = WavelenthPixelTransform(*compparams)
    # params, errs = curve_fit(
    #     fitwrapper,
    #     pixel,
    #     np.zeros(pixel.shape),
    #     # p0 = [3605.7455517169524, 0.734833446586154, 0.00025963958918475143, -2.4636866019464887e-07, 7.6347512244437e-11]
    #     p0=compparams,
    #     maxfev=10000
    # )
    #
    # wpt = WavelenthPixelTransform(*params)

    # print(params, np.sqrt(np.diag(errs)))

    final_wl_arr = wpt.px_to_wl(pixel)

    if VIEW_DEBUG_PLOTS:
        # initial_guess_trafo = WavelenthPixelTransform(*compparams)
        # initial_guess_array = initial_guess_trafo.px_to_wl(pixel)
        plt.plot(realcwl, realcflux)
        plt.plot(final_wl_arr, compflux.min() - np.nanmax(flux) + flux, linewidth=1, color="gray")
        plt.plot(final_wl_arr, compflux, color="darkred")
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

    return final_wl_arr, flux, flx_std


if __name__ == "__main__":
    extract_spectrum()