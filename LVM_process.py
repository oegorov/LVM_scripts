#!/usr/bin/env python3
import scipy.signal
from sdss_access import RsyncAccess

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import os
import logging
import glob
from copy import deepcopy

import numpy as np
import gc
import sys
from tqdm import tqdm  # as tqdm
import shutil

import multiprocessing as mp
import yaml
from astropy.io import fits, ascii
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from functools import partial
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.modeling import fitting, models
from astropy.convolution import convolve_fft, kernels
from scipy.ndimage import median_filter, gaussian_filter, percentile_filter
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy.coordinates import EarthLocation
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', AstropyUserWarning)

drp_results_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'lvm', 'spectro', 'redux', '0.1.1.dev0', '1111')
drp_results_dir_sas = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'lvm', 'spectro', 'redux', '1.0.3')
drp_results_dir_newdrp = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'lvm', 'spectro', 'redux', '0.1.2dev')

log = logging.getLogger(name='LVM-reduction')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)


def quickflux(rsshdu, wrange, crange=None, selection=None, include_sky=False, partial_sky=False):
    naxis1 = rsshdu[1].data.shape[1]
    if selection is None:
        selection = np.arange(rsshdu[1].data.shape[0])
    naxis2 = len(selection)
    texp = rsshdu[0].header['EXPTIME']
    wcsrss = WCS(rsshdu[1].header)
    wave = np.array(wcsrss.pixel_to_world(np.arange(naxis1), 0))
    wave = np.array(wave[0]) * 1e10
    selwave = (wave >= wrange[0]) * (wave <= wrange[1])
    cwave = np.mean(wave[selwave])
    cwave1 = np.nan

    selwavemask = np.tile(selwave, (naxis2, 1))
    flux_all = rsshdu[1].data
    if include_sky:
        partial_sky = False
    if include_sky or partial_sky:
        flux_all += rsshdu["SKY"].data

    if partial_sky:
        intepolated_sky = sky_mask_line(rsshdu["SKY"].data[selection,:], wave, [np.mean(wave[selwave])], wid=3.5)
        flux_all[selection,:] -= intepolated_sky

    nbadpix = np.nansum(rsshdu["MASK"].data[selection,:][:,selwave] == 1, axis=1)
    nallpix = np.nansum(selwave)
    rec_badrows = nbadpix/nallpix > 0.5
    # Replace NaN and zero values to median
    rec_finite = np.zeros_like(flux_all, dtype=bool)
    rec_finite[selection, :] = True
    rec_finite = rec_finite & (rsshdu["MASK"].data != 1) & (flux_all != 0)
    flux_all[~rec_finite] = np.nan
    flux_all_med = median_filter(flux_all, (21, 1))
    rec_med_upd = ~np.isfinite(flux_all_med)
    flux_all_med1 = median_filter(flux_all_med, (1, 21))
    flux_all_med[rec_med_upd] = flux_all_med1[rec_med_upd]
    flux_all[~rec_finite] = flux_all_med[~rec_finite]

    rssmasked = flux_all[selection] * selwavemask
    rssmasked[rssmasked == 0] = np.nan

    flux_rss = np.nansum(rssmasked, axis=1)

    # if not rsshdu[0].header.get('FLUXCAL'):
    #     flux_rss[:] = np.nan
    #     return flux_rss

    nselpix = np.sum(np.isfinite(rssmasked), axis=1)
    nselpix1 = np.zeros_like(nselpix)

    if len(wrange) == 4:
        selwave1 = ((wave >= wrange[2]) * (wave <= wrange[3]))
        if partial_sky:
            flux_all[selection,:] += intepolated_sky
            intepolated_sky = sky_mask_line(rsshdu["SKY"].data[selection, :], wave,
                                            [np.mean(wave[selwave])], wid=3.5)
            flux_all[selection,:] -= intepolated_sky
        nbadpix = np.nansum(rsshdu["MASK"].data[selection,:][:,selwave1] == 1, axis=1)
        nallpix = np.nansum(selwave1)
        rec_badrows = rec_badrows & (nbadpix / nallpix > 0.5)

        cwave1 = np.mean(wave[selwave1])
        selwavemask = np.tile(selwave1, (naxis2, 1))
        rssmasked = flux_all[selection] * selwavemask
        rssmasked[rssmasked == 0] = np.nan
        flux_rss += np.nansum(rssmasked, axis=1)
        nselpix1 = np.sum(np.isfinite(rssmasked), axis=1)

    # Optional continuum subtraction
    if crange is not None and (len(crange) >= 2):
        cselwave = (wave >= crange[0]) * (wave <= crange[1])

        if len(crange) == 2:
            cflux = np.nanmedian(flux_all[selection, cselwave], axis=1)
            flux_rss -= (cflux * nselpix)
            flux_rss -= (cflux * nselpix1)
        else:
            cselwave1 = (wave >= crange[2]) * (wave <= crange[3])
            for cur_ind, cur_spec in enumerate(flux_all[selection]):
                mask_fit = np.isfinite(cur_spec) & (cselwave | cselwave1)
                if np.sum(mask_fit) >= 5:
                    res = np.polyfit(wave[mask_fit], cur_spec[mask_fit], 1)
                    p = np.poly1d(res)
                    flux_rss[cur_ind] -= (p(cwave) * nselpix[cur_ind])
                    if np.isfinite(cwave1):
                        flux_rss[cur_ind] -= (p(cwave1) * nselpix1[cur_ind])

    flux_rss[(flux_rss == 0) | rec_badrows] = np.nan
    return flux_rss#/texp


def fit_cur_spec(data, wave=None, lines=None, fix_ratios=None, velocity=0, mean_bounds=(-5.,5.),
                 ax=None, return_plot_data=False, subtract_lsf=True):
    spectrum, errors, lsf = data
    rec = np.flatnonzero(np.isfinite(spectrum) & (spectrum != 0) &
                         np.isfinite(errors) & (np.isfinite(lsf)) & (lsf > 0) & np.isfinite(wave))  # & (errors > 0)
    if len(rec) < 10:
        if return_plot_data:
            return [np.nan]*len(lines), np.nan, np.nan, np.nan, [np.nan]*len(lines), np.nan, np.nan, np.nan, None
        else:
            return [np.nan] * len(lines), np.nan, np.nan, np.nan, [np.nan] * len(lines), np.nan, np.nan, np.nan

    spectrum = spectrum[rec]
    errors = errors[rec]
    lsf = lsf[rec]
    wave = wave[rec]
    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    g_init = None
    cont = np.nanmin(spectrum)
    max_ampl = np.nanmax(spectrum) - cont
    gaussians = {}
    mean_std = np.nanmedian(lsf/2.35428)
    for l_id, l in enumerate(lines):
        gaussians[l_id] = models.Gaussian1D(stddev=mean_std, amplitude=max_ampl, mean=l*(1+velocity/3e5))
        gaussians[l_id].amplitude.bounds = (0, 1.1*abs(max_ampl))
        gaussians[l_id].stddev.bounds = (mean_std*0.6, mean_std*5.)
        if mean_bounds is not None:
            gaussians[l_id].mean.bounds = (l+mean_bounds[0], l+mean_bounds[1])
        if g_init is None:
            g_init = gaussians[l_id]
        else:
            g_init += gaussians[l_id]

    nz = len(spectrum)
    rec_spec_cont = (wave < wave[int(nz / 3) + 1]) | (wave > wave[int(2 * nz / 3) + 1])
    spec_cont = spectrum[rec_spec_cont]
    const_model = models.Const1D(np.nanmedian(spec_cont))
    g_init += const_model

    def tie_std(model):
        return model.stddev_0

    l_id = 1
    if len(lines) > 1:
        t1 = lambda model: (model.mean_0 * lines[1]/lines[0])
        gaussians[l_id].mean.tied = t1
        gaussians[l_id].stddev.tied = tie_std
        l_id += 1
    if len(lines) > 2:
        t2 = lambda model: (model.mean_0 * lines[2] / lines[0])
        gaussians[l_id].mean.tied = t2
        gaussians[l_id].stddev.tied = tie_std
        l_id += 1
    if len(lines) > 3:
        t3 = lambda model: (model.mean_0 * lines[3] / lines[0])
        gaussians[l_id].mean.tied = t3
        gaussians[l_id].stddev.tied = tie_std

    if fix_ratios is not None:
        ref_id = np.flatnonzero(np.array(fix_ratios)==1)[0]
        for fr_id, fr in enumerate(fix_ratios):
            if fr != 1 and fr > 0:
                gaussians[fr_id].amplitude.tied = lambda model: (model.__getattr__(f'amplitude_{ref_id}') * fr)
    # const_model.amplitude.fixed = True

    try:
        weights = 1./(np.abs(errors))
        # p = interp1d(wave, spectrum, bounds_error=False)
        # p_w = interp1d(wave, weights, bounds_error=False)
        # wave_oversampled = np.linspace(wave[0], wave[-1], len(wave)*10)
        # spectrum_oversampled = p(wave_oversampled)
        # weights_oversampled = p_w(wave_oversampled)
        res = fitter(g_init, wave, spectrum)#, weights=weights)
    except fitting.NonFiniteValueError:
        if return_plot_data:
            return [np.nan]*len(lines), np.nan, np.nan, np.nan, [np.nan]*len(lines), np.nan, np.nan, np.nan, None
        else:
            return [np.nan] * len(lines), np.nan, np.nan, np.nan, [np.nan] * len(lines), np.nan, np.nan, np.nan
    vel = (res.mean_0/lines[0]-1)*3e5
    if not subtract_lsf:
        disp = res.stddev_0 / lines[0] * 3e5
    else:
        disp = np.sqrt(res.stddev_0 ** 2 - mean_std ** 2) / lines[0] * 3e5  # - mean_std**2

    cont = res.__getattr__(f"amplitude_{len(lines)}").value
    fluxes = [res.__getattr__(f"amplitude_{l_id}")*res.stddev_0*np.sqrt(2 * np.pi) for l_id in range(len(lines))]
    if fitter.fit_info['param_cov'] is None:
        cov_diag = np.zeros(shape=(len(lines)+3,), dtype=float)
        cov_diag[:] = np.nan
    else:
        cov_diag = np.diag(fitter.fit_info['param_cov'])
    fluxes_err = [np.sqrt(cov_diag[0]*res.stddev_0**2 + res.__getattr__(f"amplitude_{0}")**2 * cov_diag[2]) * np.sqrt(2 * np.pi)]
    if len(lines)>1:
        for l_id in range(len(lines)-1):
            fluxes_err.append(np.sqrt(
                cov_diag[3+l_id] * res.stddev_0 ** 2 + res.__getattr__(f"amplitude_{l_id+1}") ** 2 * cov_diag[2]) * np.sqrt(
                2 * np.pi))
    v_err = np.sqrt(cov_diag[1])/lines[0]*3e5
    sigma_err = np.sqrt(cov_diag[2]) / lines[0] * 3e5
    cont_err = np.sqrt(cov_diag[-1])

    if ax is not None:
        ax.plot(wave, spectrum, 'k-', label='Obs')
        ax.plot(wave, res(wave), 'r--', label=f'Fit')
        ax.legend()
    if return_plot_data:
        plot_data = (wave, spectrum, res(wave))
        return fluxes, vel, disp, cont, fluxes_err, v_err, sigma_err, cont_err, plot_data
    else:
        return fluxes, vel, disp, cont, fluxes_err, v_err, sigma_err, cont_err


def sky_mask_line(data, wave, lines, wid=3.5):
    d_out = data.copy()
    rec_masked = np.zeros_like(wave).astype(bool)
    for l in lines:
        rec_masked = rec_masked | ((wave < (l+wid)) & (wave > (l-wid)))
    data[:, rec_masked] = np.nan
    for d_id, d in enumerate(data):
        rec = np.isfinite(d)
        d_out[d_id, rec_masked] = np.interp(wave[rec_masked], wave[rec], d[rec])
    return d_out


def fit_spectra(rsshdu, wrange, selection=None, lines=None, fix_ratios=None, flux_add=None, flux_corr_cf=None, expnum=None,
                mean_bounds=(-5,5), mc_errors=False, n_mc=30, do_helio_corr=True, sky_only=False, save_plot_test=None,
                consider_as_comp=None, velocity=0, mask_wl=None, ra_fib=None, dec_fib=None, include_sky=False,
                partial_sky=False):
    if not mc_errors:
        n_mc = 1
    naxis1 = rsshdu[1].data.shape[1]
    if include_sky:
        partial_sky = False
    if selection is None:
        selection = np.arange(rsshdu[1].data.shape[0])
    texp = rsshdu[0].header['EXPTIME']
    obstime = Time(rsshdu[0].header['OBSTIME'])
    loc = EarthLocation.of_site('lco')
    wcsrss = WCS(rsshdu[1].header)
    wave = np.array(wcsrss.pixel_to_world(np.arange(naxis1), 0))
    wave = np.array(wave[0]) * 1e10
    sel_wave = (wave >= wrange[0]) * (wave <= wrange[1])
    wave = wave[sel_wave]

    if sky_only:
        flux_all = rsshdu["SKY"].data[selection]
    else:
        flux_all = rsshdu["FLUX"].data[selection]#/texp
        if include_sky or partial_sky:
            flux_all += (rsshdu["SKY"].data[selection])#/texp)
    lsf_all = rsshdu['FWHM'].data[selection]
    if sky_only:
        errors_all = rsshdu['SKY_ERROR'].data[selection]#/texp
    else:
        errors_all = rsshdu['ERROR'].data[selection]

    # Replace NaN and zero values to median
    rec_finite = (rsshdu["MASK"].data[selection, :] != 1) & (flux_all != 0)
    flux_all[~rec_finite] = np.nan
    flux_all_med = median_filter(flux_all, (21, 1))
    rec_med_upd = ~np.isfinite(flux_all_med)
    flux_all_med1 = median_filter(flux_all_med, (1, 21))
    flux_all_med[rec_med_upd] = flux_all_med1[rec_med_upd]
    flux_all[~rec_finite] = flux_all_med[~rec_finite]

    flux_all = flux_all[:, sel_wave]
    lsf_all = lsf_all[:, sel_wave]
    errors_all = errors_all[:, sel_wave]

    if partial_sky:
        intepolated_sky = sky_mask_line(rsshdu["SKY"].data[selection,:][:, sel_wave], wave, lines, wid=3.5)
        flux_all -= intepolated_sky

    if mask_wl is not None:
        rec_mask = (wave > mask_wl[0]) & (wave < mask_wl[1])
        flux_all[:, rec_mask] = np.nan
        errors_all[:, rec_mask] = np.nan

    if flux_corr_cf is not None:
        flux_all = flux_all * flux_corr_cf
        errors_all = errors_all * flux_corr_cf
    if flux_add is not None:
        flux_all = flux_all + flux_add

    log.info(f"Fit spectra for exp={expnum}")
    n_uniq_comp = 1
    if consider_as_comp is not None:
        consider_as_comp = np.array(consider_as_comp)
        n_uniq_comp = np.max(np.unique(consider_as_comp[consider_as_comp >= 0]+1))
    fluxes = np.zeros(shape=(flux_all.shape[0], n_uniq_comp, n_mc+1))
    fluxes[:,:,:] = np.nan
    velocities = np.zeros(shape=(flux_all.shape[0], n_uniq_comp, n_mc+1))
    velocities[:, :, :] = np.nan
    dispersions = np.zeros(shape=(flux_all.shape[0], n_uniq_comp, n_mc+1))
    dispersions[:,:, :] = np.nan
    continuums = np.zeros(shape=(flux_all.shape[0], n_uniq_comp,n_mc+1))
    continuums[:,:, :] = np.nan

    spec_ids_to_plot = np.random.choice(range(len(flux_all)), 24)
    if save_plot_test is not None:
        fig = plt.figure(figsize=(20,30))
        gs = GridSpec(6, 4, fig, 0.1, 0.1, 0.99, 0.95, wspace=0.1, hspace=0.1,
                      width_ratios=[1]*4, height_ratios=[1]*6)
        cur_ax_id = 0
    for spec_id, cur_data in enumerate(zip(flux_all, errors_all, lsf_all)):
        if do_helio_corr:
            sc = SkyCoord(ra=ra_fib[spec_id] * u.degree, dec=dec_fib[spec_id] * u.degree, frame='icrs')
            vcorr = sc.radial_velocity_correction(kind='heliocentric', obstime=obstime, location=loc).value
        else:
            vcorr = 0.
        if mc_errors:
            mc_noise = np.random.randn(n_mc,len(wave))*np.median(cur_data[1])
            mc_noise[0,: ] = 0
        else:
            mc_noise = np.zeros(shape=(1,len(wave)), dtype=float)

        for mc_id, cur_mc_noise in enumerate(mc_noise):
            cur_flux, cur_err, cur_lsf = cur_data
            cur_flux += cur_mc_noise
            cur_data_upd = (cur_flux, cur_err, cur_lsf)
            if save_plot_test is not None and (spec_id in spec_ids_to_plot) and (mc_id == 0):
                ax = fig.add_subplot(gs[cur_ax_id])
                cur_ax_id += 1
            else:
                ax = None
            results = fit_cur_spec(cur_data_upd, wave=wave, lines=lines, fix_ratios=fix_ratios, velocity=velocity,
                                       mean_bounds=mean_bounds, ax=ax)
            if ax is not None:
                ax.set_title(f"ID={cur_ax_id+1}")
            cur_fluxes = np.array(results[0])
            cur_fluxes_err = np.array(results[4])
            for uc_id in range(n_uniq_comp):
                if consider_as_comp is None:
                    fluxes[spec_id, 0, mc_id] = np.nansum(cur_fluxes)
                    if mc_id == 0:
                        fluxes[spec_id, 0, -1] = np.sqrt(np.nansum(cur_fluxes_err**2))
                else:
                    fluxes[spec_id,  uc_id, mc_id] = np.nansum(cur_fluxes[consider_as_comp == uc_id])
                    velocities[spec_id, uc_id, mc_id] = results[1] + vcorr
                    dispersions[spec_id, uc_id, mc_id] = results[2]
                    continuums[spec_id, uc_id, mc_id] = results[3]
                    if mc_id == 0:
                        fluxes[spec_id, uc_id, -1] = np.sqrt(np.nansum(cur_fluxes_err[consider_as_comp == uc_id]**2))
                        velocities[spec_id, uc_id, -1] = results[5]
                        dispersions[spec_id, uc_id, -1] = results[6]
                        continuums[spec_id, uc_id, -1] = results[7]
    if fluxes.shape[2] > 2:
        fluxes[:, :, 1] = np.nanstd(fluxes[:, :, 1:-1], axis=2)
        fluxes = fluxes[:, :, :2]
        if consider_as_comp is not None:
            velocities[:, :, 1] = np.nanstd(velocities[:, :, 1:-1], axis=2)
            velocities = velocities[:, :, :2]
            dispersions[:, :, 1] = np.nanstd(dispersions[:, :, 1:-1], axis=2)
            dispersions = dispersions[:, :, :2]
            continuums[:, :, 1] = np.nanstd(continuums[:, :, 1:-1], axis=2)
            continuums = continuums[:, :, :2]

    if save_plot_test is not None:
        fig.savefig(save_plot_test)
    return fluxes, velocities, dispersions, continuums


def mkifuimage(x, y, flux, fibid, outfile, posang=0, RAobs=0, DECobs=0, output=None, outfibcoords=None):
    # Create fiber image
    platescale = 112.36748321030637  # Focal plane platescale in "/mm

    pscale = 0.01  # IFU image pixel scale in mm/pix
    rspaxel = 35.3 / platescale / 2  # spaxel radius in mm assuming 35.3" diameter chromium mask
    npix = 1800  # size of IFU image
    ima = np.zeros((npix, npix)) + np.nan
    xima, yima = np.meshgrid(np.arange(npix) - npix / 2, np.arange(npix) - npix / 2)
    xima = xima * pscale  # x coordinate in mm of each pixel in image
    yima = yima * pscale  # y coordinate in mm of each pixel in image
    for i in range(len(x)):
        sel = (xima - x[i]) ** 2 + (yima - y[i]) ** 2 <= rspaxel ** 2
        ima[sel] = flux[i]
    # flag CRPIX for visual reference
    # ima[int(npix / 2), int(npix / 2)] = 0

    # Create WCS for IFU image
    w = WCS(naxis=2)
    w.wcs.crpix = [int(npix / 2) + 1, int(npix / 2) + 1]
    skypscale = pscale * platescale / 3600  # IFU image pixel scale in deg/pix
    posangrad = posang * np.pi / 180
    w.wcs.cd = np.array([[skypscale * np.cos(posangrad), -1 * skypscale * np.sin(posangrad)],
                            [-1 * skypscale * np.sin(posangrad), -1 * skypscale * np.cos(posangrad)]])
    # w.wcs.cd=np.array([[skypscale*np.cos(posangrad), skypscale*np.sin(posangrad)],[-1*-1*skypscale*np.sin(posangrad), skypscale*np.cos(posangrad)]])
    w.wcs.crval = [RAobs, DECobs]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    # Save IFU image as fits
    hdu = fits.PrimaryHDU(ima, header=header)
    hdul = fits.HDUList([hdu])
    if output is None:
        hdul.writeto(outfile, overwrite=True)
    else:
        hdul.writeto(output, overwrite=True)

        # Calculate RA,DEC of each individual fiber
        xfib = x / pscale + int(npix / 2)  # pixel x coordinates of fibers
        yfib = y / pscale + int(npix / 2)  # pixel y coordinates of fibers
        fibcoords = w.pixel_to_world(xfib, yfib)
        if outfibcoords is not None:
            ascii.write(fibcoords.to_table(), outfibcoords, overwrite=True)

def mkmfimage(x, y, flux, fibid, agcfile, outfile, outmffile, radec_offset=None):
    # Read focal plane master frame (proc file) WCS and compute IFU center coordinates, position angle, and platescale
    mfagc = fits.open(agcfile)
    mfheader = mfagc[1].header
    outw = WCS(mfheader)
    CDmatrix = outw.pixel_scale_matrix
    posangrad = -1 * np.arctan(CDmatrix[1, 0] / CDmatrix[0, 0])
    posang = posangrad * 180 / np.pi
    IFUcencoords = outw.pixel_to_world(2500, 1000)
    RAobs = IFUcencoords.ra.value
    DECobs = IFUcencoords.dec.value
    if radec_offset is not None:
        RAobs += radec_offset[0]
        DECobs += radec_offset[1]

    # Calculate platescale ["/mm] from master frame WCS solutions
    platescale1 = np.sqrt(CDmatrix[0, 0] ** 2 + CDmatrix[0, 1] ** 2) * 3600 / 0.009  # pixels are 9 microns
    platescale2 = np.sqrt(CDmatrix[1, 0] ** 2 + CDmatrix[1, 1] ** 2) * 3600 / 0.009
    platescale = np.mean([platescale1, platescale2])
    mkifuimage(x, y, flux, fibid, outfile=outfile, posang=posang, RAobs=RAobs, DECobs=DECobs)

    # Correct native AGC image headers WCS parameters based on master frame WCS and AGC to master frame transformation parameters from Tom
    # Read Native AGC images
    agcw = mfagc[3]
    agce = mfagc[2]

    # Reproject all three images to master frame WCS
    naxis1out = 2000
    naxis2out = 5000
    outw.wcs.crpix = [int(naxis2out / 2) + 1, int(naxis1out / 2) + 1]
    outw.wcs.crval = [RAobs, DECobs]

    hdu = fits.open(outfile)[0]

    ifuout = reproject_interp(hdu, outw, shape_out=(naxis1out, naxis2out), order='nearest-neighbor')
    agcwout = reproject_interp(agcw, outw, shape_out=(naxis1out, naxis2out))
    agceout = reproject_interp(agce, outw, shape_out=(naxis1out, naxis2out))

    # Coadd the three reprojected images and save master frame to fits file
    # Scale AGC images to match IFU image scale
    auxifuout = ifuout
    auxagcwout = (agcwout - np.nanmedian(agcw.data)) / np.nanstd(agcw.data) * np.nanstd(hdu.data)
    auxagceout = (agceout - np.nanmedian(agce.data)) / np.nanstd(agcw.data) * np.nanstd(hdu.data)

    auxifuout = np.nan_to_num(auxifuout[0])
    auxagcwout = np.nan_to_num(auxagcwout[0])
    auxagceout = np.nan_to_num(auxagceout[0])

    # Coadd and save to FITS
    outima = auxifuout + auxagcwout + auxagceout
    outima[outima == 0] = np.nan
    outheader = outw.to_header()
    outhdu = fits.PrimaryHDU(outima, header=outheader)
    outhdul = fits.HDUList([outhdu])
    outhdul.writeto(outmffile, overwrite=True)
    outhdul.close()


def quickmap(mjd: int, expnum: int, do_astrom=False, wrange=(6558,6565), crange=None, skip_bad_fibers=False,
             output=None, output_dir=None, suffix='', first_exp=None, dithering=False, include_sky=False):
    '''
    Modified version of Guillermo's quickmap
    '''

    # =============================================================================

    # Input filenames
    LVMDATA_DIR = drp_results_dir  # os.path.join(SAS_BASE_DIR, 'sdsswork','lvm','lco')
    rssfile = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{expnum:0>8}.fits"

    # fibermap
    LVMCORE_DIR = os.environ['LVMCORE_DIR']
    fibermapfile = os.path.join(LVMCORE_DIR,'metrology','lvm_fiducial_fibermap.yaml')

    if do_astrom:
        LVMAGCAM_DIR = os.environ.get('LVMAGCAM_DIR')
        coadds_folder = os.path.join(LVMAGCAM_DIR, str(mjd), 'coadds')
        if not os.path.exists(coadds_folder):
            coadds_folder = os.path.join(LVMAGCAM_DIR, str(mjd))

        agcscifile = f"{coadds_folder}/lvm.sci.coadd_s{expnum:0>8}.fits"
        if not os.path.exists(agcscifile):
            agcscifile = f"{coadds_folder}/lvm.sci.coadd_s{expnum:0>8}.fits"
        if not os.path.exists(agcscifile):
            log.warning(f'{agcscifile} does not exist for mjd={mjd} exp={expnum}, skipping astrometry')
            do_astrom = False
        else:
            h=fits.getheader(agcscifile, ext=1)
            astrom_first_exp = False
            if not h.get('SOLVED') and first_exp is not None:
                agcscifile = f"{coadds_folder}/lvm.sci.coadd_s{first_exp:0>8}.fits"
                astrom_first_exp = True
                if not os.path.exists(agcscifile):
                    do_astrom = False
                else:
                    h = fits.getheader(agcscifile, ext=1)
                    if not h.get('SOLVED'):
                        log.warning(f'Astrometry is missing in mjd={mjd} exp={expnum} and in '
                                    f'the first exposure in a set. skipping astrometry')
                        do_astrom = False
                    else:
                        log.warning(f'Astrometry is missing in mjd={mjd} exp={expnum}. '
                                    f'Will use information from the first exposure in a set (exp={first_exp})')
    if output_dir is None:
        output_dir = os.path.join(LVMDATA_DIR, mjd, 'maps')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        uid = os.stat(output_dir).st_uid
        os.chown(output_dir, uid=uid, gid=10699)
        os.chmod(output_dir, 0o775)

    # Output image
    outscifile = os.path.join(output_dir, f"quickmap_{expnum:0>8}{suffix}.fits")
    outmfscifile = os.path.join(output_dir,f'mfimage_sci_{expnum:0>8}{suffix}.fits')
    # Output fiber coordinates ascii file
    outfibcoords = os.path.join(output_dir, f'fibcoords_sci_{expnum:0>8}.txt')

    # Read fibermap and get x,y coordinates of fibers

    with open(fibermapfile, 'r') as file:
        fibermap = np.array(yaml.safe_load(file)['fibers'])

    # Read fibermap into numpy arrays to get fibid, taregttype, x, y, ypix
    nfib = fibermap.shape[0]
    fibid = fibermap[:, 0]
    targettype = fibermap[:, 4]
    spectrograph = fibermap[:, 1]
    telescope = fibermap[:, 7]
    x = fibermap[:, 8].astype(float)
    y = fibermap[:, 9].astype(float)
    ringnum = fibermap[:, 10].astype(float)

    # select fibers of interest
    selsci = (telescope == 'Sci')
    selskye = (telescope == 'SkyE')
    selskyw = (telescope == 'SkyW')
    selspec = (telescope == 'Spec')

    # =============================================================================

    # Read RSS QL file and average spectra over wrange and crange and make flux array

    rss = fits.open(rssfile)
    rss['FLUX'].data[rss['MASK'] == 1] = np.nan
    rss['ERROR'].data[rss['MASK'] == 1] = np.nan
    rss['SKY'].data[rss['MASK'] == 1] = np.nan

    if skip_bad_fibers:
        tab = Table(rss['SLITMAP'].data)
        selsci = selsci & (tab['fibstatus'] == 0)


    flux = quickflux(rss, wrange, crange=crange, include_sky=include_sky)
    # tab = Table.read("/home/egorov/Dropbox/LVM/Orion_S2_densities.txt", format='ascii')
    # flux = tab["ne_[SII]_6716_6731"][tab["Exp"] == expnum]
    #
    # rec = np.isfinite(flux)

    if do_astrom:
        radec_offset = None
        if astrom_first_exp and dithering and first_exp is not None:
            h = fits.getheader(rssfile,ext=0)
            rssfile_first = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{first_exp:0>8}.fits"
            h_ref = fits.getheader(rssfile_first, ext=0)
            radec_offset = (h.get("TESCIRA")-h_ref.get("TESCIRA"), h.get("TESCIDE")-h_ref.get("TESCIDE"))


        mkmfimage((x[selsci]), (y[selsci]), flux[selsci], (fibid[selsci]), agcscifile,
                  outfile=outscifile, outmffile=outmfscifile, radec_offset=radec_offset)

    else:
        mkifuimage(x[selsci], y[selsci], flux[selsci], fibid[selsci],
                   outfile=outscifile, output=output, outfibcoords=outfibcoords)


    rss.close()


def reduce_parallel(exp_pairs):
    exp, weights = exp_pairs
    try:
        if weights is not None:
            add_weights = f"--sky-weights {weights[0]} {weights[1]}"
        else:
            add_weights = ""
        os.system(f"drp quick-reduction -fe {exp} {add_weights}")# >/dev/null 2>&1")
        # os.system(f"drp run --no-sci --with-cals -m 60291 -e {exp}")
    except Exception as e:
        log.error(f"Something wrong with quick data reduction: {e}")
        return False
    return True


def metadata_parallel(mjd):
    try:
        os.system(f"drp metadata regenerate -m {mjd}")
    except Exception as e:
        log.error(f"Something wrong with metadata regeneration: {e}")
        return False
    return True


def parse_config(config_filename):
    try:
        with open(config_filename, "rb") as f:
            config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        log.error(f"Something wrong with parsing config file: {e}")
        return None
    log.info(
        f"Config file is parsed. Will process {len(config['object'])} objects through the following stages: "
        f"{','.join([s for s in config['steps'] if config['steps'][s]])}")

    return config


def download_from_sas(config):
    cams = ['b1', 'b2', 'b3', 'r1', 'r2', 'r3', 'z1', 'z2', 'z3']
    rsync = RsyncAccess(verbose=True)
    # sets a remote mode to the real SAS
    rsync.remote()
    rsync.set_remote_base("--no-perms --omit-dir-times rsync")
    rsync.base_dir = os.environ['SAS_BASE_DIR']+"/"
    counter = 0
    counter_exist = 0
    # add all the file(s) you want to download
    new_files = []
    if not config.get('download_raw'):
        log.warning("Will skip downloading raw data")
    if config.get('download_reduced'):
        log.warning("Will download reduced spectra")
    f_reduced = {}
    for cur_obj in config['object']:
        f_reduced[cur_obj['name']] = {}
        for cur_pointing in cur_obj['pointing']:
            if cur_pointing['name'] not in f_reduced[cur_obj['name']]:
                f_reduced[cur_obj['name']][cur_pointing['name']] = []
            if cur_pointing['skip'].get('download'):
                log.info(f"Skip download for object = {cur_obj['name']}, pointing = {cur_pointing['name']}")
                continue
            if config.get('download_reduced'):
                download_current_reduced = True
            else:
                download_current_reduced = False

            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']

                tileids = None
                if 'tileid' not in cur_pointing and ('tileid' not in data) and download_current_reduced:
                    log.warning(
                        f"Tile ID is not present for object = {cur_obj['name']}, pointing = {cur_pointing['name']}. "
                        f"Can't download the reduced data for it")
                    download_current_reduced = False
                    tileid = None
                else:
                    tileid = cur_pointing.get('tileid')
                    if not tileid:
                        tileid = data.get('tileid')

                if tileid is not None:
                    if not (isinstance(tileid, list) or isinstance(tileid, tuple)):
                        tileids = [str(tileid)] * len(exps)
                    else:
                        tileids = [str(ti) for ti in tileid]


                # create new directory with correct permissions
                d_agcam_root = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'data', 'agcam', 'lco',
                                            str(data['mjd']))
                d_agcam = os.path.join(d_agcam_root, 'coadds')
                d_data = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'data', 'lvm', 'lco', str(data['mjd']))
                for check_dir in [d_agcam_root, d_agcam, d_data]:
                    if not os.path.exists(check_dir):
                        os.makedirs(check_dir)
                        uid = os.stat(check_dir).st_uid
                        os.chown(check_dir, uid=uid, gid=10699)
                        os.chmod(check_dir, 0o775)

                for exp_ind,exp in enumerate(exps):
                    if config.get('download_raw'):
                        for cam in cams:
                            f = glob.glob(
                                os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'data', 'lvm', 'lco', str(data['mjd']),
                                             f'*-{cam}-*{exp}.fits.gz'))
                            if not config.get('force_download') and (len(f) == 1):
                                counter_exist += 1
                            else:
                                new_files.append(os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork',
                                                              'data', 'lvm', 'lco', str(data['mjd']),
                                                              f'sdR-s-{cam}-{exp:08d}.fits.gz'))
                                counter += 1
                                rsync.add('lvm_raw', camspec=cam, expnum=str(exp), hemi='s', mjd=str(data['mjd']))

                    if download_current_reduced:
                        if tileids is not None and tileids[exp_ind] is not None:
                            if cur_obj['name'] == 'Orion':
                                if (int(tileids[exp_ind]) < 1027000) & (int(tileids[exp_ind]) != 11111):
                                    tileids[exp_ind] = str(int(tileids[exp_ind])+27748)

                            if tileids[exp_ind] == '11111':
                                short_tileid = '0011XX'
                            elif tileids[exp_ind] == '999':
                                short_tileid = '0000XX'
                            else:
                                short_tileid = tileids[exp_ind][:4] + 'XX'
                        f = glob.glob(
                            os.path.join(drp_results_dir_sas, short_tileid, tileids[exp_ind], str(data['mjd']),
                                         f'lvmSFrame-*{exp}.fits'))
                        if not config.get('force_download') and (len(f) == 1):
                            counter_exist += 1
                        else:
                            cur_f = os.path.join(drp_results_dir_sas, short_tileid, tileids[exp_ind],
                                         str(data['mjd']), f'lvmSFrame-{exp:08d}.fits')
                            f_reduced[cur_obj['name']][cur_pointing['name']].append(cur_f)
                            new_files.append(cur_f)
                            counter += 1
                            rsync.add('lvm_frame', expnum=str(exp), mjd=str(data['mjd']), tileid=tileids[exp_ind],
                                      kind='SFrame', drpver=f'1.0.3/{short_tileid}')

                    if not config.get('skip_download_agcam'):
                        # add corresponding agcam coadd images
                        f = os.path.join(d_agcam, f'lvm.sci.coadd_s{exp:08d}.fits')
                        if not config.get('force_download') and os.path.exists(f):
                            counter_exist += 1
                        else:
                            new_files.append(f)
                            counter += 1
                        rsync.initial_stream.append_task(sas_module='sdsswork',
                                                         location=f'data/agcam/lco/{str(data["mjd"])}/coadds/lvm.sci.coadd_s{exp:08d}.fits',
                                                         source=f'--no-perms --omit-dir-times rsync://sdss5@dtn.sdss.org/sdsswork/data/agcam/lco/{str(data["mjd"])}/coadds/lvm.sci.coadd_s{exp:08d}.fits',
                                                         destination=f'{d_agcam}/lvm.sci.coadd_s{exp:08d}.fits')

    if counter_exist > 0:
        log.warning(f"{counter_exist} files exist and will be not downloaded")
    if counter == 0:
        log.warning(f"Nothing to download")
        return True
    log.info(f"Number of files to download from SAS: {counter}")
    log.info(
        f"Start downloading files from SAS. It can take several minutes if you ask for many files, please be patient!")
    try:
        rsync.set_stream()

        # start the download(s)
        rsync.commit()
    except Exception as e:
        log.error(f"Something wrong with rsync: {e}")
        return False
    for f in new_files:
        uid = os.stat(f).st_uid
        os.chown(f, uid=uid, gid=10699)
        os.chmod(f, 0o664)

    if config.get('download_reduced'):
        output_dir = config.get('default_output_dir')
        if not output_dir:
            log.error("Output directory is not set up. Cannot copy files")
            return False

        for obj_name in f_reduced:
            for pointing_name in f_reduced[obj_name]:
                if len(f_reduced[obj_name][pointing_name]) == 0:
                    log.warning(f"Nothing to copy for object = {obj_name}, pointing = {pointing_name}")
                    continue
                if not pointing_name:
                    curdir = os.path.join(output_dir, obj_name)
                else:
                    curdir = os.path.join(output_dir, obj_name, pointing_name)
                if not os.path.exists(curdir):
                    os.makedirs(curdir)
                    uid = os.stat(curdir).st_uid
                    os.chown(curdir, uid=uid, gid=10699)
                    os.chmod(curdir, 0o775)
                log.info(f"Copy {len(f_reduced[obj_name][pointing_name])} for object = {obj_name}, pointing = {pointing_name}")
                for sf in f_reduced[obj_name][pointing_name]:
                    fname = os.path.join(curdir, os.path.split(sf)[-1])
                    if os.path.exists(fname):
                        os.remove(fname)
                    shutil.copy(sf, curdir)
    return True


def regenerate_metadata(config):
    mjds = []
    for cur_obj in config['object']:
        for cur_pointing in cur_obj['pointing']:
            if cur_pointing['skip'].get('download'):
                continue
            for data in cur_pointing['data']:
                mjds.append(data['mjd'])
    mjds = np.unique(mjds)
    log.info(f"Regenerate metadata for {len(mjds)} nights")
    procs = np.nanmin([config['nprocs'], len(mjds)])
    statuses = []
    with mp.Pool(processes=procs) as pool:

        for status in tqdm(pool.imap_unordered(metadata_parallel, mjds),
                           ascii=True, desc="Metadata regeneration",
                           total=len(mjds), ):
            statuses.append(status)
        pool.close()
        pool.join()
        gc.collect()
    if not np.all(statuses):
        return False

    return True


def do_sky_correction(config, w_dir=None):
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    files = []
    for cur_obj in config['object']:
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'))
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        for cur_pointing in cur_obj['pointing']:
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']
                for exp in exps:
                    cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
                    if not os.path.exists(cur_fname):
                        log.warning(f"Can't find {cur_fname}")
                        continue
                    files.append(cur_fname)

    for ind_f, cur_f in tqdm(enumerate(files), total=len(files), ascii=True, desc='Spectra done:'):
        rss_orig = fits.open(cur_f)
        if not rss_orig[0].header.get('FLUXCAL'):
            log.error(f"Missing flux calibration for file {cur_f}. Skip it.")
            rss_orig.close()
            statuses.append(False)
            continue
        rss = deepcopy(rss_orig)
        rss['FLUX'].data[rss['MASK'] == 1] = np.nan
        rss['ERROR'].data[rss['MASK'] == 1] = np.nan
        rss['SKY'].data[rss['MASK'] == 1] = np.nan
        tab = Table(rss['SLITMAP'].data)
        sci = np.flatnonzero(tab['targettype'] == 'science')
        # rss["FLUX"].data += rss['SKY'].data

        flux = quickflux(rss, [5572, 5582], crange=[5545, 5565, 5585, 5600], selection=sci, include_sky=True)

        rss["FLUX"].data = rss['SKY'].data
        flux_sky = quickflux(rss, [5572, 5582], crange=[5545, 5565, 5585, 5600], selection=sci, include_sky=False)

        # flux, _, _, cont = fit_spectra(rss, [5545, 5600], selection=sci, mean_bounds=(-2,2),
        #                                consider_as_comp=[0],do_helio_corr=False,
        #                                lines=[5577.], velocity=0)
        #
        # flux_sky, _, _, cont_sky = fit_spectra(rss, [5545, 5600], selection=sci, mean_bounds=(-2, 2),
        #                                        consider_as_comp=[0], do_helio_corr=False,
        #                                        lines=[5577.], velocity=0)
        rss.close()
        cur_header = rss_orig['FLUX'].header
        cur_data = rss_orig['FLUX'].data+rss_orig['SKY'].data
        cur_data[sci] -= (rss_orig['SKY'].data[sci] * (flux/flux_sky)[:, None])
        rss_orig.append(fits.ImageHDU(data= cur_data, header=cur_header, name='FLUX_SKYCORR'))
        rss_orig.writeto(cur_f.split(".fits")[0]+'_test.fits', overwrite=True)
        statuses.append(True)


def do_coadd_spectra(config, w_dir=None):
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    for cur_obj in config['object']:
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'))
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue

        for ind_pointing, cur_pointing in tqdm(enumerate(cur_obj['pointing']), total=len(cur_obj['pointing']),
                                               ascii=True, desc='Pointings done:'):
            files = []
            corrections = []
            if cur_pointing['skip'].get('combine_spec'):
                log.info(f"Skip coadding spectra for object = {cur_obj['name']}, pointing = {cur_pointing['name']}")
                continue
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']
                if not data.get('flux_correction'):
                    corrections.extend([1.]*len(exps))
                else:
                    if isinstance(data['flux_correction'], float) or isinstance(data['flux_correction'], int):
                        corrections.append(data['flux_correction'])
                    else:
                        corrections.extend(data['flux_correction'])
                for exp in exps:
                    cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
                    if not os.path.exists(cur_fname):
                        log.warning(f"Can't find {cur_fname}")
                        continue
                    files.append(cur_fname)

            if len(files) < 2:
                log.warning(f"Nothing to coadd for object = {cur_obj['name']}, pointing = {cur_pointing['name']}")
                continue

            fout = os.path.join(cur_wdir, cur_pointing['name'], f'combined_spectrum_{ind_pointing:02d}.fits')
            with fits.open(files[0]) as hdu_ref:
                for ext in ['FLUX', 'ERROR', 'SKY', 'SKY_ERROR']:
                    hdu_ref[ext].data[hdu_ref['MASK'] == 1] = np.nan
                    hdu_ref[ext].data = np.expand_dims(hdu_ref[ext].data, axis=0) * corrections[0]
                for f_ind, f in enumerate(files[1:]):
                    with fits.open(f) as hdu:
                        for ext in ['FLUX', 'ERROR', 'SKY', 'SKY_ERROR']:
                            hdu[ext].data[hdu['MASK'] == 1] = np.nan
                            hdu[ext].data = np.expand_dims(hdu[ext].data, axis=0) * corrections[f_ind+1] / hdu[0].header['EXPTIME'] * hdu_ref[0].header['EXPTIME']
                            hdu_ref[ext].data = np.vstack([hdu_ref[ext].data, hdu[ext].data])
                for ext in ['FLUX', 'ERROR', 'SKY', 'SKY_ERROR']:
                    if "ERROR" in ext:
                        hdu_ref[ext].data = np.sqrt(np.nansum(hdu_ref[ext].data**2, axis=0))/np.sum(np.isfinite(hdu_ref[ext].data),axis=0)
                    else:
                        hdu_ref[ext].data = sigma_clip(hdu_ref[ext].data, sigma=1.3, axis=0, masked=False)
                        hdu_ref[ext].data = np.nanmean(hdu_ref[ext].data, axis=0)
                hdu_ref.writeto(fout, overwrite=True)
                statuses.append(True)

    if not np.all(statuses):
        return False
    else:
        return True


def do_quick_reduction(config):
    exps = []
    mjds = []
    for cur_obj in config['object']:
        for cur_pointing in cur_obj['pointing']:
            if cur_pointing['skip'].get('reduction'):
                log.info(f"Skip reduction for object = {cur_obj['name']}, pointing = {cur_pointing['name']}")
                continue
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exp = [data['exp']]
                else:
                    exp = data['exp']
                mjds.extend([data['mjd']]*len(exp))
                exps.extend(exp)

                # Remove old drp products since they can break new reduction
                if config.get('drp_old'):
                    for exp in exps:
                        files_check = glob.glob(os.path.join(drp_results_dir, str(data['mjd']), "ancillary", f"*{exp}.fits"))
                        for f in files_check:
                            if os.path.isfile(f):
                                os.remove(f)
                        files_check = glob.glob(os.path.join(drp_results_dir, str(data['mjd']), f"*{exp}.fits"))
                        for f in files_check:
                            if os.path.isfile(f):
                                os.remove(f)
    mjds = np.array(mjds)
    exps, exp_ids = np.unique(exps, return_index=True)
    mjds = mjds[exp_ids]

    exp_pairs = []
    # == Derive sky weights:
    for exp_id, exp in enumerate(exps):
        if config.get("wham_sky_only"):
            f = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'data', 'lvm', 'lco', str(mjds[exp_id]),
                         f'sdR-s-b1-{exp:08d}.fits.gz')
            header = fits.getheader(f)
            skye_name = header.get('SKYENAME')
            skyw_name = header.get('SKYWNAME')
            if skye_name is not None or skyw_name is not None:
                if skye_name is None:
                    skye_name = 'none'
                if skyw_name is None:
                    skyw_name = 'none'
                weights_e = 0
                weights_w = 0
                if 'WHAM' in skye_name:
                    weights_e = 1
                if 'WHAM' in skyw_name:
                    weights_w = 1
                if (weights_e + weights_w) != 1:
                    weights = None
                else:
                    weights = (weights_e, weights_w)
            else:
                weights = None
        else:
            weights = None
        # if weights is not None:
        #     log.warning(f"Skipping exp={exp} as it was already reduced with weights={weights}")
        #     continue
        # if weights is None:
        #     weights = [1, 0]
        if weights is not None:
            log.info(f"For exp={exp} assume weights=({weights_e}, {weights_w})")

        exp_pairs.append((exp, weights))

    statuses = []
    if not config.get('reduce_parallel'):
        log.info(f"Start reduction of {len(exp_pairs)} exposures")
        for exp in tqdm(exp_pairs, ascii=True, desc="Quick data reduction", total=len(exp_pairs)):
            status = reduce_parallel(exp)
            if not status:
                log.error(f"Something went wrong with mjd={str(data['mjd'])}, exposure={exp[0]}")
            statuses.append(status)
    else:
        procs = np.nanmin([config['nprocs'], len(exps)])
        log.info(f"Start reduction of {len(exp_pairs)} exposures in {procs} parallel processes")
        with mp.Pool(processes=procs) as pool:

            for status in tqdm(pool.imap_unordered(reduce_parallel, exp_pairs),
                               ascii=True, desc="Quick data reduction",
                               total=len(exp_pairs), ):
                statuses.append(status)
            pool.close()
            pool.join()
            gc.collect()
    if not np.all(statuses):
        return False

    return True


def copy_reduced_data(config, output_dir=None):
    if output_dir is None:
        output_dir = config.get('default_output_dir')
    if not output_dir:
        log.error("Output directory is not set up. Cannot copy files")
        return False

    for cur_obj in config['object']:

        for cur_pointing in cur_obj['pointing']:
            if cur_pointing['skip'].get('reduction'):
                log.warning(
                    f"Skip copy reduced spectra for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
                continue
            if not cur_pointing.get('name'):
                curdir = os.path.join(output_dir, cur_obj['name'])
            else:
                curdir = os.path.join(output_dir, cur_obj['name'], cur_pointing.get('name'))
            if not os.path.exists(curdir):
                os.makedirs(curdir)
                uid = os.stat(curdir).st_uid
                os.chown(curdir, uid=uid, gid=10699)
                os.chmod(curdir, 0o775)

            source_files = []
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']
                if not config.get('drp_old'):
                    tileids = None
                    if 'tileid' not in cur_pointing and ('tileid' not in data) and download_current_reduced:
                        log.warning(
                            f"Tile ID is not present for object = {cur_obj['name']}, pointing = {cur_pointing['name']}. "
                            f"Can't copy the reduced data for it")
                        download_current_reduced = False
                        tileid = None
                    else:
                        tileid = cur_pointing.get('tileid')
                        if not tileid:
                            tileid = data.get('tileid')

                    if tileid is not None:
                        if not (isinstance(tileid, list) or isinstance(tileid, tuple)):
                            tileids = [str(tileid)] * len(exps)
                        else:
                            tileids = [str(ti) for ti in tileid]

                for exp_ind, exp in enumerate(exps):
                    if config.get('drp_old'):
                        source_files.append(os.path.join(drp_results_dir, str(data['mjd']), f'lvmCFrame-{exp:08d}.fits'))
                    else:
                        if tileids is not None and tileids[exp_ind] is not None:
                            if cur_obj['name'] == 'Orion':
                                if (int(tileids[exp_ind]) < 1027000) & (int(tileids[exp_ind]) != 11111):
                                    tileids[exp_ind] = str(int(tileids[exp_ind])+27748)

                            if tileids[exp_ind] == '1111':
                                short_tileid = '0011XX'
                            elif tileids[exp_ind] == '999':
                                short_tileid = '0000XX'
                            else:
                                short_tileid = tileids[exp_ind][:4] + 'XX'
                        source_files.append(os.path.join(drp_results_dir_newdrp, tileids[exp_ind], #short_tileid
                                                         str(data['mjd']), f'lvmCFrame-{exp:08d}.fits'))

            if len(source_files) == 0:
                log.warning(f"Nothing to copy for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
                continue
            log.info(f"Copy {len(source_files)} for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
            for sf in tqdm(source_files, total=len(source_files)):
                fname = os.path.join(curdir, os.path.split(sf)[-1])
                if os.path.exists(fname):
                    os.remove(fname)
                shutil.copy(sf, curdir)


def quickmap_parallel(data, wl_range=None, cont_range=None, output_dir=None, suffix=None, include_sky=False,
                      skip_bad_fibers=False):
    mjd, exp, dithering, first_exp = data
    try:
        quickmap(mjd, exp, output_dir=output_dir, suffix=suffix,skip_bad_fibers=skip_bad_fibers,
                 wrange=wl_range, crange=cont_range, do_astrom=True,
                 dithering=dithering, first_exp=first_exp, include_sky=include_sky)
    except Exception as e:
        return False
    return True


def derive_radec_ifu(mjd, expnum, first_exp=None, objname=None, pointing_name=None):
    # Input filenames
    LVMDATA_DIR = drp_results_dir  # os.path.join(SAS_BASE_DIR, 'sdsswork','lvm','lco')
    if objname is None or pointing_name is None:
        rssfile = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{expnum:0>8}.fits"
    else:
        rssfile = f"/data/LVM/Reduced/{objname}/{pointing_name}/lvmCFrame-{expnum:0>8}.fits"
    hdr = fits.getheader(rssfile)
    pa_hdr = hdr.get('POSCIPA')
    if not pa_hdr:
        pa_hdr = 0
    LVMAGCAM_DIR = os.environ.get('LVMAGCAM_DIR')
    coadds_folder = os.path.join(LVMAGCAM_DIR, str(mjd), 'coadds')
    if not os.path.exists(coadds_folder):
        coadds_folder = os.path.join(LVMAGCAM_DIR, str(mjd))
    agcscifile = f"{coadds_folder}/lvm.sci.coadd_s{expnum:0>8}.fits"
    if not os.path.exists(agcscifile):
        log.warning(f'{agcscifile} does not exist for mjd={mjd} exp={expnum}, skipping astrometry')
        if not first_exp:
            return hdr['POSCIRA'], hdr['POSCIDE'], pa_hdr
        else:
            solved = False
    else:
        h = fits.getheader(agcscifile, ext=1)
        solved = h.get('SOLVED')
    radec_corr = [0,0,0]
    # if expnum in [8115, 8118]:
    #     radec_corr[2] = -60
    if not solved and first_exp is not None:
        agcscifile = f"{coadds_folder}/lvm.sci.coadd_s{first_exp:0>8}.fits"
        if not os.path.exists(agcscifile):
            log.warning(f'{agcscifile} does not exist for mjd={mjd} exp={expnum}, skipping astrometry')
            return hdr['POSCIRA'], hdr['POSCIDE'], pa_hdr
        else:
            h = fits.getheader(agcscifile, ext=1)
            if not h.get('SOLVED'):
                log.warning(f'Astrometry is missing in mjd={mjd} exp={expnum} and in '
                                    f'the first exposure in a set. skipping astrometry')
                return hdr['POSCIRA'], hdr['POSCIDE'], pa_hdr
            log.warning(f'Astrometry is missing in mjd={mjd} exp={expnum}. '
                        f'Will use information from the first exposure in a set (exp={first_exp})')
            if objname is None or pointing_name is None:
                rssfile_ref = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{first_exp:0>8}.fits"
            else:
                rssfile_ref = f"/data/LVM/Reduced/{objname}/{pointing_name}/lvmCFrame-{first_exp:0>8}.fits"
            hdr_ref = fits.getheader(rssfile_ref)
            pa_hdr_ref = hdr_ref.get('POSCIPA')
            if not pa_hdr_ref:
                pa_hdr_ref = 0
            radec_corr = [hdr['TESCIRA'] - hdr_ref['TESCIRA'], hdr['TESCIDE'] - hdr_ref['TESCIDE'], pa_hdr - pa_hdr_ref]

    agcam_hdr = fits.getheader(agcscifile, ext=1)
    w = WCS(agcam_hdr)
    cen = w.pixel_to_world(2500,1000)
    # print(cen.ra.deg+radec_corr[0], cen.dec.deg+radec_corr[1], (agcam_hdr['PAMEAS'] + 180. + radec_corr[2]) % 360.)
    return cen.ra.deg+radec_corr[0], cen.dec.deg+radec_corr[1], (agcam_hdr['PAMEAS'] + 180. + radec_corr[2]) % 360. #agcam_hdr['PAMEAS'] - 180.


def create_line_image_from_rss(file_fluxes=None, lines=None, pxscale_out=3., r_lim=50, sigma=2.,
                                output_dir=None, do_median_masking=False,
                                outfile_prefix=None):
    lvm_fiber_diameter = 35.3
    if not os.path.isfile(file_fluxes):
        log.error(f"File {file_fluxes} not found")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        uid = os.stat(output_dir).st_uid
        os.chown(output_dir, uid=uid, gid=10699)
        os.chmod(output_dir, 0o775)

    table_fluxes = Table.read(file_fluxes, format='ascii.fixed_width_two_line')
    ras = table_fluxes['fib_ra']
    decs = table_fluxes['fib_dec']

    dec_0 = np.min(decs)-37./2/3600
    ra_0 = np.max(ras)+37./2/3600/np.cos(dec_0/180*np.pi)
    dec_1 = np.max(decs) + 37. / 2 / 3600
    ra_1 = np.min(ras) - 37. / 2 / 3600 / np.cos(dec_1 / 180 * np.pi)

    ra_cen = (ra_0+ra_1)/2.
    dec_cen = (dec_0 + dec_1) / 2.
    nx = np.ceil((ra_0 - ra_1)*np.cos(dec_cen/2./180.*np.pi)/pxscale_out*3600./2.).astype(int)*2+1
    ny = np.ceil((dec_1 - dec_0) / pxscale_out * 3600./2.).astype(int)*2+1
    ra_0 = np.round(ra_cen + (nx-1)/2 * pxscale_out / 3600. * np.cos(dec_cen/2./180.*np.pi),6)
    dec_0 = np.round(dec_cen - (ny - 1) / 2 * pxscale_out/ 3600., 6)

    # Create a new WCS object.  The number of axes must be set
    # from the start
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = [1, 1]
    wcs_out.wcs.cdelt = np.array([-np.round(pxscale_out/3600.,6), np.round(pxscale_out/3600.,6)])
    wcs_out.wcs.crval = [ra_0, dec_0]
    wcs_out.wcs.cunit = ['deg', 'deg']
    wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    shape_out = (ny, nx)
    grid_scl = wcs_out.proj_plane_pixel_scales()[0].value * 3600.

    log.info(f"Grid scale: {grid_scl}")

    # img_arr = np.zeros((ny, nx), dtype=float)
    # img_arr[:, :] = np.nan
    values = None
    names_out = []

    for cur_line in lines:
        lns = cur_line.get('line')
        if isinstance(lns, str):
            lns = [lns]
        for ln in lns:
            if values is None:
                if f'{ln}_flux' in table_fluxes.colnames:
                    values = table_fluxes[f'{ln}_flux'] / (np.pi * lvm_fiber_diameter ** 2 / 4)
                else:
                    continue
            else:
                if f'{ln}_flux' in table_fluxes.colnames:
                    values = np.vstack([values.T, table_fluxes[f'{ln}_flux'].T]).T
                else:
                    continue
            names_out.append(f'{ln}_flux')
            for suff in ['vel', 'disp', 'cont']:
                if suff == 'cont':
                    fluxcorr = (np.pi * lvm_fiber_diameter ** 2 / 4)
                else:
                    fluxcorr = 1
                if f'{ln}_{suff}' in table_fluxes.colnames:
                    values = np.vstack([values.T, table_fluxes[f'{ln}_{suff}'].T/fluxcorr]).T
                    names_out.append(f'{ln}_{suff}')
    if values is None:
        log.error('Nothing to show.')
        return False
    if len(values.shape) == 1:
        values = values.reshape((-1, 1))

    img_arr = shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs, show_values=values,
                               r_lim=r_lim, sigma=sigma)
    header = wcs_out.to_header()
    for ind in range(img_arr.shape[2]):
        outfile_suffix = names_out[ind]
        if do_median_masking:
            img_arr[:, :, ind] = median_filter(img_arr[:, :, ind], (11, 11))
        fits.writeto(os.path.join(output_dir, f"{outfile_prefix}_{outfile_suffix}.fits"),
                     data=img_arr[:, :, ind], header=header, overwrite=True)
    return True


def fit_all_from_current_spec(params, header=None,
                              line_fit_params=None, mean_bounds=None, velocity=0):
    flux, error, sky, sky_error, lsf, vhel_corr, spec_id = params

    vhel = np.mean([float(vh) for vh in vhel_corr.split(',')])

    wave = ((np.arange(header['NAXIS1']) - header['CRPIX1'] + 1) * header['CDELT1'] + header['CRVAL1']) * 1e10
    res_output = {}
    all_plot_data = []
    status = True

    wid = 10
    vel_sky_correct = 0
    for sky_line in [5577.338, 6300.304]:
        sel_wave = np.flatnonzero((wave >= (sky_line - wid)) & (wave <= (sky_line + wid)))
        (fluxes, vel, disp, cont, fluxes_err, v_err,
         sigma_err, cont_err) = fit_cur_spec((sky[sel_wave], sky_error[sel_wave], lsf[sel_wave]),
                                             wave=wave[sel_wave],
                                             mean_bounds=(-1.5, 1.5),
                                             lines=[float(sky_line)],
                                             velocity=0, return_plot_data=False, subtract_lsf=False)

        res_output[f'SKY{int(sky_line)}_flux'] = fluxes[0]
        res_output[f'SKY{int(sky_line)}_fluxerr'] = fluxes_err[0]
        res_output[f'SKY{int(sky_line)}_vel'] = vel
        res_output[f'SKY{int(sky_line)}_velerr'] = v_err
        res_output[f'SKY{int(sky_line)}_disp'] = disp
        res_output[f'SKY{int(sky_line)}_disperr'] = sigma_err
        res_output[f'SKY{int(sky_line)}_cont'] = disp
        res_output[f'SKY{int(sky_line)}_conterr'] = sigma_err
        if int(sky_line) == 5577:
            vel_sky_correct = vel

    for cur_line_params in line_fit_params:
        (line_name, wl_range, mask_wl, line_fit,
         include_comp, fix_ratios, _, save_plot_ids) = cur_line_params
        sel_wave = np.flatnonzero((wave >= wl_range[0]) & (wave <= wl_range[1]))

        if save_plot_ids is not None and spec_id in save_plot_ids:
            (fluxes, vel, disp, cont, fluxes_err, v_err,
             sigma_err, cont_err, plot_data) = fit_cur_spec((flux[sel_wave], error[sel_wave], lsf[sel_wave]),
                                                            wave=wave[sel_wave], mean_bounds=mean_bounds,
                                                            lines=line_fit, fix_ratios=fix_ratios,
                                                            velocity=velocity+vel_sky_correct-vhel, return_plot_data=True)
            all_plot_data.append(plot_data)
        else:
            (fluxes, vel, disp, cont, fluxes_err, v_err,
             sigma_err, cont_err) = fit_cur_spec((flux[sel_wave], error[sel_wave], lsf[sel_wave]),
                                                 wave=wave[sel_wave], mean_bounds=mean_bounds,
                                                            lines=line_fit, fix_ratios=fix_ratios,
                                                            velocity=velocity+vel_sky_correct-vhel, return_plot_data=False,
                                                 )
        if ~np.isfinite(vel):
            continue

        for ln_id, ln in enumerate(line_name):
            rec_comp = np.flatnonzero(np.array(include_comp) == ln_id)
            if len(rec_comp) == 0:
                continue

            res_output[f'{ln}_flux'] = np.nansum(np.array(fluxes)[rec_comp])
            res_output[f'{ln}_fluxerr'] = np.sqrt(np.nansum(np.array(fluxes_err)[rec_comp]**2))
            res_output[f'{ln}_vel'] = vel + vhel - vel_sky_correct
            res_output[f'{ln}_velerr'] = v_err
            res_output[f'{ln}_disp'] = disp
            res_output[f'{ln}_disperr'] = sigma_err
            res_output[f'{ln}_cont'] = disp
            res_output[f'{ln}_conterr'] = sigma_err



    return status, res_output, all_plot_data, spec_id


def quickflux_single_rss(rsshdu, wrange, crange=None, selection=None, include_sky=False, partial_sky=True):
    naxis1 = rsshdu[1].data.shape[1]
    if selection is None:
        selection = np.arange(rsshdu[1].data.shape[0])
    naxis2 = len(selection)
    wcsrss = WCS(rsshdu[1].header)
    wave = np.array(wcsrss.pixel_to_world(np.arange(naxis1), 0))
    wave = np.array(wave[0]) * 1e10
    dw = wave[1]-wave[0]

    selwave_extended = (wave >= (min(wrange)-100)) * (wave <= (max(wrange)+100))
    wave = wave[selwave_extended]
    if include_sky:
        flux_all = (rsshdu['FLUX_ORIG'].data + rsshdu['SKY'].data)[:, selwave_extended]
    elif partial_sky:
        flux_all = rsshdu['FLUX'].data[:, selwave_extended]
    else:
        flux_all = rsshdu["FLUX_ORIG"].data[:, selwave_extended]
    errors_all = rsshdu["ERROR"].data[:, selwave_extended]
    mask_all = rsshdu["MASK"].data[:, selwave_extended]

    selwave = (wave >= wrange[0]) * (wave <= wrange[1])
    cwave = np.mean(wave[selwave])
    cwave1 = np.nan

    selwavemask = np.tile(selwave, (naxis2, 1))



    nbadpix = np.nansum(mask_all[:,selwave] == 1, axis=1)
    nallpix = np.nansum(selwave)
    rec_badrows = nbadpix/nallpix > 0.5
    # Replace NaN and zero values to median
    rec_finite = np.zeros_like(flux_all, dtype=bool)
    rec_finite[selection, :] = True
    rec_finite = rec_finite & (mask_all != 1) & (flux_all != 0)
    flux_all[~rec_finite] = np.nan
    flux_all_med = median_filter(flux_all, (21, 1))
    rec_med_upd = ~np.isfinite(flux_all_med)
    flux_all_med1 = median_filter(flux_all_med, (1, 21))
    flux_all_med[rec_med_upd] = flux_all_med1[rec_med_upd]
    flux_all[~rec_finite] = flux_all_med[~rec_finite]
    rssmasked = flux_all[selection] * selwavemask
    errormasked = errors_all[selection] * selwavemask
    rssmasked[rssmasked == 0] = np.nan

    flux_rss = np.nansum(rssmasked, axis=1)*dw
    error_rss = np.nansum(errormasked**2, axis=1)*dw**2

    nselpix = np.sum(np.isfinite(rssmasked), axis=1)
    nselpix1 = np.zeros_like(nselpix)

    if len(wrange) == 4:
        selwave1 = ((wave >= wrange[2]) * (wave <= wrange[3]))
        nbadpix = np.nansum(mask_all[:,selwave1] == 1, axis=1)
        nallpix = np.nansum(selwave1)
        rec_badrows = rec_badrows & (nbadpix / nallpix > 0.5)

        cwave1 = np.mean(wave[selwave1])
        selwavemask = np.tile(selwave1, (naxis2, 1))
        rssmasked = flux_all[selection] * selwavemask
        errormasked = errors_all[selection] * selwavemask
        rssmasked[rssmasked == 0] = np.nan
        flux_rss += (np.nansum(rssmasked, axis=1)*dw)
        error_rss += (np.nansum(errormasked ** 2, axis=1)*dw**2)
        nselpix1 = np.sum(np.isfinite(rssmasked), axis=1)

    # Optional continuum subtraction
    if crange is not None and (len(crange) >= 2):
        cselwave = (wave >= crange[0]) * (wave <= crange[1])

        if len(crange) == 2:
            cflux = np.nanmedian(flux_all[selection, cselwave], axis=1)
            flux_rss -= (cflux * nselpix * dw)
            flux_rss -= (cflux * nselpix1 * dw)
        else:
            cselwave1 = (wave >= crange[2]) * (wave <= crange[3])
            for cur_ind, cur_spec in enumerate(flux_all[selection]):
                mask_fit = np.isfinite(cur_spec) & (cselwave | cselwave1)
                if np.sum(mask_fit) >= 5:
                    res = np.polyfit(wave[mask_fit], cur_spec[mask_fit], 1)
                    p = np.poly1d(res)
                    flux_rss[cur_ind] -= (p(cwave) * nselpix[cur_ind] * dw)
                    if np.isfinite(cwave1):
                        flux_rss[cur_ind] -= (p(cwave1) * nselpix1[cur_ind] * dw)

    flux_rss[(flux_rss == 0) | rec_badrows] = np.nan
    return flux_rss, np.sqrt(error_rss)


def process_single_rss(config, output_dir=None):
    """
    Create table with fluxes from a single rss file
    :param config:
    :param output_dir:
    :return:
    """
    if output_dir is None:
        output_dir = config.get('default_output_dir')
    if not output_dir:
        log.warning(
            "Output directory is not set up. Images will be created in the "
            "individual mjd directories where the drp results are stored")
        output_dir = None

    statuses = []
    for cur_obj in config['object']:
        status_out = True
        f_rss = os.path.join(output_dir, cur_obj['name'], f"{cur_obj['name']}_all_RSS.fits")
        f_tab = os.path.join(output_dir, cur_obj['name'], f"{cur_obj['name']}_fluxes.txt")
        if not os.path.isfile(f_rss):
            log.error(f"File {f_rss} doesn't exist.")
            statuses.append(False)
            continue

        if cur_obj['name'] == 'Orion':
            mean_bounds = (-2, 2)
        else:
            mean_bounds = (-5, 5)
        vel = cur_obj.get('velocity')

        rss = fits.open(f_rss)
        table_fibers = Table(rss['SLITMAP'].data)
        if not config['imaging'].get('override_flux_table') and os.path.isfile(f_tab):
            table_fluxes = Table.read(f_tab, format='ascii.fixed_width_two_line')
        else:
            table_fluxes = table_fibers['fiberid', 'fib_ra', 'fib_dec'].copy()

        line_fit_params = []

        log.info(f"...Compute noise and median continuum level at 6380-6480 AA")

        naxis1 = rss["FLUX"].data.shape[1]
        naxis2 = rss[1].data.shape[0]
        wcsrss = WCS(rss[1].header)
        wave = np.array(wcsrss.pixel_to_world(np.arange(naxis1), 0))
        wave = np.array(wave[0]) * 1e10
        dw = wave[1] - wave[0]
        selwave = (wave >= (6380*(vel / 3e5 + 1))) & (wave <= (6480*(vel / 3e5 + 1)))
        wave = wave[selwave]
        if config['imaging'].get('include_sky'):
            flux_all = (rss['FLUX_ORIG'].data + rss['SKY'].data)[:, selwave]
        elif config['imaging'].get('partial_sky'):
            flux_all = rss['FLUX'].data[:, selwave]
        else:
            flux_all = rss["FLUX_ORIG"].data[:, selwave]
        mask_all = rss["MASK"].data[:, selwave]
        flux_all[mask_all > 0] = np.nan
        flux_all[flux_all == 0] = np.nan
        for kw in ['R_cont_med', 'R_cont_err']:
            if kw not in table_fluxes.colnames:
                table_fluxes.add_column(np.nan, name=kw)
        table_fluxes['R_cont_med'][:] = np.nanmedian(flux_all, axis=1)
        table_fluxes['R_cont_err'][:] = np.nanstd(flux_all, axis=1)

        for line in config['imaging'].get('lines'):
            wl_range = line.get('wl_range')
            if not wl_range or (len(wl_range) < 2):
                log.error(f"Incorrect wavelength range for line {line}")
                statuses.append(False)
                continue
            line_name = line.get('line')

            wl_range = np.array(wl_range) * (vel / 3e5 + 1)
            cont_range = line.get('cont_range')
            if not cont_range or (len(cont_range) < 2):
                cont_range = None
            else:
                cont_range = np.array(cont_range) * (vel / 3e5 + 1)

            mask_wl = line.get('mask_wl')
            if not mask_wl or (len(mask_wl) < 2):
                mask_wl = None
            else:
                mask_wl = np.array(mask_wl) * (vel / 3e5 + 1)

            if 'line_fit' not in line:
                # simple integration
                if not isinstance(line_name, str):
                    line_name = line_name[0]
                log.info(f"...Compute fluxes in {line_name}")
                for kw in ['flux', 'fluxerr']:
                    if f'{line_name}_{kw}' not in table_fluxes.colnames:
                        table_fluxes.add_column(np.nan, name=f'{line_name}_{kw}')
                flux, err = quickflux_single_rss(rss, wl_range, crange=cont_range,
                                                 include_sky=config['imaging'].get('include_sky'),
                                                 partial_sky=config['imaging'].get('partial_sky'))
                table_fluxes[f'{line_name}_flux'][:] = flux
                table_fluxes[f'{line_name}_fluxerr'][:] = err
            else:
                # spectra fitting
                if isinstance(line_name, str):
                    line_name = [line_name]
                for cur_line_name in line_name:
                    for kw in ['flux', 'fluxerr', 'vel', 'velerr', 'disp', 'disperr', 'cont', 'conterr']:
                        if f'{cur_line_name}_{kw}' not in table_fluxes.colnames:
                            table_fluxes.add_column(np.nan, name=f'{cur_line_name}_{kw}')
                for cur_line_name in ['SKY5577', 'SKY6300']:
                    # add some sky lines
                    for kw in ['flux', 'fluxerr', 'vel', 'velerr', 'disp', 'disperr', 'cont', 'conterr']:
                        if f'{cur_line_name}_{kw}' not in table_fluxes.colnames:
                            table_fluxes.add_column(np.nan, name=f'{cur_line_name}_{kw}')
                line_fit = line.get('line_fit')
                fix_ratios = line.get('fix_ratios')
                if not fix_ratios:
                    fix_ratios = None
                include_comp = line.get('include_comp')
                if include_comp is None:
                    include_comp = np.arange(len(line_fit)).astype(int)
                if line.get('show_fit_examples'):
                    save_plot_test = line.get('show_fit_examples')
                else:
                    save_plot_test = None
                if save_plot_test:
                    save_plot_ids = np.random.choice(range(rss['FLUX'].header['NAXIS2']), 24)
                else:
                    save_plot_ids = None
                t = (line_name, wl_range, mask_wl, line_fit,
                                        include_comp, fix_ratios, save_plot_test, save_plot_ids)
                line_fit_params.append(t)

        if len(line_fit_params) > 0:
            nprocs = np.min([np.max([config.get('nprocs'), 1]), rss['FLUX'].header['NAXIS2']])
            spec_ids = np.arange(rss['FLUX'].header['NAXIS2']).astype(int)
            local_statuses = []
            all_plot_data = []

            if config['imaging'].get('include_sky'):
                flux = rss['FLUX_ORIG'].data + rss['SKY'].data
                flux[~np.isfinite(rss['FLUX_ORIG'].data) | (
                            rss['FLUX_ORIG'].data == 0)] = np.nan
            elif config['imaging'].get('partial_sky'):
                flux = rss['FLUX'].data
                flux[flux == 0] = np.nan
            else:
                flux = rss['FLUX_ORIG'].data
                flux[flux == 0] = np.nan
            if mean_bounds is None:
                mean_bounds = (-5, 5)
            sky = rss['SKY'].data
            sky_err = rss['SKY_ERROR'].data
            sky[(rss['MASK'].data > 0) | (rss['SKY'].data == 0)] = np.nan
            error = rss['ERROR'].data
            lsf = rss['FWHM'].data
            flux[(rss['MASK'].data > 0)] = np.nan
            header = rss['FLUX'].header

            vhel = Table(rss['SLITMAP'].data)['vhel_corr']
            params = zip(flux, error, sky, sky_err, lsf, vhel, spec_ids)
            with mp.Pool(processes=nprocs) as pool:
                for status, fit_res, plot_data, spec_id in tqdm(
                                pool.imap_unordered(
                                    partial(fit_all_from_current_spec, header=header,
                                            line_fit_params=line_fit_params,
                                            mean_bounds=mean_bounds, velocity=vel,
                                            ), params),
                                ascii=True, desc="Fit lines in combined RSS",
                                total=rss['FLUX'].header['NAXIS2'], ):
                    local_statuses.append(status)
                    if not status:
                        continue

                    all_plot_data.append(plot_data)
                    for kw in fit_res.keys():
                        table_fluxes[kw][spec_id] = fit_res[kw]

                pool.close()
                pool.join()
                gc.collect()
                if not np.all(local_statuses):
                    status_out = False
                else:
                    status_out = status_out & True
        rss.close()
        table_fluxes.write(f_tab, overwrite=True, format='ascii.fixed_width_two_line')
        statuses.append(status_out)
    return np.all(statuses)


def do_imaging(config, output_dir=None, use_shepard=False):
    if output_dir is None:
        output_dir = config.get('default_output_dir')
    if not output_dir:
        log.warning(
            "Output directory is not set up. Images will be created in the "
            "individual mjd directories where the drp results are stored")
        output_dir = None

    statuses = []
    for cur_obj in config['object']:
        if cur_obj['name'] == 'Orion':
            mean_bounds = (-2, 2)
        else:
            mean_bounds = (-5, 5)
        vel = cur_obj.get('velocity')
        tab_fluxes = None
        for line in config['imaging'].get('lines'):
            wl_range = line.get('wl_range')
            if not wl_range or (len(wl_range) < 2):
                log.error(f"Incorrect wavelength range for line {line}")
                statuses.append(False)
                continue
            wl_range = np.array(wl_range) * (vel / 3e5 + 1)
            cont_range = line.get('cont_range')
            if not cont_range or (len(cont_range) < 2):
                cont_range = None
            else:
                cont_range = np.array(cont_range) * (vel / 3e5 + 1)

            mask_wl = line.get('mask_wl')
            if not mask_wl or (len(mask_wl) < 2):
                mask_wl = None
            else:
                mask_wl = np.array(mask_wl) * (vel / 3e5 + 1)

            if 'line_fit' not in line:
                line_fit = None
                fix_ratios = None
                include_comp = None
                save_plot_test = None
            else:
                line_fit = line.get('line_fit')
                fix_ratios = line.get('fix_ratios')
                if not fix_ratios:
                    fix_ratios = None
                include_comp = line.get('include_comp')
                if isinstance(line.get('save_plot_test'), dict):
                    save_plot_test = line.get('save_plot_test')
                else:
                    save_plot_test = None

            # if cur_obj.get('use_single_rss') and os.path.isfile(output_dir, cur_obj['name'], f"{cur_obj['name']}_all_RSS.fits"):
            #     status = create_line_image_from_rss(file=os.path.isfile(output_dir, cur_obj['name'], f"{cur_obj['name']}_all_RSS.fits"),
            #                                         wdir=os.path.join(output_dir, cur_obj['name']),
            #                                         outfile_prefix=os.path.join(cur_output_dir,
            #                                                                     f'{cur_obj["name"]}_singleRSS'),
            #                                         line_names=line.get("line"), lines_fit=line_fit,
            #                                         fix_ratios=fix_ratios, include_comp=include_comp,
            #                                         nprocs=config['nprocs'],
            #                                         wrange=wl_range, crange=cont_range, mask_wl=mask_wl,
            #                                         skip_bad_fibers=config['imaging'].get('skip_bad_fibers'),
            #                                         include_sky=config['imaging'].get('include_sky'),
            #                                         partial_sky=config['imaging'].get('partial_sky'),
            #                                         mean_bounds=mean_bounds,
            #                                         pxscale_out=pxscale_out,
            #                                         save_fluxes=config['imaging'].get('save_fluxes_table'),
            #                                         r_lim=r_lim, sigma=sigma,
            #                                         do_median_masking=line.get('median_filter')
            #                                         )
            #     statuses.append(status)


            # === Code below is producing combined interpolated images using Kathryn's adaptation of Shepard's method
            if use_shepard:
                cur_output_dir = os.path.join(output_dir, cur_obj['name'], 'maps')
                if not os.path.exists(cur_output_dir):
                    os.makedirs(cur_output_dir)
                    uid = os.stat(cur_output_dir).st_uid
                    os.chown(cur_output_dir, uid=uid, gid=10699)
                    os.chmod(cur_output_dir, 0o775)
                pairs = []
                for ind_point, cur_pointing in enumerate(cur_obj['pointing']):
                    if cur_pointing['skip'].get('imaging'):
                        log.warning(f"Skip imaging for object = {cur_obj['name']}, "
                                    f"pointing = {cur_pointing.get('name')}")
                        continue

                    for data in cur_pointing['data']:
                        if isinstance(data['exp'], int):
                            exps = [data['exp']]
                        else:
                            exps = data['exp']
                        if not data.get('flux_correction'):
                            flux_correction = [1.] * len(exps)
                        else:
                            if isinstance(data['flux_correction'], float) or isinstance(data['flux_correction'], int):
                                flux_correction = [data['flux_correction']]
                            else:
                                flux_correction = data['flux_correction']
                        if not data.get('flux_add') or not line.get('use_add_correction'):
                            flux_add = [None] * len(exps)
                        else:
                            if isinstance(data['flux_add'], float) or isinstance(data['flux_add'], int):
                                flux_add = [data['flux_add']]
                            else:
                                flux_add = data['flux_add']
                        for exp_ind, exp in enumerate(exps):
                            if data.get('dithering'):
                                ref_exp = exps[0]
                            else:
                                ref_exp = None
                            radec_center = derive_radec_ifu(data['mjd'], exp, ref_exp,
                                                            objname=cur_obj['name'],
                                                            pointing_name=cur_pointing.get('name'))
                            # if (cur_pointing.get('name') == 'orion' and
                            #         ((radec_center[0] < 81.8586) or (radec_center[0] > 87.4898) or
                            #          (radec_center[1] < -4.67))):
                            #     #hack for orion
                            #     log.warning(f"MJD={data['mjd']}, expnum={exp} is outside the orion. Skip.")
                            # else:
                            pairs.append((data['mjd'], exp, radec_center,
                                          cur_pointing.get('name'), ind_point, flux_correction[exp_ind], flux_add[exp_ind]))

                if config['imaging'].get('pxscale'):
                    pxscale_out = float(config['imaging'].get('pxscale'))
                else:
                    pxscale_out = 3.
                r_lim=50.
                sigma=2.
                if config['imaging'].get('r_lim'):
                    r_lim = config['imaging'].get('r_lim')
                if config['imaging'].get('r_lim'):
                    sigma = config['imaging'].get('sigma')

                status, fluxes = reconstruct_shepards_method(
                    pairs, wdir=os.path.join(output_dir, cur_obj['name']),
                    outfile_prefix=os.path.join(cur_output_dir, f'{cur_obj["name"]}_combined_shepard'),
                    line_names=line.get("line"), lines_fit=line_fit, fix_ratios=fix_ratios, include_comp=include_comp,
                    nprocs=np.min([config['nprocs'], len(pairs)]),
                    wrange=wl_range, crange=cont_range, mask_wl=mask_wl, save_plot_test=save_plot_test,
                    skip_bad_fibers=config['imaging'].get('skip_bad_fibers'),
                    include_sky=config['imaging'].get('include_sky'),
                    partial_sky=config['imaging'].get('partial_sky'), mean_bounds=mean_bounds,
                    pxscale_out=pxscale_out, save_fluxes=config['imaging'].get('save_fluxes_table'),
                    r_lim=r_lim, sigma=sigma, do_median_masking=line.get('median_filter'))
                if not status:
                    log.error(f"Something wrong with the imaging of {cur_obj['name']} in {line.get('line')} line")
                elif config['imaging'].get('save_fluxes_table'):
                    if tab_fluxes is None:
                        names = ["Exp", "Noise", "RA", "DEC"]
                        dtype = [int, float, float, float]
                        d = [fluxes[0], fluxes[5], fluxes[2], fluxes[3]]
                        for c_id, n in enumerate(fluxes[4]):
                            names.append(n)
                            dtype.append(float)
                            if c_id == 0 and (len(fluxes[1].shape)==1):
                                d.append(fluxes[1])
                            else:
                                d.append(fluxes[1][:, c_id])
                        tab_fluxes = Table(data=d,
                                           names=names, dtype=dtype)
                    else:
                        for c_id, n in enumerate(fluxes[4]):
                            if c_id == 0 and (len(fluxes[1].shape)==1):
                                d = fluxes[1]
                            else:
                                d = fluxes[1][:, c_id]
                            tab_fluxes.add_column(d, name=n)

                statuses.append(status)


            else:
                # === Code below is producing single images using Guillermo's quickmap
                suffix = line.get('line')
                if not isinstance(suffix, str):
                    log.error("Fitting is not implemented into Guillermo's quickmap. Exit.")
                    return False
                else:
                    suffix = f'_{suffix}'
                for cur_pointing in cur_obj['pointing']:
                    if cur_pointing['skip'].get('imaging'):
                        log.warning(f"Skip imaging for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
                        continue
                    if output_dir is None:
                        cur_output_dir = None
                    else:
                        if not cur_pointing.get('name'):
                            cur_output_dir = os.path.join(output_dir, cur_obj['name'], 'maps')
                        else:
                            cur_output_dir = os.path.join(output_dir, cur_obj['name'], cur_pointing.get('name'), 'maps')

                    pairs = []
                    for data in cur_pointing['data']:
                        if isinstance(data['exp'], int):
                            exps = [data['exp']]
                        else:
                            exps = data['exp']
                        for exp in exps:
                            pairs.append((data['mjd'], exp, data['dithering'], exps[0]))

                    if not os.path.exists(cur_output_dir):
                        os.makedirs(cur_output_dir)
                        uid = os.stat(cur_output_dir).st_uid
                        os.chown(cur_output_dir, uid=uid, gid=10699)
                        os.chmod(cur_output_dir, 0o775)

                    log.warning(f"Start imaging in {line.get('line')} line for individual exposures of "
                                f"object = {cur_obj['name']},"
                                f"pointing = {cur_pointing.get('name')}")
                    procs = np.nanmin([config['nprocs'], len(pairs)])
                    local_statuses=[]
                    with mp.Pool(processes=procs) as pool:
                        for status in tqdm(pool.imap_unordered(
                                partial(quickmap_parallel, wl_range=wl_range, cont_range=cont_range,
                                        output_dir=cur_output_dir, suffix=suffix,
                                        include_sky=config['imaging'].get('include_sky'),
                                        skip_bad_fibers=config['imaging'].get('skip_bad_fibers')), pairs),
                                           ascii=True, desc="Create images",
                                           total=len(pairs),):
                            local_statuses.append(status)
                        pool.close()
                        pool.join()
                        gc.collect()
                        if not np.all(local_statuses):
                            log.error(
                                f"Something wrong with the imaging of {cur_obj['name']}, "
                                f"pointing {cur_pointing.get('name')} in {line.get('line')} line")
                            statuses.append(False)
                        else:
                            statuses.append(True)

        if tab_fluxes is not None:
            tab_fluxes.write(os.path.join(output_dir, cur_obj['name'], 'maps', f"{cur_obj['name']}_fibers_fluxes.txt"),
                             overwrite=True, format='ascii.fixed_width_two_line')

    return np.all(statuses)


def do_cube_construction(config, output_dir=None):
    if output_dir is None:
        output_dir = config.get('default_output_dir')
    if not output_dir:
        log.warning(
            "Output directory is not set up. Images will be created in the "
            "individual mjd directories where the drp results are stored")
        output_dir = None

    statuses = []
    for cur_obj in config['object']:
        vel = cur_obj.get('velocity')
        cube_rec_params = config.get('cube_reconstruction')
        suffix = ''
        wrange = None
        if isinstance(cube_rec_params, dict):
            wrange = cube_rec_params.get('wl_range')
            if not wrange or (len(wrange) < 2):
                wrange = None
                log.error(f"Incorrect wavelenght range cube extraction. Using full spectral range")
            else:
                wrange = np.array(wrange) * (vel / 3e5 + 1)
                suffix = cube_rec_params.get('suffix')
                if suffix:
                    suffix = f'_{suffix}'
        cur_output_dir = os.path.join(output_dir, cur_obj['name'], 'maps')
        if not os.path.exists(cur_output_dir):
            os.makedirs(cur_output_dir)
            uid = os.stat(cur_output_dir).st_uid
            os.chown(cur_output_dir, uid=uid, gid=10699)
            os.chmod(cur_output_dir, 0o775)
        pairs = []
        for ind_point, cur_pointing in enumerate(cur_obj['pointing']):
            if cur_pointing['skip'].get('create_cube'):
                log.warning(f"Skip pointing = {cur_pointing.get('name')} for {cur_obj['name']} cube construction")
                continue
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']
                for ind_exp, exp in enumerate(exps):
                    if data.get('dithering'):
                        ref_exp = exps[0]
                    else:
                        ref_exp = None
                    if not data.get('flux_correction'):
                        flux_correction = [1.] * len(exps)
                    else:
                        if isinstance(data['flux_correction'], float) or isinstance(data['flux_correction'], int):
                            flux_correction = [data['flux_correction']]
                        else:
                            flux_correction = data['flux_correction']
                    radec_center = derive_radec_ifu(data['mjd'], exp, ref_exp,
                                                    objname=cur_obj['name'],
                                                    pointing_name=cur_pointing.get('name'))
                    # radec_center = derive_radec_ifu(data['mjd'], exp, ref_exp)
                    pairs.append((data['mjd'], exp, radec_center, cur_pointing.get('name'),
                                  ind_point, flux_correction[ind_exp]))
                    

        if config['imaging'].get('pxscale'):
            pxscale_out = float(config['imaging'].get('pxscale'))
        else:
            pxscale_out = 3.
        if config['imaging'].get('r_lim'):
            r_lim = config['imaging'].get('r_lim')
        else:
            r_lim=50
        if config['imaging'].get('r_lim'):
            sigma = config['imaging'].get('sigma')
        else:
            sigma=2
        status = reconstruct_cube_shepards_method(pairs, outfile=os.path.join(cur_output_dir,
                                                                              f'{cur_obj["name"]}_cube{suffix}.fits'),
                                                  wdir=os.path.join(output_dir,cur_obj["name"]),
                                                  nprocs=np.min([config['nprocs'], len(pairs)]),
                                                  wrange=wrange,
                                                  skip_bad_fibers=config['imaging'].get('skip_bad_fibers'),
                                                  include_sky=config['imaging'].get('include_sky'),
                                                  partial_sky=config['imaging'].get('partial_sky'),
                                                  pxscale_out=pxscale_out, r_lim=r_lim, sigma=sigma)
        if not status:
            log.error(f"Something wrong with the cube construction for {cur_obj['name']}")
        statuses.append(status)


    return np.all(statuses)


def combine_images(config, w_dir=None):
    if w_dir is None:
        w_dir = config.get('default_output_dir')
    if not os.path.exists(w_dir):
        log.error("Work directory is not set or does not exists. Cannot proceed with combining of the images")
        return False
    for cur_obj in config['object']:
        for line in config['imaging'].get('lines'):
            files_in = []
            suffix = f"_{line.get('line')}"
            if config['imaging'].get('interpolate'):
                suffix += '_interp'
            dir_out = os.path.join(w_dir, cur_obj['name'], 'maps')
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)
                uid = os.stat(dir_out).st_uid
                os.chown(dir_out, uid=uid, gid=10699)
                os.chmod(dir_out, 0o775)
            fileout = os.path.join(dir_out, f'combined{suffix}.fits')
            for cur_pointing in cur_obj['pointing']:
                if cur_pointing['skip'].get('imaging'):
                    log.warning(f"Skip imaging for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
                    continue
                if not cur_pointing.get('name'):
                    cur_in_dir = os.path.join(w_dir, cur_obj['name'], 'maps')
                else:
                    cur_in_dir = os.path.join(w_dir, cur_obj['name'], cur_pointing.get('name'), 'maps')
                for data in cur_pointing['data']:
                    if isinstance(data['exp'], int):
                        exps = [data['exp']]
                    else:
                        exps = data['exp']

                    for exp in exps:
                        f = os.path.join(cur_in_dir, f"quickmap_{exp:0>8}{suffix}.fits")
                        if os.path.exists(f):
                            files_in.append(f)
            if len(files_in) == 0:
                log.warning(f'No images found for object = {cur_obj["name"]} => nothing to combine.')
                continue

            wcs_out, shape_out = find_optimal_celestial_wcs(files_in, hdu_in=0)
            log.info(f"Start combining {len(files_in)} images for object {cur_obj['name']} in {line.get('line')}")
            for f_ind, f in tqdm(enumerate(files_in), total=len(files_in), ascii=True):
                with fits.open(f) as hdu:
                    wcs = WCS(hdu[0].header)
                    data = hdu[0].data
                    data[data == 0] = np.nan
                    data[data < -1000] = np.nan
                    img_out, fp = reproject_interp((data, wcs), wcs_out, shape_out)
                    fp = fp.astype(bool)
                    img_out[~fp] = np.nan
                    if f_ind == 0:
                        header_out = hdu[0].header
                        for kw in ['PC1_1','PC1_2','PC2_1','PC2_2']:
                            if kw in header_out:
                                del header_out[kw]
                        header_out.update(wcs_out.to_header())
                        img_combined = img_out.reshape(1, img_out.shape[0], img_out.shape[1])
                        continue
                    img_combined = np.vstack([img_combined, img_out.reshape(1, img_out.shape[0], img_out.shape[1])])
            img_combined = np.nanmedian(img_combined, axis=0)
            fits.writeto(fileout, data=img_combined, header=header_out, overwrite=True)
    return True


def rotate(xx,yy,angle):
    # rotate x and y cartesian coordinates by angle (in degrees)
    # about the point (0,0)
    theta = -np.radians(angle)
    xx1 = np.cos(theta) * xx - np.sin(theta) * yy
    yy1 = np.sin(theta) * xx + np.cos(theta) * yy

    return xx1, yy1


def make_radec(xx0,yy0,ra,dec,pa):
    platescale = 112.36748321030637  # Focal plane platescale in "/mm

    xx, yy = rotate(xx0, yy0, pa)
    ra_fib = ra + xx * platescale/3600./np.cos(np.radians(dec))
    dec_fib = dec - yy * platescale/3600.
    return ra_fib, dec_fib


def extract_flux_and_coords_parallel(data, wrange=None, crange=None, wrange_cube=None, mask_wl=None,
                                     skip_bad_fibers=False, include_sky=False, partial_sky=False, consider_as_comp=None,save_plot_test=None,
                                     wdir=None, lines_fit=None, fix_ratios=None, velocity=0, mean_bounds=(-5,5)):
    mjd = data[0]
    expnum = data[1]
    cur_save_plot_test = None
    if save_plot_test is not None:
        if str(mjd) in save_plot_test:
            if str(expnum) in save_plot_test.get(str(mjd)):
                cur_save_plot_test = save_plot_test[str(mjd)][str(expnum)]['fileout']

    flux_corr_cf = data[5]
    if len(data) > 6:
        flux_add = data[6]
    else:
        flux_add=None
    if flux_add is not None:
        flux_add = flux_add*35.3 ** 2 * np.pi / 4
    LVMDATA_DIR = drp_results_dir  # os.path.join(SAS_BASE_DIR, 'sdsswork','lvm','lco')
    use_comb = False
    if wdir is not None and os.path.exists(wdir):
        f_comb = glob.glob(os.path.join(wdir, data[3], f'combined_spectrum_*.fits'))
        if len(f_comb) == 0:
            rssfile = os.path.join(wdir, data[3], f'lvmCFrame-{expnum:0>8}.fits')
            # if not os.path.exists(rssfile):
            #     rssfile = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{expnum:0>8}.fits"
        elif len(f_comb) > 1:
            f_comb = glob.glob(os.path.join(wdir, data[3], f'combined_spectrum_{data[4]:0>2}.fits'))
            if len(f_comb) == 0:
                rssfile = os.path.join(wdir, data[3], f'lvmCFrame-{expnum:0>8}.fits')
                # if not os.path.exists(rssfile):
                #     rssfile = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{expnum:0>8}.fits"
            else:
                use_comb = True
                rssfile = f_comb[0]
                flux_corr_cf = 1.
        else:
            use_comb = True
            rssfile = f_comb[0]
            flux_corr_cf = 1.
    else:
        rssfile = f"{LVMDATA_DIR}/{mjd}/lvmCFrame-{expnum:0>8}.fits"

    if not os.path.exists(rssfile):
        log.error(f"{rssfile} is not found! Skip it.")
        return False, None, None, None, None, None, None, None, None
    rss = fits.open(rssfile)
    rss['FLUX'].data[rss['MASK'] ==1] = np.nan
    rss['ERROR'].data[rss['MASK'] == 1] = np.nan
    rss['SKY'].data[rss['MASK'] == 1] = np.nan
    if not rss[0].header.get('FLUXCAL'):
        log.error(f"Missing flux calibration for mjd={mjd} and expnum={expnum}. Skip it.")
        rss.close()
        return False, None, None, None, None, None, None, None, None
    tab = Table(rss['SLITMAP'].data)
    sci = np.flatnonzero(tab['targettype'] == 'science')
    if skip_bad_fibers:
        sci = np.flatnonzero((tab['targettype'] == 'science') & (tab['fibstatus'] == 0))
    ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], data[2][0], data[2][1], data[2][2])

    if wrange is not None:
        texp = rss[0].header['EXPTIME']
        wave = ((np.arange(rss[1].header['NAXIS1'])-rss[1].header['CRPIX1']+1)*rss[1].header['CDELT1'] +
                rss[1].header['CRVAL1']) * 1e10
        sel_wave = (wave >= 6400) * (wave <= 6500)
        estimate_noise_array = rss[1].data[sci]
        estimate_noise_array = estimate_noise_array[:, sel_wave] / texp
        noise = np.nanmedian(np.absolute(estimate_noise_array - np.nanmedian(estimate_noise_array)))

        if lines_fit is not None and len(lines_fit)>0:
            cur_wrange = wrange
            if crange is not None:
                cur_wrange[0] = min([crange[0], cur_wrange[0]])
                cur_wrange[1] = min([crange[1], cur_wrange[1]])
                if len(crange) == 4:
                    cur_wrange[0] = min([crange[2], cur_wrange[0]])
                    cur_wrange[1] = min([crange[3], cur_wrange[1]])
            flux, vel, disp, cont = fit_spectra(rss, cur_wrange, selection=sci, mask_wl=mask_wl, mean_bounds=mean_bounds,
                                                ra_fib=ra_fib, dec_fib=dec_fib, include_sky=include_sky,
                                                partial_sky=partial_sky, expnum=expnum,
                                                save_plot_test=cur_save_plot_test, flux_add=flux_add,
                                                flux_corr_cf=flux_corr_cf, consider_as_comp=consider_as_comp,
                                                lines=lines_fit, fix_ratios=fix_ratios, velocity=velocity)

            cur_wrange_sky = [5560, 5590]#[6280, 6320]
            _, vel_sky, disp_sky, _ = fit_spectra(rss, cur_wrange_sky, selection=sci, mask_wl=mask_wl,
                                                mean_bounds=(-2,2),sky_only=True,
                                                ra_fib=ra_fib, dec_fib=dec_fib, include_sky=True,
                                                flux_corr_cf=flux_corr_cf, consider_as_comp=[0],
                                                lines=[5577.], fix_ratios=None, velocity=0, do_helio_corr=False) #6300.304

            if flux.shape[2] > 1:
                flux_err = flux[:, :, 1]
                vel_err = vel[:, :, 1]
                disp_err = disp[:,:,1]
                cont_err = cont[:, :, 1]
            else:
                flux_err = np.zeros_like(flux)[:, :, 0]
                vel_err = np.zeros_like(vel)[:, :, 0]
                disp_err = np.zeros_like(disp)[:, :, 0]
                cont_err = np.zeros_like(cont)[:, :, 0]
            flux = flux[:, :, 0]
            vel = vel[:, :, 0] - vel_sky[:, :, 0] #vel_sky[:, :, 0]#
            disp = disp[:, :, 0]
            cont = cont[:, :, 0]

            n_good = np.sum([f > 0 for f in flux[:, 0]])
            n_tot = len(flux[:, 0])
            if n_good / n_tot < 0.3:
                log.warning(f"less than 30% success for expnum={expnum}")
            flux[flux == 0] = np.nan
        else:
            flux = quickflux(rss, wrange, crange, selection=sci, include_sky=include_sky, partial_sky=partial_sky)
            flux = flux * flux_corr_cf
            if flux_add is not None:
                flux += flux_add
            flux_err = np.zeros_like(flux)
            vel_err = None
            disp_err = None
            cont_err = None
            vel = None
            disp = None
            cont = None
        wave_dict = None
    else:
        # assume cube reconstruction
        flux = rss['FLUX'].data[sci]
        if include_sky:
            flux += rss['SKY'].data[sci]
        flux = flux * flux_corr_cf  #/rss[0].header['EXPTIME']
        if wrange_cube is not None:
            wl_grid = ((np.arange(rss['FLUX'].header['NAXIS1'])-
                       rss['FLUX'].header['CRPIX1']+1) * rss['FLUX'].header['CDELT1']*1e10 +
                       rss['FLUX'].header['CRVAL1']*1e10)
            rec = np.flatnonzero((wl_grid >= wrange_cube[0]) & (wl_grid <= wrange_cube[1]))
            flux = flux[:,rec]
            crval = wl_grid[rec[0]]
        else:
            crval = rss['FLUX'].header['CRVAL1']*1e10

        wave_dict = {"CRVAL3": crval,
                     'CDELT3': rss['FLUX'].header['CDELT1']*1e10,
                     'CRPIX3': rss['FLUX'].header['CRPIX1'],
                     "CTYPE3": rss['FLUX'].header['CTYPE1'],
                     'CUNIT3': 'Angstrom',
                     'BUNIT': rss['FLUX'].header['BUNIT']}

    rss.close()
    if not use_comb:
        # # ## Fix for Orion only!!!
        if expnum == 5355:
            if wrange is not None and (wrange[0] > 9000):
                flux = flux * 0.75#/ 5.3 *3.4
        if expnum == 7328:
            if wrange is not None and (wrange[0] > 9000):
                flux = flux * 1.31
        if expnum == 7329:
            if wrange is not None and (wrange[0] > 9000):
                flux = flux * 1.9
    else:
        if expnum == 5355:
            if wrange_cube is not None and (wrange_cube[0] > 9000):
                flux = flux / 5.3 *3.4
    if use_comb:
        comb_id = rssfile.split('_')[-1].split('.fits')[0]
    else:
        comb_id = None
    if wave_dict is not None:
        return True, ra_fib, dec_fib, flux, comb_id, wave_dict
    else:
        return True, ra_fib, dec_fib, flux, comb_id, vel, disp, cont, noise, flux_err, vel_err, disp_err, cont_err


def shepard_convolve(wcs_out, shape_out, ra_fibers=None, dec_fibers=None, show_values=None, r_lim=50., sigma=2.,
                     cube=False, header=None, outfile=None, do_median_masking=False):
    if not cube:
        if len(show_values.shape) ==1:
            show_values = show_values.reshape((-1, 1))
        if show_values.shape[1] > 7:
            rec_fibers = np.isfinite(show_values[:,0]) & (show_values[:,0] != 0) #& np.isfinite(show_values[:,-4]) & (show_values[:,-4] < 35) & (show_values[:,-4] > 18)
        else:
            rec_fibers = np.isfinite(show_values[:, 0]) & (show_values[:, 0] != 0)
        show_values = show_values[rec_fibers,:]
    else:
        rec_fibers = np.isfinite(np.nansum(show_values.T, axis=0)) & (np.nansum(show_values.T, axis=0) != 0)
        flux_fibers = show_values[rec_fibers, :]
    pxsize = wcs_out.proj_plane_pixel_scales()[0].value * 3600.
    radec = SkyCoord(ra=ra_fibers[rec_fibers], dec=dec_fibers[rec_fibers], unit='deg', frame='icrs')
    x_fibers, y_fibers = wcs_out.world_to_pixel(radec)
    # x_fibers = np.round(x_fibers).astype(int)
    # y_fibers = np.round(y_fibers).astype(int)
    chunk_size = min([int(r_lim * 15 / pxsize), 100])
    khalfsize = int(np.ceil(r_lim / pxsize))
    kernel_size = khalfsize * 2 + 1

    xx_kernel, yy_kernel = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    # dist_2 = ((xx_kernel - kernel_size) ** 2 + (yy_kernel - kernel_size) ** 2) * pxsize ** 2
    # kernel = np.exp(-0.5 * dist_2 / sigma ** 2)
    # kernel[dist_2 > (r_lim ** 2)] = 0
    # kernel = np.expand_dims(kernel, 0)

    nchunks_x = int(np.ceil(shape_out[1]/chunk_size))
    nchunks_y = int(np.ceil(shape_out[0] / chunk_size))
    if not cube:
        img_out = np.zeros(shape=(shape_out[0], shape_out[1], show_values.shape[1]), dtype=float)
    else:
        # img_out = np.zeros(shape=(len(flux_fibers), shape_out[0], shape_out[1]), dtype=np.float32)
        shape = tuple(header[f'NAXIS{ii}'] for ii in range(1, header['NAXIS'] + 1))
        with open(outfile, 'rb+') as fobj:
            fobj.seek(len(header.tostring()) + (np.prod(shape) * np.abs(header['BITPIX'] // 8)) - 1)
            fobj.write(b'\0')

        hdu_out = fits.open(outfile, ignore_missing_simple=True)

    last_x = -1
    last_y = -1
    log.info(f"Create image in {nchunks_x * nchunks_y} chunks")

    for _ in tqdm(range(nchunks_x*nchunks_y), total=nchunks_x*nchunks_y, ascii=True, desc="Chunks done"):

        nx = np.min([shape_out[1] - 1 - last_x, chunk_size]).astype(int)
        dx0 = np.min([khalfsize, last_x+1]).astype(int)
        dx1 = np.min([shape_out[1] - 1 - last_x - nx, khalfsize]).astype(int)
        nx += (dx0 + dx1)
        ny = np.min([shape_out[0] - 1 - last_y, chunk_size]).astype(int)
        dy0 = np.min([khalfsize, last_y + 1]).astype(int)
        dy1 = np.min([shape_out[0] - 1 - last_y - ny, khalfsize]).astype(int)
        ny += (dy0 + dy1)

        rec_fibers = np.flatnonzero((#(np.isfinite(flux_fibers) & (flux_fibers != 0) &
                      (x_fibers >= (last_x + 1 - dx0)) & (x_fibers < (last_x + 1 - dx0 + nx)) &
                      (y_fibers >= (last_y + 1 - dy0)) & (y_fibers < (last_y + 1 - dy0 + ny))))
        if len(rec_fibers) == 0:
            last_y += chunk_size
            if last_y >= shape_out[0]:
                last_x += chunk_size
                last_y = -1
            continue

        weights = np.zeros(shape=(len(rec_fibers), ny, nx), dtype=float)

        for ind, xy in enumerate(zip(x_fibers[rec_fibers], y_fibers[rec_fibers])):
            frac_x = xy[0] - np.floor(xy[0])
            frac_y = xy[1] - np.floor(xy[1])
            cur_center_y = np.round(xy[1]).astype(int) - last_y - 1 + dy0
            cur_center_x = np.round(xy[0]).astype(int) - last_x - 1 + dx0
            cur_x0 = max([0, cur_center_x - khalfsize])
            cur_y0 = max([0, cur_center_y - khalfsize])
            cur_x1 = min([nx-1, cur_center_x + khalfsize])
            cur_y1 = min([ny-1, cur_center_y + khalfsize])

            dist_2 = ((xx_kernel - khalfsize - frac_x) ** 2 + (yy_kernel - khalfsize - frac_y) ** 2) * pxsize ** 2
            kernel = np.exp(-0.5 * dist_2 / sigma ** 2)
            kernel[dist_2 > (r_lim ** 2)] = 0
            # if ~np.isfinite(show_values[rec_fibers][ind, 0]) or show_values[rec_fibers][ind, 0] == 0:
            #     kernel = kernel * 0
            weights[ind, cur_y0:cur_y1+1, cur_x0: cur_x1+1] = kernel[khalfsize - (cur_center_y-cur_y0): khalfsize - (cur_center_y-cur_y0)+ cur_y1 - cur_y0+1,
                                                                     khalfsize - (cur_center_x-cur_x0): khalfsize - (cur_center_x-cur_x0)+cur_x1 - cur_x0+1]

        weights_norm = np.sum(weights, axis=0)
        weights_norm[weights_norm == 0] = 1
        weights = weights / weights_norm[None, :, :]
        n_used_fib = np.sum(weights > 0, axis=0)
        n_used_fib = np.broadcast_to(n_used_fib, weights.shape)
        weights[n_used_fib < 3] = 0
        if not cube:
            img_chunk = np.nansum(weights[:, :, :, None] * show_values[rec_fibers, None, None, :], axis=0)

            img_out[last_y + 1: last_y + 1 + ny - dy0 - dy1,
                    last_x + 1: last_x + 1 + nx - dx0 - dx1, :] = img_chunk[dy0: ny - dy1, dx0: nx - dx1, :]
        else:
            # img_chunk = np.nansum(weights[:,None, :, :] * flux_fibers[rec_fibers, :, None, None], axis=0)
            sw = flux_fibers.shape[1]
            # flux_fibers = flux_fibers.T
            # for wl_ind in range(sw):
            hdu_out[0].data[:, last_y + 1: last_y + 1 + ny - dy0 - dy1,
            last_x + 1: last_x + 1 + nx - dx0 - dx1] = np.nansum(weights[:, None, dy0: ny - dy1, dx0: nx - dx1] * flux_fibers[rec_fibers,:, None, None], axis=0)


        last_y += chunk_size
        if last_y >= shape_out[0]:
            last_x += chunk_size
            last_y = -1
    # img_out = median_filter(img_out, (20,20))#median_filter(img_out, (25, 25))
    if cube:
        hdu_out.writeto(outfile, overwrite=True, output_verify='silentfix')
    else:
        return img_out


# def apply_shepard_parallel(radec_grid, ra_fibers=None, dec_fibers=None, flux_fibers=None, r_lim=50., sigma=2.):
#     ra_grid, dec_grid = radec_grid
#     dds = sphdist(ra_grid, dec_grid, ra_fibers, dec_fibers) * 3600.
#
#     iis = np.flatnonzero(dds < r_lim)
#     if len(iis) == 0:
#         return np.nan
#     # rr = sphdist(ra_grid, dec_grid, ra_fibers[iis], dec_fibers[iis]) * 3600.  # arcsec
#     weight_arr = np.exp(-0.5 * (dds[iis] / sigma) ** 2)
#     weight_arr[~np.isfinite(flux_fibers[iis]) | (flux_fibers[iis]==0)] = 0
#     if np.nansum(weight_arr) == 0:
#         return np.nan
#     norm = 1. / np.nansum(weight_arr)
#     return np.nansum(norm * weight_arr * flux_fibers[iis])



def reconstruct_cube_shepards_method(data, pxscale_out=3.,r_lim=50, sigma=2.,wdir=None,wrange=None,
                                     outfile=None, skip_bad_fibers=False, nprocs=8, include_sky=False, partial_sky=False):
    # data: [(mjds, exps, (ra_cent_ifu, dec_cent_ifu, pa_ifu))]
    status_out = True
    platescale = 112.36748321030637  # Focal plane platescale in "/mm
    pscale = 0.01  # IFU image pixel scale in mm/pix
    lvm_fiber_diameter = 35.3
    rspaxel = lvm_fiber_diameter / platescale / 2  # spaxel radius in mm assuming 35.3" diameter chromium mask
    fov = 1500  # size of IFU image (arcsec)
    skypscale = pxscale_out/3600. #pscale * platescale / 3600  # IFU image pixel scale in deg/pix
    npix = int(fov/3600./skypscale)
    shape = (npix, npix)
    # create wcs for individual ifu images
    wcs_in = []

    for d in data:
        w = WCS(naxis=2)
        w.wcs.crpix = [int(npix / 2) + 1, int(npix / 2) + 1]
        posangrad = np.radians(d[2][2])
        w.wcs.cd = np.array([[skypscale * np.cos(posangrad), -1 * skypscale * np.sin(posangrad)],
                         [-1 * skypscale * np.sin(posangrad), -1 * skypscale * np.cos(posangrad)]])
        w.wcs.crval = [d[2][0], d[2][1]]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        #header = w.to_header()
        wcs_in.append((shape, w))

    # construct grid for combined image
    wcs_out, shape_out = find_optimal_celestial_wcs(wcs_in)

    grid_scl = wcs_out.proj_plane_pixel_scales()[0].value * 3600.

    log.info(f"Grid scale: {grid_scl}")

    ny, nx = shape_out
    xx = np.arange(nx)
    yy = np.arange(ny)

    img_arr = np.zeros((ny, nx), dtype=float)
    img_arr[:, :] = np.nan

    fluxes = None
    ras = np.array([])
    decs = np.array([])

    local_statuses = []
    log.info(f"Extract fluxes and astrometry for {len(data)} IFU fields")
    comb_id_seen = []

    with mp.Pool(processes=nprocs) as pool:
        for status, ra, dec, flux, comb_id, wave_dict in tqdm(pool.imap_unordered(
                partial(extract_flux_and_coords_parallel,
                        skip_bad_fibers=skip_bad_fibers, include_sky=include_sky, partial_sky=partial_sky, wdir=wdir, wrange_cube=wrange),
                data),
                ascii=True, desc="Extract fluxes and astrometry",
                total=len(data), ):
            local_statuses.append(status)

            if status:
                if comb_id is None or comb_id not in comb_id_seen:
                    if fluxes is None:
                        fluxes = flux
                    else:
                        fluxes = np.vstack([fluxes, flux])
                    ras = np.append(ras, ra)
                    decs = np.append(decs, dec)
                    if comb_id is not None:
                        comb_id_seen.append(comb_id)
        # fluxes = fluxes.T
        pool.close()
        pool.join()
        gc.collect()

        if not np.all(local_statuses):
            status_out = status_out & False
        else:
            status_out = status_out & True

    fluxes = np.array(fluxes)
    ras = np.array(ras)
    decs = np.array(decs)
    log.info(f"Construct interpolated cube")

    fake_data = np.zeros((100, 100), dtype=np.float64)
    hdu = fits.PrimaryHDU(data=fake_data)
    header = hdu.header
    while len(header) < (36 * 4 - 1):
        header.append()

    header.update(wcs_out.to_header())
    # header['BITPIX'] = 8
    header['NAXIS'] = 3
    header['NAXIS1'] = shape_out[1]
    header['NAXIS2'] = shape_out[0]
    header['NAXIS3'] = fluxes.shape[1]
    for kw in wave_dict.keys():
        header[kw] = wave_dict[kw]

    header.tofile(outfile, overwrite=True)

    shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs, show_values=fluxes,
                               r_lim=r_lim, sigma=sigma, cube=True, outfile=outfile, header=header)


    return status_out


def reconstruct_shepards_method(data, wrange=None, crange=None, mask_wl=None, pxscale_out=3., r_lim=50, sigma=2.,
                                wdir=None, velocity=0, lines_fit=None, fix_ratios=None, do_median_masking=False,
                                outfile_prefix=None, line_names=None, include_comp=None, mean_bounds=(-5,5),
                                save_plot_test=None,
                                skip_bad_fibers=False, nprocs=8, include_sky=False, partial_sky=False, save_fluxes=False):
    # data: [(mjds, exps, (ra_cent_ifu, dec_cent_ifu, pa_ifu))]
    if line_names is None:
        if lines_fit is not None:
            line_names = [f'someline_{ind+1}' for ind in range(len(lines_fit))]
        else:
            line_names = ['someline']
    if not isinstance(line_names, tuple) and not isinstance(line_names, list):
        line_names = [line_names]

    status_out = True
    platescale = 112.36748321030637  # Focal plane platescale in "/mm
    pscale = 0.01  # IFU image pixel scale in mm/pix
    lvm_fiber_diameter = 35.3
    rspaxel = lvm_fiber_diameter / platescale / 2  # spaxel radius in mm assuming 35.3" diameter chromium mask
    fov = 1800  # size of IFU image (arcsec)
    skypscale = pxscale_out/3600. #pscale * platescale / 3600  # IFU image pixel scale in deg/pix
    npix = int(fov/3600./skypscale)
    shape = (npix, npix)
    # create wcs for individual ifu images
    wcs_in = []

    for d in data:
        w = WCS(naxis=2)
        w.wcs.crpix = [int(npix / 2) + 1, int(npix / 2) + 1]
        posangrad = np.radians(d[2][2])
        w.wcs.cd = np.array([[skypscale * np.cos(posangrad), -1 * skypscale * np.sin(posangrad)],
                         [-1 * skypscale * np.sin(posangrad), -1 * skypscale * np.cos(posangrad)]])
        w.wcs.crval = [d[2][0], d[2][1]]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs_in.append((shape, w))

    # construct grid for combined image
    wcs_out, shape_out = find_optimal_celestial_wcs(wcs_in)
    grid_scl = wcs_out.proj_plane_pixel_scales()[0].value * 3600.

    log.info(f"Grid scale: {grid_scl}")

    ny, nx = shape_out
    xx = np.arange(nx)
    yy = np.arange(ny)
    coord_x, coord_y = np.meshgrid(xx, yy)

    img_arr = np.zeros((ny, nx), dtype=float)
    img_arr[:, :] = np.nan

    if lines_fit is not None:
        n_other_params = 3
    else:
        n_other_params = 0
    values = None
    ras = np.array([])
    decs = np.array([])
    exps = np.array([])
    noises = np.array([])

    local_statuses = []
    log.info(f"Extract fluxes and astrometry for {len(data)} IFU fields")

    comb_id_seen = []
    bgr_flux = []
    with mp.Pool(processes=nprocs) as pool:
        cur_ind = 0
        for status, ra, dec, flux, comb_id, vel, disp, cont, noise, flux_err, vel_err, disp_err, cont_err in tqdm(
                pool.imap(
                partial(extract_flux_and_coords_parallel, wrange=wrange, crange=crange, velocity=velocity,
                        save_plot_test=save_plot_test,
                        lines_fit=lines_fit, fix_ratios=fix_ratios,consider_as_comp=include_comp, mean_bounds=mean_bounds,
                        skip_bad_fibers=skip_bad_fibers, include_sky=include_sky, partial_sky=partial_sky,
                        wdir=wdir, mask_wl=mask_wl), data),
                ascii=True, desc="Extract fluxes and astrometry",
                total=len(data), ):

            # if np.sum(flux_err) != 0:
            #     bad_rec = abs(flux/flux_err) < 10
            #     flux[bad_rec] = np.nan
            #     if vel is not None:
            #         vel[bad_rec] = np.nan
            #     if disp is not None:
            #         disp[bad_rec] = np.nan
            # if vel_err is not None and vel is not None:
            #     bad_rec = vel_err > 4.
            #     vel[bad_rec] = np.nan
            #     flux[bad_rec] = np.nan
            #     if disp is not None:
            #         disp[bad_rec] = np.nan
            # if disp_err is not None and disp is not None:
            #     bad_rec = disp_err > 7.
            #     disp[bad_rec] = np.nan
            #     if vel is not None:
            #         vel[bad_rec] = np.nan
            #     flux[bad_rec] = np.nan
            bgr_flux.append(np.mean(flux[(flux>np.nanpercentile(flux,15)) & (flux<np.nanpercentile(flux,30))]))
            local_statuses.append(status)
            if status:
                if comb_id is None or comb_id not in comb_id_seen:
                    cur_values = flux / (np.pi * lvm_fiber_diameter ** 2 / 4)
                    if n_other_params > 0:
                        for fe in flux_err.T:
                            cur_values = np.vstack([cur_values.T, fe / (np.pi * lvm_fiber_diameter ** 2 / 4)]).T
                        cur_values = np.vstack([cur_values.T, vel[:,0].T]).T
                        cur_values = np.vstack([cur_values.T, vel_err[:, 0].T]).T
                        cur_values = np.vstack([cur_values.T, disp[:, 0].T]).T
                        cur_values = np.vstack([cur_values.T, disp_err[:, 0].T]).T
                        cur_values = np.vstack([cur_values.T, cont[:,0].T / (np.pi * lvm_fiber_diameter ** 2 / 4)]).T
                        cur_values = np.vstack([cur_values.T, cont_err[:, 0].T / (np.pi * lvm_fiber_diameter ** 2 / 4)]).T
                    if len(cur_values.shape)==1:
                        cur_values = cur_values.reshape((-1,1))
                    if values is None:
                        values = cur_values
                    else:
                        values = np.vstack([values, cur_values])
                    ras = np.append(ras, ra)
                    decs = np.append(decs, dec)
                    noises = np.append(noises, np.array([noise / (np.pi * lvm_fiber_diameter ** 2 / 4)]*len(dec)))
                    exps = np.append(exps, [data[cur_ind][1]]*len(flux))
                    if comb_id is not None:
                        comb_id_seen.append(comb_id)

            cur_ind += 1
        pool.close()
        pool.join()
        gc.collect()
        if not np.all(local_statuses):
            status_out = status_out & False
        else:
            status_out = status_out & True

    bgr_flux = np.array(bgr_flux)
    print(np.nanmedian(bgr_flux)/bgr_flux)

    # log.info(f"Construct interpolated image")
    # tab = Table.read("/home/egorov/Dropbox/LVM/Orion_S2_densities.txt", format='ascii')
    # fluxes = tab["ne_[SII]_6716_6731"]
    #
    # rec = np.isfinite(fluxes)
    # fluxes = fluxes[rec]
    # ras = ras[rec]
    # decs = decs[rec]
    # if do_median_masking:
    #     sort_seq = np.lexsort((ras, decs))
    #     ras = ras[sort_seq]
    #     decs = decs[sort_seq]
    #     values = values[sort_seq]
    #     # values = sigma_clip(values, sigma=10, masked=False)
        # values = median_filter(values, 7)
    img_arr = shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs, show_values=values,
                               r_lim=r_lim, sigma=sigma)

    header = wcs_out.to_header()
    add_params_names=['vel', 'disp', 'cont']
    outfile_suffixes = []
    for ind in range(img_arr.shape[2]):
        if ind < (len(line_names)):
            outfile_suffix = f'{line_names[ind]}'
        elif ind < (2 * len(line_names)):
            outfile_suffix = f'{line_names[ind - len(line_names)]}_err'
        else:
            outfile_suffix = f'{line_names[0]}_{add_params_names[ind // 2-len(line_names)]}'
            if ind % 2 == 1:
                outfile_suffix += "_err"
        outfile_suffixes.append(outfile_suffix)
        if do_median_masking:
            img_arr[:, :, ind] = median_filter(img_arr[:, :, ind], (11,11))
        fits.writeto(f"{outfile_prefix}_{outfile_suffix}.fits",
                     data=img_arr[:,:,ind], header=header, overwrite=True)
    if save_fluxes:
        return_values = (exps, values, ras, decs, outfile_suffixes, noises)
    else:
        return_values = None
    return status_out, return_values


def get_noise_one_exp(filename):
    with fits.open(filename) as rss:
        texp = rss[0].header['EXPTIME']
        wave = ((np.arange(rss[1].header['NAXIS1']) - rss[1].header['CRPIX1'] + 1) * rss[1].header['CDELT1'] +
                rss[1].header['CRVAL1']) * 1e10
        sel_wave = (wave >= 6400) * (wave <= 6500)
        sel_wave_sky = (wave >= 6295) * (wave <= 6305)
        sel_wave_sky_cont = ((wave > 6250) & (wave < 6285)) | ((wave > 6320) & (wave < 6350))
        tab = Table(rss['SLITMAP'].data)
        sci = np.flatnonzero((tab['targettype'] == 'science') & (tab['fibstatus'] == 0))
        estimate_noise_array = rss[1].data[sci] / texp
        estimate_noise_array_sky = (rss['SKY'].data[sci]) / texp #rss[1].data[sci] +
        estimate_noise_array_0 = estimate_noise_array[:, sel_wave]
        noise = np.nanmedian(np.absolute(estimate_noise_array_0 - np.nanmedian(estimate_noise_array_0)))

        sky_line_sn = np.nanmedian(np.nanmedian(estimate_noise_array_sky[:, sel_wave_sky], axis=1)) #- (
                #np.nanmedian(estimate_noise_array_sky[:, sel_wave_sky_cont], axis=1) * np.sum(sel_wave_sky)))

    return noise, sky_line_sn/noise

def check_noise_level(config, w_dir=None):
    log.info("Check noise level in the spectra to make adjustment of fluxes easier")
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    for cur_obj in config['object']:
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'))
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue

        files = []
        mjds = []
        expnames = []
        for ind_pointing, cur_pointing in tqdm(enumerate(cur_obj['pointing']), total=len(cur_obj['pointing']),
                                               ascii=True, desc='Pointings done:'):
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']
                for exp in exps:
                    cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
                    if not os.path.exists(cur_fname):
                        log.warning(f"Can't find {cur_fname}")
                        continue
                    files.append(cur_fname)
                    expnames.append(exp)
                    mjds.append(data.get('mjd'))
        if len(files) == 0:
            log.warning(f"Nothing to check for noise for object {cur_obj.get('name')}.")
            continue
        procs = np.nanmin([config['nprocs'], len(files)])
        tab = Table(names=['mjd', 'expnum', 'noise', 'sn_sky', 'correction', 'sn_sky_frac'],
                    dtype=[int, int, float, float, float, float])
        with mp.Pool(processes=procs) as pool:
            for spec_id, results in tqdm(enumerate(pool.imap_unordered(get_noise_one_exp, files)),
                    ascii=True, desc="Calculate noise levels",
                    total=len(files),):
                noise, sky_sn = results
                tab.add_row(vals=[mjds[spec_id], expnames[spec_id], noise, sky_sn, 1., 1.])
            pool.close()
            pool.join()
            gc.collect()
        tab['correction'] = np.nanmedian(tab['noise']) / tab['noise']
        tab['sn_sky_frac'] = tab['sn_sky']/np.nanmedian(tab['sn_sky'])
        tab = tab[tab['expnum'].argsort()]
        tab.write(os.path.join(cur_wdir, 'maps', f"{cur_obj['name']}_noise_levels.txt"),
                         overwrite=True, format='ascii.fixed_width_two_line')

        fig = plt.figure(figsize=(15,5))
        ax=fig.add_subplot(111)
        myplot = ax.scatter(range(len(expnames)), tab['correction'], c=np.log10(tab['sn_sky_frac']), vmin=-0.3, vmax=0.3)
        plt.colorbar(myplot, label='Log (Dev. of sky S/N) @ 6300A')
        ax.plot([0,len(expnames)], [1.,1.],'--')
        plt.xticks(range(len(expnames)), [str(exp) for exp in tab['expnum']], rotation=90)
        ax.set_ylabel("Recommended flux correction", fontsize=14)
        ax.set_xlabel("Exp. number", fontsize=14)
        fig.savefig(os.path.join(cur_wdir, 'maps', f"{cur_obj['name']}_corrections.pdf"), dpi=200,
                    bbox_inches='tight')
        statuses.append(True)
    return np.all(statuses)


# def extract_astrometry(params):
#     exp, exp_id, mjd, cur_wdir, objname, cur_pname, skip_bad_fibers,
#     for exp_id, exp in tqdm(enumerate(exps), total=len(exps), ascii=True,
#                             desc=f'Extraction fiber coordinates for pointing {cur_pointing["name"]}'):
#         cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
#         if not os.path.exists(cur_fname):
#             log.warning(f"Can't find {cur_fname}")
#             continue
#
#         with fits.open(cur_fname) as rss:
#             if not rss[0].header.get('FLUXCAL'):
#                 log.error(f"Missing flux calibration for exp={exp}. Skip it.")
#                 statuses.append(False)
#                 continue
#
#             tab = Table(rss['SLITMAP'].data)
#             sci = np.flatnonzero(tab['targettype'] == 'science')
#             if config.get('skip_bad_fibers'):
#                 sci = np.flatnonzero((tab['targettype'] == 'science') & (tab['fibstatus'] == 0))
#
#         radec_center = derive_radec_ifu(data['mjd'], exp, ref_exp,
#                                         objname=cur_obj['name'],
#                                         pointing_name=cur_pointing.get('name'))
#         ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], radec_center[0], radec_center[1],
#                                      radec_center[2])
#         radec_array = SkyCoord(ra=ra_fib, dec=dec_fib, unit=('degree', 'degree'))
#         for trow_id, trow in enumerate(tab[sci]):
#             cur_radec = radec_array[trow_id]
#             radec_tab = SkyCoord(ra=tab_summary['fib_ra'], dec=tab_summary['fib_dec'], unit=('degree', 'degree'))
#             rec = np.flatnonzero(radec_tab.separation(cur_radec) < (1 * u.arcsec))
#             fib_id = f"{exp:08d}_{trow['fiberid']:04d}"
#             if len(rec) > 0:
#                 tab_summary[rec[0]]['sourceid'] += f', {fib_id}'
#                 tab_summary[rec[0]]['fluxcorr'] += f', {cur_flux_corr[exp_id]}'
#             else:
#                 tab_summary.add_row([len(tab_summary) + 1, ra_fib[trow_id], dec_fib[trow_id],
#                                      trow['targettype'], trow['fibstatus'], fib_id, str(cur_flux_corr[exp_id])])

def create_single_rss(config, w_dir=None):
    """
    Create single RSS file with combined spectra from different exposures
    """
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    for cur_obj in config['object']:
        log.error(f"Creating single RSS file for {cur_obj.get('name')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'))
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue

        files = []
        all_exps = []
        all_ref_exps = []
        corrections = []
        all_mjds = []
        all_pnames = []
        nx = None
        f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fiberpos.txt")

        for cur_pointing in cur_obj['pointing']:
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']

                if not data.get('flux_correction'):
                    cur_flux_corr = [1.] * len(exps)
                else:
                    cur_flux_corr = data['flux_correction']
                if isinstance(cur_flux_corr, float) or isinstance(cur_flux_corr, int):
                    cur_flux_corr = [cur_flux_corr]
                corrections.extend(cur_flux_corr)

                for exp_id, exp in enumerate(exps):
                    cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
                    if not os.path.exists(cur_fname):
                        log.warning(f"Can't find {cur_fname}")
                        continue
                    files.append(cur_fname)
                    all_exps.append(exp)
                    all_mjds.append(data['mjd'])
                    all_pnames.append(cur_pointing['name'])
                    if data.get('dithering'):
                        all_ref_exps.append(exps[0])
                    else:
                        all_ref_exps.append(None)
                    if nx is None:
                        with fits.open(cur_fname) as rss:
                            nx = rss['FLUX'].header['NAXIS1']

        if config.get('load_fiberpos_table') and os.path.isfile(f_tab_summary):
            tab_summary = Table.read(f_tab_summary, format='ascii.fixed_width_two_line',
                                     converters={'sourceid': str, 'fluxcorr': str, 'vhel_corr': str})
            log.info(f"Table with fibers positions is loaded from {f_tab_summary}")
        else:
            tab_summary = Table(data=None,
                                names=['fiberid', 'fib_ra', 'fib_dec', 'targettype',
                                       'fibstatus', 'sourceid', 'fluxcorr', 'vhel_corr'],
                                dtype=(int, float, float, str, int, 'object', 'object', 'object'))

            for cur_pointing in cur_obj['pointing']:
                for data in cur_pointing['data']:
                    if isinstance(data['exp'], int):
                        exps = [data['exp']]
                    else:
                        exps = data['exp']

                    if data.get('dithering'):
                        ref_exp = exps[0]
                    else:
                        ref_exp = None

                    if not data.get('flux_correction'):
                        cur_flux_corr = [1.]*len(exps)
                    else:
                        cur_flux_corr = data['flux_correction']
                    if isinstance(cur_flux_corr, float) or isinstance(cur_flux_corr, int):
                        cur_flux_corr = [cur_flux_corr]
                    # corrections.extend(cur_flux_corr)

                    for exp_id, exp in tqdm(enumerate(exps), total=len(exps), ascii=True,
                                            desc=f'Extraction fiber coordinates for pointing {cur_pointing["name"]}'):
                        cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
                        if not os.path.exists(cur_fname):
                            log.warning(f"Can't find {cur_fname}")
                            continue

                        with fits.open(cur_fname) as rss:
                            if not rss[0].header.get('FLUXCAL'):
                                log.error(f"Missing flux calibration for exp={exp}. Skip it.")
                                statuses.append(False)
                                continue

                            obstime = Time(rss[0].header['OBSTIME'])
                            loc = EarthLocation.of_site('lco')

                            tab = Table(rss['SLITMAP'].data)
                            sci = np.flatnonzero(tab['targettype'] == 'science')
                            if config['imaging'].get('skip_bad_fibers'):
                                sci = np.flatnonzero((tab['targettype'] == 'science') & (tab['fibstatus'] == 0))

                        # kostyl for NGC6822
                        if exp == 3602:
                            ref_exp = 3601
                        radec_center = derive_radec_ifu(data['mjd'], exp, ref_exp,
                                                        objname=cur_obj['name'],
                                                        pointing_name=cur_pointing.get('name'))


                        ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci],
                                                     radec_center[0], radec_center[1],radec_center[2])
                        radec_array = SkyCoord(ra=ra_fib, dec=dec_fib, unit=('degree','degree'))
                        vcorr = np.round(radec_array[0].radial_velocity_correction(kind='heliocentric', obstime=obstime,
                                                                        location=loc).to(u.km/u.s).value, 1)
                        for trow_id, trow in enumerate(tab[sci]):
                            cur_radec = radec_array[trow_id]
                            radec_tab = SkyCoord(ra=tab_summary['fib_ra'], dec=tab_summary['fib_dec'],
                                                 unit=('degree','degree'))
                            rec = np.flatnonzero(radec_tab.separation(cur_radec) < (1*u.arcsec))
                            fib_id = f"{exp:08d}_{trow['fiberid']:04d}"
                            if len(rec) > 0:
                                tab_summary["sourceid"][rec[0]] = f'{tab_summary["sourceid"][rec[0]]}, {fib_id}'
                                tab_summary["fluxcorr"][rec[0]] = f'{tab_summary["fluxcorr"][rec[0]]}, {cur_flux_corr[exp_id]}'
                                tab_summary['vhel_corr'][rec[0]] = f'{tab_summary["vhel_corr"][rec[0]]}, {vcorr}'
                            else:
                                tab_summary.add_row([len(tab_summary)+1, ra_fib[trow_id], dec_fib[trow_id],
                                                     trow['targettype'], trow['fibstatus'], fib_id,
                                                     str(cur_flux_corr[exp_id]), str(vcorr)])
            tab_summary.write(f_tab_summary, overwrite=True, format='ascii.fixed_width_two_line')
            tab_summary = Table.read(f_tab_summary, format='ascii.fixed_width_two_line',
                                     converters={'sourceid': str, 'fluxcorr': str, 'vhel_corr': str})
            statuses.append(True)

        fout = os.path.join(cur_wdir, f"{cur_obj['name']}_all_RSS.fits")

        # rss_out = fits.HDUList([fits.PrimaryHDU(data=None),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='FLUX'),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='ERROR'),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='MASK'),
        #                        fits.ImageHDU(data=np.zeros(shape=(nx), dtype=float), name='WAVE'),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='FWHM'),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='SKY'),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='SKY_ERROR'),
        #                        fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='FLUX_ORIG'),
        #                        fits.BinTableHDU(data=tab_summary, name='SLITMAP')])

        if config.get('keep_existing_single_rss') and os.path.isfile(fout):
            log.info('...Consider existing RSS file as container')
        else:
            log.info('...Start writing dummy RSS file which will be served as container')
            rss_out = fits.HDUList([fits.PrimaryHDU(data=None),
                                    fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='FLUX'),
                                    fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='ERROR')])
            rss_out.writeto(fout, overwrite=True)
            rss_out.close()
            log.info('......FLUX and ERROR extensions are added')
            for kw_block in [('MASK', 'WAVE', 'FWHM'), ('SKY', 'SKY_ERROR', 'FLUX_ORIG')]:
                rss_out = fits.open(fout)
                for kw in kw_block:
                    if kw != 'WAVE':
                        shape = (len(tab_summary), nx)
                    else:
                        shape = nx
                    rss_out.append(fits.ImageHDU(data=np.zeros(shape=shape, dtype=float), name=kw))

                if kw == 'FLUX_ORIG':
                    rss_out.append(fits.BinTableHDU(data=tab_summary, name='SLITMAP'))
                rss_out.writeto(fout, overwrite=True)
                rss_out.close()
                log.info(f'......{",".join(kw_block)} extensions are added')
                if kw == 'FLUX_ORIG':
                    log.info('......SLITMAP extensions is added')
            # rss_out = fits.open(fout)

            # rss_out.writeto(fout, overwrite=True)
            # rss_out.close()
            # log.info(f'......SLITMAP extension is added')

        # corrections = np.array(corrections)
        all_exps = np.array(all_exps)
        files = np.array(files)

        n_fib_per_block = 70000
        rss_open = False
        for ind_row, cur_row in tqdm(enumerate(tab_summary), total=len(tab_summary), ascii=True, desc='Spectra done:'):
            if not rss_open:
                rss_out = fits.open(fout)
                rss_open = True
                if ind_row == 0:
                    rss_out['SLITMAP'] = fits.BinTableHDU(data=tab_summary, name='SLITMAP')
            source_ids = cur_row['sourceid'].split(',')
            cur_corr_flux = np.array([float(corr) for corr in cur_row['fluxcorr'].split(',')]).astype(float)
            cur_vhel_corr = np.array([float(corr) for corr in cur_row['vhel_corr'].split(',')]).astype(float)
            fs = [files[np.flatnonzero(all_exps == int(sid.split('_')[0]))[0]] for sid in source_ids]
            # cur_corr_flux = [corrections[np.flatnonzero(all_exps == int(sid.split('_')[0]))[0]] for sid in source_ids]
            sp_ids = [int(sid.split('_')[1])-1 for sid in source_ids]
            if len(source_ids) == 1:
                # ===== Copy spectrum from the source
                f = fs[0]
                spec_id = sp_ids[0]
                with fits.open(f) as hdu:
                    wl_grid = ((np.arange(hdu['FLUX'].header['NAXIS1'])-
                       hdu['FLUX'].header['CRPIX1']+1) * hdu['FLUX'].header['CDELT1']*1e10 +
                       hdu['FLUX'].header['CRVAL1']*1e10)
                    if ind_row == 0:
                        rss_out[0].header = hdu[0].header
                        ref_h_for_fsc = hdu['FLUX'].header.copy()
                        ref_h_for_fsc['EXTNAME'] = 'FLUX_ORIG'
                        rss_out['FLUX_ORIG'].header = ref_h_for_fsc
                        rss_out['WAVE'].data = hdu['WAVE'].data
                        rss_out['WAVE'].header = hdu['WAVE'].header
                    for kw in ['FLUX', 'ERROR', 'MASK', 'FWHM', 'SKY', 'SKY_ERROR']:
                        if ind_row == 0:
                            rss_out[kw].header = hdu[kw].header
                        rss_out[kw].data[ind_row, :] = hdu[kw].data[spec_id, :]
                    rss_out['FLUX_ORIG'].data[ind_row, :] = hdu['FLUX'].data[spec_id, :].copy()
                    rss_out['FLUX'].data[ind_row, :] = (hdu['FLUX'].data[spec_id, :] + hdu['SKY'].data[spec_id, :]) - \
                                                    mask_sky_at_bright_lines(hdu['SKY'].data[spec_id, :], wave=wl_grid, vel=cur_obj['velocity'], mask=hdu['MASK'].data[spec_id, :])
                    rec_bad = np.flatnonzero((hdu['FLUX'].data[spec_id, :] == 0) | ~np.isfinite(hdu['FLUX'].data[spec_id, :]) | (hdu['MASK'].data[spec_id, :]>0))
                    for kw in ['FLUX', 'ERROR', 'SKY', 'FLUX_ORIG', 'SKY_ERROR']:
                        rss_out[kw].data[ind_row,:] = rss_out[kw].data[ind_row,:] * cur_corr_flux[0]
                    rss_out['FLUX'].data[ind_row, rec_bad] = np.nan
                    rss_out['FLUX_ORIG'].data[ind_row, rec_bad] = np.nan

            elif len(source_ids) > 1:
                # ===== Combine using sigma-clipping
                fluxes = np.zeros(shape=(len(source_ids), nx), dtype=float)
                errors = np.zeros(shape=(len(source_ids), nx), dtype=float)
                masks = np.zeros(shape=(nx), dtype=bool)
                skies = np.zeros(shape=(len(source_ids), nx), dtype=float)
                sky_errors = np.zeros(shape=(len(source_ids), nx), dtype=float)
                fluxes_skycorr = np.zeros(shape=(len(source_ids), nx), dtype=float)
                for f_id, f in enumerate(fs):
                    spec_id = sp_ids[f_id]
                    with fits.open(f) as hdu:
                        if f_id == 0:
                            wl_grid = ((np.arange(hdu['FLUX'].header['NAXIS1'])-
                            hdu['FLUX'].header['CRPIX1']+1) * hdu['FLUX'].header['CDELT1']*1e10 +
                            hdu['FLUX'].header['CRVAL1']*1e10)
                            rss_out['FWHM'].data[ind_row, :] = hdu['FWHM'].data[spec_id, :]
                        if (ind_row == 0) & (f_id == 0):
                            rss_out[0].header = hdu[0].header
                            ref_h_for_fsc = hdu['FLUX'].header.copy()
                            ref_h_for_fsc['EXTNAME'] = 'FLUX_ORIG'
                            rss_out['FLUX_ORIG'].header = ref_h_for_fsc
                            rss_out['WAVE'].data = hdu['WAVE'].data
                            for kw in ['FLUX', 'ERROR', 'MASK', "WAVE", 'FWHM', 'SKY', 'SKY_ERROR']:
                                rss_out[kw].header = hdu[kw].header

                        rec = np.flatnonzero(hdu['MASK'].data[spec_id,:] | ~np.isfinite(hdu['FLUX'].data[spec_id,:]) | (hdu['FLUX'].data[spec_id,:] == 0))
                        rec_sky = np.flatnonzero(hdu['MASK'].data[spec_id,:] | ~np.isfinite(hdu['SKY'].data[spec_id,:]))
                        rec_skycorr = np.flatnonzero(hdu['MASK'].data[spec_id,:] | ~np.isfinite(hdu['SKY'].data[spec_id,:]) | ~np.isfinite(hdu['FLUX'].data[spec_id,:]) | (hdu['FLUX'].data[spec_id,:] ==0))
                        fluxes[f_id, :] = hdu['FLUX'].data[spec_id, :] * cur_corr_flux[f_id]
                        errors[f_id, :] = abs(hdu['ERROR'].data[spec_id, :]) * cur_corr_flux[f_id]
                        skies[f_id, :] = hdu['SKY'].data[spec_id, :] * cur_corr_flux[f_id]
                        masks = masks | hdu['MASK'].data[spec_id,:]
                        sky_errors[f_id, :] = hdu['SKY_ERROR'].data[spec_id, :] * cur_corr_flux[f_id]
                        fluxes_skycorr[f_id, :] = ((hdu['FLUX'].data[spec_id, :] + hdu['SKY'].data[spec_id, :]) -
                                                  mask_sky_at_bright_lines(hdu['SKY'].data[spec_id, :], wave=wl_grid, vel=cur_obj['velocity'], mask=hdu['MASK'].data[spec_id, :]))* cur_corr_flux[f_id]
                        fluxes[f_id, rec] = np.nan
                        fluxes_skycorr[f_id, rec_skycorr] = np.nan
                        skies[f_id, rec_sky] = np.nan

                # rec_test_wl = np.flatnonzero((rss_out['WAVE'].data > 5900) & (rss_out['WAVE'].data < 6200))
                # cont_err = np.nanstd(fluxes_skycorr[:,rec_test_wl], axis=1)
                # cont_err = np.round(cont_err/np.nanmedian(cont_err),2)
                #
                # fluxes = fluxes / cont_err[:, None]
                # fluxes_skycorr = fluxes_skycorr / cont_err[:, None]
                # skies = skies / cont_err[:, None]
                # errors = errors / cont_err[:, None]
                # sky_errors = sky_errors / cont_err[:, None]

                rss_out['FLUX'].data[ind_row, :] = np.nanmean(sigma_clip(fluxes_skycorr, sigma=1.3, axis=0, masked=False), axis=0)
                rss_out['FLUX_ORIG'].data[ind_row, :] = np.nanmean(sigma_clip(fluxes, sigma=1.3, axis=0, masked=False), axis=0)
                rss_out['MASK'].data[ind_row, :] = masks
                rss_out['ERROR'].data[ind_row, :] = np.sqrt(np.nansum(errors**2, axis=0))/np.sum(np.isfinite(errors),axis=0)
                rss_out['SKY'].data[ind_row, :] = np.nansum(skies, axis=0)/np.sum(np.isfinite(skies),axis=0)
                rss_out['SKY_ERROR'].data[ind_row, :] = np.sqrt(np.nansum(sky_errors**2, axis=0))/np.sum(np.isfinite(sky_errors),axis=0)

            if ((ind_row+1) % n_fib_per_block == 0) or (ind_row == (len(tab_summary)-1)):
                rss_out.writeto(fout, overwrite=True)
                rss_out.close()
                rss_open = False
            statuses.append(True)

    return statuses


def mask_sky_at_bright_lines(sky_spec, mask=None, wave=None, vel=0):
    """
    Masks sky lines at the location of the bright nebular emission lines
    """
    lines = [6562.8, 4861., 5006.8, 4959., 3726., 3729., 6548., 6583.8, 6716.4, 6730.8, 9532.]
    wid = 3.
    if mask is None:
        mask = np.zeros_like(sky_spec, dtype=bool)
    rec_masked = mask > 0
    spec_out = sky_spec.copy()
    for l in lines:
        cur_wl_mask = np.flatnonzero((wave < ((l+wid)*(1+vel/3e5))) & (wave > ((l-wid)*(1+vel/3e5))))
        cur_wl_source = np.flatnonzero(~rec_masked & (wave < ((l + wid*5) * (1 + vel / 3e5))) & (wave > ((l - wid*5) * (1 + vel / 3e5))))
        if len(cur_wl_source) > 10:
            spec_out[cur_wl_mask] = np.percentile(sky_spec[cur_wl_source], 30)
    # rec_masked = np.flatnonzero(rec_masked)
    # spec_out = sky_spec.copy()
    # spec_out[rec_masked] = np.nan
    # rec = np.flatnonzero(np.isfinite(spec_out))
    # if len(rec) > 100:
    #     spec_out[rec_masked] = median_filter(spec_out, 200)[rec_masked]
    #     # spec_out[rec_masked] = np.interp(wave[rec_masked], wave[rec], spec_out[rec])
    return spec_out
    

def LVM_process(config_filename=None, output_dir=None):
    if not os.path.exists(config_filename):
        log.error("Config file is not found!")
        return
    config = parse_config(config_filename)
    if not config:
        log.error("Critical errors occurred. Exit.")
        return

    if len(config['object']) == 0:
        log.error("Can't find any object. Nothing to process.")
        return

    # === Step 1 - download all raw files listed in config from SAS. After that - regenerates metadata
    if config['steps'].get('download'):
        status = download_from_sas(config)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
        log.info("Downloading from SAS complete")

    else:
        log.info("Skip download step")

    # === Step 2 - Regenerate metadata (necessary after downloading new files or update of drp)
    if config['steps'].get('update_metadata'):
        status = regenerate_metadata(config)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
    else:
        log.info("Skip updating metadata step")

    # === Step 3 - reduce all exposures listed in config
    if config['steps'].get('reduction'):
        status = do_quick_reduction(config)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
        log.info("Reduction complete")

        copy_reduced_data(config, output_dir=output_dir)

    else:
        log.info("Skip reduction step")

        # === Step 3.5 - Optional step checking the noise level in the spectra (to evaluate potential correction in abs.cal)
        if config['steps'].get('check_noise_level'):
            cur_wdir = output_dir
            if cur_wdir is None:
                cur_wdir = config.get('default_output_dir')
            status = check_noise_level(config, w_dir=cur_wdir)
            if not status:
                log.error("Critical errors occurred. Exit.")
                return

    # === Step 4 - Combine spectra with sigma-clipping
    if config['steps'].get('coadd_spectra'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = do_coadd_spectra(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
        log.info("Combining spectra from individual exposures complete")

    else:
        log.info("Skip combining spectra from individual exposures")

    # === Step xx - Correct sky
    if config['steps'].get('correct_sky'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = do_sky_correction(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
        log.info("Correction of sky subtraction complete")

    else:
        log.info("Skip correction of sky subtraction")

    # === Step XXX - derive best-fit normalization coefficients
    if config['steps'].get('estimate_flux_correction'):
        if not config['imaging'].get('lines') or (len(config['imaging'].get('lines')) == 0):
            log.error("No lines are present in config file. Exit.")
            return

        status = do_imaging(config, output_dir=output_dir, use_shepard=True)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
    else:
        log.info('Skip imaging step')


    # === Step xxx - Create single RSS file
    if config['steps'].get('create_single_rss'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = create_single_rss(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
        log.info("Creating a single RSS file")

    else:
        log.info("Skip creating a single RSS file")

    # === Step xxx - Create single RSS file
    if config['steps'].get('analyse_single_rss'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        log.info("Analysing a single RSS file")
        status = process_single_rss(config, output_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
    else:
        log.info("Skip analysing a single RSS file")


    if config['steps'].get('create_image_from_single_rss'):
        w_dir = output_dir
        if w_dir is None:
            w_dir = config.get('default_output_dir')
        log.info("Create images from measurements from RSS file")
        status = True
        if config['imaging'].get('lines') is None:
            log.info("Nothing to show. Exit")
            return
        for cur_obj in config['object']:
            cur_wdir = os.path.join(w_dir, cur_obj.get('name'))
            if not os.path.exists(cur_wdir):
                log.error(
                    f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
                status = False
                continue
            file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes.txt")

            cur_status = create_line_image_from_rss(file_fluxes=file_fluxes, lines=config['imaging'].get('lines'),
                                                    pxscale_out=config['imaging'].get('pxscale'),
                                                    r_lim=config['imaging'].get('r_lim'),
                                                    sigma=config['imaging'].get('sigma'),
                                                    output_dir=os.path.join(cur_wdir,'maps_singleRSS'),
                                                    outfile_prefix=f"{cur_obj.get('name')}_{config['imaging'].get('pxscale')}asec")
            status = status & cur_status
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
    else:
        log.info("Skip imaging from a single RSS file")

    # === Step 5 - create maps in different lines
    if config['steps'].get('imaging'):
        if not config['imaging'].get('lines') or (len(config['imaging'].get('lines')) == 0):
            log.error("No lines are present in config file. Exit.")
            return

        if config['imaging'].get('interpolate'):
            # reconstruct an image from an RSS-style input (positions and fluxes)
            # Sanchez+2012 - modified version of shepard's method
            log.info("Create combined image in emission lines using shepard's method")
            use_shepard = True
        else:
            use_shepard = False
            log.info("Create images in lines for individual exposures")

        status = do_imaging(config, output_dir=output_dir, use_shepard=use_shepard)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

        if not use_shepard:
            log.info("Combine images")
            status = combine_images(config, w_dir=output_dir)
            if not status:
                log.error("Critical errors occurred. Exit.")
                return
    else:
        log.info('Skip imaging step')

    # === Step 6 - create cubes in different lines
    if config['steps'].get('create_cube'):
        status = do_cube_construction(config, output_dir=output_dir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

    else:
        log.info('Skip cube creation')

    log.info("Done!")

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        log.error("Please provide path to toml config file!")

    else:
        config_file = args[1]
        if len(args) == 2:
            output_dir = None
        else:
            output_dir = args[2]
        LVM_process(config_filename=config_file, output_dir=output_dir)
