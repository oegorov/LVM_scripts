#!/usr/bin/env python3
import yaml
from sdss_access import RsyncAccess

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import os
import logging
import glob

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import gc
import sys
from tqdm import tqdm
import shutil

import multiprocessing as mp
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from functools import partial
from astropy.table import Table, vstack, Column, join
from astropy.coordinates import SkyCoord
from astropy.modeling import fitting, models
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy.coordinates import EarthLocation
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from lmfit.models import GaussianModel, ConstantModel
from regions import Regions
from shapely.geometry import Polygon

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning

from argparse import Namespace

warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', AstropyUserWarning)

log = logging.getLogger(name='LVM-process')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

dap_ok = True
try:
    from pyFIT3D.common.auto_ssp_tools import dump_rss_output
    from pyFIT3D.common.auto_ssp_tools import load_rss
except ModuleNotFoundError:
    logging.error("pyFIT3D is not installed. Please install it to use DAP fitting with this script.")
    dap_ok = False

try:
    from lvmdap._cmdline.dap import auto_rsp_elines_rnd
except ModuleNotFoundError:
    logging.error("LVMDAP is not installed. Please install it to use DAP fitting with this script.")


# ========================
# ======== Setup =========
# ========================
red_data_version = '1.1.1'
dap_version = '1.1.0'
drp_version = '1.1.1dev'
# os.environ['LVMDRP_VERSION'] = drp_version
agcam_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'data', 'agcam', 'lco')
raw_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'data', 'lvm', 'lco')
drp_results_dir_sas = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'lvm', 'spectro', 'redux', red_data_version)
dap_results_dir_sas = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'lvm', 'spectro',
                                   'analysis', dap_version)
drp_results_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'sdsswork', 'lvm', 'spectro', 'redux', drp_version)
server_group_id = 10699  # ID of the group on the server to run 'chgrp' on all new/downloaded files. Skipped if None
obs_loc = EarthLocation.of_site('lco')  # Observatory location
fiber_d = 35.3 #diameter of the fiber in arcsec

dap_results_correspondence = {
    "Ha": 6562.85, 'Hb': 4861.36, 'SII6717': 6716.44, "SII6731": 6730.82, 'NII6584': 6583.45, 'SIII9532': 9531.1,
    'OIII5007': 5006.84,
    "OII3727": 3726.03, "OII3729": 3728.82, "Hg": 4340.49,
    'OI': 6300.3,
    "HeII": "HeII_4685.68",
    "OIII4363": '[OIII]_4363.21', 'NII5755': '[NII]_5754.59',
    '[SII]4068': '[SII]_4068.6', 'NeIII3869': '[NeIII]_3868.75',
    'OII7320': '[OII]_7318.92', 'OII7330': '[OII]_7329.66', 'ArIII7136': '[ArIII]_7135.8'
}

# ================================================
# ======== Main functions for data processing ====
# ================================================
def LVM_process(config_filename=None, output_dir=None):
    """
    Main function regulating all the LVM processing steps
    :param config_filename: path to TOML config file
    :param output_dir: output directory for all processed files
    """
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

    # === Create folders for different objects/pointings
    cur_wdir = output_dir
    if cur_wdir is None:
        cur_wdir = config.get('default_output_dir')
    status = create_folders_tree(config, w_dir=cur_wdir)
    if not status:
        log.error("Critical errors occurred. Exit.")
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
        if not config['reduction'].get('only_copy_files'):
            status = do_reduction(config)
            if not status:
                log.error("Critical errors occurred. Exit.")
                return
            log.info("Reduction complete")
        else:
            log.info("Skip reduction, only copy reduced files")
        copy_reduced_data(config, output_dir=output_dir)

    else:
        log.info("Skip reduction step")

    # === Step 4.1 - Optional step checking the noise level in the spectra (to evaluate potential correction in abs.cal)
    if config['steps'].get('check_noise_level'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = check_noise_level(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

    # === Step 4.2 - Optional step - combine spectra with sigma-clipping
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

    # === Step 5.1 - Create single RSS file
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

    # === Step 5 - Analyse RSS file(s)
    if config['steps'].get('analyse_rss'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')

        if config['imaging'].get('use_single_rss_file'):
            log.info("Analysing a single RSS file and measuring emission lines")
            status = process_single_rss(config, output_dir=cur_wdir, dap=config['imaging'].get('use_dap'))
            if not status:
                return
            if config['imaging'].get('use_dap'):
                log.info("Processing DAP results")
                status = parse_dap_results(config, w_dir=cur_wdir, local_dap_results=True)
        elif config['imaging'].get('use_binned_rss_file') and 'binning' in config:
            log.info(f"Analysing binned RSS file ({config['binning'].get('target_sn')} "
                     f"in {config['binning'].get('bin_line')} line) and measuring emission lines")
            status = process_single_rss(config, output_dir=cur_wdir, binned=True, dap=config['imaging'].get('use_dap'))
            if not status:
                return
            if config['imaging'].get('use_dap'):
                log.info("Processing DAP results")
                status = parse_dap_results(config, w_dir=cur_wdir, local_dap_results=True)
        elif config['imaging'].get('use_binned_rss_file'):
            log.error("'binning' block is not present. Exit.")
            return
        elif config['imaging'].get('use_dap'):
            log.info("Processing DAP results")
            status = parse_dap_results(config, w_dir=cur_wdir)
        else:
            log.info("Analysing all individual RSS files and measuring emission lines")
            status = process_all_rss(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
    else:
        log.info("Skip analysing RSS files")

    # === Step 6 - Create maps in different lines from table with fit results
    if config['steps'].get('create_images'):
        w_dir = output_dir
        if w_dir is None:
            w_dir = config.get('default_output_dir')
        log.info("Create images from measurements from RSS file")
        status = True
        if config['imaging'].get('lines') is None:
            log.info("Nothing to show. Exit")
            return
        for cur_obj in config['object']:
            if not cur_obj.get('version'):
                version = ''
            else:
                version = cur_obj.get('version')
            cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
            f_binmap = None
            if not os.path.exists(cur_wdir):
                log.error(
                    f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
                status = False
                continue
            if config['imaging'].get('use_single_rss_file'):
                if config['imaging'].get('use_dap'):
                    file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes_singleRSS_dap.txt")
                    cur_wdir = os.path.join(cur_wdir, 'maps_singleRSS_dap')
                else:
                    file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes_singleRSS.txt")
                    cur_wdir = os.path.join(cur_wdir, 'maps_singleRSS')
                line_list = config['imaging'].get('lines')
                filter_sn = [l.get('filter_sn') for l in config['imaging'].get('lines')]
            elif config['imaging'].get('use_binned_rss_file') and 'binning' in config:
                bin_line = config['binning'].get('line')
                if not bin_line:
                    bin_line = 'Ha'
                target_sn = config['binning'].get('target_sn')
                if not target_sn:
                    target_sn = 30.
                else:
                    target_sn = float(target_sn)
                suffix_binmap = config['binning'].get('binmap_suffix')
                if not suffix_binmap:
                    suffix_binmap = '_binmap.fits'
                if config['imaging'].get('use_dap'):
                    file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_binfluxes_{bin_line}_sn{target_sn}_dap.txt")
                    cur_wdir = os.path.join(cur_wdir, 'maps_binnedRSS_dap')
                else:
                    file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_binfluxes_{bin_line}_sn{target_sn}.txt")
                    cur_wdir = os.path.join(cur_wdir, 'maps_binnedRSS')
                f_binmap = os.path.join(cur_wdir, 'maps',
                                        f"{cur_obj.get('name')}_{config['imaging'].get('pxscale')}asec_{bin_line}_sn{target_sn}{suffix_binmap}")
                line_list = config['imaging'].get('lines')
                if (not os.path.isfile(file_fluxes)) or (not os.path.isfile(f_binmap)):
                    log.error("Either table with fluxes (from binned RSS) or binmap do not exist. Exit.")
                    return
                filter_sn = None
            elif config['imaging'].get('use_binned_rss_file'):
                log.error("'binning' block is not present. Exit.")
                return
            elif config['imaging'].get('use_dap'):
                file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes_dap.txt")
                cur_wdir = os.path.join(cur_wdir, 'maps_dap')
                line_list = dap_results_correspondence.keys()
                filter_sn = None
            else:
                file_fluxes = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes.txt")
                cur_wdir = os.path.join(cur_wdir, 'maps')
                line_list = config['imaging'].get('lines')
                filter_sn = [l.get('filter_sn') for l in config['imaging'].get('lines')]

            cur_status = create_line_image_from_table(file_fluxes=file_fluxes, lines=line_list,
                                                      pxscale_out=config['imaging'].get('pxscale'),
                                                      r_lim=config['imaging'].get('r_lim'),
                                                      sigma=config['imaging'].get('sigma'),
                                                      output_dir=cur_wdir, ra_lims=config['imaging'].get('ra_limits'),
                                                      dec_lims=config['imaging'].get('dec_limits'),
                                                      outfile_prefix=f"{cur_obj.get('name')}_{config['imaging'].get('pxscale')}asec",
                                                      filter_sn=filter_sn, binmap=f_binmap)
            status = status & cur_status
        if not status:
            log.error("Critical errors occurred. Exit.")
            return
    else:
        log.info("Skip imaging from a single RSS file")

    # === Step 7 - Voronoi binning of the RSS based on the image in a single line
    if config['steps'].get('binning'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = vorbin_rss(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

    # === Step 8 - Extract spectra in ds9 regions
    if config['steps'].get('spectra_extraction'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = extract_spectra_ds9(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

    # === Step 9 - fit extracted, binned or single RSS spectra with DAP
    if config['steps'].get('fit_with_dap'):
        if 'dap_fitting' not in config:
            log.error("No DAP fitting parameters are set. Exit.")
            return
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')

        if config['dap_fitting'].get('fit_mode') == 'rss':
            log.info("Analysing a single RSS file and measuring emission lines")
            status = process_single_rss(config, output_dir=cur_wdir, dap=True)
            if not status:
                return
            log.info("Processing DAP results")
            status = parse_dap_results(config, w_dir=cur_wdir, local_dap_results=True, mode='single_rss')
        elif config['dap_fitting'].get('fit_mode') == 'binned':
            if 'binning' not in config:
                log.error("No binning parameters are set. Exit.")
                return
            log.info(f"Analysing binned RSS file ({config['binning'].get('target_sn')} "
                     f"in {config['binning'].get('bin_line')} line) and measuring emission lines")
            status = process_single_rss(config, output_dir=cur_wdir, binned=True, dap=True)
            if not status:
                return
            log.info("Processing DAP results")
            status = parse_dap_results(config, w_dir=cur_wdir, local_dap_results=True, mode='binned')
        elif config['dap_fitting'].get('fit_mode') == 'extracted':
            log.info("Fitting extracted spectra with DAP")
            status = process_single_rss(config, w_dir=cur_wdir, extracted=True, dap=True)
            if not status:
                log.error("Critical errors occurred. Exit.")
                return
            status = parse_dap_results(config, w_dir=cur_wdir, local_dap_results=True, mode='extracted')
        else:
            log.error("Wrong DAP fit mode. It can be either 'rss', 'binned' or 'extracted'. Exit.")
            return
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

    # === Step 10 - create cubes in different lines
    if config['steps'].get('create_cube'):
        cur_wdir = output_dir
        if cur_wdir is None:
            cur_wdir = config.get('default_output_dir')
        status = reconstruct_cube(config, w_dir=cur_wdir)
        if not status:
            log.error("Critical errors occurred. Exit.")
            return

    # # == Ancillary test steps
    # if config['steps'].get('test_pixel_shifts'):
    #     status = do_test_pix_shift(config)
    #     if not status:
    #         log.error("Critical errors occurred. Exit.")
    #         return
    #     log.info("Testing of pixel shifts complete")
    # else:
    #     log.info("Skip testing")
    #
    # else:
    #     log.info('Skip pixel shift testing')

    log.info("Done!")

def create_folders_tree(config, w_dir=None):
    try:
        for cur_obj in config['object']:
            if not cur_obj.get('version'):
                version = ''
            else:
                version = cur_obj.get('version')
            check_dir = os.path.join(w_dir, cur_obj.get('name'), version)
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
                uid = os.stat(check_dir).st_uid
                if server_group_id is not None:
                    os.chown(check_dir, uid=uid, gid=server_group_id)
                    os.chown(os.path.join(w_dir, cur_obj.get('name')), uid=uid, gid=server_group_id)
                os.chmod(check_dir, 0o775)
                os.chmod(os.path.join(w_dir, cur_obj.get('name')), 0o775)
            for cur_pointing in cur_obj['pointing']:
                check_dir = os.path.join(w_dir, cur_obj.get('name'), version, cur_pointing.get('name'))
                if not os.path.exists(check_dir):
                    os.makedirs(check_dir)
                    uid = os.stat(check_dir).st_uid
                    if server_group_id is not None:
                        os.chown(check_dir, uid=uid, gid=server_group_id)
                    os.chmod(check_dir, 0o775)
        status=True
    except Exception as e:
        log.error("Something wrong with creation of the work folders tree:" + str(e))
        status = False
    return status
def fit_cur_spec_lmfit(data, wave=None, lines=None, fix_ratios=None, velocity=0, mean_bounds=(-10., 10.),
                       ax=None, return_plot_data=False, subtract_lsf=True, max_n_comp=None, tie_disp=None, tie_vel=None):
    """
    Fit current spectrum with multiple Gaussians. Each line can have several components.
    First components may be tied together or not. Second etc. components are always tied
    :param tie_vel:
    :param tie_disp:
    :param data:
    :param wave:
    :param lines:
    :param fix_ratios:
    :param velocity:
    :param mean_bounds:
    :param ax:
    :param return_plot_data:
    :param subtract_lsf:
    :param max_n_comp:
    :return:
    """
    spectrum, errors, lsf = data
    rec = np.flatnonzero(np.isfinite(spectrum) & (spectrum != 0) &
                         np.isfinite(errors) & (np.isfinite(lsf)) & (lsf > 0) & np.isfinite(wave))  # & (errors > 0)
    if len(rec) < 10:
        if return_plot_data:
            return [np.nan] * len(lines), [np.nan] * len(lines), [np.nan] * len(lines), np.nan, [np.nan] * len(
                lines), np.nan, np.nan, np.nan, None, None, None
        else:
            return [np.nan] * len(lines), [np.nan] * len(lines), [np.nan] * len(lines), np.nan, [np.nan] * len(
                lines), np.nan, np.nan, np.nan, None, None

    spectrum = spectrum[rec]/1e-16
    errors = errors[rec]/1e-16
    lsf = lsf[rec]
    wave = wave[rec]
    cur_n_comps = np.zeros_like(lines, dtype=int)
    if max_n_comp is None:
        max_n_comp = np.ones_like(lines)
    max_n_comp = np.array(max_n_comp)
    max_n_comp[max_n_comp > 3] = 3
    if tie_disp is None:
        tie_disp = np.ones_like(lines, dtype=bool)
    if tie_vel is None:
        tie_vel = np.ones_like(lines, dtype=bool)
    n_max_lines = np.sum(max_n_comp)

    cont_guess = np.nanmin(spectrum)
    max_ampl = np.nanmax(spectrum) - cont_guess
    mean_std = np.nanmedian(lsf / 2.35428)
    flux_guess = max_ampl * mean_std * np.sqrt(2*np.pi)
    res = None

    while np.sum(cur_n_comps) <= n_max_lines:

        if np.sum(cur_n_comps) == 0:
            cur_n_comps = np.ones_like(lines, dtype=int)
        else:
            rec = np.flatnonzero(cur_n_comps < max_n_comp)
            if len(rec) == 0:
                break
            cur_n_comps[rec] += 1

        components = []
        my_model = None
        min_lid_tie_vel = -1
        min_lid_tie_disp = -1
        if np.any(tie_vel):
            min_lid_tie_vel = np.flatnonzero(tie_vel)[0]
        if np.any(tie_disp):
            min_lid_tie_disp = np.flatnonzero(tie_disp)[0]
        for l_id, l in enumerate(lines):
            for cmp in range(cur_n_comps[l_id]):
                ids_with_this_number_of_comp = np.flatnonzero(cur_n_comps >= (cmp+1))
                components.append(GaussianModel(nan_policy='omit', prefix=f'gaus{l_id}{cmp}_'))
                if (l_id == 0) and (cmp == 0):
                    pars = components[0].make_params()
                else:
                    pars.update(components[-1].make_params())

                pars[f"gaus{l_id}{cmp}_amplitude"].set(value=flux_guess, min=0, max=flux_guess * 10, vary=True)
                if (cmp == 0) and (l_id > min_lid_tie_vel) and tie_vel[l_id]:
                    pars[f"gaus{l_id}{cmp}_center"].set(expr=f'gaus{min_lid_tie_vel}0_center*{lines[l_id]/lines[min_lid_tie_vel]}')
                elif (cmp>0) and (l_id > ids_with_this_number_of_comp[0]):
                    pars[f"gaus{l_id}{cmp}_center"].set(
                        expr=f'gaus{ids_with_this_number_of_comp[0]}{cmp}_center*{lines[ids_with_this_number_of_comp[cmp]]/lines[0]}')
                elif cmp > 0:
                    l0 = res.params[f'gaus{l_id}0_center'].value
                    if l0 is None or ~np.isfinite(l0):
                        l0 = l*(1+velocity/3e5)
                    pars[f"gaus{l_id}{cmp}_center"].set(value=l0,
                                                        min=l0 - 3.,
                                                        max=l0 + 3., vary=True)
                else:
                    pars[f"gaus{l_id}{cmp}_center"].set(value=l*(1+velocity/3e5),
                                                        min=l*(1+velocity/3e5)+mean_bounds[0],
                                                        max=l*(1+velocity/3e5)+mean_bounds[1], vary=True)

                if (cmp == 0) and (l_id>min_lid_tie_disp) and tie_disp[l_id]:
                    pars[f"gaus{l_id}{cmp}_sigma"].set(expr=f'gaus{min_lid_tie_disp}0_sigma')
                elif (cmp>0) and (l_id > ids_with_this_number_of_comp[0]):
                    pars[f"gaus{l_id}{cmp}_sigma"].set(
                        expr=f'gaus{ids_with_this_number_of_comp[0]}{cmp}_sigma')
                else:
                    pars[f"gaus{l_id}{cmp}_sigma"].set(value=mean_std, min=0.6 * mean_std, max=15 * mean_std, vary=True)

                if my_model is None:
                    my_model = components[0]
                else:
                    my_model += components[-1]

        if fix_ratios is not None:
            fix_ratios = np.array(fix_ratios)
            rat_reference = np.flatnonzero(fix_ratios == 1)
            if len(rat_reference) > 0:
                rat_reference = rat_reference[0]
                for l_id in range(len(lines)):
                    if (l_id == rat_reference) or (fix_ratios[l_id] == -1):
                        continue
                    for cmp in range(cur_n_comps[l_id]):
                        ids_with_this_number_of_comp = np.flatnonzero(cur_n_comps >= (cmp + 1))
                        if rat_reference not in ids_with_this_number_of_comp:
                            continue
                        pars[f"gaus{l_id}{cmp}_amplitude"].set(
                            expr=f'gaus{rat_reference}{cmp}_amplitude * {fix_ratios[l_id]}')

        cont_model = ConstantModel(nan_policy='omit', prefix='contin_')
        pars.update(cont_model.make_params(c=cont_guess))
        my_model += cont_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = my_model.fit(spectrum, pars, x=wave, weights=1/errors)

        # print(res.redchi, res.chisqr, (np.nansum(errors.ravel()**2)/np.nansum(spectrum.ravel()**2)))
        if (res.redchi > 1) and (np.sum(cur_n_comps) < n_max_lines):
            continue
        else:
            if not subtract_lsf:
                disp = [np.array(res.params[f'gaus{l_id}{0}_sigma'].value).astype(float)/lines[l_id]*3e5 for l_id in range(len(lines))]
            else:
                disp = [np.sqrt(np.array(res.params[f'gaus{l_id}{0}_sigma'].value).astype(float)**2 - mean_std**2)/ lines[l_id] * 3e5 for l_id in range(len(lines))]

            disp_err = [(np.array(res.params[f'gaus{l_id}{0}_sigma'].stderr).astype(float) / lines[l_id]) * 3e5 for l_id in range(len(lines))]
            vel = [(np.array(res.params[f'gaus{l_id}{0}_center'].value).astype(float) / lines[l_id] - 1) * 3e5 for l_id in range(len(lines))]
            vel_err = [(np.array(res.params[f'gaus{l_id}{0}_center'].stderr).astype(float) / lines[l_id]) * 3e5 for l_id in range(len(lines))]
            fluxes = [np.array(res.params[f'gaus{l_id}{0}_amplitude'].value).astype(float)*1e-16/np.pi/(fiber_d/2)**2 for l_id in range(len(lines))]
            fluxes_err = [np.array(res.params[f'gaus{l_id}{0}_amplitude'].stderr).astype(float)*1e-16/np.pi/(fiber_d/2)**2 for l_id in range(len(lines))]
            cont = [np.array(res.params[f'contin_c'].value).astype(float)*1e-16/np.pi/(fiber_d/2)**2]*len(lines)
            cont_err = [np.array(res.params[f'contin_c'].stderr).astype(float)*1e-16/np.pi/(fiber_d/2)**2]*len(lines)

            other_comps = [None, None]
            if max(max_n_comp) > 1:
                other_comps[0] = {
                    'disp': [np.nan]*len(lines),
                    'disp_err': [np.nan] * len(lines),
                    'vel': [np.nan] * len(lines),
                    'vel_err': [np.nan] * len(lines),
                    'fluxes': [np.nan] * len(lines),
                    'fluxes_err': [np.nan] * len(lines),
                }
            if max(max_n_comp) > 2:
                    other_comps[1] = {
                        'disp': [np.nan] * len(lines),
                        'disp_err': [np.nan] * len(lines),
                        'vel': [np.nan] * len(lines),
                        'vel_err': [np.nan] * len(lines),
                        'fluxes': [np.nan] * len(lines),
                        'fluxes_err': [np.nan] * len(lines),
                    }
            for ind in range(2):
                if other_comps[ind] is None:
                    continue
                rec = np.flatnonzero(cur_n_comps == (ind+2))
                for l_id in rec:
                    if not subtract_lsf:
                        other_comps[ind]['disp'][l_id] = np.array(res.params[f'gaus{l_id}{ind+1}_sigma'].value).astype(float)/lines[l_id]*3e5
                    else:
                        other_comps[ind]['disp'][l_id] = np.sqrt(np.array(res.params[f'gaus{l_id}{ind+1}_sigma'].value).astype(float)**2-mean_std**2)/ lines[l_id] * 3e5
                    other_comps[ind]['disp_err'][l_id] = (np.array(res.params[f'gaus{l_id}{ind+1}_sigma'].stderr).astype(float) / lines[l_id]) * 3e5

                    other_comps[ind]['vel'][l_id] = (np.array(res.params[f'gaus{l_id}{ind+1}_center'].value).astype(float) / lines[l_id] - 1) * 3e5
                    other_comps[ind]['vel_err'][l_id] = (np.array(res.params[f'gaus{l_id}{ind+1}_center'].stderr).astype(float) / lines[l_id]) * 3e5
                    other_comps[ind]['fluxes'][l_id] = np.array(res.params[f'gaus{l_id}{ind+1}_amplitude'].value).astype(float)*1e-16/np.pi/(fiber_d/2)**2
                    other_comps[ind]['fluxes_err'][l_id] = np.array(res.params[f'gaus{l_id}{ind+1}_amplitude'].stderr).astype(float)*1e-16/np.pi/(fiber_d/2)**2

                    if other_comps[ind]['fluxes'][l_id] > fluxes[l_id]:
                        tmp = np.copy(fluxes[l_id])
                        fluxes[l_id] = np.copy(other_comps[ind]['fluxes'][l_id])
                        other_comps[ind]['fluxes'][l_id] = tmp
                        tmp = np.copy(disp[l_id])
                        disp[l_id] = np.copy(other_comps[ind]['disp'][l_id])
                        other_comps[ind]['disp'][l_id] = tmp
                        tmp = np.copy(vel[l_id])
                        vel[l_id] = np.copy(other_comps[ind]['vel'][l_id])
                        other_comps[ind]['vel'][l_id] = tmp
                        tmp = np.copy(fluxes_err[l_id])
                        fluxes_err[l_id] = np.copy(other_comps[ind]['fluxes_err'][l_id])
                        other_comps[ind]['fluxes_err'][l_id] = tmp
                        tmp = np.copy(vel_err[l_id])
                        vel_err[l_id] = np.copy(other_comps[ind]['vel_err'][l_id])
                        other_comps[ind]['vel_err'][l_id] = tmp
                        tmp = np.copy(disp_err[l_id])
                        disp_err[l_id] = np.copy(other_comps[ind]['disp_err'][l_id])
                        other_comps[ind]['disp_err'][l_id] = tmp

    if ax is not None:
        ax.plot(wave, spectrum, 'k-', label='Obs')
        ax.plot(wave, res.eval(**res.best_values, x=wave), 'r--', label=f'Fit')
        ax.legend()
    if return_plot_data:
        plot_data = (wave, spectrum, res.eval(**res.best_values, x=wave))
        return fluxes, vel, disp, cont, fluxes_err, vel_err, disp_err, cont_err, other_comps[0], other_comps[1], plot_data
    else:
        return fluxes, vel, disp, cont, fluxes_err, vel_err, disp_err, cont_err, other_comps[0], other_comps[1],


def fit_cur_spec(data, wave=None, lines=None, fix_ratios=None, velocity=0, mean_bounds=(-10., 10.),
                 ax=None, return_plot_data=False, subtract_lsf=True, multicomp=False):
    spectrum, errors, lsf = data
    rec = np.flatnonzero(np.isfinite(spectrum) & (spectrum != 0) &
                         np.isfinite(errors) & (np.isfinite(lsf)) & (lsf > 0) & np.isfinite(wave))  # & (errors > 0)
    if len(rec) < 10:
        if return_plot_data:
            return [np.nan]*len(lines), [np.nan]*len(lines), [np.nan]*len(lines), np.nan, [np.nan]*len(lines), np.nan, np.nan, np.nan, None
        else:
            return [np.nan] * len(lines), [np.nan]*len(lines), [np.nan]*len(lines), np.nan, [np.nan] * len(lines), np.nan, np.nan, np.nan

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
        if not multicomp:
            t1 = lambda model: (model.mean_0 * lines[1]/lines[0])
            gaussians[l_id].mean.tied = t1
            gaussians[l_id].stddev.tied = tie_std
        l_id += 1
    if len(lines) > 2:
        if not multicomp:
            t2 = lambda model: (model.mean_0 * lines[2] / lines[0])
            gaussians[l_id].mean.tied = t2
            gaussians[l_id].stddev.tied = tie_std
        l_id += 1
    if len(lines) > 3:
        if not multicomp:
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
        res = fitter(g_init, wave, spectrum, weights=weights)
    except fitting.NonFiniteValueError:
        if return_plot_data:
            return [np.nan]*len(lines), np.nan, np.nan, np.nan, [np.nan]*len(lines), np.nan, np.nan, np.nan, None
        else:
            return [np.nan] * len(lines), np.nan, np.nan, np.nan, [np.nan] * len(lines), np.nan, np.nan, np.nan
    if multicomp:
        vel = [(res.__getattr__(f"mean_{l_id}")/lines[0]-1)*3e5 for l_id in range(len(lines))]
    else:
        vel = [(res.__getattr__(f"mean_{l_id}") / lines[l_id] - 1) * 3e5 for l_id in range(len(lines))]

    if not subtract_lsf:
        if multicomp:
            disp = [(res.__getattr__(f"stddev_{l_id}")  / lines[0] * 3e5) for l_id in range(len(lines))]
        else:
            disp = [(res.__getattr__(f"stddev_{l_id}") / lines[l_id] * 3e5) for l_id in range(len(lines))]
    else:
        if multicomp:
            disp = [(np.sqrt(res.__getattr__(f"stddev_{l_id}") ** 2 - mean_std ** 2) / lines[0] * 3e5) for l_id in range(len(lines))]
        else:
            disp = [(np.sqrt(res.__getattr__(f"stddev_{l_id}") ** 2 - mean_std ** 2) / lines[l_id] * 3e5) for l_id in range(len(lines))]

    cont = res.__getattr__(f"amplitude_{len(lines)}").value
    fluxes = [res.__getattr__(f"amplitude_{l_id}")*res.__getattr__(f"stddev_{l_id}")*np.sqrt(2 * np.pi) for l_id in range(len(lines))]
    if fitter.fit_info['param_cov'] is None:
        cov_diag = np.zeros(shape=(len(lines)+3,), dtype=float)
        cov_diag[:] = np.nan
    else:
        cov_diag = np.diag(fitter.fit_info['param_cov'])
    fluxes_err = [np.sqrt(cov_diag[0]*res.stddev_0**2 + res.__getattr__(f"amplitude_{0}")**2 * cov_diag[2]) * np.sqrt(2 * np.pi)]
    if len(lines)>1:
        for l_id in range(len(lines)-1):
            fluxes_err.append(np.sqrt(
                cov_diag[3+l_id] * res.__getattr__(f"stddev_{l_id}") ** 2 + res.__getattr__(f"amplitude_{l_id+1}") ** 2 * cov_diag[2]) * np.sqrt(
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

def parse_config(config_filename):
    """
    Reads TOML config file
    :param config_filename: full path to the config file
    :return: dictionary with keywords controlling the data processing
    """
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
    """
    Downloads requested files (raw frames/agcam images/reduced spectra) from SAS
    :param config: dictionary with keywords controlling the data processing
    :return: status (success or failure)
    """
    tmp_for_sdss_access = os.path.join(os.environ['HOME'], 'tmp_for_sdss_access')
    if not os.path.exists(tmp_for_sdss_access):
        os.makedirs(tmp_for_sdss_access)
    cams = ['b1', 'b2', 'b3', 'r1', 'r2', 'r3', 'z1', 'z2', 'z3']
    rsync = RsyncAccess(verbose=True)
    # sets a remote mode to the real SAS
    rsync.remote()
    rsync.set_remote_base("--no-perms --omit-dir-times  rsync")
    rsync.base_dir = os.environ['SAS_BASE_DIR']+"/"
    counter = 0
    counter_exist = 0
    # add all the file(s) you want to download
    new_files = []
    if not config['download'].get('download_raw'):
        log.warning("Will skip downloading raw data")
    if config['download'].get('download_reduced'):
        log.warning("Will download reduced spectra")
    if config['download'].get('download_dap'):
        log.warning("Will download DAP outputs")
    f_reduced = {}
    f_dap = {}
    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        f_reduced[cur_obj['name']] = {'version': version}
        f_dap[cur_obj['name']] = {'version': version}
        for cur_pointing in cur_obj['pointing']:
            if cur_pointing['name'] not in f_reduced[cur_obj['name']]:
                f_reduced[cur_obj['name']][cur_pointing['name']] = []
            if cur_pointing['name'] not in f_dap[cur_obj['name']]:
                f_dap[cur_obj['name']][cur_pointing['name']] = []
            if cur_pointing['skip'].get('download'):
                log.info(f"Skip download for object = {cur_obj['name']}, pointing = {cur_pointing['name']}")
                continue
            if config['download'].get('download_reduced'):
                download_current_reduced = True
            else:
                download_current_reduced = False
            if config['download'].get('download_dap'):
                download_current_dap = True
            else:
                download_current_dap = False

            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']

                tileids = None
                if ('tileid' not in cur_pointing and ('tileid' not in data) and
                        (download_current_reduced or download_current_dap)):
                    log.warning(
                        f"Tile ID is not present for object = {cur_obj['name']}, pointing = {cur_pointing['name']}. "
                        f"Can't download the reduced data or DAP products for for it")
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
                d_agcam_root = os.path.join(agcam_dir,
                                            str(data['mjd']))
                d_agcam = os.path.join(d_agcam_root, 'coadds')
                d_data = os.path.join(raw_data_dir, str(data['mjd']))
                for check_dir in [d_agcam_root, d_agcam, d_data]:
                    if not os.path.exists(check_dir):
                        os.makedirs(check_dir)
                        uid = os.stat(check_dir).st_uid
                        if server_group_id is not None:
                            os.chown(check_dir, uid=uid, gid=server_group_id)
                        os.chmod(check_dir, 0o775)

                for exp_ind,exp in enumerate(exps):
                    if config['download'].get('download_raw'):
                        for cam in cams:
                            f = glob.glob(
                                os.path.join(raw_data_dir, str(data['mjd']),
                                             f'*-{cam}-*{exp}.fits.gz'))
                            if not config['download'].get('force_download') and (len(f) == 1):
                                counter_exist += 1
                            else:
                                new_files.append(os.path.join(raw_data_dir, str(data['mjd']),
                                                              f'sdR-s-{cam}-{exp:08d}.fits.gz'))
                                counter += 1
                                rsync.add('lvm_raw', camspec=cam, expnum=str(exp), hemi='s', mjd=str(data['mjd']))

                    if download_current_reduced or download_current_dap:
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
                        if download_current_reduced:
                            frame_types = ['SFrame']
                            if config['download'].get('include_cframes'):
                                frame_types.append('CFrame')
                            for ft in frame_types:
                                f = glob.glob(
                                    os.path.join(drp_results_dir_sas, short_tileid, tileids[exp_ind], str(data['mjd']),
                                                 f'lvm{ft}-*{exp}.fits'))
                                if not config['download'].get('force_download') and (len(f) == 1):
                                    counter_exist += 1
                                else:
                                    cur_f = os.path.join(drp_results_dir_sas, short_tileid, tileids[exp_ind],
                                                 str(data['mjd']), f'lvm{ft}-{exp:08d}.fits')
                                    f_reduced[cur_obj['name']][cur_pointing['name']].append(cur_f)
                                    new_files.append(cur_f)
                                    counter += 1
                                    rsync.add('lvm_frame', expnum=str(exp), mjd=str(data['mjd']), tileid=tileids[exp_ind],
                                              kind=ft, drpver=f'{red_data_version}')  # /{short_tileid}
                        if download_current_dap:
                            f = glob.glob(
                                os.path.join(dap_results_dir_sas, short_tileid, tileids[exp_ind], str(data['mjd']),
                                             f'{exp:08d}', f'dap-rsp*{exp:08d}.dap.fits.gz'))
                            if not config['download'].get('force_download') and (len(f) == 1):
                                counter_exist += 1
                            else:
                                cur_f = os.path.join(dap_results_dir_sas, short_tileid, tileids[exp_ind],
                                                     str(data['mjd']), f'{exp:08d}',
                                                     f'dap-rsp108-sn20-{exp:08d}.dap.fits.gz')
                                f_dap[cur_obj['name']][cur_pointing['name']].append(cur_f)
                                counter += 1
                                rsync.initial_stream.append_task(sas_module='sdsswork',
                                                             location=f'lvm/spectro/analysis/{dap_version}/'
                                                                      f'{short_tileid}/{tileids[exp_ind]}/'
                                                                      f'{str(data["mjd"])}/{exp:08d}/'
                                                                      f'dap-rsp108-sn20-{exp:08d}.dap.fits.gz',
                                                             source=f'--no-perms --omit-dir-times '
                                                                    f'rsync://sdss5@dtn.sdss.org/sdsswork/lvm/'
                                                                    f'spectro/analysis/{dap_version}/{short_tileid}/'
                                                                    f'{tileids[exp_ind]}/{str(data["mjd"])}/'
                                                                    f'{exp:08d}/dap-rsp108-sn20-{exp:08d}.dap.fits.gz',
                                                             destination=f'{dap_results_dir_sas}/{short_tileid}/'
                                                                         f'{tileids[exp_ind]}/{str(data["mjd"])}/'
                                                                         f'{exp:08d}/'
                                                                         f'dap-rsp108-sn20-{exp:08d}.dap.fits.gz')

                    if config['download'].get('download_agcam'):
                        # add corresponding agcam coadd images
                        f = os.path.join(d_agcam, f'lvm.sci.coadd_s{exp:08d}.fits')
                        if not config['download'].get('force_download') and os.path.exists(f):
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
        rsync.stream.cli.data_dir = tmp_for_sdss_access
        # start the download(s)
        rsync.commit()
        shutil.rmtree(tmp_for_sdss_access)
    except Exception as e:
        log.error(f"Something wrong with rsync: {e}")
        try:
            shutil.rmtree(tmp_for_sdss_access)
        except Exception as e:
            pass
        return False
    for f in new_files:
        uid = os.stat(f).st_uid
        if server_group_id is not None:
            os.chown(f, uid=uid, gid=server_group_id)
        os.chmod(f, 0o664)

    if config['download'].get('download_reduced') or config['download'].get('download_dap'):
        output_dir = config.get('default_output_dir')
        if not output_dir:
            log.error("Output directory is not set up. Cannot copy files")
            return False
        copying_type = ['reduced frames', 'DAP results']
        for ind, cur_dict in enumerate([f_reduced, f_dap]):
            for obj_name in cur_dict:
                version = cur_dict[obj_name]['version']
                for pointing_name in cur_dict[obj_name]:
                    if pointing_name == 'version':
                        continue
                    if len(cur_dict[obj_name][pointing_name]) == 0:
                        log.warning(f"Nothing to copy for object = {obj_name}, pointing = {pointing_name} ({copying_type[ind]})")
                        continue
                    if not pointing_name:
                        curdir = os.path.join(output_dir, obj_name, version)
                    else:
                        curdir = os.path.join(output_dir, obj_name, version, pointing_name)
                    if not os.path.exists(curdir):
                        os.makedirs(curdir)
                        uid = os.stat(curdir).st_uid
                        if server_group_id is not None:
                            os.chown(curdir, uid=uid, gid=server_group_id)
                        os.chmod(curdir, 0o775)
                    log.info(f"Copy {len(cur_dict[obj_name][pointing_name])} {copying_type[ind]} for object = {obj_name}, pointing = {pointing_name}")
                    for sf in cur_dict[obj_name][pointing_name]:
                        fname = os.path.join(curdir, os.path.split(sf)[-1])
                        if os.path.isfile(fname):
                            os.remove(fname)
                        elif os.path.islink(fname):
                            os.unlink(fname)
                        # os.symlink(sf, fname)
                        shutil.copy(sf, curdir)
                        if server_group_id is not None:
                            uid = os.stat(fname).st_uid
                            os.chown(fname, uid=uid, gid=server_group_id)
                        os.chmod(fname, 0o664)

    return True


def metadata_parallel(mjd):
    """
    Aux. function to launch metadata regeneration in parallel threads
    :param mjd: MJD to analyse
    :return: status (success or failure)
    """
    try:
        os.system(f"drp metadata regenerate -m {mjd}")
    except Exception as e:
        log.error(f"Something wrong with metadata regeneration: {e}")
        return False
    return True


def regenerate_metadata(config):
    """
    Regenerates metadata for all MJDs in the current reduction
    :param config: dictionary with keywords controlling the data processing
    :return: status (success or failure)
    """
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


def reduce_parallel(exp_pairs):
    """
    Aux. function to launch data reduction in parallel threads
    :param exp_pairs: tuple with exp.id, mjd and sky weights
    :return: status (success or failure)
    """
    exp, mjd, weights = exp_pairs
    try:
        if weights is not None:
            add_weights = f"--sky-weights {weights[0]} {weights[1]}"
        else:
            add_weights = ""
        os.system(f"drp run -m {mjd} -e {exp} {add_weights}")# >/dev/null 2>&1")
        # os.system(f"drp run --no-sci --with-cals -m 60291 -e {exp}")
    except Exception as e:
        log.error(f"Something wrong with data reduction: {e}")
        return False
    return True


def do_reduction(config):
    """
    Launch data reduction for selected exposures according to config file
    :param config: dictionary with keywords defining data processing
    :return: status (success or failure)
    """
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

    mjds = np.array(mjds)
    exps, exp_ids = np.unique(exps, return_index=True)
    mjds = mjds[exp_ids]

    exp_pairs = []
    # == Derive sky weights:
    for exp_id, exp in enumerate(exps):
        if config['reduction'].get("wham_sky_only"):
            f = os.path.join(raw_data_dir, str(mjds[exp_id]),
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

        exp_pairs.append((exp, mjds[exp_id], weights))

    statuses = []
    if not config['reduction'].get('reduce_parallel'):
        log.info(f"Start reduction of {len(exp_pairs)} exposures")
        for exp in tqdm(exp_pairs, ascii=True, desc="Data reduction", total=len(exp_pairs)):
            status = reduce_parallel(exp)
            if not status:
                log.error(f"Something went wrong with mjd={str(data['mjd'])}, exposure={exp[0]}")
            statuses.append(status)
    else:
        procs = np.nanmin([config['nprocs'], len(exps)])
        log.info(f"Start reduction of {len(exp_pairs)} exposures in {procs} parallel processes")
        with mp.Pool(processes=procs) as pool:

            for status in tqdm(pool.imap_unordered(reduce_parallel, exp_pairs),
                               ascii=True, desc="Data reduction",
                               total=len(exp_pairs), ):
                statuses.append(status)
            pool.close()
            pool.join()
            gc.collect()
    if not np.all(statuses):
        return False

    return True


def fix_astrometry(file, first_exp=None):
    with fits.open(file) as hdu:
        slitmap = Table(hdu['SLITMAP'].data)
        h = hdu[0].header
        h_tab = hdu['SLITMAP'].header
        check_fiber = np.flatnonzero((slitmap['xpmm'] == 0) & (slitmap['ypmm'] == 0) & (slitmap['telescope'] == 'Sci'))
        if not (np.isclose(slitmap['ra'][check_fiber], 0) & np.isclose(slitmap['dec'][check_fiber], 0)):
            return
        log.warning(f"Update astrometry for mjd={h.get('MJD')}, exp={h.get('EXPOSURE')} from agcam image")
        pointing = os.path.split(file)[0].split(os.sep)[-1]
        w_dir=os.sep.join(os.path.split(file)[0].split(os.sep)[:-1])
        ra, dec, pa = derive_radec_ifu(h.get('MJD'), h.get('EXPOSURE'), w_dir=w_dir,
                                   pointing_name=pointing, objname='', first_exp=first_exp)
        rec_sci = (slitmap['telescope'] == 'Sci')

        ra_fib, dec_fib = make_radec(slitmap['xpmm'][rec_sci], slitmap['ypmm'][rec_sci],
                                     ra, dec, pa)

        slitmap['ra'][rec_sci] = ra_fib
        slitmap['dec'][rec_sci] = dec_fib
        hdu['SLITMAP'] = fits.BinTableHDU(data=slitmap, header=h_tab, name='SLITMAP')
        hdu.writeto(file, overwrite=True)

def copy_reduced_data(config, output_dir=None):
    if output_dir is None:
        output_dir = config.get('default_output_dir')
    if not output_dir:
        log.error("Output directory is not set up. Cannot copy files")
        return False

    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        for cur_pointing in cur_obj['pointing']:
            if cur_pointing['skip'].get('reduction'):
                log.warning(
                    f"Skip copy reduced spectra for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
                continue
            if not cur_pointing.get('name'):
                curdir = os.path.join(output_dir, cur_obj['name'], version)
            else:
                curdir = os.path.join(output_dir, cur_obj['name'], version, cur_pointing.get('name'))
            if not os.path.exists(curdir):
                os.makedirs(curdir)
                uid = os.stat(curdir).st_uid
                if server_group_id is not None:
                    os.chown(curdir, uid=uid, gid=server_group_id)
                os.chmod(curdir, 0o775)

            source_files = []
            first_exps = []
            for data in cur_pointing['data']:
                if isinstance(data['exp'], int):
                    exps = [data['exp']]
                else:
                    exps = data['exp']
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
                    if tileids is not None and tileids[exp_ind] is not None:
                        if cur_obj['name'] == 'Orion':
                            if (int(tileids[exp_ind]) < 1027000) & (int(tileids[exp_ind]) != 11111):
                                tileids[exp_ind] = str(int(tileids[exp_ind])+27748)

                        if (tileids[exp_ind] == '1111') or (tileids[exp_ind] == '11111'):
                            short_tileid = '0011XX'
                        elif tileids[exp_ind] == '999':
                            short_tileid = '0000XX'
                        else:
                            short_tileid = tileids[exp_ind][:4] + 'XX'
                    source_files.append(os.path.join(drp_results_dir, short_tileid, str(tileids[exp_ind]),
                                                     str(data['mjd']), f'lvmSFrame-{exp:08d}.fits'))
                    first_exps.append(exps[0])
                    if config['reduction'].get('copy_cframes'):
                        source_files.append(os.path.join(drp_results_dir, short_tileid, str(tileids[exp_ind]),
                                                         str(data['mjd']), f'lvmCFrame-{exp:08d}.fits'))
                        first_exps.append(exps[0])
            if len(source_files) == 0:
                log.warning(f"Nothing to copy for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
                continue
            log.info(f"Copy {len(source_files)} for object = {cur_obj['name']}, pointing = {cur_pointing.get('name')}")
            bad_expnums = []
            for file_id, sf in tqdm(enumerate(source_files), total=len(source_files)):
                if not os.path.isfile(sf):
                    bad_expnums.append(sf.split('-')[-1].split('.')[0])
                    continue
                fname = os.path.join(curdir, os.path.split(sf)[-1])
                if os.path.exists(fname):
                    os.remove(fname)
                shutil.copy(sf, curdir)
                fix_astrometry(fname, first_exp=first_exps[file_id])
            if len(bad_expnums) > 0:
                log.warning(f"{len(bad_expnums)} files were not copied because they are not found in "
                            f"the directory with the reduced data. Probably, data reduction failed, check logs. "
                            f"Their exnums are: {', '.join(bad_expnums)}")


def do_coadd_spectra(config, w_dir=None):
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
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
                    if config['imaging']['include_sky']:
                        cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
                    else:
                        cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmSFrame-{exp:08d}.fits')
                    if not os.path.exists(cur_fname):
                        log.warning(f"Can't find {cur_fname}")
                        continue
                    files.append(cur_fname)

            if len(files) < 2:
                log.warning(f"Nothing to coadd for object = {cur_obj['name']}, pointing = {cur_pointing['name']}")
                continue

            fout = os.path.join(cur_wdir, cur_pointing['name'], f'combined_spectrum_{ind_pointing:02d}.fits')
            with fits.open(files[0]) as hdu_ref:
                for ext in ['FLUX', 'IVAR', 'SKY', 'SKY_IVAR']:
                    hdu_ref[ext].data[hdu_ref['MASK'] == 1] = np.nan
                    hdu_ref[ext].data = np.expand_dims(hdu_ref[ext].data, axis=0) * corrections[0]
                for f_ind, f in enumerate(files[1:]):
                    with fits.open(f) as hdu:
                        for ext in ['FLUX', 'IVAR', 'SKY', 'SKY_IVAR']:
                            hdu[ext].data[hdu['MASK'] == 1] = np.nan
                            hdu[ext].data = np.expand_dims(hdu[ext].data, axis=0) * corrections[f_ind+1] / hdu[0].header['EXPTIME'] * hdu_ref[0].header['EXPTIME']
                            hdu_ref[ext].data = np.vstack([hdu_ref[ext].data, hdu[ext].data])
                for ext in ['FLUX', 'IVAR', 'SKY', 'SKY_IVAR']:
                    if "IVAR" in ext:
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


def initialize_hist_layout():
    fig = plt.figure(figsize=(15, 25))
    nregs = 6
    nlines = 3
    gs = GridSpec(nregs, nlines, height_ratios=[1]*nregs, width_ratios=[1]*nlines, wspace=0.2, hspace=0.3,
                  figure=fig, left=0.1, right=0.99, top=0.95, bottom=0.1)
    axes = [[plt.subplot(gs[i, j]) for j in range(nlines)] for i in range(nregs)]
    return fig, axes


def parse_dap_results(config, w_dir=None, local_dap_results=False, mode=None):
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False
    status_out = True
    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        log.info(f"Parsing DAP results for {cur_obj.get('name')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            status_out = status_out & False
            continue
        if local_dap_results:
            if ((config['imaging'].get('use_single_rss_file') and not config['imaging'].get('use_binned_rss_file'))
                    or mode == 'single_rss'):
                f_dap = os.path.join(cur_wdir, 'dap_output', f"{cur_obj.get('name')}_all_RSS.dap.fits.gz")
                f_rss = os.path.join(cur_wdir, f"{cur_obj['name']}_all_RSS.fits")
                f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes_singleRSS_dap.txt")
            elif ((config['imaging'].get('use_binned_rss_file') or (mode == 'binned'))
                  and config.get('binning') is not None):
                bin_line = config['binning'].get('line')
                if not bin_line:
                    bin_line = 'Ha'
                target_sn = config['binning'].get('target_sn')
                if not target_sn:
                    target_sn = 30.
                suffix_out = config['binning'].get('rss_output_suffix')
                if not suffix_out:
                    suffix_out = '_vorbin_rss.fits'
                f_dap = os.path.join(cur_wdir, f"dap_output_binfluxes_{bin_line}_sn{target_sn}",
                                     f"{cur_obj.get('name')}_{bin_line}_sn{target_sn}"
                                     f"{suffix_out.replace('.fits', '.dap.fits.gz')}")
                f_rss = os.path.join(cur_wdir, f"{cur_obj.get('name')}_{bin_line}_sn{target_sn}{suffix_out}")
                f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_binfluxes_{bin_line}_sn{target_sn}_dap.txt")

            elif mode == 'extracted':
                suffix_out = config['extraction'].get('file_output_suffix')
                if not suffix_out:
                    suffix_out = '_extracted.fits'
                f_dap = os.path.join(cur_wdir, f"dap_output_extracted",
                                     f"{cur_obj.get('name')}"
                                     f"{suffix_out.replace('.fits', '.dap.fits.gz')}")
                f_rss = os.path.join(cur_wdir, f"{cur_obj.get('name')}{suffix_out}")
                f_tab_summary = os.path.join(cur_wdir,
                                             f"{cur_obj.get('name')}_extracted_dap.txt")

        else:
            f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes_dap.txt")

        dtypes = [float]*(len(dap_results_correspondence)*6+5)
        dtypes[2] = str
        names = ['fib_ra', 'fib_dec', 'id', 'fluxcorr', 'vhel_corr']
        for kw in dap_results_correspondence.keys():
            names.extend([kw+"_flux", kw+"_fluxerr", kw+"_vel", kw+"_velerr",kw+"_disp", kw+"_disperr"])

        tab_summary = Table(data=None, names=names,
                            dtype=dtypes)

        if config['imaging'].get('save_hist_dap'):
            f_hist_out = os.path.join(cur_wdir, 'compare_flux_levels.pdf')
            pdf = PdfPages(f_hist_out)
            fig_hist, axes = initialize_hist_layout()
            nregs_hist_done = 0

        for cpnt_id, cur_pointing in enumerate(cur_obj['pointing']):
            if local_dap_results and cpnt_id > 0:
                break

            if cur_pointing['skip'].get('imaging') and not local_dap_results:
                log.warning(f"Skip DAP processing (and imaging) for object = {cur_obj['name']}, "
                            f"pointing = {cur_pointing.get('name')}")
                continue
            if local_dap_results:
                all_the_data = [[f_dap,f_rss]]
            else:
                all_the_data = cur_pointing['data']
            for data in tqdm(all_the_data, ascii=True, desc="MJDs done", total=len(all_the_data)):
                if not local_dap_results:
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
                else:
                    exps = [0]
                    cur_flux_corr = [1.]
                for exp_id, exp in enumerate(exps):
                    if local_dap_results:
                        cur_fname = data[0]
                        cur_fname_spec = data[1]
                    else:
                        cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'dap-rsp108-sn20-{exp:08d}.dap.fits.gz')
                        cur_fname_spec = os.path.join(cur_wdir, cur_pointing['name'], f'lvmSFrame-{exp:08d}.fits')
                    if not os.path.exists(cur_fname):
                        log.warning(f"Can't find {cur_fname}")
                        status_out = status_out & False
                        continue
                    with fits.open(cur_fname_spec) as rss:
                        cur_table_fibers = Table(rss['SLITMAP'].data)
                        sci = cur_table_fibers['targettype'] == 'science'
                        if config['imaging'].get('skip_bad_fibers'):
                            sci = sci & (cur_table_fibers['fibstatus'] == 0)
                        sci = np.flatnonzero(sci)

                    with fits.open(cur_fname) as rss:
                        # cur_table_fibers = Table(rss['PT'].data)
                        cur_table_fluxes = Table(rss['PM_ELINES'].data)
                        cur_table_fluxes_faint_b = Table(rss['NP_ELINES_B'].data)
                        cur_table_fluxes_faint_r = Table(rss['NP_ELINES_R'].data)
                        cur_table_fluxes_faint_i = Table(rss['NP_ELINES_I'].data)
                        cur_obstime = Time(rss[0].header['OBSTIME'])
                        try:
                            radec_central = SkyCoord(ra=rss[0].header['POSCIRA'],
                                                     dec=rss[0].header['POSCIDE'],
                                                     unit=('degree', 'degree'))
                        except ValueError:
                            radec_central = SkyCoord(ra=np.nanmedian(cur_table_fibers[sci]['ra']),
                                                     dec=np.nanmedian(cur_table_fibers[sci]['dec']),
                                                     unit=('degree', 'degree'))

                    if not local_dap_results:
                        vcorr = np.round(radec_central.radial_velocity_correction(kind='heliocentric', obstime=cur_obstime,
                                                                                  location=obs_loc).to(u.km / u.s).value,1)
                        id_prefix = int(exp)

                    else:
                        vcorr = 0.
                        id_prefix = int(str(cur_table_fluxes[0]['id']).split('.')[0])
                        #TODO: check vcorr for single RSS

                    cur_table_summary = cur_table_fibers[sci]['ra', 'dec'].copy() #
                    cur_table_summary.rename_columns(['ra', 'dec'], ['fib_ra', 'fib_dec'])

                    cur_table_summary.add_columns(
                         [Column(np.array([f'{id_prefix}.{int(cur_fibid+1)}' for cur_fibid in sci]),
                               name='id', dtype=str),
                         Column(np.array([str(cur_flux_corr[exp_id])] * len(cur_table_summary)),
                                name='fluxcorr', dtype=float),
                         Column(np.array([str(vcorr)] * len(cur_table_summary)), name='vhel_corr',
                                dtype=float)]
                         )
                    for kw in dap_results_correspondence.keys():
                        if isinstance(dap_results_correspondence[kw], str):
                            curline_wl = float(dap_results_correspondence[kw].split('_')[-1])
                            if curline_wl > 7500:
                                cur_table_fluxes_faint = cur_table_fluxes_faint_i
                            elif curline_wl > 5800:
                                cur_table_fluxes_faint = cur_table_fluxes_faint_r
                            else:
                                cur_table_fluxes_faint = cur_table_fluxes_faint_b
                            cur_table_summary = join(cur_table_summary, cur_table_fluxes_faint['id',
                                "flux_" + dap_results_correspondence[kw], "e_flux_"+dap_results_correspondence[kw],
                                "vel_" + dap_results_correspondence[kw], "e_vel_"+dap_results_correspondence[kw],
                                "disp_" + dap_results_correspondence[kw], "e_disp_"+dap_results_correspondence[kw]
                            ], keys='id')

                            cur_table_summary["flux_"+dap_results_correspondence[kw]] *= cur_flux_corr[exp_id]
                            cur_table_summary["e_flux_"+dap_results_correspondence[kw]] *= cur_flux_corr[exp_id]
                            cur_table_summary["vel_" + dap_results_correspondence[kw]] += vcorr
                            cur_table_summary["disp_" + dap_results_correspondence[kw]] /= (curline_wl / 3e5)
                            cur_table_summary["e_disp_"+dap_results_correspondence[kw]] /= (curline_wl / 3e5)
                            cur_table_summary.rename_columns(["flux_"+dap_results_correspondence[kw],
                                                              "e_flux_"+dap_results_correspondence[kw],
                                                              "vel_" + dap_results_correspondence[kw],
                                                              "e_vel_" + dap_results_correspondence[kw],
                                                              "disp_" + dap_results_correspondence[kw],
                                                              "e_disp_"+dap_results_correspondence[kw]],
                                                             [kw + "_flux", kw + "_fluxerr", kw + "_vel",
                                                              kw + "_velerr", kw + "_disp", kw + "_disperr"])

                        else:
                            rec_cur_line = np.flatnonzero(cur_table_fluxes['wl'] == dap_results_correspondence[kw])
                            cur_table_summary = join(cur_table_summary, cur_table_fluxes[rec_cur_line]['id','flux', 'e_flux',
                                                    'vel', 'e_vel', 'disp', 'e_disp'], keys='id')
                            if kw == 'OI':
                                cur_table_summary['flux'] -= np.nanmedian(cur_table_summary['flux'])
                            cur_table_summary['flux'] *= cur_flux_corr[exp_id]
                            cur_table_summary['e_flux'] *= cur_flux_corr[exp_id]
                            cur_table_summary['vel'] += vcorr
                            cur_table_summary['disp'] /= (dap_results_correspondence[kw] / 3e5)
                            cur_table_summary['e_disp'] /= (dap_results_correspondence[kw] / 3e5)
                            cur_table_summary.rename_columns(['flux', 'e_flux', 'vel', 'e_vel', 'disp', 'e_disp'],
                                                             [kw+"_flux", kw+"_fluxerr", kw+"_vel",
                                                              kw+"_velerr", kw+"_disp", kw+"_disperr"])

                    tab_summary = vstack([tab_summary, cur_table_summary])

                    if config['imaging'].get('save_hist_dap'):
                        if nregs_hist_done >= 6:
                            pdf.savefig(fig_hist)
                            plt.close()
                            fig_hist, axes = initialize_hist_layout()
                            nregs_hist_done = 0

                        for kw_ind, kw in enumerate(['Ha', 'SII6717', 'OIII5007']):
                            rec_cur_line = np.flatnonzero(cur_table_fluxes['wl'] == dap_results_correspondence[kw])
                            data_hist = cur_table_fluxes[rec_cur_line]['flux']
                            data_hist = data_hist[np.isfinite(data_hist) & (data_hist != 0)]
                            data_hist = data_hist[(data_hist < np.nanpercentile(data_hist, 90)) &
                                                  (data_hist > np.nanpercentile(data_hist, 10))]
                            med = np.nanmedian(data_hist)
                            mean = np.nanmean(data_hist)
                            std = np.nanstd(data_hist)
                            ax = axes[nregs_hist_done][kw_ind]
                            plt.sca(ax)
                            plt.hist(data_hist, bins=50, color='skyblue', edgecolor='black')
                            xlim = plt.xlim()
                            ylim = plt.ylim()
                            plt.title(f"Pointing: {cur_pointing['name']}, MJD={data['mjd']}, "
                                      f"exp={exp}, {kw}", fontsize=14)
                            plt.text(xlim[0]+(xlim[1]-xlim[0])*0.3,ylim[0]+(ylim[1]-ylim[0])*0.9,
                                     f'median = {np.round(med,1)}', fontsize=14)
                            plt.text(xlim[0] + (xlim[1] - xlim[0]) * 0.3, ylim[0] + (ylim[1] - ylim[0]) * 0.8,
                                     f'mean = {np.round(mean, 1)}±{np.round(std, 1)}', fontsize=14)
                            plt.xlabel('Flux')
                            plt.ylabel('Frequency')

                        nregs_hist_done+=1
        if config['imaging'].get('save_hist_dap'):
            pdf.savefig(fig_hist)
            plt.close()
            pdf.close()
        tab_summary.write(f_tab_summary, overwrite=True, format='ascii.fixed_width_two_line')
    return status_out

def calc_bgr(data):
    """
    Calculate mean background (residuals after sky subtraction)
    :param data:
    :return:
    """
    return np.nanmedian(data[(data<np.nanpercentile(data, 60)) & (data>np.nanpercentile(data, 5))])

def process_all_rss(config, w_dir=None):
    """
    Create table with fluxes from all rss files
    :param config: dictionary with configuration parameters
    :param output_dir: path to output directory
    :return:
    """
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    precision_fiber = config['imaging'].get('fiber_pos_precision')
    if not precision_fiber:
        precision_fiber = 1.5

    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        status_out = True
        log.info(f"Analysing RSS files for {cur_obj.get('name')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes.txt")

        if not config['imaging'].get('override_flux_table') and os.path.isfile(f_tab_summary):
            log.warning("Use existing table with flux measurements. "
                        "Information about fibers won't change, but the fluxes will be updated when needed")
        else:
            tab_summary = Table(data=None, names=['fib_ra', 'fib_dec', 'ra_round', 'dec_round',
                                                  'sourceid', 'fluxcorr_b','fluxcorr_r','fluxcorr_z',
                                                  'vhel_corr', 'bgr'],
                                dtype=(float, float, float, float, 'object', 'object', 'object', 'object',
                                       'object', 'object'))

            log.info("Selecting unique fibers")
            for cur_pointing in cur_obj['pointing']:
                for data in cur_pointing['data']:
                    if isinstance(data['exp'], int):
                        exps = [data['exp']]
                    else:
                        exps = data['exp']

                    if not data.get('flux_correction'):
                        cur_flux_corr_b = [1.] * len(exps)
                        cur_flux_corr_r = [1.] * len(exps)
                        cur_flux_corr_z = [1.] * len(exps)
                    else:
                        cur_flux_corr_b = data['flux_correction']
                        cur_flux_corr_r = data['flux_correction']
                        cur_flux_corr_z = data['flux_correction']
                    if isinstance(cur_flux_corr_r, float) or isinstance(cur_flux_corr_r, int):
                        cur_flux_corr_b = [cur_flux_corr_b]
                        cur_flux_corr_r = [cur_flux_corr_r]
                        cur_flux_corr_z = [cur_flux_corr_z]

                    for exp_id, exp in enumerate(exps):
                        cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmSFrame-{exp:08d}.fits')
                        if not os.path.exists(cur_fname):
                            log.warning(f"Can't find {cur_fname}")
                            tileid = data['tileid'][exp_id]
                            if cur_obj['name'] == 'Orion':
                                if (int(tileid) < 1027000) & (int(tileid) != 11111):
                                    tileid = str(int(tileid)+27748)

                            if str(tileid) == '11111':
                                short_tileid = '0011XX'
                            elif str(tileid) == '999':
                                short_tileid = '0000XX'
                            else:
                                short_tileid = str(tileid)[:4] + 'XX'
                            cur_fname_source = os.path.join(drp_results_dir_sas, short_tileid, str(tileid),
                                                     str(data['mjd']), f'lvmSFrame-{exp:08d}.fits')
                            if not os.path.exists(cur_fname_source):
                                log.warning(f"It is also not found in reduced files from SAS")
                                continue
                            else:
                                log.warning(f"Found in files fownloaded from SAS. Copying...")
                                shutil.copy(cur_fname_source, cur_fname)
                        with fits.open(cur_fname) as rss:
                            cur_table_fibers = Table(rss['SLITMAP'].data)
                            cur_obstime = Time(rss[0].header['OBSTIME'])
                            sci = cur_table_fibers['targettype'] == 'science'
                            if config['imaging'].get('skip_bad_fibers'):
                                sci = sci & (cur_table_fibers['fibstatus'] == 0)
                            sci = np.flatnonzero(sci)
                            if rss[0].header.get('STDSENMR') and rss[0].header.get('SCISENMR'):
                                if ((rss[0].header['STDSENMR'] < 0) or (rss[0].header['STDSENMR'] < 0) or
                                    (rss[0].header['STDSENMR'] < 0) or (rss[0].header['STDSENRR']/rss[0].header['STDSENMR'] > 0.2) or
                                    abs(np.log10(float(rss[0].header['SCISENMR'])/float(rss[0].header['STDSENMR'])))>0.1):
                                    log.warning(f"{exp} has potential problems with flux calib: "
                                                f"stderr/med={np.round(rss[0].header['STDSENRB']/rss[0].header['STDSENMB'],2)}, "
                                                f"{np.round(rss[0].header['STDSENRR']/rss[0].header['STDSENMR'],2)}, "
                                                f"{np.round(rss[0].header['STDSENRZ']/rss[0].header['STDSENMZ'],2)}, "
                                                f"sci/std = {np.round(float(rss[0].header['SCISENMB'])/float(rss[0].header['STDSENMB']),2)}, "
                                                f"{np.round(float(rss[0].header['SCISENMR'])/float(rss[0].header['STDSENMR']),2)},"
                                                f"{np.round(float(rss[0].header['SCISENMZ'])/float(rss[0].header['STDSENMZ']),2)} in b,r,z")

                                    cur_flux_corr_r[exp_id] = float(rss[0].header['SCISENMR']) / float(
                                        rss[0].header['STDSENMR'])
                                    cur_flux_corr_b[exp_id] = float(rss[0].header['SCISENMB']) / float(
                                        rss[0].header['STDSENMB'])
                                    cur_flux_corr_z[exp_id] = float(rss[0].header['SCISENMZ']) / float(
                                        rss[0].header['STDSENMZ'])
                            residual_background = calc_bgr(rss['FLUX'].data[sci])
                            log.info(f"Bgr for {exp}: {residual_background}")
                            try:
                                radec_central = SkyCoord(ra=rss[0].header['POSCIRA'],
                                                         dec=rss[0].header['POSCIDE'],
                                                         unit=('degree', 'degree'))
                            except ValueError:
                                radec_central = SkyCoord(ra=np.nanmedian(cur_table_fibers[sci]['ra']),
                                                         dec=np.nanmedian(cur_table_fibers[sci]['dec']),
                                                         unit=('degree', 'degree'))

                        vcorr = np.round(radec_central.radial_velocity_correction(kind='heliocentric', obstime=cur_obstime,
                                                                               location=obs_loc).to(u.km / u.s).value,
                                         1)
                        cur_table_summary = cur_table_fibers[sci]['ra', 'dec'].copy()
                        cur_table_summary.rename_columns(['ra', 'dec'],['fib_ra','fib_dec'])
                        cf_ra = abs(3600./np.cos(np.radians(radec_central.dec.degree)))/precision_fiber
                        cf_dec = 3600. / precision_fiber
                        ndigits_ra = np.ceil(np.log10(cf_ra)).astype(int)
                        ndigits_dec = np.ceil(np.log10(cf_dec)).astype(int)

                        cur_table_summary.add_columns((Column(np.round(np.round(cur_table_fibers[sci]['ra']*cf_ra)/cf_ra,
                                                                ndigits_ra), name='ra_round', dtype=float),
                                                       Column(np.round(np.round(cur_table_fibers[sci]['dec']*cf_dec)/cf_dec,
                                                                ndigits_dec), name='dec_round', dtype=float),
                                                       Column(np.array([f'{cur_pointing["name"]}_{exp:08d}_{cur_fibid:04d}'
                                                                        for cur_fibid in cur_table_fibers[sci]["fiberid"]]),
                                                              name='sourceid', dtype=object),
                                                       Column(np.array([str(cur_flux_corr_b[exp_id])] * len(sci)),
                                                              name='fluxcorr_b', dtype=object),
                                                       Column(np.array([str(cur_flux_corr_r[exp_id])] * len(sci)),
                                                              name='fluxcorr_r', dtype=object),
                                                       Column(np.array([str(cur_flux_corr_z[exp_id])] * len(sci)),
                                                              name='fluxcorr_z', dtype=object),
                                                       Column(np.array([str(vcorr)]*len(sci)), name='vhel_corr',
                                                              dtype=object),
                                                       Column(np.array([str(residual_background)] * len(sci)), name='bgr',
                                                              dtype=object)
                                                      ))

                        tab_summary = vstack([tab_summary, cur_table_summary])

            radec_compare = np.array([tab_summary['ra_round'], tab_summary['dec_round']]).T
            order = np.lexsort(radec_compare.T)
            radec_compare = radec_compare[order]
            diff = np.diff(radec_compare, axis=0)
            ui = np.ones(len(radec_compare), 'bool')
            ui[1:] = (diff != 0).any(axis=1)
            repeated_radec = radec_compare[~ui]

            if len(repeated_radec)>0:
                log.info(f"Found {len(repeated_radec)} fibers with the same position (within {precision_fiber} arcsec)")
                rec_remove = []
                for cur_dublicate in tqdm(repeated_radec, ascii=True,
                                     total=len(repeated_radec), desc=f'Merging coinciding fibers'):
                    rec = np.flatnonzero((tab_summary['ra_round'] == cur_dublicate[0]) &
                           (tab_summary['dec_round'] == cur_dublicate[1]))
                    if len(rec) <= 1:
                        continue
                    tab_summary["sourceid"][rec[0]] = ', '.join(tab_summary["sourceid"][rec])
                    for chan in ['_b', '_r', '_z']:
                        tab_summary[f"fluxcorr{chan}"][rec[0]] = ', '.join(tab_summary[f"fluxcorr{chan}"][rec])
                    tab_summary["vhel_corr"][rec[0]] = ', '.join(tab_summary["vhel_corr"][rec])
                    tab_summary["bgr"][rec[0]] = ', '.join(tab_summary["bgr"][rec])
                    rec_remove.extend(rec[1:])
                if len(rec_remove)>0:
                    tab_summary.remove_rows(rec_remove)
            else:
                log.info(f"All {len(tab_summary)} fibers are unique")
            tab_summary.remove_columns(['ra_round','dec_round'])
            tab_summary.write(f_tab_summary, overwrite=True, format='ascii.fixed_width_two_line')

        #==============================
        log.info("Analysing spectra")
        table_fluxes = Table.read(f_tab_summary, format='ascii.fixed_width_two_line',
                                  converters={'sourceid': str, 'fluxcorr_b': str, 'fluxcorr_r': str,
                                              'fluxcorr_z': str, 'vhel_corr': str, 'bgr': str})
        mean_bounds = (-30, 30)
        vel = cur_obj.get('velocity')
        line_fit_params = []
        line_quicksum_params = []
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
                for kw in ['flux', 'fluxerr']:
                    if f'{line_name}_{kw}' not in table_fluxes.colnames:
                        table_fluxes.add_column(np.nan, name=f'{line_name}_{kw}')
                t = (line_name, wl_range, mask_wl, cont_range)
                line_quicksum_params.append(t)
            else:
                # spectra fitting
                if isinstance(line_name, str):
                    line_name = [line_name]
                max_comps = line.get('max_comps')
                if max_comps is None:
                    n_max_comps = 1
                else:
                    n_max_comps = max(max_comps)

                for cur_line_name in line_name:
                    for kw in ['flux', 'fluxerr', 'vel', 'velerr', 'disp', 'disperr', 'cont', 'conterr']:
                        if f'{cur_line_name}_{kw}' not in table_fluxes.colnames:
                            table_fluxes.add_column(np.nan, name=f'{cur_line_name}_{kw}')
                        if n_max_comps > 1 and f'{cur_line_name}_c2_{kw}' not in table_fluxes.colnames:
                            table_fluxes.add_column(np.nan, name=f'{cur_line_name}_c2_{kw}')
                        if n_max_comps > 2 and f'{cur_line_name}_c3_{kw}' not in table_fluxes.colnames:
                            table_fluxes.add_column(np.nan, name=f'{cur_line_name}_c3_{kw}')

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
                if save_plot_test is not None:
                    save_plot_ids = np.random.choice(range(len(table_fluxes)), 24)
                else:
                    save_plot_ids = None

                tie_vel = line.get('tie_vel')
                if tie_vel is None:
                    tie_vel = [True] * len(line_fit)
                tie_disp = line.get('tie_disp')
                if tie_disp is None:
                    tie_disp = [True] * len(line_fit)
                max_comps = line.get('max_comps')
                if max_comps is None:
                    max_comps = [1] * len(line_fit)
                t = (line_name, wl_range, mask_wl, line_fit,
                     include_comp, fix_ratios, tie_vel, tie_disp, max_comps, save_plot_test, save_plot_ids)

                line_fit_params.append(t)

        nprocs = np.min([np.max([config.get('nprocs'), 1]), len(table_fluxes)])
        if len(line_quicksum_params) > 0:
            params = zip(table_fluxes['sourceid'], table_fluxes['fluxcorr_b'], table_fluxes['fluxcorr_r'],
                         table_fluxes['fluxcorr_z'],
                         table_fluxes['vhel_corr'], table_fluxes['bgr'], np.arange(len(table_fluxes)))
            with mp.Pool(processes=nprocs) as pool:
                for status, res, spec_id in tqdm(
                        pool.imap(
                            partial(quickflux_all_lines,
                                    line_params=line_quicksum_params,
                                    include_sky=config['imaging'].get('include_sky'),
                                    partial_sky=config['imaging'].get('partial_sky'),
                                    path_to_fits=cur_wdir, velocity=vel,
                                    ), params),
                        ascii=True, desc="Calculate moment0 in all RSS",
                        total=len(table_fluxes), ):
                    statuses.append(status)
                    if not status:
                        continue

                    for kw in res.keys():
                        table_fluxes[kw][spec_id] = res[kw]

                pool.close()
                pool.join()
                gc.collect()
                if not np.all(statuses):
                    status_out = False
                else:
                    status_out = status_out & True

        if len(line_fit_params) > 0:
            all_plot_data = []
            params = zip(table_fluxes['sourceid'], table_fluxes['fluxcorr_b'], table_fluxes['fluxcorr_r'],
                         table_fluxes['fluxcorr_z'],
                         table_fluxes['vhel_corr'], table_fluxes['bgr'], np.arange(len(table_fluxes)))
            with mp.Pool(processes=nprocs) as pool:
                for status, fit_res, plot_data, spec_id in tqdm(
                        pool.imap(
                            partial(fit_all_from_current_spec,
                                    line_fit_params=line_fit_params, single_rss=False,
                                    mean_bounds=mean_bounds, velocity=vel,
                                    include_sky=config['imaging'].get('include_sky'),
                                    partial_sky=config['imaging'].get('partial_sky'),
                                    path_to_fits=cur_wdir
                                    ), params),
                        ascii=True, desc="Fit lines in all RSS",
                        total=len(table_fluxes), ):
                    statuses.append(status)
                    if not status:
                        continue

                    all_plot_data.append(plot_data)
                    for kw in fit_res.keys():
                        table_fluxes[kw][spec_id] = fit_res[kw]

                pool.close()
                pool.join()
                gc.collect()
                if not np.all(statuses):
                    status_out = False
                else:
                    status_out = status_out & True
            if save_plot_test is not None and (len(all_plot_data) > 0):
                fig = plt.figure(figsize=(20, 30))
                gs = GridSpec(6, 4, fig, 0.1, 0.1, 0.99, 0.95, wspace=0.1, hspace=0.1,
                              width_ratios=[1] * 4, height_ratios=[1] * 6)
                cur_ax_id = 0

                for cur_id in range(len(all_plot_data)):
                    # TODO This is very rough fix
                    if all_plot_data[cur_id] is None or (all_plot_data[cur_id][0] is None):
                        continue
                    ax = fig.add_subplot(gs[cur_ax_id])

                    ax.plot(all_plot_data[cur_id][0][0], all_plot_data[cur_id][0][1], 'k-', label='Obs')
                    ax.plot(all_plot_data[cur_id][0][0], all_plot_data[cur_id][0][2],
                            'r--', label=f'Fit')
                    ax.legend()
                    ax.set_title(f"Row #{cur_id}", fontsize=16)
                    cur_ax_id += 1
                fig.savefig(save_plot_test, bbox_inches='tight')
        table_fluxes.write(f_tab_summary, overwrite=True, format='ascii.fixed_width_two_line')
        statuses.append(status_out)
    if not np.all(statuses):
        status_out = False
    else:
        status_out = status_out & True
    return status_out


def quickflux_all_lines(params, path_to_fits=None, include_sky=False, partial_sky=False, line_params=None, velocity=0):
    source_ids, flux_cors_b, flux_cors_r, flux_cors_z, vhel_corrs, bgr, spec_id = params
    source_ids = source_ids.split(', ')
    flux_cors_b = flux_cors_b.split(', ')
    flux_cors_r = flux_cors_r.split(', ')
    flux_cors_z = flux_cors_z.split(', ')
    vhel_corrs = vhel_corrs.split(', ')
    bgr = bgr.split(', ')
    wave = None
    for ind, source_id in enumerate(source_ids):
        pointing, expnum, fib_id = source_id.split('_')
        expnum = int(expnum)
        fib_id = int(fib_id) - 1
        if include_sky:
            rssfile = os.path.join(path_to_fits, pointing, f'lvmCFrame-{expnum:0>8}.fits')
        else:
            rssfile = os.path.join(path_to_fits, pointing, f'lvmSFrame-{expnum:0>8}.fits')

        with fits.open(rssfile) as hdu:
            if wave is None:
                wave = ((np.arange(hdu['FLUX'].header['NAXIS1']) -
                            hdu['FLUX'].header['CRPIX1'] + 1) * hdu['FLUX'].header['CDELT1'] +
                           hdu['FLUX'].header['CRVAL1'])
                dw = hdu['FLUX'].header['CDELT1']
                flux = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                ivar = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)

            rec_b = wave < 5700
            rec_r = (wave >= 5700) & (wave < 7000)
            rec_z = wave >= 7000
            if partial_sky:
                flux[ind, :] = ((hdu['FLUX'].data[fib_id, :] + hdu['SKY'].data[fib_id, :]) -
                                mask_sky_at_bright_lines(hdu['SKY'].data[fib_id, :], wave=wave,
                                                         vel=velocity, mask=hdu['MASK'].data[fib_id, :]))
            else:
                flux[ind, rec_b] =(hdu['FLUX'].data[fib_id, rec_b] - float(bgr[ind])) * float(flux_cors_b[ind])
                flux[ind, rec_r] = (hdu['FLUX'].data[fib_id, rec_r] - float(bgr[ind])) * float(flux_cors_r[ind])
                flux[ind, rec_z] = (hdu['FLUX'].data[fib_id, rec_z] - float(bgr[ind])) * float(flux_cors_z[ind])
            ivar[ind, rec_b] = hdu['IVAR'].data[fib_id, rec_b] / float(flux_cors_b[ind]) ** 2
            ivar[ind, rec_r] = hdu['IVAR'].data[fib_id, rec_r] / float(flux_cors_r[ind]) ** 2
            ivar[ind, rec_z] = hdu['IVAR'].data[fib_id, rec_z] / float(flux_cors_z[ind]) ** 2
            flux[ind, hdu['MASK'].data[fib_id, :] == 1] = np.nan

        if ind > 1:
            delta_v = float(vhel_corrs[ind]) - float(vhel_corrs[0])
            if delta_v > 3.:
                flux[ind, :] = np.interp(wave, wave*(1-delta_v/3e5), flux[ind, :])

    if len(flux.shape) == 2 and flux.shape[0] > 1:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
            )
            flux = np.nanmean(sigma_clip(flux, sigma=1.3, axis=0, masked=False), axis=0)
            ivar = 1 / (np.nansum(1 / ivar, axis=0) / np.sum(np.isfinite(ivar), axis=0)**2)
    elif len(flux.shape) == 2:
        flux = flux[0, :]
        ivar = ivar[0, :]

    res_out = {}

    for cur_line_params in line_params:
        (line_name, wl_range, mask_wl, cont_range) = cur_line_params
        selwave_extended = np.flatnonzero((wave >= (min(wl_range) - 70)) * (wave <= (max(wl_range) + 70)))
        cur_wave = wave[selwave_extended]
        cur_flux = flux[selwave_extended]
        if mask_wl is not None:
            mask_wave = (cur_wave >= mask_wl[0]) & (cur_wave <= mask_wl[1])
            cur_flux[mask_wave] = np.nan
        cur_errors = np.sqrt(1/ivar[selwave_extended])
        sel_wave = np.flatnonzero((cur_wave >= wl_range[0]) & (cur_wave <= (wl_range[1])))

        cwave = np.mean(cur_wave[sel_wave])
        if len(wl_range) == 4:
            sel_wave1 = np.flatnonzero((cur_wave >= wl_range[2]) * (cur_wave <= wl_range[3]))
            cwave1 = np.mean(cur_wave[sel_wave1])
        else:
            cwave1 = np.nan
            sel_wave1 = []
        nallpix = len(sel_wave) + len(sel_wave1)
        ngoodpix = np.sum(np.isfinite(cur_flux[sel_wave])) + np.sum(np.isfinite(cur_flux[sel_wave1]))
        if ngoodpix/nallpix < 0.5:
            res_out[f'{line_name}_flux'] = np.nan
            res_out[f'{line_name}_fluxerr'] = np.nan
            continue
        cur_flux_orig = cur_flux.copy()
        cur_flux[~np.isfinite(cur_flux)] = np.nanmedian(cur_flux)

        flux_rss = np.nansum(cur_flux[sel_wave]) * dw
        error_rss = np.nansum(cur_errors[sel_wave] ** 2) * dw ** 2

        if len(sel_wave1) > 0:
            flux_rss += (np.nansum(cur_flux[sel_wave1]) * dw)
            error_rss += np.nansum(cur_errors[sel_wave1] ** 2) * dw ** 2

        # Optional continuum subtraction
        if cont_range is not None and (len(cont_range) >= 2):
            cselwave = (cur_wave >= cont_range[0]) * (cur_wave <= cont_range[1])

            if len(cont_range) == 2:
                cflux = np.nanmedian(cur_flux[cselwave])
                flux_rss -= (cflux * nallpix * dw)
            else:
                cselwave1 = (cur_wave >= cont_range[2]) * (cur_wave <= cont_range[3])
                mask_fit = np.isfinite(cur_flux_orig) & (cselwave | cselwave1)
                if np.sum(mask_fit) >= 5:
                    res = np.polyfit(cur_wave[mask_fit], cur_flux_orig[mask_fit], 1)
                    p = np.poly1d(res)
                    flux_rss -= (p(cwave) * len(sel_wave) * dw)
                    if np.isfinite(cwave1):
                        flux_rss -= (p(cwave1) * len(sel_wave1) * dw)

        res_out[f'{line_name}_flux'] = flux_rss/np.pi/(fiber_d/2)**2
        res_out[f'{line_name}_fluxerr'] = np.sqrt(error_rss)/np.pi/(fiber_d/2)**2
    return True, res_out, spec_id

def derive_radec_ifu(mjd, expnum, first_exp=None, objname=None, pointing_name=None, w_dir=None):
    # Input filenames
    LVMDATA_DIR = drp_results_dir  # os.path.join(SAS_BASE_DIR, 'sdsswork','lvm','lco')
    if w_dir is None:
        w_dir = f"/data/LVM/Reduced/{objname}"
    if objname is None or pointing_name is None:
        rssfile = f"{LVMDATA_DIR}/{mjd}/lvmSFrame-{expnum:0>8}.fits"
    else:
        rssfile = os.path.join(w_dir, pointing_name, f'lvmSFrame-{expnum:0>8}.fits')
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
                rssfile_ref = f"{LVMDATA_DIR}/{mjd}/lvmSFrame-{first_exp:0>8}.fits"
            else:
                rssfile_ref = os.path.join(w_dir, pointing_name, f'lvmSFrame-{first_exp:0>8}.fits')
                # f"/data/LVM/Reduced/{objname}/{pointing_name}/lvmSFrame-{first_exp:0>8}.fits"
            hdr_ref = fits.getheader(rssfile_ref)
            pa_hdr_ref = hdr_ref.get('POSCIPA')
            if not pa_hdr_ref:
                pa_hdr_ref = 0
            radec_corr = [hdr['POSCIRA'] - hdr_ref['POSCIRA'], hdr['POSCIDE'] - hdr_ref['POSCIDE'], pa_hdr - pa_hdr_ref]

    agcam_hdr = fits.getheader(agcscifile, ext=1)
    w = WCS(agcam_hdr)
    CDmatrix = w.pixel_scale_matrix
    posangrad = -1 * np.arctan(CDmatrix[1, 0] / CDmatrix[0, 0])
    PAobs = posangrad * 180 / np.pi
    cen = w.pixel_to_world(2500, 1000)
    # cen = w.pixel_to_world(2500,1000)
    # print(cen.ra.deg+radec_corr[0], cen.dec.deg+radec_corr[1], (agcam_hdr['PAMEAS'] + 180. + radec_corr[2]) % 360.)
    return cen.ra.deg+radec_corr[0], cen.dec.deg+radec_corr[1], (PAobs + radec_corr[2]) % 360.#(agcam_hdr['PAMEAS'] + 180. + radec_corr[2]) % 360. #agcam_hdr['PAMEAS'] - 180.


def create_line_image_from_table(file_fluxes=None, lines=None, pxscale_out=3., r_lim=50, sigma=2.,
                                output_dir=None, do_median_masking=False, filter_sn=None,
                                outfile_prefix=None, ra_lims=None, dec_lims=None, binmap=None):
    lvm_fiber_diameter = 35.3
    if not os.path.isfile(file_fluxes):
        log.error(f"File {file_fluxes} not found")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        uid = os.stat(output_dir).st_uid
        if server_group_id is not None:
            os.chown(output_dir, uid=uid, gid=server_group_id)
        os.chmod(output_dir, 0o775)

    table_fluxes = Table.read(file_fluxes, format='ascii.fixed_width_two_line')

    if binmap is not None:
        binmap_head = fits.getheader(binmap)
        binmap = fits.getdata(binmap)
        shape_out = binmap.shape
        wcs_out = WCS(binmap_head)
        grid_scl = wcs_out.proj_plane_pixel_scales()[0].value * 3600.
    else:
        ras = table_fluxes['fib_ra']
        decs = table_fluxes['fib_dec']

        rec_tiles = np.ones_like(ras, dtype=bool)
        if ra_lims is not None and type(ra_lims) in [list, tuple]:
            if ra_lims[0] is not None and np.isfinite(ra_lims[0]):
                rec_tiles = rec_tiles & (ras >= ra_lims[0])
            if ra_lims[1] is not None and np.isfinite(ra_lims[1]):
                rec_tiles = rec_tiles & (ras <= ra_lims[1])
        if dec_lims is not None and type(dec_lims) in [list, tuple]:
            if dec_lims[0] is not None and np.isfinite(dec_lims[0]):
                rec_tiles = rec_tiles & (decs >= dec_lims[0])
            if dec_lims[1] is not None and np.isfinite(dec_lims[1]):
                rec_tiles = rec_tiles & (decs <= dec_lims[1])
        rec_tiles = np.flatnonzero(rec_tiles)
        if len(rec_tiles) == 0:
            log.warning("Nothing to show")
            return False
        if len(rec_tiles) < len(table_fluxes):
            ras = ras[rec_tiles]
            decs = decs[rec_tiles]
            table_fluxes = table_fluxes[rec_tiles]

        dec_0 = np.min(decs)-37./2/3600
        dec_1 = np.max(decs) + 37. / 2 / 3600
        ra_0 = np.max(ras)+37./2/3600/np.cos(dec_1/180*np.pi)
        ra_1 = np.min(ras) - 37. / 2 / 3600 / np.cos(dec_0 / 180 * np.pi)

        ra_cen = (ra_0 + ra_1)/2.
        dec_cen = (dec_0 + dec_1) / 2.
        nx = np.ceil((ra_0 - ra_1)*max([np.cos(dec_0/180.*np.pi),np.cos(dec_1/180.*np.pi)])/pxscale_out*3600./2.).astype(int)*2+1
        ny = np.ceil((dec_1 - dec_0) / pxscale_out * 3600./2.).astype(int)*2+1
        ra_0 = np.round(ra_cen + (nx-1)/2 * pxscale_out / 3600. / max([np.cos(dec_0/180.*np.pi),np.cos(dec_1/180.*np.pi)]),6)
        dec_0 = np.round(dec_cen - (ny - 1) / 2 * pxscale_out/ 3600., 6)
        ra_cen = np.round(ra_cen, 6)
        dec_cen = np.round(dec_cen, 6)
        # Create a new WCS object.  The number of axes must be set
        # from the start
        wcs_out = WCS(naxis=2)
        wcs_out.wcs.crpix = [(nx-1)/2+1, (ny-1)/2+1]
        wcs_out.wcs.cdelt = np.array([-np.round(pxscale_out/3600.,6), np.round(pxscale_out/3600.,6)])
        wcs_out.wcs.crval = [ra_cen, dec_cen]
        wcs_out.wcs.cunit = ['deg', 'deg']
        wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        shape_out = (ny, nx)
        grid_scl = wcs_out.proj_plane_pixel_scales()[0].value * 3600.

        log.info(f"Grid scale: {np.round(grid_scl,1)}, output shape: {nx}x{ny}, "
                 f"RA range: {np.round(ra_1,4)} - {np.round(ra_0,4)}, "
                 f"DEC range: {np.round(dec_0,4)} - {np.round(dec_1,4)} ")


    # img_arr = np.zeros((ny, nx), dtype=float)
    # img_arr[:, :] = np.nan
    values = None
    values_errs = None
    masks = None
    masks_errs = None
    names_out = []
    names_out_errs = []

    for cl_id, cur_line in enumerate(lines):
        if isinstance(cur_line, str):  # this is valid in case of DAP
            lns = [cur_line]
        else:
            lns = cur_line.get('line')
            if isinstance(lns, str):
                lns = [lns]
            lns_orig = np.array(lns)
            lns.extend([l+'_c2' for l in lns_orig])
            lns.extend([l + '_c3' for l in lns_orig])
        for ln in lns:
            if f'{ln}_flux' in table_fluxes.colnames:
                cur_masks = np.isfinite(table_fluxes[f'{ln}_flux']) & (table_fluxes[f'{ln}_flux'] != 0)
                if filter_sn is not None and filter_sn[cl_id] is not None and np.isfinite(filter_sn[cl_id]) and f'{ln}_fluxerr' in table_fluxes.colnames:
                    cur_masks = cur_masks & (table_fluxes[f'{ln}_flux'] / table_fluxes[f'{ln}_fluxerr'] >= filter_sn[cl_id])
            if values is None:
                if f'{ln}_flux' in table_fluxes.colnames:
                    values = table_fluxes[f'{ln}_flux'] #/ (np.pi * lvm_fiber_diameter ** 2 / 4)
                    masks = np.copy(cur_masks)
                    if f'{ln}_fluxerr' in table_fluxes.colnames:
                        values_errs =table_fluxes[f'{ln}_fluxerr']
                        masks_errs = np.copy(cur_masks)
                else:
                    continue
            else:
                if f'{ln}_flux' in table_fluxes.colnames:
                    values = np.vstack([values.T, table_fluxes[f'{ln}_flux'].T]).T #/ (np.pi * lvm_fiber_diameter ** 2 / 4)
                    masks = np.vstack([masks.T, cur_masks.T]).T
                    if f'{ln}_fluxerr' in table_fluxes.colnames:
                        values_errs = np.vstack([values_errs.T, table_fluxes[f'{ln}_fluxerr'].T]).T
                        masks_errs = np.vstack([masks_errs.T, cur_masks.T]).T
                else:
                    continue
            names_out.append(f'{ln}_flux')
            if f'{ln}_fluxerr' in table_fluxes.colnames:
                names_out_errs.append(f'{ln}_fluxerr')
            for suff in ['vel', 'disp', 'cont']:
                if f'{ln}_{suff}' in table_fluxes.colnames:
                    values = np.vstack([values.T, table_fluxes[f'{ln}_{suff}'].T]).T
                    masks = np.vstack([masks.T, cur_masks.T]).T
                    values_errs = np.vstack([values_errs.T, table_fluxes[f'{ln}_{suff}err'].T]).T
                    masks_errs = np.vstack([masks_errs.T, cur_masks.T]).T
                    names_out.append(f'{ln}_{suff}')
                    names_out_errs.append(f'{ln}_{suff}err')

    if values is None:
        log.error('Nothing to show.')
        return False

    if binmap is None:
        if len(values.shape) == 1:
            values = values.reshape((-1, 1))
        img_arr = shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs, show_values=values,
                                   r_lim=r_lim, sigma=sigma, masks=masks)
        if values_errs is not None:
            err_arr = shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs, show_values=values_errs,
                                       r_lim=r_lim, sigma=sigma, masks=masks_errs, is_error=True)
    else:
        img_arr = np.empty(shape=(shape_out[0], shape_out[1], values.shape[1]), dtype=np.float32)
        err_arr = np.empty(shape=(shape_out[0], shape_out[1], values.shape[1]), dtype=np.float32)
        img_arr[:] = np.nan
        err_arr[:] = np.nan
        xxbin, yybin = np.meshgrid(np.arange(binmap.shape[1]), np.arange(binmap.shape[0]))
        for bin_ind, bin in enumerate(table_fluxes['binnum']):
            rec = np.flatnonzero(binmap.ravel() == bin)
            img_arr[yybin.ravel()[rec], xxbin.ravel()[rec], :] = (values[bin_ind, :])[None, None, :]
            err_arr[yybin.ravel()[rec], xxbin.ravel()[rec], :] = (values_errs[bin_ind, :])[None, None, :]

    header = wcs_out.to_header()
    for ind in range(img_arr.shape[2]):
        outfile_suffix = names_out[ind]
        if do_median_masking:
            img_arr[:, :, ind] = median_filter(img_arr[:, :, ind], (11, 11))
        fits.writeto(os.path.join(output_dir, f"{outfile_prefix}_{outfile_suffix}.fits"),
                     data=img_arr[:, :, ind], header=header, overwrite=True)
    if values_errs is not None:
        for ind in range(err_arr.shape[2]):
            outfile_suffix = names_out_errs[ind]
            fits.writeto(os.path.join(output_dir, f"{outfile_prefix}_{outfile_suffix}.fits"),
                         data=err_arr[:, :, ind], header=header, overwrite=True)
    return True


def fit_all_from_current_spec(params, header=None, path_to_fits=None, include_sky=False, partial_sky=False,
                              line_fit_params=None, mean_bounds=None, single_rss=True, velocity=0):
    if not single_rss:
        source_ids, flux_cors_b, flux_cors_r, flux_cors_z, vhel_corrs, bgr, spec_id = params
        source_ids = source_ids.split(', ')
        flux_cors_b = flux_cors_b.split(', ')
        flux_cors_r = flux_cors_r.split(', ')
        flux_cors_z = flux_cors_z.split(', ')
        vhel_corrs = vhel_corrs.split(', ')
        bgr = bgr.split(', ')
        wl_grid = None
        for ind, source_id in enumerate(source_ids):
            pointing, expnum, fib_id = source_id.split('_')
            expnum = int(expnum)
            fib_id = int(fib_id) - 1
            if include_sky:
                rssfile = os.path.join(path_to_fits, pointing, f'lvmCFrame-{expnum:0>8}.fits')
            else:
                rssfile = os.path.join(path_to_fits, pointing, f'lvmSFrame-{expnum:0>8}.fits')

            with fits.open(rssfile) as hdu:
                if wl_grid is None:
                    wl_grid = ((np.arange(hdu['FLUX'].header['NAXIS1']) -
                                hdu['FLUX'].header['CRPIX1'] + 1) * hdu['FLUX'].header['CDELT1'] +
                               hdu['FLUX'].header['CRVAL1'])
                    lsf = hdu['LSF'].data[fib_id]

                    flux = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                    ivar = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                    sky = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                    sky_ivar = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                rec_b = wl_grid < 5700
                rec_r = (wl_grid >= 5700) & (wl_grid < 7000)
                rec_z = wl_grid >= 7000
                if partial_sky:
                    flux[ind, :] = ((hdu['FLUX'].data[fib_id, :] + hdu['SKY'].data[fib_id, :]) -
                     mask_sky_at_bright_lines(hdu['SKY'].data[fib_id, :], wave=wl_grid,
                                              vel=velocity, mask=hdu['MASK'].data[fib_id, :]))
                else:
                    flux[ind, rec_b] = (hdu['FLUX'].data[fib_id, rec_b] - float(bgr[ind])) * float(flux_cors_b[ind])
                    flux[ind, rec_r] = (hdu['FLUX'].data[fib_id, rec_r] - float(bgr[ind])) * float(flux_cors_r[ind])
                    flux[ind, rec_z] = (hdu['FLUX'].data[fib_id, rec_z] - float(bgr[ind])) * float(flux_cors_z[ind])
                ivar[ind, rec_b] = hdu['IVAR'].data[fib_id, rec_b] / float(flux_cors_b[ind]) ** 2
                sky[ind, rec_b] = (hdu['SKY'].data[fib_id, rec_b] + float(bgr[ind])) * float(flux_cors_b[ind])
                sky_ivar[ind, rec_b] = hdu['SKY_IVAR'].data[fib_id, rec_b] / float(flux_cors_b[ind]) ** 2
                ivar[ind, rec_r] = hdu['IVAR'].data[fib_id, rec_r] / float(flux_cors_r[ind]) ** 2
                sky[ind, rec_r] = (hdu['SKY'].data[fib_id, rec_r] + float(bgr[ind])) * float(flux_cors_r[ind])
                sky_ivar[ind, rec_r] = hdu['SKY_IVAR'].data[fib_id, rec_r] / float(flux_cors_r[ind]) ** 2
                ivar[ind, rec_z] = hdu['IVAR'].data[fib_id, rec_z] / float(flux_cors_z[ind]) ** 2
                sky[ind, rec_z] = (hdu['SKY'].data[fib_id, rec_z] + float(bgr[ind])) * float(flux_cors_z[ind])
                sky_ivar[ind, rec_z] = hdu['SKY_IVAR'].data[fib_id, rec_z] / float(flux_cors_z[ind]) ** 2
                flux[ind, hdu['MASK'].data[fib_id, :] == 1] = np.nan
                sky[ind, hdu['MASK'].data[fib_id, :] == 1] = np.nan

            if ind > 1:
                delta_v = float(vhel_corrs[ind]) - float(vhel_corrs[0])
                if delta_v > 2.:
                    flux[ind, :] = np.interp(wl_grid, wl_grid*(1-delta_v/3e5), flux[ind, :])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
            )
            flux = np.nanmean(sigma_clip(flux, sigma=1.3, axis=0, masked=False), axis=0)
            ivar = 1 / (np.nansum(1 / ivar, axis=0) / np.sum(np.isfinite(ivar), axis=0)**2)
            sky = np.nansum(sky, axis=0) / np.sum(np.isfinite(sky), axis=0)
            sky_ivar = 1 / (np.nansum(1 / sky_ivar, axis=0) / np.sum(np.isfinite(sky_ivar), axis=0)**2)
        vhel = float(vhel_corrs[0])
        wave = wl_grid
    else:
        flux, ivar, sky, sky_ivar, lsf, vhel_corr, spec_id = params
        if isinstance(vhel_corr, str):
            vhel = np.mean([float(vh) for vh in vhel_corr.split(',')])
        else:
            vhel = float(np.nanmean(vhel_corr))
        wave = ((np.arange(header['NAXIS1']) - header['CRPIX1'] + 1) * header['CDELT1'] + header['CRVAL1'])
    error = 1./np.sqrt(ivar)
    sky_error = 1./np.sqrt(sky_ivar)
    error[~np.isfinite(error)] = 1e10
    sky_error[~np.isfinite(sky_error)] = 1e10

    res_output = {}
    all_plot_data = []
    status = True

    wid = 10
    vel_sky_correct = 0
    mean_bounds_1 =np.array(mean_bounds)
    for sky_line in [5577.338, 6300.304]:
        sel_wave = np.flatnonzero((wave >= (sky_line - wid)) & (wave <= (sky_line + wid)))
        (fluxes, vel, disp, cont, fluxes_err, v_err,
         sigma_err, cont_err, _, _) = fit_cur_spec_lmfit((sky[sel_wave], sky_error[sel_wave], lsf[sel_wave]),
                                             wave=wave[sel_wave],
                                             mean_bounds=(-1.5, 1.5),
                                             lines=[float(sky_line)],
                                             velocity=0, return_plot_data=False, subtract_lsf=False)

        res_output[f'SKY{int(sky_line)}_flux'] = fluxes[0]
        res_output[f'SKY{int(sky_line)}_fluxerr'] = fluxes_err[0]
        res_output[f'SKY{int(sky_line)}_vel'] = vel[0]
        res_output[f'SKY{int(sky_line)}_velerr'] = v_err[0]
        res_output[f'SKY{int(sky_line)}_disp'] = disp[0]
        res_output[f'SKY{int(sky_line)}_disperr'] = sigma_err[0]
        res_output[f'SKY{int(sky_line)}_cont'] = cont[0]
        res_output[f'SKY{int(sky_line)}_conterr'] = cont_err[0]
        # if int(np.round(sky_line)) == 6300:
        #     vel_sky_correct = vel

    for cur_line_params in line_fit_params:
        (line_name, wl_range, mask_wl, line_fit,
         include_comp, fix_ratios, tie_vel, tie_disp, max_comps, _, save_plot_ids) = cur_line_params
        sel_wave = np.flatnonzero((wave >= wl_range[0]) & (wave <= wl_range[1]))

        if save_plot_ids is not None and spec_id in save_plot_ids:
            (fluxes, vel, disp, cont, fluxes_err, v_err,
             sigma_err, cont_err, second_comp, third_comp, plot_data) = fit_cur_spec_lmfit((flux[sel_wave], error[sel_wave], lsf[sel_wave]),
                                                            wave=wave[sel_wave], mean_bounds=mean_bounds_1,
                                                            lines=line_fit, fix_ratios=fix_ratios, tie_vel=tie_vel,
                                                            tie_disp=tie_disp, max_n_comp=max_comps,
                                                            velocity=velocity+vel_sky_correct-vhel, return_plot_data=True)
            all_plot_data.append(plot_data)
        else:
            (fluxes, vel, disp, cont, fluxes_err, v_err,
             sigma_err, cont_err, second_comp, third_comp) = fit_cur_spec_lmfit((flux[sel_wave], error[sel_wave], lsf[sel_wave]),
                                                            wave=wave[sel_wave], mean_bounds=mean_bounds_1,
                                                            lines=line_fit, fix_ratios=fix_ratios, tie_vel=tie_vel,
                                                            tie_disp=tie_disp, max_n_comp=max_comps,
                                                            velocity=velocity+vel_sky_correct-vhel, return_plot_data=False,
                                                 )
            all_plot_data.append(None)
        if np.all([~np.isfinite(v) for v in vel]):
            continue

        for ln_id, ln in enumerate(line_name):
            rec_comp = np.flatnonzero(np.array(include_comp) == ln_id)
            if len(rec_comp) == 0:
                continue

            res_output[f'{ln}_flux'] = np.nansum(np.array(fluxes)[rec_comp])
            res_output[f'{ln}_fluxerr'] = np.sqrt(np.nansum(np.array(fluxes_err)[rec_comp]**2))
            res_output[f'{ln}_vel'] = np.round(np.array(vel)[rec_comp[0]] + vhel - vel_sky_correct,1)
            res_output[f'{ln}_velerr'] = np.round(np.array(v_err)[rec_comp[0]],1)
            res_output[f'{ln}_disp'] = np.round(np.array(disp)[rec_comp[0]],1)
            res_output[f'{ln}_disperr'] = np.round(np.array(sigma_err)[rec_comp[0]],1)
            res_output[f'{ln}_cont'] = np.array(cont)[rec_comp[0]]
            res_output[f'{ln}_conterr'] = np.array(cont_err)[rec_comp[0]]

            if second_comp is not None and np.isfinite(second_comp['fluxes'][rec_comp[0]]):
                res_output[f'{ln}_c2_flux'] = np.nansum(np.array(second_comp['fluxes'])[rec_comp])
                res_output[f'{ln}_c2_fluxerr'] = np.sqrt(np.nansum(np.array(second_comp['fluxes_err'])[rec_comp] ** 2))
                res_output[f'{ln}_c2_vel'] = np.round(np.array(second_comp['vel'])[rec_comp[0]] + vhel - vel_sky_correct, 1)
                res_output[f'{ln}_c2_velerr'] = np.round(np.array(second_comp['vel_err'])[rec_comp[0]], 1)
                res_output[f'{ln}_c2_disp'] = np.round(np.array(second_comp['disp'])[rec_comp[0]], 1)
                res_output[f'{ln}_c2_disperr'] = np.round(np.array(second_comp['disp_err'])[rec_comp[0]], 1)

            if third_comp is not None and np.isfinite(third_comp['fluxes'][rec_comp[0]]):
                res_output[f'{ln}_c3_flux'] = np.nansum(np.array(third_comp['fluxes'])[rec_comp])
                res_output[f'{ln}_c3_fluxerr'] = np.sqrt(np.nansum(np.array(third_comp['fluxes_err'])[rec_comp] ** 2))
                res_output[f'{ln}_c3_vel'] = np.round(np.array(third_comp['vel'])[rec_comp[0]] + vhel - vel_sky_correct, 1)
                res_output[f'{ln}_c3_velerr'] = np.round(np.array(third_comp['vel_err'])[rec_comp[0]], 1)
                res_output[f'{ln}_c3_disp'] = np.round(np.array(third_comp['disp'])[rec_comp[0]], 1)
                res_output[f'{ln}_c3_disperr'] = np.round(np.array(third_comp['disp_err'])[rec_comp[0]], 1)

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
    if 'FLUX_ORIG' in rsshdu:
        if include_sky:
            flux_all = (rsshdu['FLUX_ORIG'].data + rsshdu['SKY'].data)[:, selwave_extended]
        elif partial_sky:
            flux_all = rsshdu['FLUX'].data[:, selwave_extended]
        else:
            flux_all = rsshdu["FLUX_ORIG"].data[:, selwave_extended]
    else:
        if include_sky:
            flux_all = (rsshdu['FLUX'].data + rsshdu['SKY'].data)[:, selwave_extended]
        else:
            flux_all = rsshdu['FLUX'].data[:, selwave_extended]
    errors_all = rsshdu["IVAR"].data[:, selwave_extended]
    if 'MASK' in rsshdu:
        mask_all = rsshdu["MASK"].data[:, selwave_extended]
    else:
        mask_all = np.zeros_like(flux_all)

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


def convert_extracted_spec_format(f_rss_in, f_rss_out, ds9_file=None):
    """
    Convert the extracted spectrum to the RSS format
    :param f_rss_in: Input file
    :param f_rss_out: Output file
    :param ds9_file: Regions in ds9 format
    :return: Status (True or False)
    """
    if not os.path.isfile(f_rss_in):
        log.error(f"File {f_rss_in} doesn't exist.")
        return False

    log.info('...Start converting extracted spectrum to RSS file')

    with fits.open(f_rss_in) as hdu:
        n_regs = len(hdu)-1
        nx = hdu[1].data.shape[1]

    if not os.path.isfile(ds9_file):
        log.warning(f"File {ds9_file} is not found.")
        ds9_regions = None
    else:
        ds9_regions = Regions.read(ds9_file, format='ds9')
        if len(ds9_regions) != n_regs:
            log.warning(f"Number of regions in the file {ds9_file} doesn't match the number of regions in the RSS file. "
                        f"Ignoring ds9.")
            ds9_regions = None

    """ Extract coordinates in RA, DEC of the centers of each region from ds9 region file """
    if ds9_regions is not None:
        ras = []
        decs = []
        for r_id, r in enumerate(ds9_regions):
            try:
                cur_ra = r.center.ra.degree
                cur_dec = r.center.dec.degree
            except AttributeError:
                v_ra = r.vertices.ra.degree
                v_deg = r.vertices.dec.degree
                polygon = Polygon(
                    [(v_ra[i], v_deg[i]) for i in range(len(v_ra))])
                cur_ra, cur_dec = polygon.centroid.x, polygon.centroid.y
            except:
                log.warning(f"Not supported type of region for reg={r_id}")
                cur_ra = 0
                cur_dec = 0
            ras.append(cur_ra)
            decs.append(cur_dec)

    tab_summary = Table(
        data=[np.array([f"{1:08d}_{reg_id:04d}" for reg_id in np.arange(n_regs)]),
              np.array(ras), np.array(decs), ['science'] * n_regs,
              [0] * n_regs], names=['fiberid', 'fib_ra', 'fib_dec', 'targettype', 'fibstatus'],
        dtype=(int, float, float, str, int))
    rss_out = fits.HDUList([fits.PrimaryHDU(data=None),
                            fits.ImageHDU(data=np.zeros(shape=(n_regs, nx), dtype=float), name='FLUX'),
                            fits.ImageHDU(data=np.zeros(shape=(n_regs, nx), dtype=float), name='IVAR')])
    rss_out.writeto(f_rss_out, overwrite=True)
    rss_out.close()
    for kw_block in [('MASK', 'WAVE', 'LSF'), ('SKY', 'SKY_IVAR')]:
        rss_out = fits.open(f_rss_out)
        for kw in kw_block:
            if kw != 'WAVE':
                shape = (n_regs, nx)
            else:
                shape = nx
            rss_out.append(fits.ImageHDU(data=np.zeros(shape=shape, dtype=float), name=kw))
        if kw == 'SKY_IVAR':
            rss_out.append(fits.BinTableHDU(data=tab_summary, name='SLITMAP'))
        rss_out.writeto(f_rss_out, overwrite=True)
        rss_out.close()

    rss_out = fits.open(f_rss_out)
    hdu_in = fits.open(f_rss_in)
    for ind in range(n_regs):
        rss_out['FLUX'].data[ind, :] = hdu_in[ind+1].data[0,:] * np.pi *fiber_d **2 / 4
        rss_out['IVAR'].data[ind, :] = 1/(hdu_in[ind+1].data[1,:] * np.pi *fiber_d **2 / 4) ** 2
        rss_out['SKY'].data[ind, :] = hdu_in[ind+1].data[2,:] * np.pi *fiber_d **2 / 4
        rss_out['SKY_IVAR'].data[ind, :] = 1/(hdu_in[ind+1].data[3,:] * np.pi *fiber_d **2 / 4) ** 2
        rss_out['LSF'].data[ind, :] = hdu_in[ind+1].data[4,:]
        rss_out['MASK'].data[ind, :] = hdu_in[ind + 1].data[4, :]
    for kw in ['FLUX', 'IVAR', 'SKY', 'SKY_IVAR', 'LSF', 'MASK', 'WAVE']:
        for kw_copy in ['CRVAL1', 'CRPIX1', 'CDELT1', 'CTYPE1']:
            rss_out[kw].header[kw_copy] = hdu_in[1].header[kw_copy]
    wave = ((np.arange(nx) - rss_out['FLUX'].header['CRPIX1'] + 1) *
            rss_out['FLUX'].header['CDELT1'] + rss_out['FLUX'].header['CRVAL1'])
    rss_out['WAVE'].data = wave
    rec = ~np.isfinite(rss_out['FLUX'].data)
    rss_out['MASK'].data[rec] = 1
    rss_out.writeto(f_rss_out, overwrite=True)
    rss_out.close()

def process_single_rss(config, output_dir=None, binned=False, dap=False, extracted=False):
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
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        status_out = True
        if binned:
            suffix_out = config['binning'].get('rss_output_suffix')
            if not suffix_out:
                suffix_out = '_vorbin_rss.fits'
            suffix_binmap = config['binning'].get('binmap_suffix')
            if not suffix_binmap:
                suffix_binmap = '_binmap.fits'
            bin_line = config['binning'].get('line')
            if not bin_line:
                bin_line = 'Ha'
            target_sn = config['binning'].get('target_sn')
            if not target_sn:
                target_sn = 30.
            else:
                target_sn = float(target_sn)
            f_rss = os.path.join(output_dir, cur_obj['name'], version, f"{cur_obj.get('name')}_{bin_line}_sn{target_sn}{suffix_out}")
            f_tab = os.path.join(output_dir, cur_obj['name'], version, f"{cur_obj.get('name')}_binfluxes_{bin_line}_sn{target_sn}.txt")
        elif extracted:
            if 'extraction' not in config:
                suffix_out = '_extracted.fits'
                ds9_file = None
            else:
                suffix_out = config['extraction'].get('file_output_suffix')
                ds9_suffix = config['extraction'].get('file_ds9_suffix')
                if not suffix_out:
                    suffix_out = '_extracted.fits'
                if not ds9_suffix:
                    ds9_suffix = '_ds9.reg'
                ds9_file = os.path.join(output_dir, cur_obj['name'], version, f"{cur_obj.get('name')}{ds9_suffix}")
            f_rss_in = os.path.join(output_dir, cur_obj['name'], version,
                                 f"{cur_obj.get('name')}{suffix_out}")
            f_rss = os.path.join(output_dir, cur_obj['name'], version,
                                 f"{cur_obj.get('name')}{suffix_out.replace('.fits', '_rss.fits')}")
            status = convert_extracted_spec_format(f_rss_in, f_rss, ds9_file)
            if not status:
                log.error("Something went wrong with converting the extracted spectrum to the RSS format. Exit")
                return False
            f_tab = os.path.join(output_dir, cur_obj['name'], version,
                                 f"{cur_obj.get('name')}_extracted.txt")
        else:
            f_rss = os.path.join(output_dir, cur_obj['name'], version, f"{cur_obj['name']}_all_RSS.fits")
            f_tab = os.path.join(output_dir, cur_obj['name'], version, f"{cur_obj['name']}_fluxes_singleRSS.txt")
        if not os.path.isfile(f_rss):
            log.error(f"File {f_rss} doesn't exist.")
            statuses.append(False)
            continue

        if dap:
            if binned:
                dap_config_file = os.path.join(output_dir, cur_obj['name'], version,
                                               f"{cur_obj.get('name')}_binfluxes_{bin_line}_sn{target_sn}_dap_config.yaml")
                dap_output_dir = os.path.join(output_dir, cur_obj['name'], version,
                                              f"dap_output_binfluxes_{bin_line}_sn{target_sn}")
            elif extracted:
                dap_config_file = os.path.join(output_dir, cur_obj['name'], version,
                                               f"{cur_obj.get('name')}_extracted_dap_config.yaml")
                dap_output_dir = os.path.join(output_dir, cur_obj['name'], version,
                                              f"dap_output_extracted")

            else:
                dap_config_file = os.path.join(output_dir, cur_obj['name'], version,
                                               f"{cur_obj.get('name')}_dap_config.yaml")
                dap_output_dir = os.path.join(output_dir, cur_obj['name'], version,
                                              f"dap_output")
            label = f_rss.split('/')[-1].replace('.fits', '')

            if os.path.isdir(dap_output_dir) and os.path.exists(os.path.join(dap_output_dir, f'm_{label}.output.fits')):
                log.warning("DAP results already exist. Skipping DAP running. "
                            "Remove the directory to rerun DAP, if necessary.")
                statuses.append(True)
                continue
            if not os.path.isdir(dap_output_dir):
                os.makedirs(dap_output_dir)
                uid = os.stat(dap_output_dir).st_uid
                if server_group_id is not None:
                    os.chown(dap_output_dir, uid=uid, gid=server_group_id)
                os.chmod(dap_output_dir, 0o775)
            # Update the SLITMAP extension of the RSS file with ra and dec columns for DAP
            rss = fits.open(f_rss)
            table_fibers = Table(rss['SLITMAP'].data)
            if ('ra' not in table_fibers.colnames) or ('dec' not in table_fibers.colnames):
                # add additional columns to the table equal to other columns ra and dec
                table_fibers.add_columns([Column(name='ra', data=table_fibers['fib_ra']),
                                          Column(name='dec', data=table_fibers['fib_dec'])])
                # save updated table to SLITMAP extension
                rss['SLITMAP'] = fits.BinTableHDU(table_fibers, name='SLITMAP')
                rss.writeto(f_rss, overwrite=True)
            rss.close()

            # Run DAP on a single RSS file
            # if neeeded, prepare DAP configuration file using the template from lvmdap/_legacy/lvm-dap_v110.yaml
            if not os.path.isfile(dap_config_file):
                # read yaml file
                with open(os.path.join(os.environ.get('LVM_DAP_CFG'),'lvm-dap_v110.yaml'), 'r') as stream:
                    try:
                        dap_config = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        log.error(exc)
                        statuses.append(False)
                        continue
                # change the output directory in dap_config
                dap_config['output_path'] = dap_output_dir
                # save the new configuration file
                with open(dap_config_file, 'w') as outfile:
                    yaml.dump(dap_config, outfile, default_flow_style=False)

            cdir = os.curdir
            os.chdir(os.environ.get('LVM_DAP'))
            try:
                os.system(f"lvm-dap-conf {f_rss} {label} {dap_config_file}")
                statuses.append(True)
            except Exception as e:
                log.error(f"Something wrong with running DAP: {e}")
                statuses.append(False)
            os.chdir(cdir)
            continue

        if cur_obj['name'] == 'Orion':
            mean_bounds = (-2, 2)
        else:
            mean_bounds = (-10, 10)
        vel = cur_obj.get('velocity')

        rss = fits.open(f_rss)
        table_fibers = Table(rss['SLITMAP'].data)

        if not config['imaging'].get('override_flux_table') and os.path.isfile(f_tab):
            table_fluxes = Table.read(f_tab, format='ascii.fixed_width_two_line')
        else:
            if binned:
                table_fluxes = table_fibers['fiberid', 'fib_ra', 'fib_dec', 'binnum'].copy()
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
        if 'FLUX_ORIG' in rss:
            if config['imaging'].get('include_sky'):
                flux_all = (rss['FLUX_ORIG'].data + rss['SKY'].data)[:, selwave]
            elif config['imaging'].get('partial_sky'):
                flux_all = rss['FLUX'].data[:, selwave]
            else:
                flux_all = rss["FLUX_ORIG"].data[:, selwave]
        else:
            if config['imaging'].get('include_sky'):
                flux_all = (rss['FLUX'].data + rss['SKY'].data)[:, selwave]
            else:
                flux_all = rss["FLUX"].data[:, selwave]
        if 'MASK' in rss:
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
                tie_vel = line.get('tie_vel')
                if tie_vel is None:
                    tie_vel = [True]*len(line_fit)
                tie_disp = line.get('tie_disp')
                if tie_disp is None:
                    tie_disp = [True]*len(line_fit)
                max_comps = line.get('max_comps')
                if max_comps is None:
                    max_comps = [1]*len(line_fit)
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
                     include_comp, fix_ratios, tie_vel, tie_disp, max_comps, save_plot_test, save_plot_ids)
                line_fit_params.append(t)

        if len(line_fit_params) > 0:
            nprocs = np.min([np.max([config.get('nprocs'), 1]), rss['FLUX'].header['NAXIS2']])
            spec_ids = np.arange(rss['FLUX'].header['NAXIS2']).astype(int)
            local_statuses = []
            all_plot_data = []

            if 'FLUX_ORIG' in rss:
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
            else:
                if config['imaging'].get('include_sky'):
                    flux = rss['FLUX'].data + rss['SKY'].data
                    flux[~np.isfinite(rss['FLUX'].data) | (
                                rss['FLUX'].data == 0)] = np.nan
                else:
                    flux = rss['FLUX'].data
                    flux[flux == 0] = np.nan
            if mean_bounds is None:
                mean_bounds = (-5, 5)
            sky = rss['SKY'].data
            sky_ivar = rss['SKY_IVAR'].data
            if 'MASK' in rss:
                sky[(rss['MASK'].data > 0) ] = np.nan #| (rss['SKY'].data == 0)
            ivar = rss['IVAR'].data
            lsf = rss['LSF'].data
            if 'MASK' in rss:
                flux[(rss['MASK'].data > 0)] = np.nan
            header = rss['FLUX'].header

            t = Table(rss['SLITMAP'].data)
            if 'vhel_corr' in t.colnames:
                vhel = t['vhel']
            else:
                vhel = np.zeros_like(flux)
            params = zip(flux, ivar, sky, sky_ivar, lsf, vhel, spec_ids)
            with mp.Pool(processes=nprocs) as pool:
                for status, fit_res, plot_data, spec_id in tqdm(
                                pool.imap(
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


def sn_func(index, signal=None, noise=None):
    sn = np.nansum(signal[index])/np.sqrt(np.nansum(noise[index]**2))

    # The following commented line illustrates, as an example, how one
    # would include the effect of spatial covariance using the empirical
    # Eq.(1) from http://adsabs.harvard.edu/abs/2015A%26A...576A.135G
    # Note however that the formula is not accurate for large bins.
    #
    # sn /= 1 + 1.07*np.log10(index.size)
    return sn


def vorbin_rss(config, w_dir=None):
    from voronoi_2d_binning import voronoi_2d_binning

    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False
    statuses = []
    if not config.get('binning'):
        log.error(f"Cannot proceed with the binning step without corresponding control block in the config file.")
        return False
    suffix_out = config['binning'].get('rss_output_suffix')
    if not suffix_out:
        suffix_out = '_vorbin_rss.fits'
    suffix_binmap = config['binning'].get('binmap_suffix')
    if not suffix_binmap:
        suffix_binmap = '_binmap.fits'
    bin_line = config['binning'].get('line')
    if not bin_line:
        bin_line = 'Ha'
    target_sn = config['binning'].get('target_sn')
    if not target_sn:
        target_sn = 30.
    else:
        target_sn = float(target_sn)
    if not config['binning'].get('maps_source'):
        map_source = 'maps'
    else:
        map_source = config['binning'].get('maps_source')
    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        log.info(f"Binning data for {cur_obj.get('name')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes.txt")
        if not os.path.isfile(f_tab_summary):
            log.error(f"Table with the results of the RSS analysis doesn't exist. "
                      f"Anylise_rss step must be run before the binning spectra. "
                      f"Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        log.info(f"Performing voronoi binning for object {cur_obj.get('name')}. "
                 f"Target SN={target_sn} in {bin_line} line.")
        f_binmap = os.path.join(cur_wdir, map_source,
                                f"{cur_obj.get('name')}_{config['imaging'].get('pxscale')}asec_{bin_line}_sn{target_sn}{suffix_binmap}")
        f_image = os.path.join(cur_wdir, map_source,
                               f"{cur_obj.get('name')}_{config['imaging'].get('pxscale')}asec_{bin_line}_flux.fits")
        f_err = os.path.join(cur_wdir, map_source,
                             f"{cur_obj.get('name')}_{config['imaging'].get('pxscale')}asec_{bin_line}_fluxerr.fits")
        if not os.path.isfile(f_image) or not os.path.isfile(f_err):
            log.error(
                f"Maps of flux and errors in {bin_line} must be created in 'maps' folder for {cur_obj.get('name')}"
                f" before the binning. If they are there already, check that pixscale is consistent "
                f"with what is indicated in 'imaging' block")
            statuses.append(False)
            continue
        header = fits.getheader(f_image)
        if not config['binning'].get('use_binmap'):
            signal = fits.getdata(f_image)
            noise = fits.getdata(f_err)
            x, y = np.meshgrid(np.arange(signal.shape[1]), np.arange(signal.shape[0]))
            rec = (np.isfinite(noise) & (noise > 0) & (signal/noise >= 1)).ravel()
            # noise[rec] = abs(np.nanmedian(signal)+np.nanstd(signal))

            binnum, _, _, x_bar, y_bar, sn, nPixels, _ = voronoi_2d_binning(x.ravel()[rec], y.ravel()[rec],
                                                                                        signal.ravel()[rec], noise.ravel()[rec],
                                                                                        target_sn, plot=0, quiet=1,
                                                                            pixelsize=1, sn_func=sn_func, cvt=False)
            bin_image = np.zeros_like(signal, dtype=int) - 1
            bin_image.ravel()[rec] = binnum
            # bin_image = binnum.reshape(signal.shape)

            wcs = WCS(header)
            radec_bin = wcs.pixel_to_world(x_bar, y_bar)
            tab_summary_bins = Table(data=[np.unique(binnum), radec_bin.ra.degree, radec_bin.dec.degree, ['science']*len(radec_bin),
                                           [0]*len(radec_bin)], names=['fiberid', 'fib_ra', 'fib_dec', 'targettype',
                                                       'fibstatus'], dtype=(int, float, float, str, int))
            hdu_out = fits.HDUList([fits.PrimaryHDU(data=bin_image, header=header), fits.BinTableHDU(tab_summary_bins)])
            hdu_out.writeto(f_binmap, overwrite=True)
        else:
            if not os.path.isfile(f_binmap):
                log.error(f"Binmap in {bin_line} is not found be created in '{map_source}' folder for {cur_obj.get('name')}."
                          f" If they are there already, check that pixscale is consistent "
                          f"with what is indicated in 'imaging' block. Otherwise, set 'use_binmap' to false")
                statuses.append(False)
                continue
            with fits.open(f_binmap) as hdu:
                bin_image = hdu[0].data
                tab_summary_bins = Table(hdu[1].data)


        tab = Table.read(f_tab_summary, format='ascii.fixed_width_two_line',
                         converters={'sourceid': str, 'fluxcorr_b': str, 'fluxcorr_r': str,
                                     'fluxcorr_z': str, 'vhel_corr': str})
        wcs = WCS(header)
        tab_x, tab_y = wcs.world_to_pixel(SkyCoord(ra=tab['fib_ra'], dec=tab['fib_dec'], unit=('degree','degree')))
        binnum_fibers = np.zeros(shape=len(tab), dtype=int)-1
        rec = np.flatnonzero((tab_x >= 0) & (tab_y >= 0) & (tab_x <= (bin_image.shape[1]-1)) & (tab_y <= (bin_image.shape[0]-1)))
        for r in rec:
            binnum_fibers[r] = bin_image[int(np.round(tab_y[r])), int(np.round(tab_x[r]))]
        if 'binnum' not in tab.colnames:
            tab.add_column(binnum_fibers, name='binnum')
        else:
            tab['binnum'] = binnum_fibers
        tab.write(f_tab_summary, format='ascii.fixed_width_two_line', overwrite=True)

        #==== Extraction of the spectrum (modified from extract_spectra_ds9)
        hdu_out = fits.HDUList([fits.PrimaryHDU()])
        uniq_bins = np.unique(tab['binnum'][tab['binnum']>=0])
        if config['binning'].get('correct_vel_line'):
            if config['binning'].get('correct_vel_line')+'_vel' in tab.colnames:
                correct_vel_line = config['binning'].get('correct_vel_line')+'_vel'
            else:
                correct_vel_line = None
        else:
            correct_vel_line = None

        for cur_bin_id, cur_bin in enumerate(uniq_bins):
            in_reg = np.flatnonzero(tab['binnum'] == cur_bin)

            if correct_vel_line is not None:
                med_vel = np.nanmedian(tab[in_reg][correct_vel_line])
                params = zip(tab[in_reg]['sourceid'], tab[in_reg]['fluxcorr_b'],
                             tab[in_reg]['fluxcorr_r'], tab[in_reg]['fluxcorr_z'],
                             tab[in_reg]['vhel_corr'],
                             tab[in_reg][correct_vel_line] - med_vel,
                             [cur_obj.get('velocity')] * len(in_reg),
                             [config['imaging'].get('include_sky')] * len(in_reg),
                             [config['imaging'].get('partial_sky')] * len(in_reg),
                             [cur_wdir] * len(in_reg)
                             )
            else:
                params = zip(tab[in_reg]['sourceid'], tab[in_reg]['fluxcorr_b'],
                             tab[in_reg]['fluxcorr_r'], tab[in_reg]['fluxcorr_z'],
                             tab[in_reg]['vhel_corr'], [0] * len(in_reg),
                             [cur_obj.get('velocity')] * len(in_reg),
                             [config['imaging'].get('include_sky')] * len(in_reg),
                             [config['imaging'].get('partial_sky')] * len(in_reg),
                             [cur_wdir] * len(in_reg))

            nprocs = np.min([np.max([config.get('nprocs'), 1]), len(in_reg)])
            with mp.Pool(processes=nprocs) as pool:
                res = list(tqdm(pool.imap(extract_spectrum_for_cur_fiber, params),
                                ascii=True, desc=f"Extract spectra from fibers in {cur_bin_id+1}/{len(uniq_bins)} bins",
                                total=len(in_reg)))

                pool.close()
                pool.join()
                gc.collect()
            res = np.array(res)

            cur_flux = np.atleast_2d(np.nanmean(res[:, 0, :], axis=0))# / np.pi / (fiber_d ** 2 / 4))
            cur_error = np.atleast_2d(np.sqrt(np.nanmean(res[:, 1, :] ** 2, axis=0)))# / np.pi / (fiber_d ** 2 / 4))
            cur_sky = np.atleast_2d(np.nanmean(res[:, 2, :], axis=0))# / np.pi / (fiber_d ** 2 / 4))
            cur_sky_error = np.atleast_2d(np.sqrt(np.nanmean(res[:, 3, :] ** 2, axis=0)))# / np.pi / (fiber_d ** 2 / 4))
            cur_lsf = np.atleast_2d(np.nanmean(res[:, 5, :], axis=0))

            if cur_bin_id == 0:
                flux = np.empty(shape=(len(uniq_bins), cur_flux.shape[1]), dtype=float)
                error = np.empty(shape=(len(uniq_bins), cur_flux.shape[1]), dtype=float)
                sky = np.empty(shape=(len(uniq_bins), cur_flux.shape[1]), dtype=float)
                sky_error = np.empty(shape=(len(uniq_bins), cur_flux.shape[1]), dtype=float)
                lsf = np.empty(shape=(len(uniq_bins), cur_flux.shape[1]), dtype=float)
                tab_summary_bins_resorted = Table(data=None, names=['fiberid', 'fib_ra',
                                                                    'fib_dec', 'targettype', 'fibstatus', 'binnum'],
                                                  dtype=(int, float, float, str, int, int))

            rec = tab_summary_bins['fiberid'] == cur_bin
            tab_summary_bins_resorted.add_row([tab_summary_bins['fiberid'][rec], tab_summary_bins['fib_ra'][rec],
                                               tab_summary_bins['fib_dec'][rec], tab_summary_bins['targettype'][rec],
                                               tab_summary_bins['fibstatus'][rec],
                                               cur_bin])
            flux[cur_bin_id] = cur_flux[0,:]
            error[cur_bin_id] = cur_error[0, :]
            sky[cur_bin_id] = cur_sky[0, :]
            sky_error[cur_bin_id] = cur_sky_error[0, :]
            lsf[cur_bin_id] = cur_lsf[0, :]

        hdu_flux = fits.ImageHDU(data=np.array(flux, dtype=np.float32), name='FLUX')
        hdu_ivar = fits.ImageHDU(data=np.array(1/error**2, dtype=float), name='IVAR')
        hdu_sky = fits.ImageHDU(data=np.array(sky, dtype=np.float32), name='SKY')
        hdu_sky_ivar = fits.ImageHDU(data=np.array(1 / sky_error ** 2, dtype=float), name='SKY_IVAR')
        hdu_lsf = fits.ImageHDU(data=np.array(lsf, dtype=np.float32), name='LSF')
        for hdu in [hdu_flux, hdu_ivar, hdu_sky, hdu_sky_ivar, hdu_lsf]:
            hdu.header['CRVAL1'] = res[0, 4, 0]
            hdu.header['CRPIX1'] = 1
            hdu.header['CDELT1'] = res[0, 4, 1] - res[0, 4, 0]
            hdu.header['CTYPE1'] = 'WAV-AWAV'
            hdu_out.append(hdu)

        hdu_out.append(fits.BinTableHDU(tab_summary_bins_resorted, name='SLITMAP'))
        f_out = os.path.join(cur_wdir, f"{cur_obj.get('name')}_{bin_line}_sn{target_sn}{suffix_out}")
        hdu_out.writeto(f_out, overwrite=True)

    return np.all(statuses)


def extract_spectra_ds9(config, w_dir=None):
    """
        Extract spectra in given ds9 regions
        :param config: dictionary with configuration parameters
        :param output_dir: path to output directory
        :return:
        """
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    if not config.get('extraction'):
        log.error(f"Cannot proceed with the extraction step without corresponding control block in the config file.")
        return False
    suffix_out = config['extraction'].get('file_output_suffix')
    if not suffix_out:
        suffix_out = '_extracted.fits'

    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        log.info(f"Extracting spectra for {cur_obj.get('name')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        if not config['imaging'].get('use_dap'):
            f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes.txt")
        else:
            f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes_dap.txt")
        if not os.path.isfile(f_tab_summary):
            log.error(f"Table with the results of the RSS analysis doesn't exist. "
                      f"Anylise_rss step must be run before the spectra extraction. "
                      f"Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        f_ds9 = os.path.join(cur_wdir, f"{cur_obj.get('name')}{config['extraction'].get('file_ds9_suffix')}")
        f_ds9_mask = os.path.join(cur_wdir, f"{cur_obj.get('name')}{config['extraction'].get('mask_ds9_suffix')}")
        f_out = os.path.join(cur_wdir, f"{cur_obj.get('name')}{suffix_out}")
        if not os.path.isfile(f_ds9):
            log.error(f"ds9 file doesn't exist. "
                      f"Create regions for the extraction first. "
                      f"Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        if not os.path.isfile(f_ds9_mask):
            reg_mask = None
        else:
            reg_mask = Regions.read(f_ds9_mask, format='ds9')
            log.info(f"Region mask for {cur_obj.get('name')} is used")

        table_fluxes = Table.read(f_tab_summary, format='ascii.fixed_width_two_line',
                                  converters={'sourceid': str, 'id': str, 'fluxcorr_b': str, 'fluxcorr_r': str,
                                              'fluxcorr_z': str, 'fluxcorr': str, 'vhel_corr': str})

        correct_vel_line = config['extraction'].get('correct_vel_line')
        if not correct_vel_line:
            correct_vel_line = None
        else:
            if correct_vel_line+'_vel' not in table_fluxes.colnames:
                log.warning(f"Can't correct spectra for velocity because "
                            f"the measurements in {correct_vel_line} are not found in RSS flux table")
                correct_vel_line = None

        ras = table_fluxes['fib_ra']
        decs = table_fluxes['fib_dec']
        radec = SkyCoord(ra=ras, dec=decs, unit=('degree', 'degree'))
        dec_0 = np.min(decs) - 37. / 2 / 3600
        dec_1 = np.max(decs) + 37. / 2 / 3600
        ra_0 = np.max(ras) + 37. / 2 / 3600 / np.cos(dec_1 / 180 * np.pi)
        ra_1 = np.min(ras) - 37. / 2 / 3600 / np.cos(dec_0 / 180 * np.pi)

        ra_cen = (ra_0 + ra_1) / 2.
        dec_cen = (dec_0 + dec_1) / 2.
        nx = np.ceil((ra_0 - ra_1) * max(
            [np.cos(dec_0 / 180. * np.pi), np.cos(dec_1 / 180. * np.pi)]) / 1 * 3600. / 2.).astype(
            int) * 2 + 1
        ny = np.ceil((dec_1 - dec_0) / 1 * 3600. / 2.).astype(int) * 2 + 1
        ra_0 = np.round(ra_cen + (nx - 1) / 2 * 1 / 3600. / max(
            [np.cos(dec_0 / 180. * np.pi), np.cos(dec_1 / 180. * np.pi)]), 6)
        dec_0 = np.round(dec_cen - (ny - 1) / 2 * 1 / 3600., 6)
        ra_cen = np.round(ra_cen, 6)
        dec_cen = np.round(dec_cen, 6)
        # Create a new WCS object.  The number of axes must be set
        # from the start
        wcs_ref = WCS(naxis=2)
        wcs_ref.wcs.crpix = [(nx - 1) / 2 + 1, (ny - 1) / 2 + 1]
        wcs_ref.wcs.cdelt = np.array([-np.round(1 / 3600., 6), np.round(1 / 3600., 6)])
        wcs_ref.wcs.crval = [ra_cen, dec_cen]
        wcs_ref.wcs.cunit = ['deg', 'deg']
        wcs_ref.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        regions = Regions.read(f_ds9, format='ds9')
        hdu_out = fits.HDUList([fits.PrimaryHDU()])
        log.info(f"Total number of extracted spectra is {len(regions)}.")
        fig = plt.figure(figsize=(20,20))
        if len(regions) > 10:
            nregs_to_show = 10
            log.warning(f"Only first {nregs_to_show} will be shown in the pdf file")
        else:
            nregs_to_show = len(regions)
        gs = GridSpec(nregs_to_show, 1, fig, 0.1, 0.1, 0.99, 0.99, 0.1, 0.15)
        for cur_reg_id, cur_reg in enumerate(regions):
            if cur_reg.meta.get('text'):
                cur_reg_name = cur_reg.meta['text']
            else:
                cur_reg_name = f'{cur_reg_id+1}'
            in_reg = cur_reg.contains(radec, wcs_ref)
            if reg_mask is not None:
                for cur_mask in reg_mask:
                    in_reg = in_reg & (~cur_mask.contains(radec, wcs_ref))
            in_reg = np.flatnonzero(in_reg)
            if len(in_reg) == 0:
                log.warning(f"No fibers within region {cur_reg_name}")
                continue

            if correct_vel_line is not None:
                med_vel = np.nanmedian(table_fluxes[in_reg][f'{correct_vel_line}_vel'])
                if not config['imaging'].get('use_dap'):
                    sourceid = table_fluxes[in_reg]['sourceid']
                    fluxcorr_b = table_fluxes[in_reg]['fluxcorr_b']
                    fluxcorr_r = table_fluxes[in_reg]['fluxcorr_r']
                    fluxcorr_z = table_fluxes[in_reg]['fluxcorr_z']
                else:
                    sourceid = []
                    for v in table_fluxes[in_reg]['id']:
                        dap_id = str(v).split('.')
                        exp_id = int(dap_id[0])
                        fib_id = int(dap_id[1])
                        pointing_id_found = False
                        for cur_pointing in cur_obj.get('pointing'):
                            for cur_data in cur_pointing.get('data'):
                                if exp_id in cur_data.get('exp'):
                                    pointing_id = cur_pointing.get('name')
                                    pointing_id_found = True
                                    break
                            if pointing_id_found:
                                break
                        sourceid.append('_'.join([str(pointing_id), f"{exp_id:08d}", f"{fib_id:04d}"]))
                    fluxcorr_b = table_fluxes[in_reg]['fluxcorr']
                    fluxcorr_r = table_fluxes[in_reg]['fluxcorr']
                    fluxcorr_z = table_fluxes[in_reg]['fluxcorr']
                params = zip(sourceid, fluxcorr_b,
                             fluxcorr_r, fluxcorr_z,
                             table_fluxes[in_reg]['vhel_corr'], table_fluxes[in_reg][f'{correct_vel_line}_vel']-med_vel,
                             [cur_obj.get('velocity')] * len(in_reg),
                             [config['imaging'].get('include_sky')] * len(in_reg),
                             [config['imaging'].get('partial_sky')] * len(in_reg),
                             [cur_wdir] * len(in_reg)
                             )
            else:
                params = zip(sourceid, fluxcorr_b,
                             fluxcorr_r, fluxcorr_z,
                             table_fluxes[in_reg]['vhel_corr'],[0]*len(in_reg), [cur_obj.get('velocity')]*len(in_reg),
                             [config['imaging'].get('include_sky')]*len(in_reg), [config['imaging'].get('partial_sky')]*len(in_reg),
                             [cur_wdir]*len(in_reg))

            nprocs = np.min([np.max([config.get('nprocs'), 1]), len(in_reg)])
            with mp.Pool(processes=nprocs) as pool:
                res = list(tqdm(pool.imap(extract_spectrum_for_cur_fiber, params),
                        ascii=True, desc=f"Extract spectra from fibers in {cur_reg_name} region",
                        total=len(in_reg)))

                pool.close()
                pool.join()
                gc.collect()
            res = np.array(res)

            flux = np.atleast_2d(np.nanmean(res[:,0,:], axis=0)/np.pi/(fiber_d**2/4))
            error = np.atleast_2d(np.sqrt(np.nanmean(res[:, 1, :]**2, axis=0))/np.pi/(fiber_d**2/4))
            sky = np.atleast_2d(np.nanmean(res[:, 2, :], axis=0)/np.pi/(fiber_d**2/4))
            sky_error = np.atleast_2d(np.sqrt(np.nanmean(res[:, 3, :] ** 2, axis=0))/np.pi/(fiber_d**2/4))
            cur_lsf = np.atleast_2d(np.nanmean(res[:, 5, :], axis=0))
            cur_hdu = fits.ImageHDU(data = np.vstack([flux,error,sky,sky_error, cur_lsf], dtype=np.float32),
                                    name=cur_reg_name)
            cur_hdu.header['CRVAL1'] = res[0,4,0]
            cur_hdu.header['CRPIX1'] = 1
            cur_hdu.header['CDELT1'] = res[0,4,1]-res[0,4,0]
            cur_hdu.header['CTYPE1'] = 'WAV-AWAV'
            hdu_out.append(cur_hdu)

            if cur_reg_id < nregs_to_show:
                ax = fig.add_subplot(gs[cur_reg_id])
                ax.plot(res[0, 4, :], flux[0,:])
                ax.set_title(cur_reg_name)
                ax.set_xlabel(r"Wavelength, $\AA$")
                ax.set_ylabel(r"Intensity, erg/s/cm^2/arcsec^2/A")

        hdu_out.writeto(f_out, overwrite=True)
        fig.savefig(f_out.replace(".fits", '.pdf'), dpi=300, bbox_inches='tight')
        statuses.append(True)

    return np.all(statuses)


def extract_spectrum_for_cur_fiber(params):

    (source_ids, flux_cors_b, flux_cors_r, flux_cors_z, vhel_corrs, corr_vel_line,
     velocity, include_sky, partial_sky, path_to_fits) = params
    source_ids = source_ids.split(', ')
    flux_cors_b = flux_cors_b.split(', ')
    flux_cors_r = flux_cors_r.split(', ')
    flux_cors_z = flux_cors_z.split(', ')
    vhel_corrs = vhel_corrs.split(', ')
    wl_grid = None
    for ind, source_id in enumerate(source_ids):
        pointing, expnum, fib_id = source_id.split('_')
        expnum = int(expnum)
        fib_id = int(fib_id) - 1
        if include_sky:
            rssfile = os.path.join(path_to_fits, pointing, f'lvmCFrame-{expnum:0>8}.fits')
        else:
            rssfile = os.path.join(path_to_fits, pointing, f'lvmSFrame-{expnum:0>8}.fits')

        with fits.open(rssfile) as hdu:
            if wl_grid is None:
                wl_grid = ((np.arange(hdu['FLUX'].header['NAXIS1']) -
                            hdu['FLUX'].header['CRPIX1'] + 1) * hdu['FLUX'].header['CDELT1'] +
                           hdu['FLUX'].header['CRVAL1'])

                flux = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                ivar = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                sky = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                sky_ivar = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
                lsf = np.zeros(shape=(len(source_ids), hdu['FLUX'].header['NAXIS1']), dtype=float)
            if partial_sky:
                flux[ind, :] = ((hdu['FLUX'].data[fib_id, :] + hdu['SKY'].data[fib_id, :]) -
                                mask_sky_at_bright_lines(hdu['SKY'].data[fib_id, :], wave=wl_grid,
                                                         vel=velocity, mask=hdu['MASK'].data[fib_id, :]))
            else:
                flux_corr = float(np.nanmean([float(flux_cors_b[ind]), float(flux_cors_r[ind]),float(flux_cors_z[ind])]) )
                flux[ind, :] = hdu['FLUX'].data[fib_id, :] * flux_corr

            ivar[ind, :] = hdu['IVAR'].data[fib_id, :] / flux_corr ** 2
            sky[ind, :] = hdu['SKY'].data[fib_id, :] * flux_corr
            sky_ivar[ind, :] = hdu['SKY_IVAR'].data[fib_id, :] / flux_corr ** 2
            lsf[ind, :] = hdu['LSF'].data[fib_id, :]
            flux[ind, hdu['MASK'].data[fib_id, :] == 1] = np.nan
            sky[ind, hdu['MASK'].data[fib_id, :] == 1] = np.nan


        delta_v = float(float(vhel_corrs[ind]) - corr_vel_line)
        if delta_v > 2.:
            flux[ind, :] = np.interp(wl_grid, wl_grid * (1 - delta_v / 3e5), flux[ind, :])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
        )
        flux = np.nanmean(sigma_clip(flux, sigma=1.3, axis=0, masked=False), axis=0)
        ivar = 1 / (np.nansum(1 / ivar, axis=0) / np.sum(np.isfinite(ivar), axis=0) ** 2)
        sky = np.nansum(sky, axis=0) / np.sum(np.isfinite(sky), axis=0)
        sky_ivar = 1 / (np.nansum(1 / sky_ivar, axis=0) / np.sum(np.isfinite(sky_ivar), axis=0) ** 2)
        error = 1/np.sqrt(ivar)
        sky_error = 1 / np.sqrt(sky_ivar)
        lsf = np.nanmean(lsf, axis=0)


    return flux, error, sky, sky_error, wl_grid, lsf

def reconstruct_cube(config, w_dir=None):
    statuses = []
    if not config['imaging'].get('pxscale'):
        pxscale_out = 15
        log.warning("Cannot find pxscale in config. Use 15 arcsec")
    else:
        pxscale_out = float(config['imaging'].get('pxscale'))
    if config['imaging'].get('r_lim'):
        r_lim = config['imaging'].get('r_lim')
    else:
        log.warning("Cannot find r_lim in config. Use 50 arcsec")
        r_lim = 50
    if config['imaging'].get('sigma'):
        sigma = config['imaging'].get('sigma')
    else:
        log.warning("Cannot find sigma in config. Use 2 arcsec")
        sigma = 2
    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        log.info(f"Constructing data cube for {cur_obj.get('name')} in {config['cube_reconstruction'].get('suffix')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
        outfile = os.path.join(cur_wdir, cur_obj.get('name')+
                               f"_cube_{config['cube_reconstruction'].get('suffix')}.fits" )
        if not os.path.exists(cur_wdir):
            log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
            statuses.append(False)
            continue
        f_tab_summary = os.path.join(cur_wdir, f"{cur_obj.get('name')}_fluxes.txt")
        if not os.path.isfile(f_tab_summary):
            log.error(f"File {f_tab_summary} does not exist. Have you run 'analyse_rss' step first?")
            statuses.append(False)
            continue

        table_fluxes = Table.read(f_tab_summary, format='ascii.fixed_width_two_line',
                                  converters={'sourceid': str, 'fluxcorr': str, 'vhel_corr': str})

        vel = cur_obj.get('velocity')
        ras = table_fluxes['fib_ra']
        decs = table_fluxes['fib_dec']

        # === Define rectangular grid
        dec_0 = np.min(decs) - 37. / 2 / 3600
        dec_1 = np.max(decs) + 37. / 2 / 3600
        ra_0 = np.max(ras) + 37. / 2 / 3600 / np.cos(dec_1 / 180 * np.pi)
        ra_1 = np.min(ras) - 37. / 2 / 3600 / np.cos(dec_0 / 180 * np.pi)

        ra_cen = (ra_0 + ra_1) / 2.
        dec_cen = (dec_0 + dec_1) / 2.
        nx = np.ceil((ra_0 - ra_1) * np.cos(dec_cen / 180. * np.pi) / pxscale_out * 3600. / 2.).astype(int) * 2 + 1
        ny = np.ceil((dec_1 - dec_0) / pxscale_out * 3600. / 2.).astype(int) * 2 + 1
        ra_0 = np.round(ra_cen + (nx - 1) / 2 * pxscale_out / 3600. / np.cos(dec_cen / 180. * np.pi), 6)
        dec_0 = np.round(dec_cen - (ny - 1) / 2 * pxscale_out / 3600., 6)
        ra_cen = np.round(ra_cen, 6)
        dec_cen = np.round(dec_cen, 6)
        # Create a new WCS object.  The number of axes must be set
        # from the start
        wcs_out = WCS(naxis=2)
        wcs_out.wcs.crpix = [(nx - 1) / 2 + 1, (ny - 1) / 2 + 1]
        wcs_out.wcs.cdelt = np.array([-np.round(pxscale_out / 3600., 6), np.round(pxscale_out / 3600., 6)])
        wcs_out.wcs.crval = [ra_cen, dec_cen]
        wcs_out.wcs.cunit = ['deg', 'deg']
        wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        shape_out = (ny, nx)
        grid_scl = wcs_out.proj_plane_pixel_scales()[0].value * 3600.

        log.info(f"Grid scale: {np.round(grid_scl, 1)}, output shape: {nx}x{ny}, "
                 f"RA range: {np.round(ra_1, 4)} - {np.round(ra_0, 4)}, "
                 f"DEC range: {np.round(dec_0, 4)} - {np.round(dec_1, 4)} ")

        # === Extract wl information from first pointing
        source_ids = table_fluxes['sourceid'][0].split(', ')
        pointing, expnum, fib_id = source_ids[0].split('_')
        expnum = int(expnum)
        rssfile = os.path.join(cur_wdir, pointing,
                               f'lvmSFrame-{expnum:0>8}.fits')
        with (fits.open(rssfile) as rss):
            wave = ((np.arange(rss['FLUX'].header['NAXIS1']) -
                     rss['FLUX'].header['CRPIX1']) * rss['FLUX'].header['CDELT1'] +
                    rss['FLUX'].header['CRVAL1'])
            try:
                rec_wave = np.flatnonzero((wave>=config['cube_reconstruction']['wl_range'][0]) &
                                          (wave<=config['cube_reconstruction']['wl_range'][1]))
            except KeyError:
                log.error('Something wrong with "cube_reconstruction" in config')
                statuses.append(False)
                continue
            if len(rec_wave) == 0:
                log.error('Wavelength range is inconsistent with the data')
                statuses.append(False)
                continue
            crval = wave[rec_wave][0]
            wave_dict = {"CRVAL3": crval,
                     'CDELT3': rss['FLUX'].header['CDELT1'],
                     'CRPIX3': 1,
                     "CTYPE3": rss['FLUX'].header['CTYPE1'],
                     'CUNIT3': 'Angstrom',
                     'BUNIT': rss['FLUX'].header['BUNIT']}


        # === Create container for data cube
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
        header['NAXIS3'] = len(rec_wave)
        for kw in wave_dict.keys():
            header[kw] = wave_dict[kw]

        header.tofile(outfile, overwrite=True)
        fluxes = None
        for row_id, source_ids in tqdm(enumerate(table_fluxes['sourceid']), total=len(table_fluxes),
                                       desc='Fibers done:', ascii=True):
            source_ids = source_ids.split(', ')
            fluxcorrs = np.array(table_fluxes['fluxcorr'][row_id].split(', ')).astype(float)
            vhel_corr = np.array(table_fluxes['vhel_corr'][row_id].split(', ')).astype(float)
            cur_fluxes = None
            for cur_ind, source_id in enumerate(source_ids):
                pointing, expnum, fib_id = source_id.split('_')
                expnum = int(expnum)
                fib_id = int(fib_id)
                if config['imaging'].get('include_sky'):
                    rssfile = os.path.join(cur_wdir, pointing, f'lvmCFrame-{expnum:0>8}.fits')
                else:
                    rssfile = os.path.join(cur_wdir, pointing, f'lvmSFrame-{expnum:0>8}.fits')
                with fits.open(rssfile) as rss:
                    flux = rss['FLUX'].data[fib_id,rec_wave] * fluxcorrs[cur_ind]
                    delta_v = vhel_corr[cur_ind] - vhel_corr[0]
                    if abs(delta_v) > 3.:
                        flux = np.interp(wave[rec_wave], wave[rec_wave]*(1-delta_v/3e5), flux)
                    if cur_fluxes is None:
                        cur_fluxes = flux
                    else:
                        cur_fluxes = np.vstack([cur_fluxes, flux])
                if len(cur_fluxes.shape) == 2 and cur_fluxes.shape[0] > 1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                        )
                        cur_fluxes = np.nanmean(sigma_clip(cur_fluxes, sigma=1.3, axis=0, masked=False), axis=0)
            if fluxes is None:
                fluxes = cur_fluxes
            else:
                fluxes = np.vstack([fluxes, cur_fluxes])

        shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs,
                         show_values=fluxes, r_lim=r_lim, sigma=sigma,
                         cube=True, outfile=outfile, header=header)
        statuses.append(True)
    status_out = np.all(statuses)
    return status_out

def rotate(xx,yy,angle):
    # rotate x and y cartesian coordinates by angle (in degrees)
    # about the point (0,0)
    theta = -np.radians(angle)
    xx1 = np.cos(theta) * xx - np.sin(theta) * yy
    yy1 = np.sin(theta) * xx + np.cos(theta) * yy

    return xx1, yy1


def make_radec(xx0,yy0,ra,dec,pa):
    platescale = 112.36748321030637  # Focal plane platescale in "/mm

    pscale = 0.01  # IFU image pixel scale in mm/pix
    skypscale = pscale * platescale / 3600  # IFU image pixel scale in deg/pix
    npix = 1800  # size of fake IFU image
    w = WCS(naxis=2)  # IFU image wcs object
    w.wcs.crpix = [int(npix / 2) + 1, int(npix / 2) + 1]
    posangrad = pa * np.pi / 180
    w.wcs.cd = np.array([[skypscale * np.cos(posangrad), -1 * skypscale * np.sin(posangrad)],
                            [-1 * skypscale * np.sin(posangrad), -1 * skypscale * np.cos(posangrad)]])
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # Calculate RA,DEC of each individual fiber
    xfib = xx0 / pscale + int(npix / 2)  # pixel x coordinates of fibers
    yfib = yy0 / pscale + int(npix / 2)  # pixel y coordinates of fibers
    fibcoords = w.pixel_to_world(xfib, yfib).to_table()
    ra_fib = fibcoords['ra'].degree
    dec_fib = fibcoords['dec'].degree


    # xx, yy = rotate(xx0, yy0, pa)
    # ra_fib = ra + xx * platescale/3600./np.cos(np.radians(dec))
    # dec_fib = dec - yy * platescale/3600.
    return ra_fib, dec_fib

def shepard_convolve(wcs_out, shape_out, ra_fibers=None, dec_fibers=None, show_values=None, r_lim=50., sigma=2.,
                     cube=False, header=None, outfile=None, do_median_masking=False, masks=None, remove_empty=False,
                     is_error=False):
    if is_error:
        show_values = show_values ** 2
    if not cube:
        if len(show_values.shape) ==1:
            show_values = show_values.reshape((-1, 1))
            masks = masks.reshape((-1,1))
        # if show_values.shape[1] > 7:
        #     rec_fibers = np.isfinite(show_values[:,0]) & (show_values[:,0] != 0) #& np.isfinite(show_values[:,-4]) & (show_values[:,-4] < 35) & (show_values[:,-4] > 18)
        # else:
        if masks is None:
            masks = np.tile(np.isfinite(show_values[:, 0]) & (show_values[:, 0] != 0), show_values.shape) # or tile??
            if len(masks.shape) == 1:
                masks = masks.reshape((-1, 1))
        # rec_fibers = np.isfinite(show_values[:, 0]) & (show_values[:, 0] != 0)
        rec_fibers = np.any(masks, axis=1)
        masks = masks[rec_fibers]
        show_values = show_values[rec_fibers,:]
    else:
        rec_fibers = np.isfinite(np.nansum(show_values.T, axis=0)) & (np.nansum(show_values.T, axis=0) != 0)
        flux_fibers = show_values[rec_fibers, :]
    pxsize = wcs_out.proj_plane_pixel_scales()[0].value * 3600.
    radec = SkyCoord(ra=ra_fibers[rec_fibers], dec=dec_fibers[rec_fibers], unit='deg', frame='icrs')
    x_fibers, y_fibers = wcs_out.world_to_pixel(radec)
    # x_fibers = np.round(x_fibers).astype(int)
    # y_fibers = np.round(y_fibers).astype(int)
    chunk_size = min([int(r_lim * 5 / pxsize), 50])
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

        if not cube:
            weights = np.zeros(shape=(len(rec_fibers), ny, nx, show_values.shape[-1]), dtype=float)
        else:
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
            if not cube:
                weights[ind, cur_y0:cur_y1+1, cur_x0: cur_x1+1, :] = (
                                                                         kernel)[khalfsize - (cur_center_y-cur_y0):
                                                                                 khalfsize - (cur_center_y-cur_y0)+ cur_y1 - cur_y0+1,
                                                                     khalfsize - (cur_center_x-cur_x0):
                                                                     khalfsize - (cur_center_x-cur_x0)+cur_x1 - cur_x0+1,
                                                                     None] * (masks[rec_fibers[ind],None,None,:]).astype(float)
            else:
                weights[ind, cur_y0:cur_y1 + 1, cur_x0: cur_x1 + 1] = kernel[
                                                                      khalfsize - (cur_center_y - cur_y0): khalfsize - (
                                                                                  cur_center_y - cur_y0) + cur_y1 - cur_y0 + 1,
                                                                      khalfsize - (cur_center_x - cur_x0): khalfsize - (
                                                                                  cur_center_x - cur_x0) + cur_x1 - cur_x0 + 1]

        weights_norm = np.sum(weights, axis=0)
        weights_norm[weights_norm == 0] = 1
        if is_error:
            weights = weights ** 2
            weights_norm = weights_norm**2
        if not cube:
            weights = weights / weights_norm[None, :, :, :]
        else:
            weights = weights / weights_norm[None, :, :]
        n_used_fib = np.sum(weights > 0, axis=0)
        n_used_fib = np.broadcast_to(n_used_fib, weights.shape)
        weights[n_used_fib < 2] = 0
        if remove_empty:
            weights[weights == 0] = np.nan
        if not cube:
            img_chunk = np.nansum(weights[:, :, :, :] * show_values[rec_fibers, None, None, :], axis=0)

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
        if is_error:
            hdu_out[0].data = np.sqrt(hdu_out[0].data)
        hdu_out.writeto(outfile, overwrite=True, output_verify='silentfix')
    else:
        if is_error:
            img_out = np.sqrt(img_out)
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
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
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
            for spec_id, results in tqdm(enumerate(pool.imap(get_noise_one_exp, files)),
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


def create_single_rss(config, w_dir=None):
    """
    Create single RSS file with combined spectra from different exposures
    """
    if w_dir is None or not os.path.exists(w_dir):
        log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
        return False

    statuses = []
    for cur_obj in config['object']:
        if not cur_obj.get('version'):
            version = ''
        else:
            version = cur_obj.get('version')
        log.info(f"Creating single RSS file for {cur_obj.get('name')}.")
        cur_wdir = os.path.join(w_dir, cur_obj.get('name'), version)
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
                    if not config['imaging']['include_sky']:
                        cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmSFrame-{exp:08d}.fits')
                    else:
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
                        cur_flux_corr = [1.] * len(exps)
                    else:
                        cur_flux_corr = data['flux_correction']
                    if isinstance(cur_flux_corr, float) or isinstance(cur_flux_corr, int):
                        cur_flux_corr = [cur_flux_corr]
                    # corrections.extend(cur_flux_corr)

                    for exp_id, exp in tqdm(enumerate(exps), total=len(exps), ascii=True,
                                            desc=f'Extraction fiber coordinates for pointing {cur_pointing["name"]}'):
                        if not config['imaging']['include_sky']:
                            cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmSFrame-{exp:08d}.fits')
                        else:
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

                            tab = Table(rss['SLITMAP'].data)
                            sci = np.flatnonzero(tab['targettype'] == 'science')
                            if config['imaging'].get('skip_bad_fibers'):
                                sci = np.flatnonzero((tab['targettype'] == 'science') & (tab['fibstatus'] == 0))

                        # kostyl for NGC6822
                        if exp == 3602:
                            ref_exp = 3601
                        radec_center = derive_radec_ifu(data['mjd'], exp, ref_exp,
                                                        objname=cur_obj['name'],
                                                        pointing_name=cur_pointing.get('name'), w_dir=cur_wdir)

                        ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci],
                                                     radec_center[0], radec_center[1], radec_center[2])
                        radec_array = SkyCoord(ra=ra_fib, dec=dec_fib, unit=('degree', 'degree'))
                        vcorr = np.round(radec_array[0].radial_velocity_correction(kind='heliocentric', obstime=obstime,
                                                                                   location=obs_loc).to(u.km / u.s).value,
                                         1)
                        for trow_id, trow in enumerate(tab[sci]):
                            cur_radec = radec_array[trow_id]
                            radec_tab = SkyCoord(ra=tab_summary['fib_ra'], dec=tab_summary['fib_dec'],
                                                 unit=('degree', 'degree'))
                            rec = np.flatnonzero(radec_tab.separation(cur_radec) < (1.5 * u.arcsec))
                            fib_id = f"{exp:08d}_{trow['fiberid']:04d}"
                            if len(rec) > 0:
                                tab_summary["sourceid"][rec[0]] = f'{tab_summary["sourceid"][rec[0]]}, {fib_id}'
                                tab_summary["fluxcorr"][rec[0]] = f'{tab_summary["fluxcorr"][rec[0]]}, {cur_flux_corr[exp_id]}'
                                tab_summary['vhel_corr'][rec[0]] = f'{tab_summary["vhel_corr"][rec[0]]}, {vcorr}'
                            else:
                                tab_summary.add_row([len(tab_summary) + 1, ra_fib[trow_id], dec_fib[trow_id],
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
                                    fits.ImageHDU(data=np.zeros(shape=(len(tab_summary), nx), dtype=float), name='IVAR')])
            rss_out.writeto(fout, overwrite=True)
            rss_out.close()
            log.info('......FLUX and IVAR extensions are added')
            for kw_block in [('MASK', 'WAVE', 'LSF'), ('SKY', 'SKY_IVAR', 'FLUX_ORIG')]:
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
            sp_ids = [int(sid.split('_')[1]) - 1 for sid in source_ids]
            if len(source_ids) == 1:
                # ===== Copy spectrum from the source
                f = fs[0]
                spec_id = sp_ids[0]
                with fits.open(f) as hdu:
                    wl_grid = ((np.arange(hdu['FLUX'].header['NAXIS1']) -
                                hdu['FLUX'].header['CRPIX1'] + 1) * hdu['FLUX'].header['CDELT1'] * 1e10 +
                               hdu['FLUX'].header['CRVAL1'] * 1e10)
                    if ind_row == 0:
                        rss_out[0].header = hdu[0].header
                        ref_h_for_fsc = hdu['FLUX'].header.copy()
                        ref_h_for_fsc['EXTNAME'] = 'FLUX_ORIG'
                        rss_out['FLUX_ORIG'].header = ref_h_for_fsc
                        rss_out['WAVE'].data = hdu['WAVE'].data
                        rss_out['WAVE'].header = hdu['WAVE'].header
                    for kw in ['FLUX', 'IVAR', 'MASK', 'LSF', 'SKY', 'SKY_IVAR']:
                        if ind_row == 0:
                            rss_out[kw].header = hdu[kw].header
                        rss_out[kw].data[ind_row, :] = hdu[kw].data[spec_id, :]
                    rss_out['FLUX_ORIG'].data[ind_row, :] = hdu['FLUX'].data[spec_id, :].copy()
                    rss_out['FLUX'].data[ind_row, :] = (hdu['FLUX'].data[spec_id, :] + hdu['SKY'].data[spec_id, :]) - \
                                                       mask_sky_at_bright_lines(hdu['SKY'].data[spec_id, :],
                                                                                wave=wl_grid, vel=cur_obj['velocity'],
                                                                                mask=hdu['MASK'].data[spec_id, :])
                    rec_bad = np.flatnonzero(
                        (hdu['FLUX'].data[spec_id, :] == 0) | ~np.isfinite(hdu['FLUX'].data[spec_id, :]) | (
                                    hdu['MASK'].data[spec_id, :] > 0))
                    for kw in ['FLUX', 'IVAR', 'SKY', 'FLUX_ORIG', 'SKY_IVAR']:
                        rss_out[kw].data[ind_row, :] = rss_out[kw].data[ind_row, :] * cur_corr_flux[0]
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
                            wl_grid = ((np.arange(hdu['FLUX'].header['NAXIS1']) -
                                        hdu['FLUX'].header['CRPIX1'] + 1) * hdu['FLUX'].header['CDELT1'] * 1e10 +
                                       hdu['FLUX'].header['CRVAL1'] * 1e10)
                            rss_out['LSF'].data[ind_row, :] = hdu['LSF'].data[spec_id, :]
                        if (ind_row == 0) & (f_id == 0):
                            rss_out[0].header = hdu[0].header
                            ref_h_for_fsc = hdu['FLUX'].header.copy()
                            ref_h_for_fsc['EXTNAME'] = 'FLUX_ORIG'
                            rss_out['FLUX_ORIG'].header = ref_h_for_fsc
                            rss_out['WAVE'].data = hdu['WAVE'].data
                            for kw in ['FLUX', 'IVAR', 'MASK', "WAVE", 'LSF', 'SKY', 'SKY_IVAR']:
                                rss_out[kw].header = hdu[kw].header

                        rec = np.flatnonzero(
                            hdu['MASK'].data[spec_id, :] | ~np.isfinite(hdu['FLUX'].data[spec_id, :]) | (
                                        hdu['FLUX'].data[spec_id, :] == 0))
                        rec_sky = np.flatnonzero(
                            hdu['MASK'].data[spec_id, :] | ~np.isfinite(hdu['SKY'].data[spec_id, :]))
                        rec_skycorr = np.flatnonzero(
                            hdu['MASK'].data[spec_id, :] | ~np.isfinite(hdu['SKY'].data[spec_id, :]) | ~np.isfinite(
                                hdu['FLUX'].data[spec_id, :]) | (hdu['FLUX'].data[spec_id, :] == 0))
                        fluxes[f_id, :] = hdu['FLUX'].data[spec_id, :] * cur_corr_flux[f_id]
                        errors[f_id, :] = abs(hdu['IVAR'].data[spec_id, :]) / cur_corr_flux[f_id] ** 2
                        skies[f_id, :] = hdu['SKY'].data[spec_id, :] * cur_corr_flux[f_id]
                        masks = masks | hdu['MASK'].data[spec_id, :]
                        sky_errors[f_id, :] = hdu['SKY_IVAR'].data[spec_id, :] / cur_corr_flux[f_id] ** 2
                        fluxes_skycorr[f_id, :] = ((hdu['FLUX'].data[spec_id, :] + hdu['SKY'].data[spec_id, :]) -
                                                   mask_sky_at_bright_lines(hdu['SKY'].data[spec_id, :], wave=wl_grid,
                                                                            vel=cur_obj['velocity'],
                                                                            mask=hdu['MASK'].data[spec_id, :])) * \
                                                  cur_corr_flux[f_id]
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
                rss_out['IVAR'].data[ind_row, :] = 1/(np.nansum(1/errors, axis=0)/np.sum(np.isfinite(errors),axis=0)**2)
                rss_out['SKY'].data[ind_row, :] = np.nansum(skies, axis=0)/np.sum(np.isfinite(skies),axis=0)
                rss_out['SKY_IVAR'].data[ind_row, :] = 1/(np.nansum(1/sky_errors, axis=0)/np.sum(np.isfinite(sky_errors),axis=0)**2)

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


# ===========================================================================================
# ======== Auxiliary functions (for testing etc., not used in general processing) ===========
# ===========================================================================================
#

# def check_pixel_shift(data):
#     from lvmdrp.functions import run_calseq as calseq
#     from lvmdrp.utils import metadata as md
#     mjd, first_exp, source_dir, messup = data
#     imagetyp = "object"
#     expnums = sorted(md.get_frames_metadata(mjd=mjd).query(
#         "imagetyp == @imagetyp and not (ldls|quartz|argon|neon|hgne|xenon)").expnum.unique())
#
#     if messup:
#         calseq.messup_frame(mjd, expnum=expnums[1], shifts=[2345], spec="2", shift_size=-2, undo_messup=False)
#
#     try:
#         calseq.fix_raw_pixel_shifts(mjd=mjd, expnums=expnums, ref_expnums=first_exp, specs="123",
#                                     create_mask_always=False, dry_run=True, display_plots=False,
#                                     wave_widths=0.6*5000, y_widths=20, flat_spikes=21, threshold_spikes=0.1)
#
#         f = glob.glob(os.path.join(source_dir, '*pixel_shifts.png'))
#
#         if len(f) > 0:
#             for sf in f:
#                 shutil.copy(sf, '/home/egorov/Science/LVM/test_pixshift/')
#             return 1
#         return 0
#     except:
#         return -1
#
# def do_test_pix_shift(config):
#     mjds = []
#     first_exp = []
#     source_dirs = []
#     for cur_obj in config['object']:
#         for cur_pointing in cur_obj['pointing']:
#             for data in cur_pointing['data']:
#                 if isinstance(data['exp'], int):
#                     exp = [data['exp']]
#                 else:
#                     exp = data['exp']
#                 if data['mjd'] not in mjds:
#                     sdir = os.path.join("/data/LVM/sdsswork/data/lvm/lco/", str(data['mjd']))
#                     if os.path.exists(sdir):
#                         f = glob.glob(os.path.join(sdir, "*.fits.gz"))
#                         if len(f) < 1:
#                             continue
#                     mjds.append(data['mjd'])
#                     first_exp.append(exp[0])
#                     source_dirs.append(os.path.join(drp_results_dir, '11111', str(data['mjd']), 'ancillary', 'qa'))
#
#     mjds = np.array(mjds)
#     first_exp = np.array(first_exp)
#     source_dirs = np.array(source_dirs)
#     if len(mjds) == 0:
#         log.warning("No mjds provided")
#         return False
#
#     procs = np.nanmin([config['nprocs'], len(mjds)])
#     log.info(f"Start testing of {len(mjds)} mjds in {procs} parallel processes")
#     statuses = []
#     # status = check_pixel_shift((mjds[6], first_exp[6], source_dirs[6]))
#     #
#     # return status
#     messup = [True]
#     messup.extend([False]*(len(mjds)-1))
#     messup = np.array(messup)
#     with mp.Pool(processes=procs) as pool:
#
#         for status in tqdm(pool.imap_unordered(check_pixel_shift, zip(mjds, first_exp, source_dirs, messup)),
#                            ascii=True, desc="Test pixel shifts",
#                            total=len(mjds), ):
#             statuses.append(status)
#         pool.close()
#         pool.join()
#         gc.collect()
#     statuses = np.array(statuses)
#     npxshifts = np.sum(statuses == 1)
#     log.info(f"{npxshifts} are detected in {len(mjds)} mjds")
#     if not np.all(statuses >= 0):
#         return False
#
#     return True
#
#
# def do_sky_correction(config, w_dir=None):
#     if w_dir is None or not os.path.exists(w_dir):
#         log.error(f"Work directory does not exist ({w_dir}). Can't proceed further.")
#         return False
#
#     statuses = []
#     files = []
#     for cur_obj in config['object']:
#         cur_wdir = os.path.join(w_dir, cur_obj.get('name'))
#         if not os.path.exists(cur_wdir):
#             log.error(f"Work directory does not exist ({cur_wdir}). Can't proceed with object {cur_obj.get('name')}.")
#             statuses.append(False)
#             continue
#         for cur_pointing in cur_obj['pointing']:
#             for data in cur_pointing['data']:
#                 if isinstance(data['exp'], int):
#                     exps = [data['exp']]
#                 else:
#                     exps = data['exp']
#                 for exp in exps:
#                     cur_fname = os.path.join(cur_wdir, cur_pointing['name'], f'lvmCFrame-{exp:08d}.fits')
#                     if not os.path.exists(cur_fname):
#                         log.warning(f"Can't find {cur_fname}")
#                         continue
#                     files.append(cur_fname)
#
#     for ind_f, cur_f in tqdm(enumerate(files), total=len(files), ascii=True, desc='Spectra done:'):
#         rss_orig = fits.open(cur_f)
#         if not rss_orig[0].header.get('FLUXCAL'):
#             log.error(f"Missing flux calibration for file {cur_f}. Skip it.")
#             rss_orig.close()
#             statuses.append(False)
#             continue
#         rss = deepcopy(rss_orig)
#         rss['FLUX'].data[rss['MASK'] == 1] = np.nan
#         rss['IVAR'].data[rss['MASK'] == 1] = np.nan
#         rss['SKY'].data[rss['MASK'] == 1] = np.nan
#         tab = Table(rss['SLITMAP'].data)
#         sci = np.flatnonzero(tab['targettype'] == 'science')
#         # rss["FLUX"].data += rss['SKY'].data
#
#         flux = quickflux(rss, [5572, 5582], crange=[5545, 5565, 5585, 5600], selection=sci, include_sky=True)
#
#         rss["FLUX"].data = rss['SKY'].data
#         flux_sky = quickflux(rss, [5572, 5582], crange=[5545, 5565, 5585, 5600], selection=sci, include_sky=False)
#
#         # flux, _, _, cont = fit_spectra(rss, [5545, 5600], selection=sci, mean_bounds=(-2,2),
#         #                                consider_as_comp=[0],do_helio_corr=False,
#         #                                lines=[5577.], velocity=0)
#         #
#         # flux_sky, _, _, cont_sky = fit_spectra(rss, [5545, 5600], selection=sci, mean_bounds=(-2, 2),
#         #                                        consider_as_comp=[0], do_helio_corr=False,
#         #                                        lines=[5577.], velocity=0)
#         rss.close()
#         cur_header = rss_orig['FLUX'].header
#         cur_data = rss_orig['FLUX'].data+rss_orig['SKY'].data
#         cur_data[sci] -= (rss_orig['SKY'].data[sci] * (flux/flux_sky)[:, None])
#         rss_orig.append(fits.ImageHDU(data=cur_data, header=cur_header, name='FLUX_SKYCORR'))
#         rss_orig.writeto(cur_f.split(".fits")[0]+'_test.fits', overwrite=True)
#         statuses.append(True)
#

# =========================================================
# ======== Run LVM_processing from command line ===========
# =========================================================

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
