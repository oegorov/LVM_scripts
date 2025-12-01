#!/usr/bin/env python3
import sys
import os
import logging
import json
import numpy as np
import requests
from requests.structures import CaseInsensitiveDict
from astropy.io import fits
from astropy.table import Table

log = logging.getLogger(name='CreateTomlFile')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

server_group_id = 10699
url_json_file = "https://raw.githubusercontent.com/sdss/lvmvis/refs/heads/main/frontend/src/assets/lvm.json"
local_json_file = '/data/LVM/lvm.json'
use_local_file = False
drp_version = '1.2.0'
dap_version = '1.2.0'
dap_local_version = '1.2.0dev'
drpall_file = os.path.join(os.environ.get('SAS_BASE_DIR'),
                      'sdsswork/lvm/spectro/redux', drp_version, f'drpall-{drp_version}.fits')

def parse_tiles(request, file_save=None, drpall=False):
    try:
        request = request.lower()
        request = request.replace('and', '&')
        request = request.replace('or', '|')
    except Exception as e:
        log.error(f"Something wrong with parsing request: {e}")
        return

    if drpall:
        if not os.path.isfile(drpall_file):
            log.error(f"Cannot access the drpall file at {drpall_file}")
            return
        with fits.open(drpall_file) as hdul:
            all_frames = Table.read(hdul[1])
            idx = np.arange(len(all_frames))
            kws = {'MJD': 'mjd', 'TILEID': 'tileid', 'EXPID': 'expnum',
                   'RA': 'sci_ra', 'DEC': 'sci_dec', 'PA': 'sci_pa'}
    else:
        try_local = False
        if os.environ.get('GITHUB_TOKEN') is not None:
            access_token = os.environ.get('GITHUB_TOKEN')
            try:
                headers = CaseInsensitiveDict()
                headers["Authorization"] = f"token {access_token}"
                resp = requests.get(url_json_file, headers=headers)
                parsed_json = json.loads(resp.content)
                all_frames = parsed_json["data"]
                meta = parsed_json["meta"]
            except Exception as e:
                log.error(f"Something wrong with access to online json file with your GitHub token: {e}")
                try_local = True
        else:
            log.warning(f"Cannot access the latest LVM dump file in lvmvis "
                        f"without your GITHUB_TOKEN as environmental variable")
            try_local = True
        if try_local:
            log.info(f"Trying local file: {local_json_file}")
            try:
                f = open(local_json_file, 'r')
                json_string = f.read()
                f.close()
                parsed_json = json.loads(json_string)
                all_frames = parsed_json["data"]
                meta = parsed_json["meta"]
            except Exception as e:
                log.error(f"Something wrong with local json file: {e}. Cannot proceed.")
                return
        log.info(f"Selecting from {len(all_frames)} frames from LVM dump generated on {meta['time_generated']}")

        kws = {'MJD': 'mjd', 'TILEID': 'tile_id', 'EXPID': 'exposure',
               'RA': 'ra', 'DEC': 'dec', 'PA': 'pa'}
        idx = np.array([fr['idx'] for fr in all_frames], dtype=int)

    mjd = np.array([fr[kws['MJD']] for fr in all_frames], dtype=int)
    tile_id = np.array([fr[kws['TILEID']] for fr in all_frames], dtype=int)
    exposure = np.array([fr[kws['EXPID']] for fr in all_frames], dtype=int)
    ra = np.array([fr[kws['RA']] for fr in all_frames], dtype=float)
    dec = np.array([fr[kws['DEC']] for fr in all_frames], dtype=float)
    pa = np.array([fr[kws['PA']] for fr in all_frames], dtype=float)


    try:
        rec = np.flatnonzero(eval(request))
    except Exception as e:
        log.error(f"Something wrong with parsing request: {e}")
        return

    if len(rec) == 0:
        log.warning('Nothing selected')
        return

    log.info(f"Selected {len(rec)} frames")

    if file_save is None:
        file_save = os.path.join(os.curdir, 'LVM_config.toml')

    header = f"""# =======================================
# ==== General control of the workflow ====
# =======================================
nprocs = 10 # number of processes to use for parallelization
default_output_dir = "/data/LVM/Reduced/" # default directory where reduced data and products will be copied/saved. Could be overwritten by the second argument in of the script
drp_sas_version = '{drp_version}'
dap_sas_version = '{dap_version}'
drp_local_version = '{dap_local_version}'
server_group_id = {server_group_id}
        
[steps] # true = run step; false = skip
download = true  # check what to download in [download] section
update_metadata = false  # for drp only
reduction = false  # run drp locally
create_single_rss = false # only if you want to create a single RSS file with all fibers (includes sigma-clipping for fibers with same coordinates)
analyse_rss = false  # fit Gaussians or calculate moments and save table with fluxes/velocities/widths (see [imaging] and [imaging.lines] sections)
create_images = false  # create fits images for individual lines (see [imaging] and [imaging.lines] sections)
binning = false  # voronoi binning of the spectra (see [binning] section)
spectra_extraction = false  # extract spectra integrated in the desired ds9 region (see [extraction] section)
fit_with_dap = false # fit extracted or binned spectrum with DAP (see [dap_fitting] section)

create_cube = false  # create datacube, if really needed. This is not well-tested. If you really need data cube - better use Hector's script      
        
# =======================================
# === Control of downloading process
# =======================================
[download]
force_download = false # will not download files if they already exist
download_reduced = true # if you want to download reduced data from SAS. No needs to do update_metadata and reduction then
include_cframes = false # Set true if CFrames should be downloaded from SAS. By default, only SFrames will be downloaded
download_raw = false # set false if you want to download reduced files and agcam images only
download_agcam = false # set true if you want to download agcam images (they are necessary for data reduction, but not necessary if you work with already reduced frames)
download_dap = false
        
# =======================================
# === Control of data reduction
# =======================================
[reduction]
reduce_parallel = false # if true, then drp will run in parallel threads. This will use all processors on the server, so other users will not able to do anything. Avoid this unless you need this done urgently 
wham_sky_only = false # set true if you want to include only sky from the darkest sky patch (wham).
copy_cframes = false # Set true if CFrames should be copied to the object folder after reduction is done. By default, only SFrames will be copied 
only_copy_files = false # Set true to skip reduction and only copy already reduced files to the work dir 

# =======================================
# === Control of imaging in individual lines
# =======================================
[imaging]
use_dap = false
fallback_fluxcal = false  # "SCI" (default), 'MOD', 'STD' or false (to skip) -> if problems detected with the current fluxcal, this fallback will be used instead (just normalization will be applied)
force_calib = false # if true, then normalization to fallback_fluxcal will be applied to all exposures
include_sky = false # if true, then subtracted sky will be added back before the flux extraction
use_single_rss_file = false # if true, then creates single RSS file with all individual fibers and process this file.
use_binned_rss_file = false # if true, then process binned RSS file.  
override_flux_table = true # if true, then overwrites table with measured fluxes every time (otherwise - tries to add new/update existing measurements instead)
pxscale = 15 # Pixel scale (in arcsec) for output image
sigma = 2 # arcsec; standard deviation for calculating of the weights of neighbouringfibers
r_lim = 50 # arcsec; maximal radius to limit the fibers to be considered in flux calculation for a given position
fiber_pos_precision = 1.5 # Spectra from the fibers with positions deviating by less than this value (in arcsec) are combined with sigma-clipping before the analysis
lines_config = "lines_to_measure.toml"  # file with lines to measure (see example below)

# =======================================
# === Control for Voronoi binning
# =======================================
[binning]
use_binmap = false
maps_source = 'maps'
line = 'Ha_p'
pxscale = 10 # Pixel scale (in arcsec) for the map used for evaluating S/N in bins
rescale_noise = true # if true, then uses a tweak to upscale noise to make it consistent with the RSS spectra
sn_prefilter = 0 # minimal S/N in pre-filtering of pixels before binning
target_sn = 30
correct_vel_line = 'Ha'
mask_ds9_suffix = '_bin_exclude.reg'

# =======================================
# === Control for spectra extraction
# =======================================
[extraction]
file_ds9_suffix = '_ds9.reg'
mask_ds9_suffix = '_mask_ds9.reg'
correct_vel_line = 'Ha'
file_output_suffix = '_extracted.fits'

# =======================================
# === Control for DAP fitting
# =======================================
[dap_fitting]
fit_mode = 'rss' # can be 'extracted' (i.e. based on ds9-masks), 'binned' (i.e. extracted in voronoi bins) or 'rss' (i.e. huge single RSS file)
skip_running_dap = false # if true, then assume dap results already exist and only extract the measurements (useful if there were external changes in DAP results)


# =======================================
# ====== Setup for lines to measure =====
# =======================================

# !!! It is better to store line list in a separate file and include it here !!!
# !!! This config assumes there is a file 'lines_to_measure.toml' with the content similar to below. 
# !!! Also, it can be uncommented from here.
# # # # === Moment0 maps =====================
# [[imaging.lines]] # One block per line for output image
# line = 'Ha_mom0' # name of the line (arbitrary)
# wl_range = [6561, 6564] # rest-frame wavelength range for the line
# cont_range = [6530, 6540, 6605, 6620]  # rest-frame wavelength range for the continuum (w00, w01, w10, w11)
# 
# # # === Gaussian fits =====================
# 
# [[imaging.lines]]
# line = ["Ha", "NII6584"]
# wl_range = [6523, 6610]
# line_fit = [6562.78, 6583.5, 6548.1]
# fix_ratios = [-1, 1, 0.333]
# tie_disp = [false, true, true]
# tie_vel = [false, true, true]
# include_comp = [0, 1, -1] # -1 means not to include this component in the output
# filter_sn = 5 # minimal signal-to-noise ratio to consider during imaging


# =======================================
# === Limits For cube reconstruction, if needed =====
# =======================================
[cube_reconstruction]
wl_range = [4990, 5030]
suffix = 'OIII5007'

# =======================================
# ====== Setup for individual objects ====
# =======================================
"""

    u_mjd = np.unique(mjd[rec])
    f = open(file_save, 'w')

    f.write(f'{header}\n# ========= New Object =============\n[[object]]\n'
            f'name = "AutoGenerated" # Name of your object, will be used as folder in output directory\n'
            f'version = "v{drp_version}"# version of the files (perhaps, it is good to tag it to the version of the drp)\n'
            'velocity = 0 # Systemic velocity in km/s (used only for imaging to adjust the flux extraction window)\n'
            '\n#### New pointing ### Blocks for individual fiels for the same object\n'
            '[[object.pointing]]\nname = "P1" # Name of current pointing. '
            'Will be used as subfolder in the object directory.\n\n')

    for mjd_id, cur_mjd in enumerate(u_mjd):
        txt = "[[object.pointing.data]]"
        if mjd_id > 0:
            txt += "\n"
        else:
            txt += " # one block per all exposures for this pointing from single mjd \n"
        f.write(txt)
        f.write(f"mjd = {cur_mjd}\n")
        f.write(f"tileid = [{', '.join(tile_id[rec][mjd[rec] == cur_mjd].astype(str))}]\n")
        f.write(f"exp = [{', '.join(exposure[rec][mjd[rec] == cur_mjd].astype(str))}]\n")
        f.write("\n")

    f.close()
    log.info(f"TOML file saved to {file_save}")

    uid = os.stat(file_save).st_uid
    if server_group_id is not None and os.stat(file_save).st_gid != server_group_id:
        os.chown(file_save, uid=uid, gid=server_group_id)
    try:
        os.chmod(file_save, 0o664)
    except:
        log.error(f"Cannot change permissions for {file_save}")

if __name__ == "__main__":
    args = sys.argv
    drpall = False
    positional = []
    if len(args) < 2:
        log.error("Please provide valid request from lvmvis tool")

    else:
        for arg in args[1:]:
            if arg == "--drpall":
                drpall = True
            else:
                positional.append(arg)
        # Check required argument: query
        if len(positional) < 1:
                raise ValueError("Missing required argument: query")

        request = positional[0]
        file_save = positional[1] if len(positional) > 1 else None
        log.info(f"Generating TOML config file for LVM processing using the request: {request}")
        parse_tiles(request, file_save, drpall=drpall)