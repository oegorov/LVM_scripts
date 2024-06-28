#!/usr/bin/env python3
import sys, os
import logging
import json
import numpy as np

log = logging.getLogger(name='CreateTomlFile')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

server_group_id = 10699
json_file = '/data/LVM/lvm.json'

def parse_tiles(request, file_save=None):
    try:
        request = request.lower()
        request = request.replace('and', '&')
        request = request.replace('or', '|')
    except Exception as e:
        log.error(f"Something wrong with parsing request: {e}")
        return
    print(request)
    try:
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()
        parsed_json = json.loads(json_string)
        all_frames = parsed_json["data"]
    except Exception as e:
        log.error(f"Something wrong with json file: {e}")
        return

    mjd = np.array([fr['mjd'] for fr in all_frames])
    tile_id = np.array([fr['tile_id'] for fr in all_frames])
    exposure = np.array([fr['exposure'] for fr in all_frames])
    ra = np.array([fr['ra'] for fr in all_frames])
    dec = np.array([fr['dec'] for fr in all_frames])
    pa = np.array([fr['pa'] for fr in all_frames])
    idx = np.array([fr['idx'] for fr in all_frames])

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

    header = """# =======================================
# ==== General control of the workflow ====
# =======================================
nprocs = 20 # number of processes to use for parallelization
default_output_dir = "/data/LVM/Reduced/" # default directory where reduced data and products will be copied/saved. Could be overwritten by the second argument in of the script
keep_existing_single_rss = false # needed only for re-creation of a single RSS file with all exposures. If true - considers the existing file as container (allows to save time)
        
[steps] # true = run step; false = skip
download = true
update_metadata = false
reduction = false
create_single_rss = false # only if you want to create a single RSS file with all fibers (includes sigma-clipping for fibers with same coordinates)
analyse_rss = false
create_images = false
        
# ==== outdated (do not use)
# check_noise_level = false
# coadd_spectra = false
# imaging = false
# create_cube = false
        
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
        
# =======================================
# === Control of imaging in individual lines
# =======================================
[imaging]
use_dap = false
interpolate = true # use interpolation (shepard's method)
skip_bad_fibers = true # skip fibers classified as bad (only for shepard's method)
include_sky = false # if true, then subtracted sky will be added back before the flux extraction
partial_sky = false # if true, then subtracted sky will be removed back at the location of the emission lines. 
use_single_rss_file = false # if true, then creates single RSS file with all individual fibers and process this file. 
override_flux_table = true # if true, then overwrites table with measured fluxes every time (otherwise - tries to add new/update existing measurements instead)
pxscale = 5 # Pixel scale (in arcsec) for output image
sigma = 2 # arcsec; standard deviation for calculating of the weights of neighbouringfibers
r_lim = 50 # arcsec; maximal radius to limit the fibers to be considered in flux calculation for a given position
fiber_pos_precision = 1.5 # Spectra from the fibers with positions deviating by less than this value (in arcsec) are combined with sigma-clipping before the analysis
        
# === Sum
[[imaging.lines]] # One block per line for output image
line = 'Ha_mom0' # name of the line (arbitrary)
wl_range = [6561, 6564]
cont_range = [6530, 6540, 6605, 6620]
median_filter = false
        
# =======================================
# ====== Setup for individual objects ====
# =======================================
        
"""

    u_mjd = np.unique(mjd[rec])
    f = open(file_save, 'w')

    f.write(f'{header}[[object]]\nname = "AutoGenerated"\nvelocity = 0\n\n[[object.pointing]]\nname = "none"\n')

    for cur_mjd in u_mjd:
        f.write("[[object.pointing.data]]\n")
        f.write(f"mjd = {cur_mjd}\n")
        f.write(f"tileid = [{', '.join(tile_id[rec][mjd[rec] == cur_mjd].astype(str))}]\n")
        f.write(f"exp = [{', '.join(exposure[rec][mjd[rec] == cur_mjd].astype(str))}]\n")
        f.write("\n")

    f.write('[object.pointing.skip]\ndownload = false\nreduction = false\nimaging = false\ncreate_cube = false\n')

    f.close()
    uid = os.stat(file_save).st_uid
    if server_group_id is not None:
        os.chown(file_save, uid=uid, gid=server_group_id)
    os.chmod(file_save, 0o664)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        log.error("Please provide valid request from lvmvis tool")

    else:
        request = args[1]
        if len(args) == 2:
            file_save = None
        else:
            file_save = args[2]
        parse_tiles(request, file_save)