# LVM_scripts
**Disclaimer: These scripts are intended for simplifying/automatizing work with the LVM data and performing some advanced analysis of these data at our local server at ARI. However, they can be easily adapted for other machines, if necessary.**

*LMV_process.py* script is used for processing the LVM data on the local computing cluster. 
For this purpose, this script takes the information about tiles/exposures of interest from TOML config file and processes them through several steps (any step can be skipped):
- Downloading data from SAS
  - User can select to download Raw, Reduced data, DAP results and/or AGCam images
- Reducing data with the actual version of drp
- Analysing SFrames or DAP results:
  - Combining the spectra with coinciding fiber positions (SFrames)
  - Fitting selected emission lines with Gaussians, and/or summing the spectrum (SFrames)
  - Applying heliocentric correction and subtracting the LSF from the kinematical data (both SFrames and DAP)
  - Optionally: correct the fluxes in individual exposures to account for the potential imperfections of sky subtraction/flux calibration
  - Creating a big table (both SFrames and DAP) or single RSS file (SFrames only) containing measurements or spectra at all unique fiber positions for the object/region of interest
- Creates images in emission lines (fluxes, velocity, dispersion), or data cube, at the rectangular homogeneous grid with the selected resolution



*LVM_config.toml* shows an example of the TOML config file

## Requirements and usage 
#### These libraries must be installed: 
lmfit, tomllib, lvmdrp, sdss-access, tqdm  + standard astronomical libraries (astropy, numpy, scipy, matplotlib) 

`pip install lmfit tomllib sdss-access tqdm astropy` must work + check how to install the latest version of lvmdrp

#### Modify code + setup the folder structure

1. Raw or reduced/analysed files will be downloaded from SAS and stored at `$SAS_BASE_DIR/sdsswork/...` (path mimicking the structure at the SAS). Thus make sure that the same structure exists on your laptop/server for Raw data, AGCam images, Reduced data and DAP results
2. Define all environmental variables necessary for `lvmdrp`, including the root directory `$SAS_BASE_DIR`
3. Setup access to the SDSS data through the `.netrc` file (see wiki)
4. Change few lines of the code in the `=== Setup ===` block of the **LVM_process.py**:
   1. `red_data_version` - current version of the reduced data on SAS
   2. `dap_version` - current version of the DAP-analysed data on SAS
   3. `drp_version` - your LOCAL version of the lvmdrp
   4. `server_group_id` - ID of the group to be used for changing access to the new files and folders. If you wish to run `chgrp` to all newly created files and folders, provide the correct group ID according to the setup on your machine. Use None to skip this.

If everything is correctly installed/setup, then just type `LVM_process.py LVM_config.toml` - and wait for the results. 
 
**More description is TBA at some point...** 
