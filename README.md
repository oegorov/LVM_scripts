# LVM_scripts
**Disclaimer: These scripts are intended for simplifying/automatizing work with the LVM data at our local server at ARI and probably are not very useful for other people. In principle, they can be easily adapted for other machines, if necessary.**

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

**More description is TBA at some point...** 
