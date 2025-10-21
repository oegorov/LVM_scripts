#!/usr/bin/env python3

# Cut stacked rss data into smaller tiles for testing.

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python cut_rss.py <input_fits> <center_ra_deg> <center_dec_deg> <tile_size_arcmin>")
        sys.exit(1)

    input_fits = sys.argv[1]
    tile_size_arcmin = float(sys.argv[4])
    center_ra = float(sys.argv[2])
    center_dec = float(sys.argv[3])
    center_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame='icrs')

    # Load the RSS file
    with fits.open(input_fits) as hdul:
        t_fibs = Table(hdul['SLITMAP'].data)
        nfibs = len(t_fibs)
        rec = ((t_fibs['ra']-center_coord.ra.deg)**2 + (t_fibs['dec']-center_coord.dec.deg)**2) < ((tile_size_arcmin/60)**2)
        hdul['SLITMAP'] = fits.BinTableHDU(data=t_fibs[rec], name='SLITMAP', header=hdul['SLITMAP'].header)
        for kw in ('FLUX', 'IVAR', 'SKY', 'SKY_IVAR', "FLUX_SKYCORR", "LSF", "FLUX_ORIG"):
            if kw in hdul:
                hdul[kw] = fits.ImageHDU(data=hdul[kw].data[rec, :], name=kw, header=hdul[kw].header)
        hdul.writeto("cut_" + os.path.basename(input_fits), overwrite=True)
    print(f"Done: selected {np.sum(rec)} out of {nfibs} fibers.")

