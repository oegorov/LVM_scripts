import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.ndimage import median_filter
import logging as log
from tqdm import tqdm

def create_line_image_from_table(file_fluxes=None, columns_to_show=None, pxscale_out=3., r_lim=50, sigma=2.,
                                output_dir=None, do_median_masking=False,
                                outfile_prefix=None, ra_lims=None, dec_lims=None, skip_zeros=True):
    """Create line images from the table of fluxes.
    Parameters
    ----------
    file_fluxes : str
        Path to the fits file with fluxes
    columns_to_show : list
        Columns to show
    pxscale_out : float
        Outupt pixel scale in arcsec/pixel (default: 3)
    r_lim : float
        Radius limit in arcsec for convolution (default: 50)
    sigma : float
        Sigma for Gaussian kernel (default: 2)
    output_dir : str
        Output directory
    do_median_masking : bool
        Apply median masking (default: False)
    outfile_prefix : str
        Output file prefix
    ra_lims : list
        RA limits in degrees (default: None)
    dec_lims : list
        Dec limits in degrees (default: None)
    skip_zeros : bool
        Skip zeros in the table (default: True)
    """

    lvm_fiber_diameter = 35.3
    if not os.path.exists(file_fluxes):
        log.error(f"File {file_fluxes} does not exist.")
        return False
    table_fluxes = Table.read(file_fluxes, format='fits')
    ras = table_fluxes['RA']
    decs = table_fluxes['Dec']

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

    # Create a new WCS object.  The number of axes must be set from the beginning
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

    values = None
    masks = None
    names_out = []

    for cl_id, cur_col in enumerate(columns_to_show):
        if cur_col not in table_fluxes.colnames:
            log.warning(f"Column {cur_col} not found in the table.")
            continue
        cur_masks = np.isfinite(table_fluxes[cur_col])
        if skip_zeros:
            cur_masks = cur_masks & (table_fluxes[cur_col] != 0)
        if values is None:
            if cur_col in table_fluxes.colnames:
                values = table_fluxes[cur_col] #/ (np.pi * lvm_fiber_diameter ** 2 / 4)
                masks = np.copy(cur_masks)
                names_out.append(cur_col)
        else:
            if cur_col in table_fluxes.colnames:
                values = np.vstack([values.T, table_fluxes[cur_col].T]).T #/ (np.pi * lvm_fiber_diameter ** 2 / 4)
                masks = np.vstack([masks.T, cur_masks.T]).T
                names_out.append(cur_col)
            else:
                continue

    if values is None:
        log.error('Nothing to show.')
        return False

    if len(values.shape) == 1:
        values = values.reshape((-1, 1))
    img_arr = shepard_convolve(wcs_out, shape_out, ra_fibers=ras, dec_fibers=decs, show_values=values,
                               r_lim=r_lim, sigma=sigma, masks=masks)

    header = wcs_out.to_header()
    for ind in range(img_arr.shape[2]):
        outfile_suffix = names_out[ind]
        if do_median_masking:
            img_arr[:, :, ind] = median_filter(img_arr[:, :, ind], (11, 11))
        f_out = os.path.join(output_dir, f"{outfile_prefix}_{outfile_suffix}.fits")
        fits.writeto(f_out, data=img_arr[:, :, ind], header=header, overwrite=True)


def shepard_convolve(wcs_out, shape_out, ra_fibers=None, dec_fibers=None, show_values=None, r_lim=50., sigma=2.,
                     cube=False, header=None, outfile=None, do_median_masking=False, masks=None, remove_empty=False):
    if len(show_values.shape) ==1:
        show_values = show_values.reshape((-1, 1))
        masks = masks.reshape((-1,1))
    if masks is None:
        masks = np.tile(np.isfinite(show_values[:, 0]) & (show_values[:, 0] != 0), show_values.shape) # or tile??
        if len(masks.shape) == 1:
            masks = masks.reshape((-1, 1))
    rec_fibers = np.any(masks, axis=1)
    masks = masks[rec_fibers]
    show_values = show_values[rec_fibers,:]

    pxsize = wcs_out.proj_plane_pixel_scales()[0].value * 3600.
    radec = SkyCoord(ra=ra_fibers[rec_fibers], dec=dec_fibers[rec_fibers], unit='deg', frame='icrs')
    x_fibers, y_fibers = wcs_out.world_to_pixel(radec)
    chunk_size = min([int(r_lim * 5 / pxsize), 50])
    khalfsize = int(np.ceil(r_lim / pxsize))
    kernel_size = khalfsize * 2 + 1

    xx_kernel, yy_kernel = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))

    nchunks_x = int(np.ceil(shape_out[1]/chunk_size))
    nchunks_y = int(np.ceil(shape_out[0] / chunk_size))

    img_out = np.zeros(shape=(shape_out[0], shape_out[1], show_values.shape[1]), dtype=float)

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

        weights = np.zeros(shape=(len(rec_fibers), ny, nx, show_values.shape[-1]), dtype=float)

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
            weights[ind, cur_y0:cur_y1+1, cur_x0: cur_x1+1, :] = (
                                                                         kernel)[khalfsize - (cur_center_y-cur_y0):
                                                                                 khalfsize - (cur_center_y-cur_y0)+ cur_y1 - cur_y0+1,
                                                                     khalfsize - (cur_center_x-cur_x0):
                                                                     khalfsize - (cur_center_x-cur_x0)+cur_x1 - cur_x0+1,
                                                                     None] * (masks[rec_fibers[ind],None,None,:]).astype(float)

        weights_norm = np.sum(weights, axis=0)
        weights_norm[weights_norm == 0] = 1
        weights = weights / weights_norm[None, :, :, :]

        n_used_fib = np.sum(weights > 0, axis=0)
        n_used_fib = np.broadcast_to(n_used_fib, weights.shape)
        weights[n_used_fib < 2] = 0
        if remove_empty:
            weights[weights == 0] = np.nan
        img_chunk = np.nansum(weights[:, :, :, :] * show_values[rec_fibers, None, None, :], axis=0)

        img_out[last_y + 1: last_y + 1 + ny - dy0 - dy1,
            last_x + 1: last_x + 1 + nx - dx0 - dx1, :] = img_chunk[dy0: ny - dy1, dx0: nx - dx1, :]


        last_y += chunk_size
        if last_y >= shape_out[0]:
            last_x += chunk_size
            last_y = -1

    return img_out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create line images from the table of fluxes.')
    parser.add_argument('file_fluxes', type=str, help='Path to the file with fluxes')
    parser.add_argument('columns_to_show', type=str, nargs='+', help='Columns to show')
    parser.add_argument('--pxscale_out', type=float, default=3., help='Pixel scale in arcsec/pixel')
    parser.add_argument('--r_lim', type=float, default=50., help='Radius limit in arcsec')
    parser.add_argument('--sigma', type=float, default=2., help='Sigma for Gaussian kernel')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--do_median_masking', action='store_true', help='Apply median masking')
    parser.add_argument('--outfile_prefix', type=str, default='', help='Output file prefix')
    parser.add_argument('--ra_lims', type=float, nargs=2, default=None, help='RA limits in degrees')
    parser.add_argument('--dec_lims', type=float, nargs=2, default=None, help='Dec limits in degrees')
    parser.add_argument('--skip_zeros', action='store_true', help='Skip zeros in the table')

    args = parser.parse_args()

    create_line_image_from_table(file_fluxes=args.file_fluxes,
                                 columns_to_show=args.columns_to_show,
                                 pxscale_out=args.pxscale_out,
                                 r_lim=args.r_lim,
                                 sigma=args.sigma,
                                 output_dir=args.output_dir,
                                 do_median_masking=args.do_median_masking,
                                 outfile_prefix=args.outfile_prefix,
                                 ra_lims=args.ra_lims,
                                 dec_lims=args.dec_lims,
                                 skip_zeros=args.skip_zeros)