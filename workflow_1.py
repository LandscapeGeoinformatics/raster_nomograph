# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:19:30 2020

@author: Alexander
"""

from osgeo import gdal, osr

import sys
import os

import numpy as np
import numpy.ma as ma

import sys
import os

import pprint

import buff_width_calc_vectorized_v4

# /media/rocket_gis/kmoch/nomograph/soil_prep/slope_5m_pzone_21.tif
# /media/rocket_gis/kmoch/nomograph/soil_prep/flowlength_5m_pzone_21.tif
# /media/rocket_gis/kmoch/nomograph/soil_prep/flowacc_5m_pzone_21.tif
# /media/rocket_gis/kmoch/nomograph/soil_prep/ls_faktor5m_overz_21.tif
# /media/rocket_gis/kmoch/nomograph/soil_prep/estsoil_labeled_pzone_21.tif
source_dir = "R:/kmoch/nomograph/soil_prep"

# veekaitsekonnd_dis_21.gpkg
veekaits_dir = "R:/kmoch/nomograph/tests/veekaitsekoond"

# pzone xx
# buffer_size  specific_slopelength
# default with_log
# _  clipped

# buffer_size_default_pzone_21.tif
# specific_slopenlength_default_pzone_21.tif

# buffer_size_with_log_pzone_21.tif
# specific_slopenlength_with_log_pzone_21.tif

# buffer_size_default_clipped_pzone_21.tif
# specific_slopenlength_default_clipped_pzone_21.tif

# buffer_size_with_log_clipped_pzone_21.tif
# specific_slopenlength_with_log_clipped_pzone_21.tif
dest_dir = "R:/kmoch/nomograph/tests/v4"


def make_filenames(pzone=1, logarythm_switch=False, clipped=False):

    base_buf_name = "buffer_size"
    base_spec_slopelen = "specific_slopenlength"
    log_def = "default" if not logarythm_switch else "with_log"
    clipped_add = "" if not clipped else "clipped"

    dest_fname = os.path.join(
        dest_dir,
        f"{base_buf_name}_{log_def}_{clipped_add}_pzone_{pzone}.tif".replace("__", "_"),
    )

    dest_fname_slopelen = os.path.join(
        dest_dir,
        f"{base_spec_slopelen}_{log_def}_{clipped_add}_pzone_{pzone}.tif".replace(
            "__", "_"
        ),
    )

    slope_raster_in = os.path.join(source_dir, f"slope_5m_pzone_{pzone}.tif")
    flowlen_raster_in = os.path.join(source_dir, f"flowlength_5m_pzone_{pzone}.tif")
    flowacc_raster_in = os.path.join(source_dir, f"flowacc_5m_pzone_{pzone}.tif")
    lsfactor_raster_in = os.path.join(source_dir, f"ls_faktor5m_pzone_{pzone}_cog.tif")
    soil_class_raster_in = os.path.join(
        source_dir, f"estsoil_labeled_pzone_{pzone}.tif"
    )

    veekaitse_gpkg = os.path.join(veekaits_dir, f"veekaitsekonnd_dis_{pzone}.gpkg")

    return {
        "inputs": {
            "INPUT_SLOPE": slope_raster_in,
            "INPUT_FLOW_LENGTH": flowlen_raster_in,
            "INPUT_FLOW_ACC": flowacc_raster_in,
            "INPUT_LS_FACTOR": lsfactor_raster_in,
            "INPUT_SOIL_CLASS": soil_class_raster_in,
            "CLIP_CUTLINE": veekaitse_gpkg,
        },
        "outputs": {
            "OUTPUT_BUF_SIZE": dest_fname,
            "OUTPUT_SPEC_SLOPELEN": dest_fname_slopelen,
        },
    }


def check_files_ex(fdict, inout="inputs"):
    all_good = True

    subd = fdict[inout]

    for k, v in subd.items():
        if os.path.exists(v):
            print(f"{k} - {v} OK")
        else:
            all_good = False
            print(f"{k} - {v} error")

    return all_good


def make_params(pzone=1, logarythm_switch=False, clipped=False):

    fnames = make_filenames(
        pzone=pzone, logarythm_switch=logarythm_switch, clipped=clipped
    )

    all_good = check_files_ex(fnames, "inputs")

    input_params = {
        "INPUT_SLOPE": fnames["inputs"]["INPUT_SLOPE"],
        "INPUT_FLOW_LENGTH": fnames["inputs"]["INPUT_FLOW_LENGTH"],
        "INPUT_FLOW_ACC": fnames["inputs"]["INPUT_FLOW_ACC"],
        "INPUT_LS_FACTOR": fnames["inputs"]["INPUT_LS_FACTOR"],
        "INPUT_SOIL_CLASS": fnames["inputs"]["INPUT_SOIL_CLASS"],
        "INPUT_NOMOGRAPH_CATCHMENTS_COEFFICIENT": 210,
        "INPUT_MIN_ABS_BUFFER_SIZE": 0,
        "INPUT_SLOPE_SCALE_FACTOR": 10000,
        "INPUT_BUFSTRIP_SCALE_FACTOR": 50,
        "INPUT_FLOWLEN_MAX": 1392,
        "INPUT_FLOWACC_MAX": 21508,
        "INPUT_LSFACTOR_MAX": 100,
        "INPUT_FLOWLEN_WEIGHT": 3,
        "INPUT_FLOWACC_WEIGHT": 3,
        "INPUT_LSFACTOR_WEIGHT": 3,
        "INPUT_LOGARYTHM_SWITCH": logarythm_switch,
        "OUTPUT_BUF_SIZE": fnames["outputs"]["OUTPUT_BUF_SIZE"],
        "OUTPUT_SPEC_SLOPELEN": fnames["outputs"]["OUTPUT_SPEC_SLOPELEN"],
    }

    return (input_params, fnames)


def clip_cutline(params_base, params_clip, buf_or_slope=0):

    in_buf = params_base["outputs"]["OUTPUT_BUF_SIZE"]
    out_buf = params_clip["outputs"]["OUTPUT_BUF_SIZE"]
    in_sloplen = params_base["outputs"]["OUTPUT_SPEC_SLOPELEN"]
    out_sloplen = params_clip["outputs"]["OUTPUT_SPEC_SLOPELEN"]
    clip_file = params_clip["inputs"]["CLIP_CUTLINE"]
    # clip_layer = params_clip["inputs"]["CLIP_CUTLINE"].replace(".gpkg", "")

    # TODO
    print("clipping baseline raster to water protection zone")

    # gdalwarp -of GTiff -cutline C:/dev/05_geodata/dem/processing_zones_single/veekaitsekonnd_dis_21.gpkg -cl veekaitsekonnd_dis_21 -crop_to_cutline -co COMPRESS=PACKBITS C:/temp/QGIS_OUTPUT_BUF_SIZE.tif C:\temp/processing/OUTPUT.tif

    warp_opts = gdal.WarpOptions(
        cutlineDSName=clip_file,
        # cutlineLayer=clip_layer,
        cropToCutline=True,
        creationOptions=["COMPRESS=LZW"],
    )

    try:
        if buf_or_slope == 0:
            clip_result = gdal.Warp(out_buf, in_buf, options=warp_opts)
            return {"outputs": {"OUTPUT_BUF_SIZE": out_buf}}

            params_clip["outputs"]
        if buf_or_slope == 1:
            clip_result = gdal.Warp(out_sloplen, in_sloplen, options=warp_opts)
            return {"outputs": {"OUTPUT_SPEC_SLOPELEN": out_sloplen}}
        if buf_or_slope == 2:
            clip_result = gdal.Warp(out_buf, in_buf, options=warp_opts)

            clip_result = gdal.Warp(out_sloplen, in_sloplen, options=warp_opts)
            return {
                "outputs": {
                    "OUTPUT_BUF_SIZE": out_buf,
                    "OUTPUT_SPEC_SLOPELEN": out_sloplen,
                }
            }
        else:
            return False

    except Exception as ex:
        print(ex)
        return False


if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=2)

    # known is already 19
    clip_errs = []

    # 3, 12, 13, 19, 21
    already_done = set([1, 3, 12, 13, 19, 21])
    todo_nums = [i for i in range(1,23) if not i in already_done]
    for i in todo_nums:

        # operations
        # 1) nomo default
        # 2) clip defaults

        # 3) nomo with_log
        # 4) clip with_log

        with_log_switch = False
        p1, fnames1 = make_params(
            pzone=i, logarythm_switch=with_log_switch, clipped=False
        )
        pp.pprint(p1)

        nomo_out = buff_width_calc_vectorized_v4.params_to_nomograph(p1)
        pp.pprint(nomo_out)
        check_files_ex(fnames1, "outputs")

        fnames2 = make_filenames(
            pzone=i, logarythm_switch=with_log_switch, clipped=True
        )
        pp.pprint(fnames2)

        clip_res = clip_cutline(fnames1, fnames2)
        if isinstance(clip_res, bool):
            if not clip_res:
                clip_errs.append((i, fnames1["outputs"]))
        else:
            check_files_ex(clip_res, "outputs")

        # ----------------------------------------------#

        with_log_switch = True
        p3, fnames3 = make_params(
            pzone=i, logarythm_switch=with_log_switch, clipped=False
        )
        pp.pprint(p3)

        nomo_out = buff_width_calc_vectorized_v4.params_to_nomograph(p3)
        pp.pprint(nomo_out)
        check_files_ex(fnames3, "outputs")

        fnames4 = make_filenames(
            pzone=i, logarythm_switch=with_log_switch, clipped=True
        )
        pp.pprint(fnames4)

        clip_res2 = clip_cutline(fnames3, fnames4)
        if isinstance(clip_res2, bool):
            if not clip_res:
                clip_errs.append((i, fnames3["outputs"]))
        else:
            check_files_ex(clip_res2, "outputs")
        
        print("cutline clip errors:")
        pp.pprint(clip_errs)
