# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:19:30 2020

@author: Alexander
"""

from osgeo import gdal, osr

import sys
import os

import numpy as np
import numpy.ma as ma


def getDataParams():

    # soil types
    k_coarse_sand = 1.0
    nom_a_k7 = 58.0

    k_fine_sand = 0.80
    nom_a_k6 = 52.0

    k_loamy_sand = 0.61
    nom_a_k5 = 45.0

    k_sandy_loam = 0.53
    nom_a_k4 = 41.0

    k_sandy_clay_sandy_loam = 0.43
    nom_a_k3 = 34.0

    k_clay_loam_sandy_clay = 0.33
    nom_a_k2 = 29.0

    k_loam = 0.21
    nom_a_k7 = 24.0

    soil_dict = {
        "coarse_sand": {"k": 1.0, "alpha": 58.0},
        "fine_sand": {"k": 0.80, "alpha": 52.0},
        "loamy_sand": {"k": 0.61, "alpha": 45.0},
        "sandy_loam": {"k": 0.53, "alpha": 41.0},
        "sandy_clay_sandy_loam": {"k": 0.43, "alpha": 34.0},
        "clay_loam_sandy_clay": {"k": 0.33, "alpha": 29.0},
        "loam": {"k": 0.21, "alpha": 24.0},
    }

    a9_limes = 45.0
    nom_a_i9 = 180.0 - 117

    a8_limes = 35.0
    nom_a_i8 = 180.0 - 122

    # slope ranges
    a7 = 25.0
    i7 = 0.47
    nom_a_i7 = 180.0 - 128

    a6 = 15.0
    i6 = 0.27
    nom_a_i6 = 180.0 - 135

    a5 = 10.0
    i5 = 0.1
    nom_a_i5 = 180.0 - 141

    a4 = 5.0
    i4 = 0.08
    nom_a_i4 = 180.0 - 150

    a3 = 2.0
    i3 = 0.035
    nom_a_i3 = 180.0 - 160

    a2 = 1.0
    i2 = 0.015
    nom_a_i2 = 180.0 - 165

    a1 = 0.5
    i1 = 0.01
    nom_a_i1 = 180.0 - 170

    x_sl = np.array([a1, a2, a3, a4, a5, a6, a7, a8_limes, a9_limes])
    y_sl = np.array(
        [
            nom_a_i1,
            nom_a_i2,
            nom_a_i3,
            nom_a_i4,
            nom_a_i5,
            nom_a_i6,
            nom_a_i7,
            nom_a_i8,
            nom_a_i9,
        ]
    )
    pred_sl_fun = np.polyfit(x_sl, y_sl, 5)

    pred_sl = np.poly1d(pred_sl_fun)

    soil_classes_list_arr = np.array([soil_dict[k]["alpha"] for k in soil_dict.keys()])

    return {"pred_sl": pred_sl, "soil_classes_list_arr": soil_classes_list_arr}


def degr_to_perc_slope(degrees):
    return (np.tan(np.radians(degrees))) * 100


def pre_vect_slope_angle_rel_y_val(slope_angle_degr, slopelen_m, slope_scale_factor):
    slope_rel = slopelen_m / slope_scale_factor
    alpha = getDataParams()["pred_sl"](slope_angle_degr)
    y = slope_rel * (np.tan(np.radians(alpha)))
    return y


def pre_vect_select_angle_for_soil_class(soil_class):
    return np.take(getDataParams()["soil_classes_list_arr"], soil_class - 1)


def pre_vect_soiltype_rel_x_val(soil_class, y, buffer_strip_scale):
    alpha = pre_vect_select_angle_for_soil_class(soil_class)
    beta = 90 - alpha
    x = y * (np.tan(np.radians(beta)))
    return x * buffer_strip_scale


def pre_vect_nomo_simple(
    slopelen_m, slope_angle_degr, soil_class, slope_scale_factor, buffer_strip_scale
):
    nomo_vals = pre_vect_soiltype_rel_x_val(
        soil_class,
        pre_vect_slope_angle_rel_y_val(
            slope_angle_degr, slopelen_m, slope_scale_factor
        ),
        buffer_strip_scale,
    )
    return nomo_vals


def nomograph(
    slope_raster_in,
    flowlen_raster_in,
    flowacc_raster_in,
    lsfactor_raster_in,
    soil_class_raster_in,
    nomo_catch_coeff,
    min_abs_buffer_size,
    slope_scale_factor,
    buffer_strip_scale,
    flowlen_max_value,
    flowacc_max_value,
    lsfactor_max_value,
    flowlen_weight,
    flowacc_weight,
    lsfactor_weight,
    dest_fname,
    dest_fname_slopelen,
    logarythm_switch=False,
):

    print("reading slope tif")
    slope_tif = gdal.Open(slope_raster_in)
    slope_nodata = slope_tif.GetRasterBand(1).GetNoDataValue()
    slope_band = slope_tif.GetRasterBand(1).ReadAsArray().astype(np.float64)

    geotrans = slope_tif.GetGeoTransform()

    height = slope_band.shape[0]
    width = slope_band.shape[1]

    print("reading flow length tif")
    flowlen_tif = gdal.Open(flowlen_raster_in)
    flowlen_nodata = flowlen_tif.GetRasterBand(1).GetNoDataValue()
    flowlen_band = flowlen_tif.GetRasterBand(1).ReadAsArray().astype(np.float64)

    print("reading flow accumulation tif")
    flowacc_tif = gdal.Open(flowacc_raster_in)
    flowacc_nodata = flowacc_tif.GetRasterBand(1).GetNoDataValue()
    flowacc_band = flowacc_tif.GetRasterBand(1).ReadAsArray().astype(np.float64)

    print("reading ls factor tif")
    lsfactor_tif = gdal.Open(lsfactor_raster_in)
    lsfactor_nodata = lsfactor_tif.GetRasterBand(1).GetNoDataValue()
    lsfactor_band = lsfactor_tif.GetRasterBand(1).ReadAsArray().astype(np.float64)

    print("reading soil class tif")
    soil_tif = gdal.Open(soil_class_raster_in)
    soil_nodata = soil_tif.GetRasterBand(1).GetNoDataValue()  # but also 0
    soil_band = soil_tif.GetRasterBand(1).ReadAsArray().astype(np.int64)

    slope_band_x = slope_band

    if logarythm_switch:
        print("applying pre-normalisation log10-smoothing")
        flowlen_band_log = np.log10(flowlen_band)
        flowlen_band = np.where(flowlen_band_log < 0, 0, flowlen_band_log)
        flowlen_max_value = np.log10(flowlen_max_value)

        flowacc_band_log = np.log10(flowacc_band)
        flowacc_band = np.where(flowacc_band_log < 0, 0, flowacc_band_log)
        flowacc_max_value = np.log10(flowacc_max_value)

        lsfactor_band_log = np.log10(lsfactor_band)
        lsfactor_band = np.where(lsfactor_band_log < 0, 0, lsfactor_band_log)
        lsfactor_max_value = np.log10(lsfactor_max_value)

    print("normalising flow length raster into 0-100 (min=0, max=1391.5403)")
    flowlen_band_x = (flowlen_band - 0) / (flowlen_max_value - 0) * 100

    print("normalising flow accumulation raster into 0-100 (min=0, max=21508)")
    flowacc_band_x = (flowacc_band - 0) / (flowacc_max_value - 0) * 100

    print("normalising LS factor raster into 0-100 (min=0, max=100)")
    lsfactor_band_x = (lsfactor_band - 0) / (lsfactor_max_value - 0) * 100

    # soil_band_x = ma.filled(soil_band, 0)
    soil_band_x = soil_band

    slope_nan = np.count_nonzero(np.isnan(slope_band_x))
    if slope_nan > 0:
        print(
            "slope tif has NaN values (e.g. infinite nodata or null), might cause problem"
        )

    flowlen_nan = np.count_nonzero(np.isnan(flowlen_band_x))
    if flowlen_nan > 0:
        print(
            "slope length tif has NaN values (e.g. infinite nodata or null), might cause problem"
        )

    flowacc_nan = np.count_nonzero(np.isnan(flowacc_band_x))
    if flowacc_nan > 0:
        print(
            "slope accumulation tif has NaN values (e.g. infinite nodata or null), might cause problem"
        )

    lsfactor_nan = np.count_nonzero(np.isnan(lsfactor_band_x))
    if lsfactor_nan > 0:
        print(
            "lsfactor tif has NaN values (e.g. infinite nodata or null), might cause problem"
        )

    soil_nan = np.count_nonzero(np.isnan(soil_band_x))
    if soil_nan > 0:
        print(
            "soil tif has NaN values (e.g. infinite nodata or null), might cause problem"
        )

    all_weights = flowlen_weight + flowacc_weight + lsfactor_weight
    print(
        """weighing rasters flow_length ({}), flow_acc ({}) and ls_factor ({})
        into 1-100 ({})""".format(
            flowlen_weight, flowacc_weight, lsfactor_weight, all_weights
        )
    )

    slope_len_band_x = (
        (
            (flowlen_band_x * flowlen_weight)
            + (flowacc_band_x * flowacc_weight)
            + (lsfactor_band_x * lsfactor_weight)
        )
        / all_weights
        * nomo_catch_coeff
    )
    slope_len_band_x_np = np.where(slope_len_band_x < 0, 0, slope_len_band_x)

    print("starting nomograph calculations")
    if min_abs_buffer_size is None:
        min_abs_buffer_size = 0
    elif min_abs_buffer_size < 0:
        min_abs_buffer_size = 0
    buf_recom_val = np.where(
        soil_band_x == 0,
        np.nan,
        pre_vect_nomo_simple(
            slope_len_band_x_np,
            slope_band_x,
            soil_band_x,
            slope_scale_factor,
            buffer_strip_scale,
        ),
    )
    buf_recom_val_np = np.where(
        buf_recom_val < min_abs_buffer_size, min_abs_buffer_size, buf_recom_val
    )

    print("writing final output raster for buffer strip size")
    buf_recom_val_x = np.nan_to_num(buf_recom_val_np, copy=False, nan=-1)

    driver = gdal.GetDriverByName("GTIFF")

    dataset = driver.Create(
        dest_fname, width, height, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"]
    )

    dataset.SetGeoTransform(geotrans)

    out_srs = osr.SpatialReference()
    out_srs.ImportFromWkt(slope_tif.GetProjectionRef())

    dataset.SetProjection(out_srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(
        buf_recom_val_x.astype(np.float32)
    )  # add ".T" if it's inverted.

    dataset.GetRasterBand(1).SetNoDataValue(-1.0)
    dataset.GetRasterBand(1).FlushCache()

    dataset = None

    print("writing final output raster for weighted specific slope length")

    slope_len_band_x_out = np.nan_to_num(slope_len_band_x_np, copy=False, nan=-1)

    dataset2 = driver.Create(
        dest_fname_slopelen,
        width,
        height,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE"],
    )
    dataset2.SetGeoTransform(geotrans)

    dataset2.SetProjection(out_srs.ExportToWkt())
    dataset2.GetRasterBand(1).WriteArray(
        slope_len_band_x_out.astype(np.float32)
    )  # add ".T" if it's inverted.

    dataset2.GetRasterBand(1).SetNoDataValue(-1.0)
    dataset2.GetRasterBand(1).FlushCache()

    dataset2 = None

    print("script finished")

    # Return the results of the algorithm, all be included in the returned
    # dictionary, with keys matching the feature corresponding parameter
    # or output names.
    return {"OUTPUT_BUF_SIZE": dest_fname, "OUTPUT_SPEC_SLOPELEN": dest_fname_slopelen}


def params_to_nomograph(params_dict):

    INPUT_SLOPE = params_dict["INPUT_SLOPE"]
    INPUT_FLOW_LENGTH = params_dict["INPUT_FLOW_LENGTH"]
    INPUT_FLOW_ACC = params_dict["INPUT_FLOW_ACC"]
    INPUT_LS_FACTOR = params_dict["INPUT_LS_FACTOR"]
    INPUT_SOIL_CLASS = params_dict["INPUT_SOIL_CLASS"]

    INPUT_NOMOGRAPH_CATCHMENTS_COEFFICIENT = params_dict[
        "INPUT_NOMOGRAPH_CATCHMENTS_COEFFICIENT"
    ]
    INPUT_MIN_ABS_BUFFER_SIZE = params_dict["INPUT_MIN_ABS_BUFFER_SIZE"]
    INPUT_SLOPE_SCALE_FACTOR = params_dict["INPUT_SLOPE_SCALE_FACTOR"]
    INPUT_BUFSTRIP_SCALE_FACTOR = params_dict["INPUT_BUFSTRIP_SCALE_FACTOR"]

    INPUT_FLOWLEN_MAX = params_dict["INPUT_FLOWLEN_MAX"]
    INPUT_FLOWACC_MAX = params_dict["INPUT_FLOWACC_MAX"]
    INPUT_LSFACTOR_MAX = params_dict["INPUT_LSFACTOR_MAX"]

    INPUT_FLOWLEN_WEIGHT = params_dict["INPUT_FLOWLEN_WEIGHT"]
    INPUT_FLOWACC_WEIGHT = params_dict["INPUT_FLOWACC_WEIGHT"]
    INPUT_LSFACTOR_WEIGHT = params_dict["INPUT_LSFACTOR_WEIGHT"]

    INPUT_LOGARYTHM_SWITCH = params_dict["INPUT_LOGARYTHM_SWITCH"]

    OUTPUT_BUF_SIZE = params_dict["OUTPUT_BUF_SIZE"]
    OUTPUT_SPEC_SLOPELEN = params_dict["OUTPUT_SPEC_SLOPELEN"]

    return nomograph(
        slope_raster_in=INPUT_SLOPE,
        flowlen_raster_in=INPUT_FLOW_LENGTH,
        flowacc_raster_in=INPUT_FLOW_ACC,
        lsfactor_raster_in=INPUT_LS_FACTOR,
        soil_class_raster_in=INPUT_SOIL_CLASS,
        nomo_catch_coeff=INPUT_NOMOGRAPH_CATCHMENTS_COEFFICIENT,
        min_abs_buffer_size=INPUT_MIN_ABS_BUFFER_SIZE,
        slope_scale_factor=INPUT_SLOPE_SCALE_FACTOR,
        buffer_strip_scale=INPUT_BUFSTRIP_SCALE_FACTOR,
        flowlen_max_value=INPUT_FLOWLEN_MAX,
        flowacc_max_value=INPUT_FLOWACC_MAX,
        lsfactor_max_value=INPUT_LSFACTOR_MAX,
        flowlen_weight=INPUT_FLOWLEN_WEIGHT,
        flowacc_weight=INPUT_FLOWACC_WEIGHT,
        lsfactor_weight=INPUT_LSFACTOR_WEIGHT,
        dest_fname=OUTPUT_BUF_SIZE,
        dest_fname_slopelen=OUTPUT_SPEC_SLOPELEN,
        logarythm_switch=INPUT_LOGARYTHM_SWITCH,
    )


if __name__ == "__main__":

    input_params = {
        "INPUT_SLOPE": r"c:\temp\slope_5m_pzone_21.tif",
        "INPUT_FLOW_LENGTH": r"c:\temp\flowlength_5m_pzone_21.tif",
        "INPUT_FLOW_ACC": r"c:\temp\flowacc_5m_pzone_21.tif",
        "INPUT_LS_FACTOR": r"c:\temp\ls_faktor5m_pzone_21_cog.tif",
        "INPUT_SOIL_CLASS": r"c:\temp\estsoil_labeled_pzone_21.tif",
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
        "INPUT_LOGARYTHM_SWITCH": True,
        "OUTPUT_BUF_SIZE": r"c:\temp\TEMPORARY_OUTPUT_BUF_SIZE_log10.tif",
        "OUTPUT_SPEC_SLOPELEN": r"c:\temp\TEMPORARY_OUTPUT_SPEC_SLOPELEN_log10.tif",
    }

    nomo_out = params_to_nomograph(input_params)

    print(nomo_out)
