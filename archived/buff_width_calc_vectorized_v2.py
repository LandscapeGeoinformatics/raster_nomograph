# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:19:30 2020

@author: Alexander
"""

import sys
import os

import numpy as np

import numpy.ma as ma
import rasterio

conf1 = {

    "work_path" : r'c:/dev/05_geodata/qgis_map_projects/kik_project',
    "slope_raster_in" : 'slope_1m_valgjarve.tif',
    "slope_len_raster_in" : 'slope_length_1m_valgjarve.tif',
    "soil_class_raster_in" : 'soil_1_valgjarve.tif',
    "buff_width_raster_out" : 'buffer_strip_width_1m_valgjarve_vect.tif'
}

conf2_grass = {

    "work_path" : r'c:/dev/05_geodata/qgis_map_projects/kik_project',
    "slope_raster_in" : 'slope_degrees_3301.tif',
    "slope_len_raster_in" : 'flow_length_grass_3301.tif',
    "soil_class_raster_in" : 'soil_text_class.tif',
    "buff_width_raster_out" : 'buffer_strip_width_flowlength_grass1.tif'
}

# for log flow_acc uncomment line 203+
conf3_saga1 = {

    "work_path" : r'c:/dev/05_geodata/qgis_map_projects/kik_project',
    "slope_raster_in" : 'slope_degrees_3301.tif',
    "slope_len_raster_in" : 'flow_distance_saga1_3301.tif',
    "soil_class_raster_in" : 'soil_text_class.tif',
    "buff_width_raster_out" : 'buffer_strip_width_flowdistance_saga1.tif'
}

conf3_saga2 = {

    "work_path" : r'c:/dev/05_geodata/qgis_map_projects/kik_project',
    "slope_raster_in" : 'slope_degrees_3301.tif',
    "slope_len_raster_in" : 'flow_acc_saga1_3301.tif',
    "soil_class_raster_in" : 'soil_text_class.tif',
    "buff_width_raster_out" : 'buffer_strip_width_log_e_flowacc_saga1.tif'
}

cf = conf3_saga2

work_path = cf['work_path']
slope_raster_in = os.path.join(work_path, cf['slope_raster_in'])
slope_len_raster_in = os.path.join(work_path, cf['slope_len_raster_in'])
soil_class_raster_in = os.path.join(work_path, cf['soil_class_raster_in'])
buff_width_raster_out = os.path.join(work_path, cf['buff_width_raster_out'])

# Specific slope length f (m) 2000 4000 6000 8000 10000
slope_scale_factor=2000

# Recommendable buffer strip's width P (m) 10 20 30 40 50
buffer_strip_scale=10

# soil types
k_coarse_sand=1.0
nom_a_k7=58.0

k_fine_sand=0.80
nom_a_k6=52.0

k_loamy_sand=0.61
nom_a_k5=45.0

k_sandy_loam=0.53
nom_a_k4=41.0

k_sandy_clay_sandy_loam=0.43
nom_a_k3=34.0

k_clay_loam_sandy_clay=0.33
nom_a_k2=29.0

k_loam=0.21
nom_a_k7=24.0

soil_dict = {
    "coarse_sand" : {"k":1.0,
                     "alpha":58.0
                    },
    "fine_sand" : {"k":0.80,
                   "alpha":52.0
                  },
    "loamy_sand" : {"k":0.61,
                    "alpha":45.0
                   },
    "sandy_loam" : {"k":0.53,
                    "alpha":41.0
                   },
    "sandy_clay_sandy_loam" : {"k":0.43,
                               "alpha":34.0
                              },
    "clay_loam_sandy_clay" : {"k":0.33,
                              "alpha":29.0
                             },
    "loam" : {"k":0.21,
              "alpha":24.0}
}

a9_limes=45.0
nom_a_i9=180.0-117

a8_limes=35.0
nom_a_i8=180.0-122

# slope ranges
a7=25.0
i7=0.47
nom_a_i7=180.0-128

a6=15.0
i6=0.27
nom_a_i6=180.0-135

a5=10.0
i5=0.1
nom_a_i5=180.0-141

a4=5.0
i4=0.08
nom_a_i4=180.0-150

a3=2.0
i3=0.035
nom_a_i3=180.0-160

a2=1.0
i2=0.015
nom_a_i2=180.0-165

a1=0.5
i1=0.01
nom_a_i1=180.0-170

x_sl = np.array([a1, a2, a3, a4, a5, a6, a7, a8_limes, a9_limes])
y_sl = np.array([nom_a_i1, nom_a_i2, nom_a_i3, nom_a_i4, nom_a_i5, nom_a_i6, nom_a_i7, nom_a_i8, nom_a_i9])
pred_sl_fun = np.polyfit(x_sl, y_sl, 5)
pred_sl = np.poly1d(pred_sl_fun)

soil_classes_list_arr = np.array([soil_dict[k]['alpha'] for k in soil_dict.keys()])


def degr_to_perc_slope(degrees):
    return (np.tan(np.radians(degrees)))*100


def pre_vect_slope_angle_rel_y_val(slope_angle_degr, slope_len_m):
    slope_rel=slope_len_m/slope_scale_factor
    alpha = pred_sl(slope_angle_degr)
    y = slope_rel * (np.tan(np.radians(alpha)))
    return y


def pre_vect_select_angle_for_soil_class(soil_class):
    return np.take(soil_classes_list_arr, soil_class-1)


def pre_vect_soiltype_rel_x_val(soil_class, y):
    alpha=pre_vect_select_angle_for_soil_class(soil_class)
    beta = 90-alpha
    x = y * (np.tan(np.radians(beta)))
    return x * buffer_strip_scale


def pre_vect_nomo_simple(slope_len_m, slope_angle_degr, soil_class):
    nomo_vals = pre_vect_soiltype_rel_x_val(soil_class, pre_vect_slope_angle_rel_y_val(slope_angle_degr, slope_len_m))
    return nomo_vals


print('reading slope tif')
slope_tif = rasterio.open(slope_raster_in)
slope_nodata = slope_tif.nodata
slope_band = slope_tif.read(1, masked=True)

print('reading slope length tif')
slope_len_tif = rasterio.open(slope_len_raster_in)
slope_len_nodata = slope_len_tif.nodata
slope_len_band = slope_len_tif.read(1, masked=True)

print('reading soil tif')
soil_tif = rasterio.open(soil_class_raster_in)
soil_nodata = soil_tif.nodata # but also 0
soil_band = soil_tif.read(1, masked=True)



print('masking nodata for numpy arrays')
slope_band_x = ma.filled(slope_band, slope_nodata)

# slope_len_band_x = ma.filled(  
#         np.log(slope_len_band), slope_len_nodata
#     )

slope_len_band_x = ma.filled(slope_len_band, slope_len_nodata)
    
soil_band_x = ma.filled(soil_band, 0)


slope_nan = np.count_nonzero(np.isnan(slope_band_x))
if slope_nan > 0:
    print('slope tif has NaN values (e.g. infinite nodata or null), aborting')
    
slope_len_nan = np.count_nonzero(np.isnan(slope_len_band_x))
if slope_len_nan > 0:
    print('slope length tif has NaN values (e.g. infinite nodata or null), aborting')
    
soil_nan =  np.count_nonzero(np.isnan(soil_band_x))
if soil_nan > 0:
    print('soil tif has NaN values (e.g. infinite nodata or null), aborting')


print('starting nomograph calculations')

buf_recom_val = np.where(soil_band_x==0, np.nan, pre_vect_nomo_simple(slope_len_band_x, slope_band_x, soil_band_x))


print('writing final buffer width raster')
buf_recom_val_x = np.nan_to_num(buf_recom_val, copy=False, nan=-1)

out_profile = slope_tif.profile.copy()

out_profile.update(dtype=rasterio.float32,
                   count=1,
                   compress='lzw',
                   nodata=-1.0)

with rasterio.open(buff_width_raster_out, "w", **out_profile) as dest:
    dest.write(buf_recom_val_x.astype(rasterio.float32), 1)

print('script finished')
