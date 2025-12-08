# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:19:30 2020

@author: Alexander
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import numpy.ma as ma

import gdal
import osr

import seaborn as sns
sns.set(style="whitegrid")

import sys
import os

import buff_width_calc_vectorized_v3

if __name__ == '__main__':

    for processing_zone in range(5, 21):
        input_params = { 
                    'INPUT_SLOPE' : f"/gpfs/rocket/samba/gis/HannaIngrid/5m_calc/zone_{processing_zone}_slope_5m.tif",
                    'INPUT_FLOW_LENGTH' : f"/gpfs/rocket/samba/gis/HannaIngrid/5m_calc/zone_{processing_zone}_flowlength_5m.tif",
                    'INPUT_FLOW_ACC' : f"/gpfs/rocket/samba/gis/HannaIngrid/5m_calc/zone_{processing_zone}_flowacc_5m.tif",
                    'INPUT_LS_FACTOR' : f"/gpfs/rocket/samba/gis/kmoch/nomograph/soil_prep/ls_faktor5m_zone_{processing_zone}.tif",
                    'INPUT_SOIL_CLASS' : f"/gpfs/rocket/samba/gis/kmoch/nomograph/soil_prep/estsoil_proc_overz_3301_{processing_zone}_slopebase_labeled.tif",
                    'INPUT_NOMOGRAPH_CATCHMENTS_COEFFICIENT' : 210,
                    'INPUT_MIN_ABS_BUFFER_SIZE': 3,
                    'INPUT_SLOPE_SCALE_FACTOR' : 10000,
                    'INPUT_BUFSTRIP_SCALE_FACTOR' : 50,
                    'INPUT_FLOWLEN_MAX' : 1392,
                    'INPUT_FLOWACC_MAX' : 21508,
                    'INPUT_LSFACTOR_MAX' : 100,
                    'INPUT_FLOWLEN_WEIGHT' : 3,
                    'INPUT_FLOWACC_WEIGHT' : 3,
                    'INPUT_LSFACTOR_WEIGHT' : 3,
                    'OUTPUT_BUF_SIZE' : os.path.join("/gpfs/hpc/home/kmoch/nomo_kik", f"QGIS_OUTPUT_BUF_SIZE_{processing_zone}.tif"),
                    'OUTPUT_SPEC_SLOPELEN' : os.path.join("/gpfs/hpc/home/kmoch/nomo_kik", f"QGIS_OUTPUT_SPEC_SLOPELEN_{processing_zone}.tif")
                    }

        nomo_out = buff_width_calc_vectorized_v3.params_to_nomograph(input_params)

        print(nomo_out)