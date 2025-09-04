# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:19:30 2020

@author: Alexander
"""

import numpy as np

import pandas as pd

import numpy.ma as ma

import gdal
import osr

import sys
import os

import buff_width_calc_vectorized_v3

import spotpy

from spotpy.objectivefunctions import rmse, rsquared, kge, nashsutcliffe, lognashsutcliffe, correlationcoefficient,mae, pbias, covariance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, mean_squared_log_error

import getopt
import logging
import datetime

log_level = logging.INFO
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)


class nomo_spotpy_setup(object):
    def __init__(self, observed_data, param_defs, parallel="seq", temp_dir=None, pzone=21):
        self.observed_data = observed_data
        self.processing_zone = int(pzone)
        
        self.params = []
        for i in range(len(param_defs)):
            self.params.append(
                spotpy.parameter.Uniform(
                    name=param_defs[i][0],
                    low=param_defs[i][1],
                    high=param_defs[i][2],
                    optguess=param_defs[i][3] ))
    
        self.temp_dir = temp_dir
        self.parallel = parallel

        if self.parallel == "seq":
            pass

        if self.parallel == "mpi":

            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            self.mpi_size = comm.Get_size()
            self.mpi_rank = comm.Get_rank()
    
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    
    # provide the available observed data
    def evaluation(self):

        # observations = [self.observed_data]
        logger.info('reading baseline calculated buffer strip tif as observed data')
        buffer_tif = gdal.Open(self.observed_data['dest_fname'])
        buffer_nodata = buffer_tif.GetRasterBand(1).GetNoDataValue()
        buffer_band_b = buffer_tif.GetRasterBand(1).ReadAsArray().flatten()
        logger.info(f"buffer_nodata: {buffer_nodata}")

        del(buffer_tif)

        return buffer_band_b # self.observed_data
        # return np.where(buffer_band_b == buffer_nodata, np.nan, buffer_band_b)


    # Simulation function must not return values besides for which evaluation values/observed data are available
    def simulation(self, parameters):
        
        logger.info(f"this iteration's parameters:")
        logger.info(parameters)
        # logger.info(len(parameters))
        # logger.info(parameters[0])
        
        # do stuff
        input_params = { 
                'INPUT_SLOPE' : f"/gpfs/rocket/samba/gis/HannaIngrid/5m_calc/zone_{self.processing_zone}_slope_5m.tif",
                'INPUT_FLOW_LENGTH' : f"/gpfs/rocket/samba/gis/HannaIngrid/5m_calc/zone_{self.processing_zone}_flowlength_5m.tif",
                'INPUT_FLOW_ACC' : f"/gpfs/rocket/samba/gis/HannaIngrid/5m_calc/zone_{self.processing_zone}_flowacc_5m.tif",
                'INPUT_LS_FACTOR' : f"/gpfs/rocket/samba/gis/kmoch/nomograph/soil_prep/ls_faktor5m_zone_{self.processing_zone}.tif",
                'INPUT_SOIL_CLASS' : f"/gpfs/rocket/samba/gis/kmoch/nomograph/soil_prep/estsoil_proc_overz_3301_{self.processing_zone}_slopebase_labeled.tif",
                'INPUT_NOMOGRAPH_CATCHMENTS_COEFFICIENT' : 210,
                'INPUT_MIN_ABS_BUFFER_SIZE': 3,
                'INPUT_SLOPE_SCALE_FACTOR' : 10000,
                'INPUT_BUFSTRIP_SCALE_FACTOR' : 50,
                'INPUT_FLOWLEN_MAX' : 1392,
                'INPUT_FLOWACC_MAX' : 21508,
                'INPUT_LSFACTOR_MAX' : 100,
                'INPUT_FLOWLEN_WEIGHT' : parameters[0],
                'INPUT_FLOWACC_WEIGHT' : parameters[1],
                'INPUT_LSFACTOR_WEIGHT' : parameters[2],
                'OUTPUT_BUF_SIZE' : os.path.join(self.temp_dir, f"TEMPORARY_OUTPUT_BUF_SIZE_{self.processing_zone}_spotpy_x.tif"),
                'OUTPUT_SPEC_SLOPELEN' : os.path.join(self.temp_dir, f"TEMPORARY_OUTPUT_SPEC_SLOPELEN_{self.processing_zone}_spotpy_x.tif")
                }
        
        logger.info(input_params)

        # {'OUTPUT_BUF_SIZE': dest_fname, 'OUTPUT_SPEC_SLOPELEN': dest_fname_slopelen}
        nomo_out = buff_width_calc_vectorized_v3.params_to_nomograph(input_params)

        logger.info('reading simulated calculated buffer strip tif')
        buffer_tif = gdal.Open(nomo_out['OUTPUT_BUF_SIZE'])
        buffer_nodata = buffer_tif.GetRasterBand(1).GetNoDataValue()
        buffer_band_b = buffer_tif.GetRasterBand(1).ReadAsArray().flatten()
        logger.info(f"buffer_nodata: {buffer_nodata}")
        
        del(buffer_tif)
        
        try:
            os.remove(nomo_out['OUTPUT_BUF_SIZE'])
            os.remove(nomo_out['OUTPUT_SPEC_SLOPELEN'])
        except:
            pass
        
        return buffer_band_b # result_values_for_tracking
        # return np.where(buffer_band_b == buffer_nodata, np.nan, buffer_band_b)
    
    
    # if we want to minimize our function, we can select a negative objective function
    def objectivefunction(self, simulation, evaluation):
        logger.info("simulation")
        logger.info(len(simulation))
        logger.info("evaluation")
        logger.info(len(evaluation))

        objectivefunction = spotpy.objectivefunctions.rsquared(evaluation,simulation)      
        return objectivefunction
    

def main(argv):
    my_pid = os.getpid()
    pzone = ""
    try:
        opts, args = getopt.getopt(
            argv, "hz:", ["pzone="]
        )
        logger.info(f"opts {opts} args ({args})")
    except getopt.GetoptError:
        logger.info("test.py -z <pzone>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            logger.info("test.py -z <pzone>")
            sys.exit()
        elif opt in ("-z", "--pzone"):
            pzone = arg
            if not int(pzone) > 0 or not int(pzone) <= 22:
                logger.info(f"{pzone} not in known range for processing zones, abort")
                sys.exit()
    
    fh = logging.FileHandler(f"nomo_sens_{pzone}_{my_pid}_output.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    repetitions=579

    parallel = 'seq'

    dbformat = "csv"

    observed_data =  { 'dest_fname' : f"QGIS_OUTPUT_BUF_SIZE_{pzone}.tif", 
            'dest_fname_slopelen' : f"QGIS_OUTPUT_SPEC_SLOPELEN_{pzone}.tif"}

    param_defs = [
                    ('FLOWLENWEIGHT',1,10,3),
                    ('FLOWACCWEIGHT',1,10,3),
                    ('LSFACTORWEIGHT',1,10,3)
                ]

    get_ready = nomo_spotpy_setup(observed_data, param_defs, parallel="seq", temp_dir='/tmp', pzone=int(pzone))

    fast_sampler = spotpy.algorithms.fast(get_ready, parallel=parallel, dbname=f"nomo_fast_sens_zone_{pzone}", dbformat=dbformat, save_sim=False)

    fast_sampler.sample(repetitions)


if __name__ == "__main__":
    main(sys.argv[1:])