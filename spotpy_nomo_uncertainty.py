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


def collect_result_values_for_tracking(nomo_out, input_params, observed_data_src, pzone, initial_obs_clearing=False):

    logger.info(f"collect_result_values_for_tracking internal tracking start (zone {pzone})")

    logger.info('reading slope tif')
    slope_tif = gdal.Open(input_params['INPUT_SLOPE'])
    slope_nodata = slope_tif.GetRasterBand(1).GetNoDataValue()
    slope_band = slope_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"slope_nodata: {slope_nodata}")

    slope_tif = None

    logger.info('reading flow length tif')
    flowlen_tif = gdal.Open(input_params['INPUT_FLOW_LENGTH'])
    flowlen_nodata = flowlen_tif.GetRasterBand(1).GetNoDataValue()
    flowlen_band = flowlen_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"flowlen_nodata: {flowlen_nodata}")

    flowlen_tif = None

    logger.info('reading flow accumulation tif')
    flowacc_tif = gdal.Open(input_params['INPUT_FLOW_ACC'])
    flowacc_nodata = flowacc_tif.GetRasterBand(1).GetNoDataValue()
    flowacc_band = flowacc_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"flowacc_nodata: {flowacc_nodata}")

    flowacc_tif = None

    logger.info('reading ls factor tif')
    lsfactor_tif = gdal.Open(input_params['INPUT_LS_FACTOR'])
    lsfactor_nodata = lsfactor_tif.GetRasterBand(1).GetNoDataValue()
    lsfactor_band = lsfactor_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"lsfactor_nodata: {lsfactor_nodata}, but also -inf sort of")

    lsfactor_tif = None

    logger.info('reading soil class tif')
    soil_tif = gdal.Open(input_params['INPUT_SOIL_CLASS'])
    soil_nodata = soil_tif.GetRasterBand(1).GetNoDataValue() # but also 0
    soil_band = soil_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"soil_nodata: {soil_nodata} , but also 0")

    soil_tif = None

    logger.info('reading calculated buffer strip tif')
    buffer_tif = gdal.Open(input_params['OUTPUT_BUF_SIZE'])
    buffer_nodata = buffer_tif.GetRasterBand(1).GetNoDataValue()
    buffer_band = buffer_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"buffer_nodata: {buffer_nodata}")

    buffer_tif = None

    logger.info('reading calculated specific slope length tif')
    calc_slopelen_tif = gdal.Open(input_params['OUTPUT_SPEC_SLOPELEN'])
    calc_slopelen_nodata = calc_slopelen_tif.GetRasterBand(1).GetNoDataValue() # but also 0
    calc_slopelen_band = calc_slopelen_tif.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"calc_slopelen_nodata: {calc_slopelen_nodata}")

    calc_slopelen_tif = None

    nomo_df = pd.DataFrame(
    {
        'slope': slope_band,
        'grass_flowlen': flowlen_band,
        'grass_flowacc': flowacc_band,
        'lsfactor': lsfactor_band,
        'soil_class': soil_band,
        'calc_slopelen': calc_slopelen_band,
        'buffer_width': buffer_band
    })

    nomo_df = nomo_df.loc[nomo_df['slope'] != slope_nodata]
    nomo_df = nomo_df.loc[nomo_df['soil_class'] != soil_nodata]
    nomo_df = nomo_df.loc[nomo_df['buffer_width'] != buffer_nodata]
    nomo_df = nomo_df.loc[nomo_df['calc_slopelen'] != calc_slopelen_nodata]
    nomo_df = nomo_df.loc[nomo_df['lsfactor'] != lsfactor_nodata]

    corr_df = nomo_df.corr()

    corr_arr = corr_df.values.flatten()

    if initial_obs_clearing == True:
        logger.info(f"writing col index for corr_df_flattened {pzone}: observed_index_{pzone}_corr_df_flattened_columns.txt")
        corr_names = []
        for idx, row in corr_df.iterrows():
            for col in corr_df.columns.tolist():
                corr_names.append(f"corr__{idx}__{col}")
                
        with open(f"observed_index_{pzone}_corr_df_flattened_columns.txt", 'w') as fh:
            for col in corr_names:
                fh.write(col + '\n')

    nomo_df = None
    slope_band = None
    flowlen_band = None
    flowacc_band = None
    lsfactor_band = None
    soil_band = None

    # logger.info('reading calculated buffer strip tif')
    # buffer_tif = gdal.Open(input_params['OUTPUT_BUF_SIZE'])
    # buffer_nodata = buffer_tif.GetRasterBand(1).GetNoDataValue()
    # buffer_band = buffer_tif.GetRasterBand(1).ReadAsArray().flatten()
    # logger.info(f"buffer_nodata: {buffer_nodata}")
# 
    # logger.info('reading calculated specific slope length tif')
    # calc_slopelen_tif = gdal.Open(input_params['OUTPUT_SPEC_SLOPELEN'])
    # calc_slopelen_nodata = calc_slopelen_tif.GetRasterBand(1).GetNoDataValue() # but also 0
    # calc_slopelen_band = calc_slopelen_tif.GetRasterBand(1).ReadAsArray().flatten()
    # logger.info(f"calc_slopelen_nodata: {calc_slopelen_nodata}")

    # comes later
    logger.info('reading baseline calculated buffer strip tif')
    buffer_tif2 = gdal.Open(observed_data_src['dest_fname'])
    buffer_nodata2 = buffer_tif2.GetRasterBand(1).GetNoDataValue()
    buffer_band_b = buffer_tif2.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"buffer_nodata: {buffer_nodata2}")

    logger.info('reading baseline calculated specific slope length tif')
    calc_slopelen_tif2 = gdal.Open(observed_data_src['dest_fname_slopelen'])
    calc_slopelen_nodata2 = calc_slopelen_tif2.GetRasterBand(1).GetNoDataValue() # but also 0
    calc_slopelen_band_b = calc_slopelen_tif2.GetRasterBand(1).ReadAsArray().flatten()
    logger.info(f"calc_slopelen_nodata: {calc_slopelen_nodata2}")

    buf_df = pd.DataFrame(
    {
        'calc_slopelen_a': calc_slopelen_band,
        'buffer_width_a': buffer_band,
        'calc_slopelen_b': calc_slopelen_band_b,
        'buffer_width_b': buffer_band_b
    })

    buf_df = buf_df.loc[buf_df['buffer_width_a'] != buffer_nodata]
    buf_df = buf_df.loc[buf_df['calc_slopelen_a'] != calc_slopelen_nodata]

    buf_df = buf_df.loc[buf_df['buffer_width_b'] != buffer_nodata2]
    buf_df = buf_df.loc[buf_df['calc_slopelen_b'] != calc_slopelen_nodata2]

    buf_df = buf_df.loc[buf_df['buffer_width_a'] >= 0]
    buf_df = buf_df.loc[buf_df['calc_slopelen_a'] >= 0]

    buf_df = buf_df.loc[buf_df['buffer_width_b'] >= 0]
    buf_df = buf_df.loc[buf_df['calc_slopelen_b'] >= 0]

    # observed minus simulated
    buf_df['buffer_width_error'] = buf_df['buffer_width_b'] - buf_df['buffer_width_a']
    buf_df['calc_slopelen_error'] = buf_df['calc_slopelen_b'] - buf_df['calc_slopelen_a']

    dstat_buffer = buf_df['buffer_width_a'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    dstat_slopelen = buf_df['calc_slopelen_a'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    dstat_buf_err = buf_df['buffer_width_error'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    dstat_slopelen_err = buf_df['calc_slopelen_error'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])

    out_records = []

    dstat_buffer = buf_df['buffer_width_a'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    for idx, val in dstat_buffer.iteritems():
        if not idx in ['count']:
            out_records.append({f"dstat_buf_{idx}" : val })
            print({f"dstat_buf_{idx}" : val })
        
    dstat_slopelen = buf_df['calc_slopelen_a'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    for idx, val in dstat_slopelen.iteritems():
        if not idx in ['count']:
            out_records.append({f"dstat_slopelen_{idx}" : val })
            print({f"dstat_slopelen_{idx}" : val })

    dstat_buf_err = buf_df['buffer_width_error'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    for idx, val in dstat_buf_err.iteritems():
        if not idx in ['count']:
            out_records.append({f"dstat_buf_err_{idx}" : val })
            print({f"dstat_buf_err_{idx}" : val })
        
    dstat_slopelen_err = buf_df['calc_slopelen_error'].dropna().describe(percentiles=[.05, .1, .2, .3,.4 ,.5, .6, .7, .8, .9, .95])
    for idx, val in dstat_slopelen_err.iteritems():
        if not idx in ['count']:
            out_records.append({f"dstat_slopelen_err_{idx}" : val })
            print({f"dstat_slopelen_err_{idx}" : val })

    # result_values_for_tracking
    out_records.append({'rmse_buf' : rmse(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    out_records.append({'rmse_slopelen' : rmse(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    logger.info('rmse: buffer / calc_slopelen')
    logger.info(out_records[-2:])
    
    # out_records.append({'rsquared_buf' : rsquared(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'rsquared_slopelen' : rsquared(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('rsquared: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    # out_records.append({'kge_buf' : kge(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'kge_slopelen' : kge(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('kge: buffer / calc_slopelen')
    # logger.info(out_records[-2:])
# 
    # out_records.append({'nashsutcliffe_buf' : nashsutcliffe(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'nashsutcliffe_slopelen' : nashsutcliffe(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('nashsutcliffe: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    # out_records.append({'correlationcoefficient_buf' : correlationcoefficient(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'correlationcoefficient_slopelen' : correlationcoefficient(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('correlationcoefficient: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    out_records.append({'mae_buf' : mae(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    out_records.append({'mae_slopelen' : mae(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    logger.info('mae: buffer / calc_slopelen')
    logger.info(out_records[-2:])

    out_records.append({'pbias_buf' : pbias(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    out_records.append({'pbias_slopelen' : pbias(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    logger.info('pbias: buffer / calc_slopelen')
    logger.info(out_records[-2:])

    # out_records.append({'covariance_buf' : covariance(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'covariance_slopelen' : covariance(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('covariance: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    # out_records.append({'r2_score_buf' : r2_score(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'r2_score_slopelen' : r2_score(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('r2_score: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    # out_records.append({'sk_mean_absolute_error_buf' : mean_absolute_error(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'sk_mean_absolute_error_slopelen' : mean_absolute_error(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('sklearn mean_absolute_error: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    # out_records.append({'mse_buf' : mean_squared_error(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'mse_slopelen' : mean_squared_error(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('mean_squared_error: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    # out_records.append({'explained_variance_score_buf' : explained_variance_score(buf_df['buffer_width_a'], buf_df['buffer_width_b']) })
    # out_records.append({'explained_variance_score_slopelen' : explained_variance_score(buf_df['calc_slopelen_a'], buf_df['calc_slopelen_b']) })
    # logger.info('explained_variance_score: buffer / calc_slopelen')
    # logger.info(out_records[-2:])

    out_records.append({'mad_buf' : buf_df['buffer_width_error'].mad() })
    out_records.append({'mad_slopelen' : buf_df['calc_slopelen_error'].mad() })
    logger.info('mad: buffer / calc_slopelen')
    logger.info(out_records[-2:])

    key_list = []
    value_list = []

    for d in out_records:
        k = list(d.keys())[0]
        v = d[k]
        key_list.append(k)
        value_list.append(v)
        print(f"{k}: {v}")

    if initial_obs_clearing == True:
        with open(f"observed_index_{pzone}_error_metrics_columns.txt", 'w') as fh:
            for col in key_list:
                fh.write(col + '\n')
            
    data_arr = np.array(value_list)

    buf_df = None
    
    calc_slopelen_band = None
    buffer_band = None
    calc_slopelen_band_b = None
    buffer_band_b = None

    buffer_tif = None
    calc_slopelen_tif = None
    buffer_tif2 = None
    calc_slopelen_tif2 = None

    # out_df = pd.DataFrame(out_records)
    # try:
    #     app_df = pd.read_csv(f"records_db_{pzone}.csv", index_col=0)
    #     write_df = pd.concat([app_df, out_df], axis=0, ignore_index=True).reset_index(drop=True)
    #     write_df.to_csv(f"records_db_{pzone}.csv")
    # except FileNotFoundError as ex:
    #     out_df.to_csv(f"records_db_{pzone}.csv")
    # except Exception as ex:
    #     logger.warn(ex)
    #  finally:
    # logger.info("-----------------------------------")
    # logger.info(out_records)
    # logger.info("-----------------------------------")


    # load or create data file csv
    # flatten all collected results from this run into one data record line
    # append to file

    logger.info(f"collect_result_values_for_tracking internal tracking stop (zone {pzone})")
    arr3 = np.concatenate([corr_arr, data_arr])
    # return out_records
    return arr3



class nomo_spotpy_setup(object):
    def __init__(self, observed_data, observed_data_src, param_defs, parallel="seq", temp_dir=None, pzone=21):
        self.temp_dir = temp_dir
        self.observed_data_src = observed_data_src
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
        # logger.info('reading baseline calculated buffer strip tif as observed data')
        # buffer_tif = gdal.Open(self.observed_data['dest_fname'])
        # buffer_nodata = buffer_tif.GetRasterBand(1).GetNoDataValue()
        # buffer_band_b = buffer_tif.GetRasterBand(1).ReadAsArray().flatten()
        # logger.info(f"buffer_nodata: {buffer_nodata}")

        # del(buffer_tif)

        # return buffer_band_b # self.observed_data
        # return np.where(buffer_band_b == buffer_nodata, np.nan, buffer_band_b)
        return self.observed_data


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
                'OUTPUT_BUF_SIZE' : os.path.join(self.temp_dir, f"TEMPORARY_OUTPUT_BUF_SIZE_uncert_{self.processing_zone}_spotpy_x.tif"),
                'OUTPUT_SPEC_SLOPELEN' : os.path.join(self.temp_dir, f"TEMPORARY_OUTPUT_SPEC_SLOPELEN_uncert_{self.processing_zone}_spotpy_x.tif")
                }
        
        logger.info(input_params)

        # {'OUTPUT_BUF_SIZE': dest_fname, 'OUTPUT_SPEC_SLOPELEN': dest_fname_slopelen}
        nomo_out = buff_width_calc_vectorized_v3.params_to_nomograph(input_params)

        # logger.info('reading simulated calculated buffer strip tif')
        # buffer_tif = gdal.Open(nomo_out['OUTPUT_BUF_SIZE'])
        # buffer_nodata = buffer_tif.GetRasterBand(1).GetNoDataValue()
        # buffer_band_b = buffer_tif.GetRasterBand(1).ReadAsArray().flatten()
        # logger.info(f"buffer_nodata: {buffer_nodata}")
        
        # del(buffer_tif)

        # corr_arr = collect_correlation_for_tracking(nomo_out, input_params, self.observed_data_src, self.processing_zone, False)
        data_arr = collect_result_values_for_tracking(nomo_out, input_params, self.observed_data_src, self.processing_zone, False)
        # arr3 = np.concatenate([corr_arr, err_arr])
        
        
        try:
            os.remove(nomo_out['OUTPUT_BUF_SIZE'])
            os.remove(nomo_out['OUTPUT_SPEC_SLOPELEN'])
        except:
            pass
        
        # return err_arr
        return data_arr
        # return buffer_band_b # result_values_for_tracking
        # return np.where(buffer_band_b == buffer_nodata, np.nan, buffer_band_b)
    
    
    # if we want to minimize our function, we can select a negative objective function
    def objectivefunction(self, simulation, evaluation):
        logger.info("simulation")
        logger.info(len(simulation))
        logger.info("evaluation")
        logger.info(len(evaluation))

        objectivefunction = spotpy.objectivefunctions.rsquared(evaluation,simulation)      
        return objectivefunction
    

def prep_observed_data(observed_data_src, processing_zone, temp_dir, initial_obs_clearing):
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
                    'OUTPUT_BUF_SIZE' : os.path.join(temp_dir, f"TEMP_OUTPUT_BUF_SIZE_uncert_obs_{processing_zone}.tif"),
                    'OUTPUT_SPEC_SLOPELEN' : os.path.join(temp_dir, f"TEMP_OUTPUT_SPEC_SLOPELEN_uncert_obs_{processing_zone}.tif")
                    }

    nomo_out = buff_width_calc_vectorized_v3.params_to_nomograph(input_params)

    # corr_arr = collect_correlation_for_tracking(nomo_out, input_params, observed_data_src, processing_zone, initial_obs_clearing)
    data_arr = collect_result_values_for_tracking(nomo_out, input_params, observed_data_src, processing_zone, initial_obs_clearing)
    # arr3 = np.concatenate([corr_arr, err_arr])
    return data_arr
    # return corr_arr


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
    
    fh = logging.FileHandler(f"nomo_uncert_{pzone}_{my_pid}_output.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    repetitions=579

    parallel = 'seq'

    dbformat = "csv"

    observed_data_src =  { 'dest_fname' : os.path.join("/gpfs/hpc/home/kmoch/nomo_kik", f"QGIS_OUTPUT_BUF_SIZE_{pzone}.tif"),
            'dest_fname_slopelen' : os.path.join("/gpfs/hpc/home/kmoch/nomo_kik", f"QGIS_OUTPUT_SPEC_SLOPELEN_{pzone}.tif") }

    logger.info('preparing observed data baseline')
    observed_data = prep_observed_data(observed_data_src, processing_zone=int(pzone), temp_dir='/tmp', initial_obs_clearing=True)

    param_defs = [
                    ('FLOWLENWEIGHT',1,10,3),
                    ('FLOWACCWEIGHT',1,10,3),
                    ('LSFACTORWEIGHT',1,10,3)
                ]

    get_ready = nomo_spotpy_setup(observed_data, observed_data_src, param_defs, parallel="seq", temp_dir='/tmp', pzone=int(pzone))

    lhs_sampler = spotpy.algorithms.lhs(get_ready, parallel=parallel, dbname=f"nomo_uncert_lhs_zone_{pzone}", dbformat=dbformat, save_sim=True)

    lhs_sampler.sample(repetitions)


if __name__ == "__main__":
    main(sys.argv[1:])