# -*- coding: utf-8 -*-

"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication

from qgis.core import (QgsProcessing, Qgis,
                       QgsMessageLog,
                       QgsRaster,
                       QgsRasterLayer,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterNumber)
from qgis import processing

from osgeo import gdal, osr

import sys
import os

import numpy as np
import numpy.ma as ma

class NomographProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT_SLOPE = 'INPUT_SLOPE'
    INPUT_FLOW_LENGTH = 'INPUT_FLOW_LENGTH'
    INPUT_FLOW_ACC = 'INPUT_FLOW_ACC'
    INPUT_LS_FACTOR = 'INPUT_LS_FACTOR'
    INPUT_SOIL_CLASS = 'INPUT_SOIL_CLASS'
    
    INPUT_SLOPE_SCALE_FACTOR = 'INPUT_SLOPE_SCALE_FACTOR'
    INPUT_BUFSTRIP_SCALE_FACTOR = 'INPUT_BUFSTRIP_SCALE_FACTOR'

    INPUT_FLOWLEN_MAX = 'INPUT_FLOWLEN_MAX'
    INPUT_FLOWACC_MAX = 'INPUT_FLOWACC_MAX'

    INPUT_FLOWLEN_WEIGHT = 'INPUT_FLOWLEN_WEIGHT'
    INPUT_FLOWACC_WEIGHT = 'INPUT_FLOWACC_WEIGHT'
    INPUT_LSFACTOR_WEIGHT = 'INPUT_LSFACTOR_WEIGHT'
    
    OUTPUT_BUF_SIZE = 'OUTPUT_BUF_SIZE'
    OUTPUT_SPEC_SLOPELEN = 'OUTPUT_SPEC_SLOPELEN'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Nomograph Processing Algorithm', string)

    def createInstance(self):
        return NomographProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'nomographrastercalc'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Nomograph Processing Algorithm Script')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Alex scripts')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'alexscripts'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Nomograph Processing Algorithm short description, calculate recommended riparian buffer width")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input raster layers. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_SLOPE,
                self.tr('Input layer Slope in degrees')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_LENGTH,
                self.tr('Input layer Grass Flow Length in m')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_ACC,
                self.tr('Input layer Grass Flow accumulation')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_LS_FACTOR,
                self.tr('Input layer LS Factor (assuming to be normalised 1-100)')
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_SOIL_CLASS,
                self.tr('Input layer Soil Class (with classes 1-7)')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_SLOPE_SCALE_FACTOR,
                self.tr('Input slope scaling factor (default 2000)'),
                defaultValue = 2000, optional = True, minValue = 0, maxValue = 10000
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_BUFSTRIP_SCALE_FACTOR,
                self.tr('Input buffer strip scaling factor (default 10)'),
                defaultValue = 10, optional = True, minValue = 0, maxValue = 1000
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_FLOWLEN_MAX,
                self.tr('Input Flow Length normalise max value (default 1391.5403)'),
                defaultValue = 1391.5403, optional = True, minValue = 1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_FLOWACC_MAX,
                self.tr('Input Flow Accumulation normalise max value (default 21508)'),
                defaultValue = 21508.0, optional = True, minValue = 1
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_FLOWLEN_WEIGHT,
                self.tr('Input Flow Length weight (default 3)'),
                defaultValue = 3.0, optional = True, minValue = 1.0, maxValue = 100
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_FLOWACC_WEIGHT,
                self.tr('Input Flow Accumulation weight (default 3)'),
                defaultValue = 3.0, optional = True, minValue = 1.0, maxValue = 100
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_LSFACTOR_WEIGHT,
                self.tr('Input LS Factor weight (default 3)'),
                defaultValue = 3.0, optional = True, minValue = 1.0, maxValue = 100
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_BUF_SIZE,
                self.tr('Output raster layer with buffer size recommendation')
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_SPEC_SLOPELEN,
                self.tr('Output raster layer with weighted calculated specific slope length'),
                optional = True
            )
        )
        
    
    def getDataParams(self):
        
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
        
        return {'pred_sl': pred_sl, 'soil_classes_list_arr': soil_classes_list_arr}


    def degr_to_perc_slope(self, degrees):
        return (np.tan(np.radians(degrees)))*100


    def pre_vect_slope_angle_rel_y_val(self, slope_angle_degr, slopelen_m, slope_scale_factor):
        slope_rel=slopelen_m/slope_scale_factor
        alpha = self.getDataParams()['pred_sl'](slope_angle_degr)
        y = slope_rel * (np.tan(np.radians(alpha)))
        return y


    def pre_vect_select_angle_for_soil_class(self, soil_class):
        return np.take(self.getDataParams()['soil_classes_list_arr'], soil_class-1)


    def pre_vect_soiltype_rel_x_val(self, soil_class, y, buffer_strip_scale):
        alpha=self.pre_vect_select_angle_for_soil_class(soil_class)
        beta = 90-alpha
        x = y * (np.tan(np.radians(beta)))
        return x * buffer_strip_scale


    def pre_vect_nomo_simple(self, slopelen_m, slope_angle_degr, soil_class, slope_scale_factor, buffer_strip_scale):
        nomo_vals = self.pre_vect_soiltype_rel_x_val(soil_class, self.pre_vect_slope_angle_rel_y_val(slope_angle_degr, slopelen_m, slope_scale_factor), buffer_strip_scale)
        return nomo_vals
    
    
    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        
        slope_raster_in = self.parameterAsRasterLayer(
            parameters,
            self.INPUT_SLOPE,
            context
        )
        
        flowlen_raster_in = self.parameterAsRasterLayer(
            parameters,
            self.INPUT_FLOW_LENGTH,
            context
        )
        
        flowacc_raster_in = self.parameterAsRasterLayer(
            parameters,
            self.INPUT_FLOW_ACC,
            context
        )
        
        lsfactor_raster_in = self.parameterAsRasterLayer(
            parameters,
            self.INPUT_LS_FACTOR,
            context
        )

        soil_class_raster_in = self.parameterAsRasterLayer(
            parameters,
            self.INPUT_SOIL_CLASS,
            context
        )
        
        slope_scale_factor = self.parameterAsInt(
            parameters,
            self.INPUT_SLOPE_SCALE_FACTOR,
            context
        )
        
        buffer_strip_scale = self.parameterAsInt(
            parameters,
            self.INPUT_BUFSTRIP_SCALE_FACTOR,
            context
        )
        
        flowlen_max_value = self.parameterAsDouble(
            parameters,
            self.INPUT_FLOWLEN_MAX,
            context
        )
        
        flowacc_max_value = self.parameterAsDouble(
            parameters,
            self.INPUT_FLOWACC_MAX,
            context
        )

        flowlen_weight = self.parameterAsDouble(
            parameters,
            self.INPUT_FLOWLEN_WEIGHT,
            context
        )
        
        flowacc_weight = self.parameterAsDouble(
            parameters,
            self.INPUT_FLOWACC_WEIGHT,
            context
        )
        
        lsfactor_weight = self.parameterAsDouble(
            parameters,
            self.INPUT_LSFACTOR_WEIGHT,
            context
        )
        
        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if not slope_raster_in.isValid():
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT_SLOPE))
        
        if not flowlen_raster_in.isValid():
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT_FLOW_LENGTH))
            
        if not flowacc_raster_in.isValid():
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT_FLOW_ACC))
            
        if not lsfactor_raster_in.isValid():
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT_LS_FACTOR))
        
        if not soil_class_raster_in.isValid():
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT_SOIL_CLASS))
        
        dest_fname = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_BUF_SIZE,
            context
        )

        dest_fname_slopelen = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_SPEC_SLOPELEN,
            context
        )
        
        # Send some information to the user
        # QgsMessageLog.logMessage(f"aiming to write raster to {dest_fname}", "the plugman", Qgis.Info)
        feedback.pushInfo(f"aiming to write raster to {dest_fname}")

        # Update the progress bar
        feedback.setProgress(5)

        feedback.pushInfo('reading slope tif')

        # slope is our reference
        slope_prov = slope_raster_in.dataProvider()
        slope_crs = slope_prov.crs()
        
        slope_tif = gdal.Open(slope_prov.dataSourceUri())
        slope_nodata = slope_prov.sourceNoDataValue(1)
        slope_band = slope_tif.GetRasterBand(1).ReadAsArray()
        
        height = slope_raster_in.height()
        width = slope_raster_in.width()
        unit_x = slope_raster_in.rasterUnitsPerPixelX()
        unit_y = slope_raster_in.rasterUnitsPerPixelY()
        
        origin_x = 0
        origin_y = 0
        
        geotrans = slope_tif.GetGeoTransform()
        
        if slope_crs.authid() is None or slope_crs.authid() == '':
            feedback.pushInfo(f"CRS is empty for slope, not good")
        else:
            feedback.pushInfo('CRS is {}'.format(slope_crs.authid()))
        

        # Update the progress bar
        feedback.setProgress(15)

        feedback.pushInfo('reading flow length tif')
        flowlen_tif = gdal.Open(flowlen_raster_in.dataProvider().dataSourceUri())
        flowlen_nodata = flowlen_raster_in.dataProvider().sourceNoDataValue(1)
        flowlen_band = flowlen_tif.GetRasterBand(1).ReadAsArray()

        # Update the progress bar
        feedback.setProgress(25)

        feedback.pushInfo('reading flow accumulation tif')
        flowacc_tif = gdal.Open(flowacc_raster_in.dataProvider().dataSourceUri())
        flowacc_nodata = flowacc_raster_in.dataProvider().sourceNoDataValue(1)
        flowacc_band = flowacc_tif.GetRasterBand(1).ReadAsArray()

        # Update the progress bar
        feedback.setProgress(35)

        feedback.pushInfo('reading ls factor tif (assuming it to be normalised 1-100)')
        lsfactor_tif = gdal.Open(lsfactor_raster_in.dataProvider().dataSourceUri())
        lsfactor_nodata = lsfactor_raster_in.dataProvider().sourceNoDataValue(1)
        lsfactor_band = lsfactor_tif.GetRasterBand(1).ReadAsArray()

        # Update the progress bar
        feedback.setProgress(45)

        feedback.pushInfo('reading soil class tif')
        soil_tif = gdal.Open(soil_class_raster_in.dataProvider().dataSourceUri())
        soil_nodata = soil_class_raster_in.dataProvider().sourceNoDataValue(1) # but also 0
        soil_band = soil_tif.GetRasterBand(1).ReadAsArray()

        # feedback.pushInfo('masking nodata for numpy arrays - is that still necessary?')
        
        # slope_nodata_mask = slope_band == slope_nodata
        # slope_band_x = ma.masked_array(slope_band, mask=slope_nodata_mask )
        
        # slope_band_x = ma.filled(slope_band, slope_nodata)
        slope_band_x = slope_band

        # Update the progress bar
        feedback.setProgress(55)

        feedback.pushInfo('normalising flow length raster into 1-100 (min=0, max=1391.5403)')
        flowlen_band_x = ( flowlen_band - 0 ) / ( flowlen_max_value - 0 )
        
        # flowlen_band_x = ma.filled(  
        #         np.log(flowlen_band), flowlen_nodata
        #     )

        # flowlen_band_x = ma.filled(flowlen_band, flowlen_nodata)
        # print('applying np.sqrt / np.log to flow_acc')
        # flowlen_band_x = ma.filled(np.sqrt(slope_len_band), slope_len_nodata)

        # Update the progress bar
        feedback.setProgress(60)

        feedback.pushInfo('normalising flow accumulation raster into 1-100 (min=0, max=21508)')
        flowacc_band_x = ( flowacc_band - 0 ) / ( flowacc_max_value - 0 )
        
        lsfactor_band_x = lsfactor_band

        # soil_band_x = ma.filled(soil_band, 0)
        soil_band_x = soil_band

        slope_nan = np.count_nonzero(np.isnan(slope_band_x))
        if slope_nan > 0:
            feedback.pushInfo('slope tif has NaN values (e.g. infinite nodata or null), might cause problem')
            
        flowlen_nan = np.count_nonzero(np.isnan(flowlen_band_x))
        if flowlen_nan > 0:
            feedback.pushInfo('slope length tif has NaN values (e.g. infinite nodata or null), might cause problem')
        
        flowacc_nan = np.count_nonzero(np.isnan(flowacc_band_x))
        if flowacc_nan > 0:
            feedback.pushInfo('slope accumulation tif has NaN values (e.g. infinite nodata or null), might cause problem')
        
        lsfactor_nan = np.count_nonzero(np.isnan(lsfactor_band_x))
        if lsfactor_nan > 0:
            feedback.pushInfo('lsfactor tif has NaN values (e.g. infinite nodata or null), might cause problem')

        soil_nan =  np.count_nonzero(np.isnan(soil_band_x))
        if soil_nan > 0:
            feedback.pushInfo('soil tif has NaN values (e.g. infinite nodata or null), might cause problem')
        
        # Update the progress bar
        feedback.setProgress(65)

        all_weights = flowlen_weight + flowacc_weight + lsfactor_weight
        feedback.pushInfo(f"weighing rasters flow_length ({flowlen_weight}), flow_acc ({flowacc_weight}) and ls_factor ({lsfactor_weight}) into 1-100 ({all_weights})")

        slope_len_band_x = ( (flowlen_band_x * flowlen_weight) + (flowacc_band_x * flowacc_weight) + (lsfactor_band_x * lsfactor_weight) ) / all_weights
        slope_len_band_x_np = np.where(slope_len_band_x < 0, 0, slope_len_band_x)

        # Update the progress bar
        feedback.setProgress(70)

        feedback.pushInfo('starting nomograph calculations')

        buf_recom_val = np.where(soil_band_x==0, np.nan, self.pre_vect_nomo_simple(slope_len_band_x_np, slope_band_x, soil_band_x, slope_scale_factor, buffer_strip_scale))

        # Update the progress bar
        feedback.setProgress(90)

        feedback.pushInfo('writing final output raster for buffer strip size')
        buf_recom_val_x = np.nan_to_num(buf_recom_val, copy=False, nan=-1)

        driver = gdal.GetDriverByName("GTIFF")

        dataset = driver.Create(
                dest_fname,
                width,
                height,
                1,
                gdal.GDT_Float32,
                options = [ 'COMPRESS=DEFLATE' ]
                )
        
        dataset.SetGeoTransform(geotrans)

        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(int(slope_crs.authid().replace('EPSG:','')))

        dataset.SetProjection(out_srs.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(buf_recom_val_x.astype(np.float32))  # add ".T" if it's inverted.
        
        dataset.GetRasterBand(1).SetNoDataValue(-1.0)
        dataset.GetRasterBand(1).FlushCache()

        dataset = None
        
        # Update the progress bar
        feedback.setProgress(95)
        
        feedback.pushInfo('writing final output raster for weighted specific slope length')

        

        slope_len_band_x_out = np.nan_to_num(slope_len_band_x_np, copy=False, nan=-1)

        dataset2 = driver.Create(
                dest_fname_slopelen,
                width,
                height,
                1,
                gdal.GDT_Float32,
                options = [ 'COMPRESS=DEFLATE' ]
                )
        dataset2.SetGeoTransform(geotrans)

        dataset2.SetProjection(out_srs.ExportToWkt())
        dataset2.GetRasterBand(1).WriteArray(slope_len_band_x_out.astype(np.float32))  # add ".T" if it's inverted.
        
        dataset2.GetRasterBand(1).SetNoDataValue(-1.0)
        dataset2.GetRasterBand(1).FlushCache()

        dataset2 = None


        # Return the results of the algorithm, all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT_BUF_SIZE: dest_fname, self.OUTPUT_SPEC_SLOPELEN: dest_fname_slopelen}
