#!/usr/bin/env python
# coding: utf-8

# In[90]:


import geopandas as gpd
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)


# In[91]:


stats = gpd.read_file('tmp_stats.gpkg')


# In[92]:


stats.dtypes


# In[93]:


stats['id'] = pd.to_numeric(stats['id'], downcast='integer')


# In[94]:


stats.head(2)


# In[95]:


source_cols = [ 'specific_slopenlength_with_log',
                'specific_slopenlength_default',
                'buffer_size_with_log',
                'buffer_size_default',
                'specific_slopenlength_with_log_clipped',
                'specific_slopenlength_default_clipped',
                'buffer_size_with_log_clipped',
                'buffer_size_default_clipped',
              ]


# In[96]:


stats.loc[stats['id'] == 3]['specific_slopenlength_with_log_pzone_3_mean'].values[0]


# In[103]:


new_df = pd.DataFrame({'zone_id': [i for i in range(1,23)], 'geometry': None})

for col in source_cols:
    for stat in ['mean', 'median', 'max', 'stdev']:
        sum_col = f"{col}_{stat}"
        new_df[sum_col] = np.nan
        
for i in range(1,23):
    new_df.loc[new_df['zone_id'] == i, 'geometry'] = stats.loc[stats['id'] == i]['geometry'].values[0]
    
    for col in source_cols:
        for stat in ['mean', 'median', 'max', 'stdev']:
            target_col = f"{col}_pzone_{i}_{stat}"
            sum_col = f"{col}_{stat}"
            
            if target_col in stats.columns:
                if sum_col in new_df.columns:
                    x = stats.loc[stats['id'] == i][target_col].values[0]
                    if 'max' in target_col and 'buffer_size' in target_col:
                        print(f"{target_col} -> {sum_col}: value {x}")
                    new_df.loc[new_df['zone_id'] == i, sum_col] = x
                


# In[98]:


new_df.sample(3)


# In[99]:


stats_geo = gpd.GeoDataFrame(new_df, geometry='geometry', crs='EPSG:3301')


# In[100]:


stats_geo.plot(column='buffer_size_with_log_clipped_mean', legend=True)


# In[117]:


specific_slopenlength_max = 0

for x in filter(lambda x: 'max' in x, stats_geo.columns.tolist()):
    if 'specific_slopenlength' in x:
        tmax = stats_geo[x].max()
        if tmax > specific_slopenlength_max:
            specific_slopenlength_max = tmax
            
buffer_size_max = 0

for x in filter(lambda x: 'max' in x, stats_geo.columns.tolist()):
    if 'buffer_size' in x:
        tmax = stats_geo[x].max()
        if tmax > buffer_size_max:
            buffer_size_max = tmax


# In[118]:


print(specific_slopenlength_max)


# In[119]:


print(buffer_size_max)


# In[120]:


stats_geo
stats_geo['cent'] = stats_geo.centroid
stats_geo['cent_x'] = stats_geo.apply(lambda g: g['cent'].x, axis=1)
stats_geo['cent_y'] = stats_geo.apply(lambda g: g['cent'].y, axis=1)


# In[101]:


eesti = gpd.read_file('../../../grid_tiles/estonia_without_lakes.shp').to_crs(3301)


# In[171]:



for x in filter(lambda x: 'max' in x, stats_geo.columns.tolist()):
    if 'buffer_size' in x:
        fig, ax = plt.subplots(figsize=(10,7))
        eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=ax)
        stats_geo.loc[stats_geo[x] < 1000].plot(column=x, legend=True, ax=ax, alpha=0.8, cmap='coolwarm')
        
        for idx, row in stats_geo.iterrows():
            t_val = row[x]
            if not np.isnan(t_val):
                plt.text(row['cent_x'], row['cent_y'],"{}\n".format(row['zone_id'],),size=10, color='black')

        plt.title(f"{x}", fontsize=16)
        plt.tight_layout()


# In[123]:


plt.close()


# In[58]:


t_val = stats_geo.loc[stats_geo['zone_id'] == i, x].values[0]


# In[59]:


t_val


# In[62]:





# In[114]:


eesti.crs


# In[82]:





# In[130]:


from rasterstats import zonal_stats
import os
import rasterio
import seaborn as sns

catcher = []
stats2 = []
for base_n in source_cols:
    for i in [3, 12,13,19, 21]:
        if 'buffer_size' in base_n:
            layer = f"{base_n}_pzone_{i}.tif"
            print(layer)
            
            dset = rasterio.open(layer, 'r')
            print(dset.meta)
            # band1 = dset.read(1)
            # catcher.append({layer: band1})
            
            # plt.figure(figsize=(4,6))

            #plot the boxplots
            # sns.boxplot(x=band1, palette="YlOrRd", saturation=0.9, showfliers=False)

            # plt.title(base_n, fontsize='x-large')
            # plt.show()

            #set min and max values for y-axis
            # plt.ylim(-70, 70)
            nodata = dset.nodata
            del dset
            t = zonal_stats(stats['geometry'], layer, stats=['max'], all_touched=True, nodata=nodata)
            
            stats2.append({layer: t})


# In[131]:


t
    


# In[135]:


new_df2 = pd.DataFrame({'zone_id': [i for i in range(1,23)], 'geometry': None})

for col in source_cols:
    for stat in ['max']:
        sum_col = f"{col}_{stat}"
        new_df[sum_col] = np.nan


# In[142]:


counter = 0

for base_n in source_cols:
    for i in [3, 12,13,19, 21]:
        if 'buffer_size' in base_n:
            layer = f"{base_n}_pzone_{i}.tif"
            print(layer)
            elem = stats2[counter]
            print(list(elem.keys())[0])
            sum_col = f"{base_n}_max"
            x = stats2[counter][layer][i-1]['max']
            new_df2.loc[new_df2['zone_id'] == i, sum_col] = x
            counter = counter +1 
        


# In[143]:


new_df2


# In[144]:


len(catcher)


# In[145]:



catcher = []

for base_n in source_cols:
    for i in [3, 12,13,19, 21]:
        if 'buffer_size' in base_n:
            layer = f"{base_n}_pzone_{i}.tif"
            print(layer)
            
            dset = rasterio.open(layer, 'r')
            print(dset.meta)
            band1 = dset.read(1, masked=True)
            nodata = dset.nodata
            
            print(band1.max())
            print(band1.mean())
            
            # plt.figure(figsize=(4,6))

            #plot the boxplots
            # sns.boxplot(x=band1, palette="YlOrRd", saturation=0.9, showfliers=False)

            # plt.title(base_n, fontsize='x-large')
            # plt.show()

            #set min and max values for y-axis
            # plt.ylim(-70, 70)
            
            del dset
            
            catcher.append({layer: band1})
            


# In[148]:


dset = rasterio.open("../../soil_prep/ls_faktor5m_pzone_12_cog.tif", 'r')
print(dset.meta)
band1 = dset.read(1, masked=True)
nodata = dset.nodata

print(band1.max())
print(band1.mean())

del dset


# In[173]:


stats_geo.columns


# In[424]:


from scipy import stats
from scipy.stats import iqr

from shapely.geometry import Point

ptest = pd.DataFrame()

ptest['zone_id'] = -1
ptest['layer'] = None
ptest['x'] = -1
ptest['y'] = -1
ptest['buf_value'] = -1
ptest['geometry'] = None
    
import numpy.ma as ma
idx = 0

# if max buffer size above
max_limit = 500
check_limit = 150

for base_n in source_cols:
    for i in [3, 12,13,19, 21]:
        if 'buffer_size' in base_n:
            
            cons = {'max': 0,
                   'mean': 0,
                   'median': 0}
            
            layer = f"{base_n}_pzone_{i}.tif"
            print(layer)

            dset = rasterio.open(layer, 'r')
            print(dset.meta)
            band1 = dset.read(1, masked=False)
            nodata = dset.nodata
            
            x = band1.flatten()
            
            condition = x >= 0
            z = np.extract(condition, x)
            # x_nan = np.nan_to_num(x, nan=-1)
            # z = ma.masked_values(x_nan, -1.0)
            # np.extract(condition, arr)
            
            # print(stats.describe(z))
            print(f"min {z.min()}")
            
            zmax = np.max(z)
            print(f"max {zmax}")
            
            zmean = np.mean(z)
            print(f"mean {zmean}")
            
            zstd = np.std(z)
            print(f"std {zstd}")
            
            zvar = np.var(z)
            print(f"var {zvar}")
            
            z_iqr = iqr(z)
            print(f"iqr {z_iqr}")
            
            zmedian = np.quantile(z, 0.5)
            print(f"q0.5 {zmedian}")
            print(f"q0.75 {np.quantile(z, 0.75)}")
            print(f"q0.85 {np.quantile(z, 0.85)}")
            print(f"q0.95 {np.quantile(z, 0.95)}")
            print(f"q0.99 {np.quantile(z, 0.99)}")
            
            plt.figure()
            plt.hist(z, bins=50)
            plt.title(layer)
            plt.show()
            
            cons['max'] = zmax
            cons['mean'] = zmean
            cons['median'] = zmedian
            
            if zmax > max_limit:
                print(f"IQR modification {layer}")
                
                iqr_condition = z < 3*zvar
                
                iz = np.extract(iqr_condition, z)
                
                oz_arr_loc = np.argwhere(band1 > 3*zvar)
                
                oz = len(oz_arr_loc) # np.extract(~iqr_condition, z)
                
                print(f"reduced by {oz} of {len(z)} values")
                
                x_coef = dset.transform.column_vectors[0][0]
                y_coef = dset.transform.column_vectors[1][1]

                x_orig = dset.transform.column_vectors[2][0]
                y_orig = dset.transform.column_vectors[2][1]
                
                ozdf = pd.DataFrame({'x': oz_arr_loc[:,1], 'y': oz_arr_loc[:,0]})
                ozdf['zone_id'] = i
                ozdf['layer'] = layer
                ozdf['px'] = ozdf['x'].apply(lambda x: x*x_coef + x_orig + x_coef/2)
                ozdf['py'] = ozdf['y'].apply(lambda y: y*y_coef+y_orig + y_coef/2)
                ozdf['buf_value'] = band1[ oz_arr_loc[:,0], oz_arr_loc[:,1] ]
                ozdf['geometry'] = ozdf.apply(lambda row: Point(row['px'], row['py']), axis=1)
                
                ptest = pd.concat([ ptest, ozdf[['zone_id', 'layer', 'x', 'y', 'buf_value','geometry']].copy() ], ignore_index=True)
                
                izmax = iz.max()

                print(f"imax {izmax}")
                
                izmean = z.mean()
                print(f"imean {izmean}")
                
                izmedian = np.quantile(iz, 0.5)

                izstd = np.std(iz)
                print(f"istd {izstd}")
                izvar = np.var(iz)
                print(f"ivar {izvar}")

                plt.figure()
                plt.hist(iz, bins=50, color='red')
                plt.title(f"{layer} IQR reduced")
                plt.show()
                
                cons['max'] = izmax
                cons['mean'] = izmean
                cons['median'] = izmedian
                
                
                
            
            # buffer_size_with_log_max
            # buffer_size_default_max
            # buffer_size_with_log_clipped_max
            # buffer_size_default_clipped_max
            stats_geo.loc[stats_geo['zone_id'] == i, f"{base_n}_max"] = cons['max']
            stats_geo.loc[stats_geo['zone_id'] == i, f"{base_n}_mean"] = cons['mean']
            stats_geo.loc[stats_geo['zone_id'] == i, f"{base_n}_median"] = cons['median']
                
            del dset
            
            idx=idx+1


# In[392]:


ptest.loc[ptest['clipped'] ==1 ].head(5)


# In[425]:


ptest['clipped'] = ptest['layer'].apply(lambda s: 1 if 'clipped' in s else 0  )


# In[426]:


ptest_geo2 = gpd.GeoDataFrame(ptest, geometry='geometry', crs='EPSG:3301')
ptest_geo2.to_file('ptest_geo2.gpkg', driver='GPKG', layer='ptest_geo2')
ptest_geo2.loc[ptest_geo2['clipped'] ==1 ].to_file('ptest_geo2.gpkg', driver='GPKG', layer='ptest_geo2_clipped')


# In[427]:


ptest_geo2_clipped = gpd.read_file('ptest_geo2.gpkg', driver='GPKG', layer='ptest_geo2_clipped')


# In[428]:


from rasterstats import point_query

ptest_geo2_clipped['soil'] = np.nan
ptest_geo2_clipped['slope'] = np.nan

ptest_geo2_clipped['ls_factor'] = np.nan
ptest_geo2_clipped['flowlen'] = np.nan
ptest_geo2_clipped['flowacc'] = np.nan

ptest_geo2_clipped['specific_slopenlength_with_log_clipped'] = np.nan
ptest_geo2_clipped['specific_slopenlength_default_clipped'] = np.nan

ptest_geo2_clipped['buffer_size_with_log_clipped'] = np.nan
ptest_geo2_clipped['buffer_size_default_clipped'] = np.nan


for i in [3, 12,13,19, 21]:

    subdf = ptest_geo2_clipped.loc[ptest_geo2_clipped['zone_id'] == i]

    soil = f"R:/kmoch/nomograph/soil_prep/estsoil_labeled_pzone_{i}.tif"
    
    t1 = point_query(subdf['geometry'].values, soil, nodata=0)
    for tp in zip(subdf.index, pd.DataFrame( { 'soil': t1 } )['soil'].values ):
        ptest_geo2_clipped.loc[tp[0],'soil'] = tp[1]
    
    
    slope = f"R:/kmoch/nomograph/soil_prep/slope_5m_pzone_{i}.tif"
    
    t1 = point_query(subdf['geometry'].values, slope, nodata=-9999)
    for tp in zip(subdf.index, pd.DataFrame( { 'slope': t1 } )['slope'].values ):
        ptest_geo2_clipped.loc[tp[0],'slope'] = tp[1]

    ls_factor = f"R:/kmoch/nomograph/soil_prep/ls_faktor5m_pzone_{i}_cog.tif"
    
    t1 = point_query(subdf['geometry'].values, ls_factor, nodata=-3.4028230607370965e+38)
    for tp in zip(subdf.index, pd.DataFrame( { 'ls_factor': t1 } )['ls_factor'].values ):
        ptest_geo2_clipped.loc[tp[0],'ls_factor'] = tp[1]
    
    flowlen = f"R:/kmoch/nomograph/soil_prep/flowlength_5m_pzone_{i}.tif"
    
    t1 = point_query(subdf['geometry'].values, flowlen)
    for tp in zip(subdf.index, pd.DataFrame( { 'flowlen': t1 } )['flowlen'].values ):
        ptest_geo2_clipped.loc[tp[0],'flowlen'] = tp[1]
        
    flowacc = f"R:/kmoch/nomograph/soil_prep/flowacc_5m_pzone_{i}.tif"
    
    t1 = point_query(subdf['geometry'].values, flowacc)
    for tp in zip(subdf.index, pd.DataFrame( { 'flowacc': t1 } )['flowacc'].values ):
        ptest_geo2_clipped.loc[tp[0],'flowacc'] = tp[1]


    for base_n in source_cols:
        if 'clipped' in base_n:
            layer = f"{base_n}_pzone_{i}.tif"
            print(layer)

            t1 = point_query(subdf['geometry'].values, layer, nodata=-1)
            for tp in zip(subdf.index, pd.DataFrame( { 'base_n': t1 } )['base_n'].values ):
                ptest_geo2_clipped.loc[tp[0], base_n ] = tp[1]


# In[429]:


ptest_geo2_clipped.sample(5)


# In[430]:


ptest_geo2_clipped.to_file('ptest_geo2.gpkg', driver='GPKG', layer='ptest_geo2_clipped_sampled')


# In[431]:


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[438]:


corr_d = ptest_geo2_clipped[['buf_value', 'soil', 'slope', 'ls_factor', 'flowlen', 'flowacc',
       'specific_slopenlength_with_log_clipped',
       'specific_slopenlength_default_clipped', 'buffer_size_with_log_clipped',
       'buffer_size_default_clipped']]


# In[442]:


sns.scatterplot(data=corr_d, x="buf_value", y="soil", hue="slope")


# In[441]:


sns.scatterplot(data=corr_d, x="buf_value", y="slope", hue="soil")


# In[449]:


import numpy as np
import numpy.ma as ma

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

soil_dict = {
            "coarse_sand": {"k": 1.0, "alpha": 58.0},
            "fine_sand": {"k": 0.80, "alpha": 52.0},
            "loamy_sand": {"k": 0.61, "alpha": 45.0},
            "sandy_loam": {"k": 0.53, "alpha": 41.0},
            "sandy_clay_sandy_loam": {"k": 0.43, "alpha": 34.0},
            "clay_loam_sandy_clay": {"k": 0.33, "alpha": 29.0},
            "loam": {"k": 0.21, "alpha": 24.0},
        }

soil_classes_list_arr = np.array(
            [soil_dict[k]["alpha"] for k in soil_dict.keys()]
        )


# In[450]:


len(y_sl)


# In[456]:


pred_sl(25)


# In[454]:


np.take(soil_classes_list_arr, 1 - 1)


# In[435]:






corr = corr_d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[434]:


ptest_geo2_clipped.loc[ptest_geo2_clipped['buf_value'] > 150].sample(3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[421]:


ptest_geo2_clipped.columns


# In[415]:


# ptest_geo2_clipped.drop(columns=['specific_slopenlength_with_log', 'specific_slopenlength_default',
       'buffer_size_with_log', 'specific_slopenlength_with_log_clipped',
       'specific_slopenlength_default_clipped', 'buffer_size_with_log_clipped',
       'buffer_size_default_clipped', 'buffer_size_default'], inplace=True)


# In[407]:


len(ptest_geo2)


# In[361]:


oz_arr_loc = np.argwhere(band1 > 3*zvar)


# In[362]:


oz_arr_loc.shape


# In[374]:



ozdf = pd.DataFrame({'x': oz_arr_loc[:,0], 'y': oz_arr_loc[:,1]})
ozdf['px'] = ozdf['x'].apply(lambda x: x*x_coef+x_orig)
ozdf['py'] = ozdf['y'].apply(lambda y: y*y_coef+y_orig)
ozdf['buf_value'] = band1.data[ oz_arr_loc[:,0], oz_arr_loc[:,1] ]
ozdf['geometry'] = ozdf.apply(lambda row: Point(row['px'], row['py']), axis=1)


# In[375]:


ozdf


# In[346]:


oz_arr_loc = np.array([[ 439, 7400],
       [ 483, 7408],
       [ 490, 7409],
       [ 549, 7415],
       [ 550, 7415],
       [ 552, 7415],
       [ 587, 7414],
       [ 743, 7425],
       [ 775, 7429],
       [ 860, 7439],
       [ 966, 6416],
       [ 992, 6728],
       [ 994, 6731]], dtype=np.int64)

oz_arr_loc[ np.array( [ 0 ] ), np.array( [ 0 ] ) ]


# In[365]:


band1.data[ oz_arr_loc[:,0], oz_arr_loc[:,1] ]


# In[350]:


oz_arr_loc[:,0]


# In[356]:



band1.data[9999][9999]


# In[251]:


dir(dset.transform)


# In[286]:


x_coef = dset.transform.column_vectors[0][0]
y_coef = dset.transform.column_vectors[1][1]

x_orig = dset.transform.column_vectors[2][0]
y_orig = dset.transform.column_vectors[2][1]


# In[287]:


oz_arr_loc.shape


# In[290]:


from shapely.geometry import Point

ptest = gpd.GeoDataFrame()
ptest['geometry'] = None
ptest['buf_value'] = 0

for counter,i in enumerate(oz_arr_loc):
    print(i)
    print(band1[i[0]][i[1]])
    x = i[0]*x_coef+x_orig
    y = i[1]*y_coef+y_orig
    p = Point(x, y)
    ptest.loc[counter, 'geometry'] = p
    ptest.loc[counter, 'buf_value'] = band1[i[0]][i[1]]


# In[377]:


ptest_geo = gpd.GeoDataFrame(ozdf, geometry='geometry', crs='EPSG:3301')

fig, ax = plt.subplots(figsize=(10,7))

eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=ax)

ptest_geo.plot(column='buf_value', legend=True, ax=ax, alpha=0.8, cmap='Spectral_r', markersize=2)

plt.tight_layout()


# In[ ]:





# In[ ]:





# In[176]:


for x in filter(lambda x: 'max' in x, stats_geo.columns.tolist()):
    if 'buffer_size' in x:
        print(x)


# In[194]:



for x in filter(lambda x: 'max' in x, stats_geo.columns.tolist()):
    if 'buffer_size' in x:
        
        fig, ax = plt.subplots(figsize=(10,7))
        eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=ax)
        stats_geo.loc[stats_geo[x] < 1000].plot(column=x, legend=True, ax=ax, alpha=0.8, cmap='Spectral_r', vmin=0, vmax=300)
        
        for idx, row in stats_geo.iterrows():
            t_val = row[x]
            if not np.isnan(t_val):
                plt.text(row['cent_x'], row['cent_y'],"{} (max: {:.0f}m)\n".format(row['zone_id'],t_val),size=10, color='black')

        plt.title(f"{x}", fontsize=16)
        plt.tight_layout()


# In[198]:



for x in filter(lambda x: 'mean' in x, stats_geo.columns.tolist()):
    if 'buffer_size' in x:
        
        fig, ax = plt.subplots(figsize=(10,7))
        eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=ax)
        stats_geo.plot(column=x, legend=True, ax=ax, alpha=0.8, cmap='Spectral_r', vmax=10)
        
        for idx, row in stats_geo.iterrows():
            t_val = row[x]
            if not np.isnan(t_val):
                plt.text(row['cent_x'], row['cent_y'],"{} (max: {:.1f}m)\n".format(row['zone_id'],t_val),size=10, color='black')

        plt.title(f"{x}", fontsize=16)
        plt.tight_layout()


# In[238]:


import math

nrow = 0
ncol = 0

max_rows = 2
max_cols = 2

j = 0

fig, axs = plt.subplots(max_rows, max_cols, figsize=(18,18))

for x in  filter(lambda x: 'median' in x, stats_geo.columns.tolist()) :
    if 'buffer_size' in x:
        
        if j % max_rows == 0 and nrow > 0:
            ncol = ncol + 1

        nrow = j % max_rows
    
        print(f"idx ({j}) row col ({ncol} / {nrow})")
        
        eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=axs[nrow][ncol])
        stats_geo.plot(column=x, legend=True, ax=axs[nrow][ncol], alpha=0.8, cmap='Spectral_r', vmax=5)
        
        for idx, row in stats_geo.iterrows():
            t_val = row[x]
            if not np.isnan(t_val):
                axs[nrow][ncol].text(row['cent_x'], row['cent_y'],"{} (max: {:.1f}m)\n".format(row['zone_id'],t_val),size=10, color='black')

        ax=axs[nrow][ncol].set_title(f"{x}", fontsize=10)
        
        j=j+1

plt.tight_layout()


# In[241]:


import math

nrow = 0
ncol = 0

max_rows = 2
max_cols = 2

j = 0

fig, axs = plt.subplots(max_rows, max_cols, figsize=(18,18))

for x in  filter(lambda x: 'max' in x, stats_geo.columns.tolist()) :
    if 'buffer_size' in x:
        
        if j % max_rows == 0 and nrow > 0:
            ncol = ncol + 1

        nrow = j % max_rows
    
        print(f"idx ({j}) row col ({ncol} / {nrow})")
        
        eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=axs[nrow][ncol])
        stats_geo.plot(column=x, legend=True, ax=axs[nrow][ncol], alpha=0.8, cmap='Spectral_r', vmax=500)
        
        for idx, row in stats_geo.iterrows():
            t_val = row[x]
            if not np.isnan(t_val):
                axs[nrow][ncol].text(row['cent_x'], row['cent_y'],"{} (max: {:.1f}m)\n".format(row['zone_id'],t_val),size=10, color='black')

        ax=axs[nrow][ncol].set_title(f"{x}", fontsize=10)
        
        j=j+1

plt.tight_layout()


# In[238]:


import math

nrow = 0
ncol = 0

max_rows = 2
max_cols = 2

j = 0

fig, axs = plt.subplots(max_rows, max_cols, figsize=(18,18))

for x in  filter(lambda x: 'median' in x, stats_geo.columns.tolist()) :
    if 'buffer_size' in x:
        
        if j % max_rows == 0 and nrow > 0:
            ncol = ncol + 1

        nrow = j % max_rows
    
        print(f"idx ({j}) row col ({ncol} / {nrow})")
        
        eesti.boundary.plot(edgecolor='grey', lw=0.8, ax=axs[nrow][ncol])
        stats_geo.plot(column=x, legend=True, ax=axs[nrow][ncol], alpha=0.8, cmap='Spectral_r', vmax=5)
        
        for idx, row in stats_geo.iterrows():
            t_val = row[x]
            if not np.isnan(t_val):
                axs[nrow][ncol].text(row['cent_x'], row['cent_y'],"{} (max: {:.1f}m)\n".format(row['zone_id'],t_val),size=10, color='black')

        ax=axs[nrow][ncol].set_title(f"{x}", fontsize=10)
        
        j=j+1

plt.tight_layout()


# In[235]:


import math

nrow = 0
ncol = 0

max_rows = 4
max_cols = 3

for j in range (0,max_cols*max_rows):
    
    if j % max_rows == 0 and nrow > 0:
        ncol = ncol + 1
    
    nrow = j % max_rows
    
    print(f"idx ({j}) row col ({ncol} / {nrow})")


# In[228]:


2 % max_rows


# In[ ]:




