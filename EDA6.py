#plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster", font_scale=1.3)
import folium

#system packages
import os, sys
import warnings
warnings.filterwarnings('ignore')

#basic wrangling
import numpy as np
import pandas as pd

#eda tools
import pivottablejs
import missingno as msno
import pandas_profiling

#File with functions from prior notebook(s)
sys.path.append('/home/niangu/桌面/数据挖掘与建模/复习/scripts/')
from aqua_helper import time_slice, country_slice, time_series, simple_regions, subregion, variable_slice

#Update matplotlib defaults to something nicer
mpl_update = {'font.size': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'figure.figsize': [12.0, 8.0],
              #'axes.edgecolor':['#0055A7', '#2C3E4F', '#26C5ED', '#00cc66', '#D34100', '#FF9700','#091D32'],
              'axes.labelsize': 20,
              'axes.labelcolor': '#677385',
              'axes.titlesize': 20,
              'lines.color': '#0055A7',
              'lines.linewidth': 3,
              #'test.color': '#677385'
              }
mpl.rcParams.update(mpl_update)
data = pd.read_csv('aquastat.csv.gzip', compression='gzip')
data.region = data.region.apply(lambda x: simple_regions[x])

#Missing data
#By variable
recent = time_slice(data, '2013-2017')
msno.matrix(recent, labels=True)
plt.show()
#Deep dive:exploitable variables
msno.matrix(variable_slice(data, 'exploitable_total'), inline=False, sort='descending')
plt.xlabel('Time period')
plt.ylabel('Country')
plt.title('Missing total exploitable water resources data across countries and time periods \n \n \n \n')
plt.show()
data = data.loc[~data.variable.str.contains('exploitable'), :]
#Deep dive:National rainfall index
msno.matrix(variable_slice(data, 'national_rainfall_index'),
            inline=False, sort='descending')
plt.xlabel('Time period')
plt.ylabel('Country')
plt.title('Missing national rainfall index data across countries and time periods \n \n \n \n')
data = data.loc[~(data.variable=='national_rainfall_index')]
#By country
north_america = subregion(data, 'North America')
msno.matrix(msno.nullity_sort(time_slice(north_america, '2013-2017'), sort='descending').T, inline=False)
plt.show()
msno.nullity_filter(country_slice(data, 'Bahamas').T, filter='bottom', p=0.1)
#By country for a single variable
geo = r'world.json'

null_data = recent['agg_to_gdp'].notnull()*1
map = folium.Map(location=[48, -102], zoom_start=2)
map.choropleth(geo_data=geo,
                data=null_data,
                columns=['country', 'agg_to_gdp'],
                key_on='feature.properties.name', reset=True,
                fill_color='GnBu', fill_opacity=1, line_opacity=0.2,
                legend_name='Missing agricltural contribution to GDP data 2013-2017')
print(map)

def plot_null_map(df, time_period, variable, legend_name=None):
    geo = r'world.json'

    ts = time_slice(df, time_period).reset_index().copy()
    ts[variable] = ts[variable].notnull()*1
    map = folium.Map(location=[48, -102], zoom_start=2)
    map.choropleth(geo_data=geo,
                   data=ts,
                   columns=['country', variable],
                   key_on='feature.properties.name', reset=True,
                   fill_color='GnBu', fill_opacity=1, line_opacity=0.2,
                   legend_name=legend_name if legend_name else variable)
    return map
plot_null_map(data, '2013-2017', 'number_undernourished', 'Number undernourished is missing')

#Over time
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(data.groupby(['time_period', 'variable']).value.count().unstack().T, ax=ax)
plt.xticks(rotation=45)
plt.xlabel('Time period')
plt.ylabel('Variable')
plt.title('Number of countries with data reported for each variable over time')
plt.show()

pivottablejs.pivot_ui(time_slice(data, '2013-2017'),)
pandas_profiling.ProfileReport(time_slice(data, '2013-2017'))
