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

#interactive
import ipywidgets as widgets

#more technical eda
import sklearn
import scipy
sys.path.append('')
from aqua_helper import time_slice, country_slice, time_series, simple_regions, subregion, variable_slice
mpl_update = {
    'font.size': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': [12.0, 8.0],
    #'axes.color_cycle':['#0055A7', '#2C3E4F', '#26C5ED', '#00cc66', '#D34100', '#FF9700', '#091D32'],
    'axes.labelsize': 16,
    'axes.labelcolor': '#677385',
    #'axes.titlesize:20': 20,
    'lines.color': '#0055A7',
    'lines.linewidth': 3,
    #'text.color': '#677385'
}
mpl.rcParams.update(mpl_update)
data = pd.read_csv('aquastat.csv.gzip', compression='gzip')

data.region = data.region.apply(lambda x: simple_regions[x])

data = data.loc[~data.variable.str.contains('explotitable'), :]
data = data.loc[~(data.variable=='national_rainfall_index')]
recent = time_slice(data, '2013-2017')

#Cross-section
#Location and spread of the data
recent[['total_pop', 'urban_pop', 'rural_pop']].describe().astype(int)
recent.sort_values('rural_pop')[['total_pop', 'urban_pop', 'rural_pop']].head()
time_series(data, 'Qatar', 'total_pop').join(time_series(data, 'Qatar', 'urban_pop')).join(time_series(data, 'Qatar', 'rural_pop'))
#Shape of the data
recent[['total_pop', 'urban_pop', 'rural_pop']].describe().astype(int)
recent[['total_pop', 'urban_pop', 'rural_pop']].apply(scipy.stats.skew)
recent[['total_pop', 'urban_pop', 'rural_pop']].apply(scipy.stats.kurtosis)
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(recent.total_pop.values, bins=50)
ax.set_xlabel('Total population')
ax.set_ylabel('Number of countries')
ax.set_title('Distribution of population of countries 2013-2017')
plt.show()
recent[['total_pop']].apply(np.log).apply(scipy.stats.skew)
recent[['total_pop']].apply(np.log).apply(scipy.stats.kurtosis)

def plot_hist(df, variable, bins=20, xlabel=None, by=None, ylabel=None, title=None, logx=False, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 8))
    if logx:
        if df[variable].min() <=0:
            df[variable] = df[variable] - df[variable].min() + 1
            print('Warning: data <=0 exists, data transformed by %0.2g before plotting' % (- df[variable].min() + 1))
        bins = np.logspace(np.log10(df[variable].min()),
                           np.log10(df[variable].max()), bins)
        ax.set_xscale("log")
    ax.hist(df[variable].dropna().values, bins=bins)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


plot_hist(recent, 'total_pop', bins=25, logx=True, xlabel='Log of total population', ylabel='Number of countries',
          title='Distribution of total population of countries 2013-2017')
plt.show()
recent['population_density'] = recent.total_pop.divide(recent.total_area)

#One country
plt.plot(time_series(data, 'United States of America', 'total_pop'))
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('United States polpulation over time')
plt.show()

with sns.color_palette(sns.diverging_palette(220, 280, s=85, l=25, n=23)):
    north_america = time_slice(subregion(data, 'North America'), '1958-1962').sort_values('total_pop').index.tolist()
    for country in north_america:
        plt.plot(time_series(data, country, 'total_pop'), label=country)
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title('North American populations over time')
    plt.legend(loc=2, prop={'size': 10})
    plt.show()

with sns.color_palette(sns.diverging_palette(220, 280, s=85, l=25, n=23)):
    for country in north_america:
        ts = time_series(data, country, 'total_pop')
        ts['norm_pop'] = ts.total_pop/ts.total_pop.min()*100
        plt.plot(ts['norm_pop'], label=country)

north_america_pop = variable_slice(subregion(data, 'North America'), 'total_pop')
north_america_norm_pop = north_america_pop.div(north_america_pop.min(axis=1), axis=0)*100
north_america_norm_pop = north_america_norm_pop.loc[north_america]

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(north_america_norm_pop, ax=ax, cmap=sns.light_palette((214, 90, 60), input="husl", as_cmap=True))
plt.xticks(rotation=45)
plt.xlabel('Time period')
plt.ylabel('Country, ordered by population in 1960 (<- greatest to least ->)')
plt.title('Percent increase in population from 1960')
plt.show()

#Exploring total renewable water resources
plot_hist(recent, 'total_renewable', bins=50, xlabel='Total renewable water resources ($10^9 m^3/yr$)',
          ylabel='Number of countries',
          title='Distribution of total renewable water resources, 2013-2017')
plt.show()

plot_hist(recent, 'total_renewable', bins=50, xlabel='Total renewable water resources ($10^9 m^3/yr$)',
          ylabel='Number of countries', logx=True,
          title='Distribution of total renewable water resources, 2013-2017')
plt.show()
north_america_renew = variable_slice(subregion(data, 'North America'), 'total_renewable')

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(north_america_renew, ax=ax, cmap=sns.light_palette((214, 90, 60), input="husl", as_cmap=True))
plt.xticks(rotation=45)
plt.xlabel('Time period')
plt.ylabel('Country, ordered by Total renewable water resources in 1960 (<- greatest to least ->)')
plt.title('Total renewable water resources increase in population fro 1960')
plt.show()
#Assessing many variables
def two_hist(df, variable, bins=50,
             ylabel='Number of countries', title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1 = plot_hist(df, variable, bins=bins,
                    xlabel=variable, ylabel=ylabel, ax=ax1, title=variable if not title else title)
    ax2 = plot_hist(df, variable, bins=bins, xlabel='Log of'+ variable, ylabel=ylabel, logx=True, ax=ax2,
                    title='Log of' + variable if not title else title)
    plt.close()
    return fig


def hist_over_var(df, variables, bins=50, ylabel='Number of countries', title=None):
    variable_slider = widgets.Dropdown(options=variables.tolist(),
                                       value=variables[0],
                                       description='Variable',
                                       disabled=False,
                                       button_style='')
    widgets.interact(two_hist, df=widgets.fixed(df),
                     variable=variable_slider, ylabel=widgets.fixed(ylabel),
                     title=widgets.fixed(title), bins=widgets.fixed(bins))
    plt.show()

hist_over_var(recent, recent.columns, bins=20)

