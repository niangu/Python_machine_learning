from __future__ import absolute_import, division, print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import warnings
import pandas_profiling
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

import missingno as msno
from sklearn.datasets import make_blobs
import time


def save_subgroup(dataframe, g_index, subgroup_name, prefix='raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    dataframe.to_csv(save_subgroup_filename, compression='gzip', encoding='UTF-8')
    test_df = pd.read_csv(save_subgroup_filename, compression='gzip', index_col=g_index, encoding='UTF-8')
    if dataframe.equals(test_df):
        print("Test-passed: we recover the equivalent subgroup dataframe")
    else:
        print("Warning -- equivalence test!!! Double-check")


def load_subgroup(filename, index_col=[0]):
    return pd.read_csv(filename, compression='gzip', index_col=index_col)


#Players
players = load_subgroup("raw_players.csv.gz")
print(players.head())
print(players.shape)

#Visualize the missing-ness of data
msno.matrix(players.sample(500), figsize=(16, 7), width_ratios=(15, 1))

msno.heatmap(players.sample(500), figsize=(16, 7))
plt.show()
print("All players:", len(players))
print("rater1 nulls:", len(players[(players.rater1.isnull())]))
print("rater2 nulls:", len(players[players.rater2.isnull()]))
print("Both nulls", len(players[(players.rater1.isnull()) & (players.rater2.isnull())]))

#modifying dataframe
players = players[players.rater1.notnull()]
print(players.shape[0])
print(2053-1585)#removed 468 个数据
msno.matrix(players.sample(500),
            figsize=(16, 7),
            width_ratios=(15, 1))
plt.show()
pd.crosstab(players.rater1, players.rater2)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pd.crosstab(players.rater1, players.rater2), cmap='Blues', annot=True, fmt='d', ax=ax)
ax.set_title("Correlation between Rater 1 and Rater 2\n")
fig.tight_layout()

print(players.head())
#modifying dataframe
players['skintone'] = players[['rater1', 'rater2']].mean(axis=1)
plt.show()
print(players.head())


#Visualize distributions of univariate features
sns.distplot(players.skintone, kde=False);
plt.show()
#Positions
MIDSIZE = (12, 8)
fig, ax = plt.subplots(figsize=MIDSIZE)
players.position.value_counts(dropna=False, ascending=True).plot(kind='barh', ax=ax)
ax.set_ylabel("Position")
ax.set_xlabel("Counts")
fig.tight_layout()
plt.show()

#Create higher level categories
position_types = players.position.unique()
print(position_types)

defense = ['Center Back', 'Defensive Midfierlder', 'Left Fullback', 'Right Fullback', ]
midfield = ['Right Midfielder', 'Center Midfierlder', 'Left Midfielder', ]
forward = ['Attacking Midfielder', 'Left Winger', 'Right Winger', 'Center Forward']
keeper = 'Goalkeeper'

players.loc[players['position'].isin(defense), 'position_agg'] = "Defense"
players.loc[players['position'].isin(midfield), 'position_agg'] = "Midfield"
players.loc[players['position'].isin(forward), 'position_agg'] = "Forward"
players.loc[players['position'].eq(keeper), 'position_agg'] = "Keeper"

MIDSIZE = (12, 8)
fig, ax = plt.subplots(figsize=MIDSIZE)
players['position_agg'].value_counts(dropna=False, ascending=True).plot(kind='barh', ax=ax)
ax.set_ylabel("position_agg")
ax.set_xlabel("Counts")
fig.tight_layout()
plt.show()


from pandas.tools.plotting import scatter_matrix

fig, ax = plt.subplots(figsize=(10, 10))
scatter_matrix(players[['height', 'weight', 'skintone']], alpha=0.2, diagonal='hist', ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=MIDSIZE)
sns.regplot('weight', 'height', data=players, ax=ax)
ax.set_ylabel("Height [cm]")
ax.set_xlabel("Weight [kg]")
fig.tight_layout()
plt.show()

#Create_quantile bins for continuous variables
weight_categories = ["vlow_weight",
                     "low_weight",
                     "mid_weight",
                     "high_weight",
                     "vhigh_weight",
                     ]
players['weightclass'] = pd.qcut(players['weight'],
                                 len(weight_categories),
                                 weight_categories)
print(players.head())
height_categories = ["vlow_height",
                     "low_height",
                     "mid_height",
                     "high_height",
                     "vhigh_height"]
players['heightclass'] = pd.qcut(players['height'],
                                 len(height_categories),
                                 height_categories)
print(players['skintone'])
pd.qcut(players['skintone'], 3)
players['skintoneclass'] = pd.qcut(players['skintone'], 3)

#Pandas profiling
pandas_profiling.ProfileReport(players)
plt.show()

#Question -- What to do with birthday column?
print(players.birthday.head())
players['birth_date'] = pd.to_datetime(players.birthday, format='%d.%m.%Y')
players['age_years'] = ((pd.to_datetime("2013-01-01") - players['birth_date']).dt.days)/365.25

#Select vtiables to (possibly) use
print(players.head())
players_cleaned_variables = players.columns.tolist()
print(players_cleaned_variables)
players_cleaned_variables = [#'birthday',
                             'height',
                             'weight',
#                              'position',
#                              'photoID',
#                              'rater1',
#                              'rater2',
                             'skintone',
                             'position_agg',
                             'weightclass',
                             'heightclass',
                             'skintoneclass',
#                              'birth_date',
                             'age_years']
#pandas_profiling.ProfileReport(players[players_cleaned_variables])
#players[players_cleaned_variables].to_csv("cleaned_players.csv.gz", compression='gzip')
