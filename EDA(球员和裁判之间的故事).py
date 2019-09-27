from __future__ import absolute_import, division, print_function
import matplotlib as plt
from matplotlib.pyplot import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

import missingno as msno

from sklearn.datasets import make_blobs
import time

df = pd.read_csv("redcard.csv.gz", compression='gzip')
print(df.shape)
print(df.head())
print(df.describe().T)
print(df.dtypes)
all_columns = df.columns.tolist()
print(all_columns)

df["height"].mean()
df['height'].mean()
np.mean(df.groupby('playerShort').height.mean())
df2 = pd.DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
print(df2)
grouped = df2['data1'].groupby(df2['key1'])
grouped.mean()

player_index = 'playerShort'
player_cols = ['birthday', 'height', 'position', 'photoID', 'rater1', 'rater2']
all_cols_unique_players = df.groupby('playerShort').agg({col:'nunique' for col in player_cols})
print(all_cols_unique_players.head())

all_cols_unique_players[all_cols_unique_players > 1].dropna().head()
all_cols_unique_players[all_cols_unique_players > 1].dropna().shape[0] == 0


def get_subgroup(dataframe, g_index, g_columns):
    g = dataframe.groupby(g_index).agg({col:'nunique' for col in g_columns})
    if g[g > 1].dropna().shape[0] != 0:
        print("Warning：you probably assumed this had all unique values but it doesn't.")
    return dataframe.groupby(g_index).agg({col:'max' for col in g_columns})


players = get_subgroup(df, player_index, player_cols)
print(players.head())


def save_subgroup(dataframe, g_index, subgroup_name, prefix='raw'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    dataframe.to_csv(save_subgroup_filename, compression='gzip', encoding='UTF-8')
    test_df = pd.read_csv(save_subgroup_filename, compression='gzip', index_col=g_index, encoding='UTF-8')
    if dataframe.equals(test_df):
        print("Test-passed: we recover the equivalent subgroup dataframe.")
    else:
        print("Warning -- equivalence test !!! Double-check")


players = get_subgroup(df, player_index, player_cols)
print(players.head())
save_subgroup(players, player_index, "players")


#Create Tidy Clubs Table
club_index = 'club'
club_cols = ['leagueCountry']
clubs = get_subgroup(df, club_index, club_cols)
print(clubs.head())
clubs['leagueCountry'].value_counts()
save_subgroup(clubs, club_index, 'clubs',)


#Create Tidy Referees Table
referee_index = 'refNum'
referee_cols = ['refCountry']
referees = get_subgroup(df, referee_index, referee_cols)
print(referees.head())
print(referees.refCountry.nunique())
print(referees.tail())
print(referees.shape)
save_subgroup(referees, referee_index, "referees")

#Create Tidy Countries Table
country_index = 'refCountry'
country_cols = ['Alpha_3',#rename this name country
                'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp']
countries = get_subgroup(df, country_index, country_cols)
print(countries.head())

rename_columns={'Alpha_3':'countryName'}
countries = countries.rename(columns=rename_columns)
print(countries.head())
print(countries.shape)
save_subgroup(countries, country_index, "countries")

#Crete separate(not yet Tidy)Dyads Table
dyad_index = ['refNum', 'playerShort']
dyad_cols = ['games',
             'victories',
             'ties',
             'defeats',
             'goals',
             'yellowCards',
             'yellowReds',
             'redCards']
dyads = get_subgroup(df, g_index=dyad_index, g_columns=dyad_cols)
print(dyads.head())
print(dyads.shape)
print(dyads[dyads.redCards > 1].head(10))
save_subgroup(dyads, dyad_index, "dyads")
print(dyads.redCards.max())
