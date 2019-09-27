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
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

import missingno as msno
import pandas_profiling

from sklearn.datasets import make_blobs
import time

def save_subgroup(dataframe, g_index, subgroup_name, prefix='raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    dataframe.to_csv(save_subgroup_filename, compression='gzip', index_col=g_index, encoding='UTF-8')
    test_df = pd.read_csv(save_subgroup_filename, compression='gzip', index_col=g_index, encoding='UTF-8')
    if dataframe.equals(test_df):
        print("Test=passed:we recover the equivalent subgroup dataframe.")
    else:
        print("warning -- equivalence test!!! Double-check.")

def load_subgroup(filename, index_col=[0]):
    return pd.read_csv(filename, compression='gzip', index_col=index_col)


clean_players = load_subgroup("cleaned_players.csv.gz")
players = load_subgroup("raw_players.csv.gz")
countries = load_subgroup("raw_countries.csv.gz")
referees = load_subgroup("raw_referees.csv.gz")
agg_dyads = pd.read_csv("raw_dyads.csv.gz", compression='gzip', index_col=[0, 1])
tidy_dyads = pd.read_csv("cleaned_dyads.csv.gz", compression='gzip', index_col=[0, 1])

#Joining and further considrtations
clean_player = load_subgroup("cleaned_players.csv.gz")
temp = tidy_dyads.reset_index().set_index('playerShort').merge(clean_players, left_index=True, right_index=True)
print(temp.shape)
print(tidy_dyads.head())
(tidy_dyads.groupby(level=0)
           .sum()
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total redcards given'})).head()
(tidy_dyads.groupby(level=1)
           .sum().sort_values('redcard', ascending=False)
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total redcards received'})).head()
print(tidy_dyads.head())
total_ref_games = tidy_dyads.groupby(level=0).size().sort_values(ascending=False)
total_player_games = tidy_dyads.groupby(level=1).size().sort_values(ascending=False)
total_ref_given = tidy_dyads.groupby(level=0).sum().sort_values(ascending=False, by='redcard')
total_player_received = tidy_dyads.groupby(level=1).sum().sort_values(ascending=False, by='redcard')
sns.distplot(total_player_received, kde=False)
plt.show()
sns.distplot(total_ref_given, kde=False)
plt.show()
tidy_dyads.groupby(level=1).sum().sort_values(ascending=False, by='redcard').head()
print(tidy_dyads.sum())
print(tidy_dyads.sum(), tidy_dyads.count(), tidy_dyads.sum()/tidy_dyads.count())
player_ref_game = (tidy_dyads.reset_index().set_index('playerShort').merge(clean_players,
                                                                           left_index=True,
                                                                           right_index=True))
print(player_ref_game.head())
print(player_ref_game.shape)
bootstrap = pd.concat([player_ref_game.sample(replace=True,
                                               n=10000).groupby('skintone').mean() for _ in range(100)])
player_ref_game.sample(replace=True, n=10000).groupby('skintone').mean()
ax = sns.regplot(bootstrap.index.values,y='redcard',data=bootstrap, lowess=True,scatter_kws={'alpha':0.4,},
                 x_jitter=(0.125 / 4.0))
ax.set_xlabel("Skintone")
plt.show()