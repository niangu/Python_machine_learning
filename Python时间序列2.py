import pandas as pd
import numpy as np

rng = pd.date_range('1/1/2011', periods=90, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.head())
print(ts.resample('M').sum())
print(ts.resample('3D').sum())
day3Ts = ts.resample('3D').mean()
print(day3Ts)
print(day3Ts.resample('D').asfreq())
print(day3Ts.resample('D').ffill(1))
print(day3Ts.resample('D').bfill(1))
print(day3Ts.resample('D').interpolate('linear'))

