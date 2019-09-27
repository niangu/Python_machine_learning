import matplotlib.pylab
import numpy as np
import pandas as pd

df = pd.Series(np.random.randn(600), index= pd.date_range('7/1/2016', freq='D', periods=600))
print(df.head())
r = df.rolling(window=10)
print(r)
print(r.mean())#r.max, r.median, r.std, r.skew, r.sum, r.var

import matplotlib.pyplot as plt1

plt1.figure(figsize=(15, 5))
df.plot(style='r--')
df.rolling(window=10).mean().plot(style='b')
plt1.show()