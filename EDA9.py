import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
dailyData = pd.read_csv("train.csv")

print(dailyData.shape)
print(dailyData.head(2))
print(dailyData.dtypes)
dailyData["date"] = dailyData.datetime.apply(lambda x: x.split()[0])
dailyData["hour"] = dailyData.datetime.apply(lambda x: x.split()[1].split(":")[0])
dailyData["weekday"] = dailyData.date.apply(lambda dateString: calendar.day_name[datetime.strptime(dateString, "%Y-%m-%d").weekday()])
dailyData["month"] = dailyData.date.apply(lambda dateString: calendar.month_name[datetime.strptime(dateString, "%Y-%m-%d").month])
dailyData["season"] = dailyData.season.map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})
dailyData["weather"] = dailyData.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
categoryVariableList = ["hour", "weekday", "month", "season", "weather", "holiday", "workingday"]
for var in categoryVariableList:
    dailyData[var] = dailyData[var].astype("category")

dailyData = dailyData.drop(["datetime"], axis=1)
dataTypeDf = pd.DataFrame(dailyData.dtypes.value_counts()).reset_index().rename(columns={"index": "variableType", 0:"count"})
fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
sn.barplot(data=dataTypeDf, x="variableType", y="count", ax=ax)
ax.set(xlabel='variableTypeariable Type', ylabel='Count', title="Variables DataType Count")
plt.show()
#Missing Values Analysis
msno.matrix(dailyData, figsize=(12, 5))
plt.show()
#Outliers Analysis
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)
sn.boxplot(data=dailyData, y="count", orient="v", ax=axes[0][0])
sn.boxplot(data=dailyData, y="count", x="season", orient="v", ax=axes[0][1])
sn.boxplot(data=dailyData, y="count", x="hour", orient="v", ax=axes[1][0])
sn.boxplot(data=dailyData, y="count", x="workingday", orient="v", ax=axes[1][1])

axes[0][0].set(ylabel='Count', title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count', title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count', title="Box Plot On Count Across Season")
axes[1][1].set(xlabel='Working Day', ylabel='Count', title="Box Plot On Count Across Hour Of The Day")
plt.show()

dailyDataWithoutOutliers = dailyData[np.abs(dailyData["count"]-dailyData["count"].mean())<=(3*dailyData["count"].std())]
print("Shape Of The Before Ouliers:", dailyData.shape)
print("Shape Of The After Ourliers:", dailyDataWithoutOutliers.shape)
corrMatt = dailyData[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
plt.show()

fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)
fig.set_size_inches(12,20)
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

monthAggregated = pd.DataFrame(dailyData.groupby("month")["count"].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by="count",ascending=False)
sn.barplot(data=monthSorted,x="month",y="count",ax=ax1,order=sortOrder)
ax1.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")

hourAggregated = pd.DataFrame(dailyData.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')

hourAggregated = pd.DataFrame(dailyData.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax3)
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')

hourTransformed = pd.melt(dailyData[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax4)
ax4.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across User Type",label='big')
plt.show()
