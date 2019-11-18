from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
import statsmodels.graphics.tsaplots as st
#import seaborn as sns
import math


##-------------------------------------Part-1---------------------------
#
series = read_csv('/home//Desktop/sem3/ds3/data_science_3/lab9/inLab/SoilForce.csv', header=0,
index_col=0)
values = DataFrame(series.values)
plt.xlabel('Date')
plt.ylabel('Force')
plt.plot(values)
plt.show()
#print(values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
#print(dataframe) spilted the data
result = dataframe.corr()
print(result)



##-------------------------------------Part-2
#----------------------------
#if lag =0 then error: index 1 is out of bounds for axis 0 with size 1
st.plot_acf(series,lags=35)
plt.show()
#we limited the lag varaiables

#--------------------------------------Part-3----------------------------


from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
series = read_csv('/home//Desktop/sem3/ds3/data_science_3/lab9/inLab/SoilForce.csv', header=0,
index_col=0)
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[0:71], X[71:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
        return x

# walk-forward validation
predictions = list()
for x in test_X:
        yhat = model_persistence(x)
        predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % math.sqrt(test_score))
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
#
#
#----------------------Part
#-4--------------------------------------------

from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
series = read_csv('/home/Desktop/sem3/ds3/data_science_3/lab9/inLab/SoilForce.csv', header=0,
index_col=0)
# split dataset
X = series.values
train, test = X[0:71], X[71:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
                yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % math.sqrt(error))
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
