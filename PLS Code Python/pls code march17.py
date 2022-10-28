from sys import stdout
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("C:/Users/EME_Firmware/Desktop/R soft training/Analysis_Data/water analysis/water_13-21mar/water_sop3.csv")
#data.head()

y = data['Water'].values
X = data.values[:, 1:]


print(y.shape) 
print(X.shape)

# Plot the data
wl = np.arange(950, 1651, 3.53)
print(len(wl))
'''
wl = data.values[:,:]
print(wl)
'''

#plt.figure(figsize=(8,4.5))
with plt.style.context('ggplot'):
    plt.plot(wl, X.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Absorbance")
    plt.show()


# Perform second derivative

X2 = savgol_filter(X, 9, polyorder=2, deriv=2)

# plot and see
#plt.figure(figsize=(8, 4.5))
with plt.style.context('ggplot'):
    plt.plot(wl, X2.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("D2 Absorbance")
    plt.show()


def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rmse = math.sqrt(mse)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rmse, rpd)


# test with 40 components
r2s = []
mses = []
rmses = []
rpds = []
xticks = np.arange(1, 41)
for n_comp in xticks:
    y_cv, r2, mse, rmse, rpd = optimise_pls_cv(X2, y, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rmses.append(rmse)
    rpds.append(rpd)


# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()

plot_metrics(mses, 'MSE', 'min')
plot_metrics(rmses, 'RMSE', 'min')
plot_metrics(rpds, 'RPD', 'max')
plot_metrics(r2s, 'R2', 'max')

#create the optimize pls
y_cv, r2, mse, rmse, rpd = optimise_pls_cv(X2, y, 15)
print('R2: %0.4f, MSE: %0.4f, RMSE: %0.4f, RPD: %0.4f' %(r2, mse, rmse, rpd))

plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y, y_cv, color='red')
    plt.plot(y, y, '-g', label='Expected regression line')
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()
    plt.show()

'''
'''