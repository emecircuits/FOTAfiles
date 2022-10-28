from sys import stdout
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('C:/Users/EME_Firmware/Downloads/nirpyresearch-master/nirpyresearch-master/data/peach_spectra_brix.csv')

# Get reference values
y = data['Brix'].values

# Get spectra
X = data.drop(['Brix'], axis=1).values
# Get wavelengths
wl = np.arange(1100,2300,2)


#Plot absorbance
plt.figure(figsize=(8,4.5))
with plt.style.context(('ggplot')):
   plt.plot(wl,X.T)
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Absorbance')
   plt.show()


# Calculate second derivative
X2 = savgol_filter(X, 17, polyorder = 2,deriv=2)
 
# Plot second derivative
plt.figure(figsize=(8,4.5))
with plt.style.context(('ggplot')):
    plt.plot(wl, X2.T)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('D2 Absorbance')
    plt.show()
