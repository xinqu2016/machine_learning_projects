from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model
from scipy import stats
reg = linear_model.LinearRegression()

dir_path = '/Users/xinqu/Projects/gig2_backup/LCC_EIS_SST_relationship'
filename = '/Users/xinqu/Projects/gig2_backup/low_cloud_IPCC/LCC_EIS_SST_for_ridge_regression.nc'
f_cl = Dataset(filename, 'r')
LCC = f_cl.variables['LCC'][:]
EIS = f_cl.variables['EIS'][:]
SST = f_cl.variables['SST'][:]
f_cl.close()

LCC_dim = LCC.shape;
print(LCC_dim[0])
X = np.ma.zeros((LCC_dim[0],2))
X[:,0]=EIS
X[:,1]=SST
reg.fit(X, LCC.reshape(-1,1))
print(reg.coef_)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=2500)
clf.fit(X, LCC.reshape(-1,1))
print(clf.coef_)
