# This program investigates the interactions between clouds and sea ice in climate models using various machine learning algorithms

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import detrend
import itertools
from sklearn import linear_model
from scipy import stats
from mpl_toolkits.basemap import Basemap, cm, interp, shiftgrid
reg = linear_model.LinearRegression()
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import copy
import random
# change default matplotlib resources
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('mathtext', fontset='custom')
rc('axes',linewidth='1.5')
rc('text.latex', preamble = r'\usepackage[tx]{sfmath}')
#params = {'mathtext.default': 'bf'}
#plt.rcParams.update(params)
dir_path = '/Users/xinqu/Projects/gig2_backup/SAF_SWcld_feedback_correlation'

# Annual-mean analysis
model_name_26 = ("ACCESS1-3", "bcc-csm1-1", "bcc-csm1-1-m", "BNU-ESM", "CCSM4", "CanESM2", "FGOALS-s2", "GFDL-CM3","HadGEM2-ES", "inmcm4", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC-ESM", "MIROC5","MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P", "MRI-CGCM3", "NorESM1-M","ACCESS1-0", "CSIRO-Mk3-6-0","GFDL-ESM2G","GFDL-ESM2M","GISS-E2-H","GISS-E2-R")
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)

filename6 = "/Users/xinqu/Projects/gig2_backup/SAF_SWcld_feedback_correlation/data/rsdt_Amon_HadGEM2-ES_piControl_r1i1p1_185912-188411.nc"
f_rsdt = Dataset(filename6, 'r')
rsdt_mean = np.ma.average(f_rsdt['rsdt'][0:12,:,:],0)
(lat_rsdt, lon_rsdt) = (f_rsdt.variables['lat'][:], f_rsdt.variables['lon'][:])
rsdt_mean_regridded = np.ma.zeros((90,144))
lon_2d, lat_2d = np.meshgrid(lon, lat)
rsdt_mean_regridded = interp(rsdt_mean,lon_rsdt, lat_rsdt, lon_2d, lat_2d, False, False, 1)

rsdt_mean_regridded_annual = rsdt_mean_regridded

# 2D surface albedo and cloud shortwave feedbacks

Albedo_2d = np.ma.zeros((90, 144, 5, 26))
SW_cloud_2d = np.ma.zeros((90, 144, 5, 26))
SW_cloudamt_2d = np.ma.zeros((90, 144, 5, 26))
SW_cloudscat_2d = np.ma.zeros((90, 144, 5, 26))
for i in range(26):
    print(i)
    filename1 = "/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc".format(model_name_26[i])
    f_cl_1 = Dataset(filename1, 'r')
    Albedo_2d[:, :, 0, i] = np.ma.average(f_cl_1.variables['sfcalb_APRP_fbk'][:, :, :],0)/rsdt_mean_regridded
    Albedo_2d[:, :, 1, i] = np.ma.average(f_cl_1.variables['sfcalb_APRP_fbk'][[0, 1,11], :, :], 0)
    Albedo_2d[:, :, 2, i] = np.ma.average(f_cl_1.variables['sfcalb_APRP_fbk'][[2, 3, 4], :, :], 0)
    Albedo_2d[:, :, 3, i] = np.ma.average(f_cl_1.variables['sfcalb_APRP_fbk'][[5, 6, 7], :, :], 0)
    Albedo_2d[:, :, 4, i] = np.ma.average(f_cl_1.variables['sfcalb_APRP_fbk'][[8, 9, 10], :, :], 0)
    SW_cloud_2d[:, :, 0, i] = np.ma.average(f_cl_1.variables['cloud_APRP_fbk'][:, :, :],0)/rsdt_mean_regridded
    SW_cloud_2d[:, :, 1, i] = np.ma.average(f_cl_1.variables['cloud_APRP_fbk'][[0, 1,11], :, :], 0)
    SW_cloud_2d[:, :, 2, i] = np.ma.average(f_cl_1.variables['cloud_APRP_fbk'][[2, 3, 4], :, :], 0)
    SW_cloud_2d[:, :, 3, i] = np.ma.average(f_cl_1.variables['cloud_APRP_fbk'][[5, 6, 7], :, :], 0)
    SW_cloud_2d[:, :, 4, i] = np.ma.average(f_cl_1.variables['cloud_APRP_fbk'][[8, 9, 10], :, :], 0)
    SW_cloudamt_2d[:, :, 0, i] = np.ma.average(f_cl_1.variables['cloud_amt_APRP_fbk'][:, :, :],0)/rsdt_mean_regridded
    SW_cloudamt_2d[:, :, 1, i] = np.ma.average(f_cl_1.variables['cloud_amt_APRP_fbk'][[0, 1, 11], :, :], 0)
    SW_cloudamt_2d[:, :, 2, i] = np.ma.average(f_cl_1.variables['cloud_amt_APRP_fbk'][[2, 3, 4], :, :], 0)
    SW_cloudamt_2d[:, :, 3, i] = np.ma.average(f_cl_1.variables['cloud_amt_APRP_fbk'][[5, 6, 7], :, :], 0)
    SW_cloudamt_2d[:, :, 4, i] = np.ma.average(f_cl_1.variables['cloud_amt_APRP_fbk'][[8, 9, 10], :, :], 0)
    SW_cloudscat_2d[:, :, 0, i] = np.ma.average(f_cl_1.variables['cloud_scat_APRP_fbk'][:, :, :]+f_cl_1.variables['cloud_abs_APRP_fbk'][:, :, :],0)/rsdt_mean_regridded
    SW_cloudscat_2d[:, :, 1, i] = np.ma.average(f_cl_1.variables['cloud_scat_APRP_fbk'][[0, 1,11], :, :]+f_cl_1.variables['cloud_abs_APRP_fbk'][[0, 1,11], :, :], 0)
    SW_cloudscat_2d[:, :, 2, i] = np.ma.average(f_cl_1.variables['cloud_scat_APRP_fbk'][[2, 3, 4], :, :]+f_cl_1.variables['cloud_abs_APRP_fbk'][[2, 3, 4], :, :], 0)
    SW_cloudscat_2d[:, :, 3, i] = np.ma.average(f_cl_1.variables['cloud_scat_APRP_fbk'][[5, 6, 7], :, :]+f_cl_1.variables['cloud_abs_APRP_fbk'][[5, 6, 7], :, :], 0)
    SW_cloudscat_2d[:, :, 4, i] = np.ma.average(f_cl_1.variables['cloud_scat_APRP_fbk'][[8, 9, 10], :, :]+f_cl_1.variables['cloud_abs_APRP_fbk'][[8, 9, 10], :, :], 0)
    # filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_2d/{0}_forcings_fdbks_v15.nc'.format(model_name_25[i])
    # f_cl = Dataset(filename, 'r')
    # Albedo_25_2d[:, :, i] = f_cl.variables['alb_fbk'][0, :, :]
    # f_cl_1.close()

# model_name_28=('CNRM-CM5','BNU-ESM','FGOALS-s2','INMCM4','IPSL-CM5A-LR','IPSL-CM5A-MR','CanESM2','MIROC5','MRI-CGCM3','GISS-E2-H','MIROC-ESM','GFDL-ESM2M','IPSL-CM5B-LR','HadGEM2-ES','GFDL-CM3','MPI-ESM-LR','BCC-CSM1-1-M','ACCESS1-0','MPI-ESM-P','MPI-ESM-MR','BCC-CSM1-1','ACCESS1-3','CCSM4','FGOALS-g2','NorESM1-M','GFDL-ESM2G','CSIRO-Mk3-6-0','GISS-E2-R')
# filename1 = "/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_2d/CMIP5_fdbks_v15_3periods.nc"
# f_cl_1 = Dataset(filename1, 'r')
# jj=[21,20,16,1,22,6,23,14,13,3,4,5,12,10,7,15,19,18,8,24,17,26,25,11,9]
# model_name_28_reordered = [model_name_28[i] for i in jj]
# for i in range(3):
#     for j in range(5):
#         Albedo_25_2d[:,:,j,i,:] = f_cl_1.variables['ALBfbk3'][i,:,:,jj]
# plot 1
lon_len = len(lon)
meanSAF = 100*np.ma.average(Albedo_2d[:, :, 0, :],axis=2)
meanSAF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSAF_new[:,0:lon_len]=meanSAF[:,:]
meanSAF_new[:,lon_len]  =meanSAF[:,0]
meanSAF = meanSAF_new
del meanSAF_new

meanSWCF = 100*np.ma.average(SW_cloud_2d[:, :, 0, :],axis=2)
meanSWCF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSWCF_new[:,0:lon_len]=meanSWCF[:,:]
meanSWCF_new[:,lon_len]  =meanSWCF[:,0]
meanSWCF = meanSWCF_new
meanSWCF[-1,:]=meanSWCF[-2,:]
del meanSWCF_new

meanSWCAF = 100*np.ma.average(SW_cloudamt_2d[:, :, 0, :],axis=2)
meanSWCAF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSWCAF_new[:,0:lon_len]=meanSWCAF[:,:]
meanSWCAF_new[:,lon_len]  =meanSWCAF[:,0]
meanSWCAF = meanSWCAF_new
meanSWCAF[-1,:]=meanSWCAF[-2,:]
del meanSWCAF_new

meanSWCSAF = 100*np.ma.average(SW_cloudscat_2d[:, :, 0, :],axis=2)
meanSWCSAF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSWCSAF_new[:,0:lon_len]=meanSWCSAF[:,:]
meanSWCSAF_new[:,lon_len]  =meanSWCSAF[:,0]
meanSWCSAF = meanSWCSAF_new
meanSWCSAF[-1,:]=meanSWCSAF[-2,:]
del meanSWCSAF_new

lon_new = np.ones((lon_len+1))
lon_new[0:lon_len]=lon[:]
lon_new[lon_len]=lon[0]
lon=lon_new
del lon_new

lat[0]=-90
lat[-1]=90
lon, lat = np.meshgrid(lon,lat)

#SH
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.8, 9.6))
fig.subplots_adjust(wspace=0.1,hspace=0.3)
m1 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0,0])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m1.fillcontinents(color='0.8')
    # draw parallels and meridians, but don't bother labelling them.
m1.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m1.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap1 = copy.deepcopy(plt.cm.jet)
cmap1.set_under((0.000000,0.0,0.500000))
im1 = m1.contourf(lon,lat,meanSAF,shading='flat',cmap=cmap1,latlon=True, vmin=-2, levels=[-1,-0.5, 0,0.5, 1, 1.5,2,2.5], extend="both")
cb1 = m1.colorbar(im1,"bottom", size="5%", pad="8%", cmap=cmap1,label="\% K$^{-1}$")
axes[0,0].set_title('(a) SAF',y=1.08)

m2 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0,1])
#m2.drawcoastlines(linewidth=0.3)
m2.fillcontinents(color='0.8')
m2.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m2.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im2 = m2.contourf(lon, lat, meanSWCF, shading='flat', cmap=cmap2, latlon=True, vmin=-1.2,vmax=1,levels=[-1,-0.75, -0.5, -0.25, 0,0.25, 0.5], extend="both")
cb2 = m2.colorbar(im2,"bottom", size="5%", pad="8%", label="\% K$^{-1}$")
axes[0,1].set_title('(b) SWCF',y=1.08)

m3 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1,0])
#m2.drawcoastlines(linewidth=0.3)
m3.fillcontinents(color='0.8')
m3.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m3.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im3 = m3.contourf(lon, lat, meanSWCAF, shading='flat', cmap=cmap2, latlon=True, vmin=-1.2,vmax=1,levels=[-1,-0.75, -0.5, -0.25, 0,0.25, 0.5], extend="both")
cb3 = m3.colorbar(im3,"bottom", size="5%", pad="8%", label="\% K$^{-1}$")
axes[1,0].set_title('(c) SWCAF',y=1.08)

m4 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1,1])
#m2.drawcoastlines(linewidth=0.3)
m4.fillcontinents(color='0.8')
m4.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m4.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im4 = m4.contourf(lon, lat, meanSWCSAF, shading='flat', cmap=cmap2, latlon=True, vmin=-1.2,vmax=1,levels=[-1,-0.75, -0.5, -0.25, 0,0.25, 0.5], extend="both")
cb4 = m4.colorbar(im4,"bottom", size="5%", pad="8%", label="\% K$^{-1}$")
axes[1,1].set_title('(d) SWCSAF',y=1.08)
plt.savefig('SAF_cloud_APRP_SH.pdf', papertype='letter', \
            orientation='landscape', bbox_inches='tight',pad_inches=0.5)


filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)

reg_coef = np.ma.zeros(26)
reg_intercept = np.ma.zeros(26)
reg_coef_cloudamt = np.ma.zeros(26)
reg_intercept_cloudamt = np.ma.zeros(26)
reg_coef_cloudscat = np.ma.zeros(26)
reg_intercept_cloudscat = np.ma.zeros(26)
Albedo_2d_saved  = Albedo_2d.copy()
SW_cloud_2d_saved  = SW_cloud_2d.copy()
SW_cloudamt_2d_saved  = SW_cloudamt_2d.copy()
SW_cloudscat_2d_saved  = SW_cloudscat_2d.copy()

f_landsea = Dataset(dir_path + '/sftlf_ncl.nc','r')
sftlf     = f_landsea.variables['sftlf'][:]
Albedo_2d_saved[lat >= -60, :, :,:] = ma.masked
Albedo_2d_saved[sftlf >= 30, :,:] = ma.masked
SW_cloud_2d_saved[lat >= -60, :,:,:] = ma.masked
SW_cloud_2d_saved[sftlf >= 30, :,:] = ma.masked
SW_cloudamt_2d_saved[lat >= -60, :,:,:] = ma.masked
SW_cloudamt_2d_saved[sftlf >= 30, :,:] = ma.masked
SW_cloudscat_2d_saved[lat >= -60, :,:,:] = ma.masked
SW_cloudscat_2d_saved[sftlf >= 30, :,:] = ma.masked
for i in range(26):
    x1 = Albedo_2d_saved[:, :, 0, i].flatten()
    y1 = SW_cloud_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef[i]=reg.coef_[0][0]
    reg_intercept[i]=reg.intercept_[:]
    y1 = SW_cloudamt_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_cloudamt[i]=reg.coef_[0][0]
    reg_intercept_cloudamt[i]=reg.intercept_[:]
    y1 = SW_cloudscat_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_cloudscat[i]=reg.coef_[0][0]
    reg_intercept_cloudscat[i]=reg.intercept_[:]
Albedo_SO = np.zeros((5,26))
SW_cloud_SO    = np.zeros((5,26))
SW_cloudamt_SO    = np.zeros((5,26))
SW_cloudscat_SO    = np.zeros((5,26))
for i in range(26):
  for imonth in range(5):
    Albedo_SO[imonth,i] = np.ma.average(Albedo_2d_saved[:,:,imonth,i], None, weight_2d)
    SW_cloud_SO[imonth,i]    = np.ma.average(SW_cloud_2d_saved[:,:,imonth,i], None, weight_2d)
    SW_cloudamt_SO[imonth, i] = np.ma.average(SW_cloudamt_2d_saved[:, :, imonth, i], None, weight_2d)
    SW_cloudscat_SO[imonth, i] = np.ma.average(SW_cloudscat_2d_saved[:, :, imonth, i], None, weight_2d)
print(stats.mstats.pearsonr(SW_cloud_SO[0,:],SW_cloudamt_SO[0,:])[0])
print(stats.mstats.pearsonr(SW_cloud_SO[0,:],SW_cloudscat_SO[0,:])[0])
print(stats.mstats.pearsonr(reg_coef,reg_coef_cloudamt)[0])
print(stats.mstats.pearsonr(reg_coef,reg_coef_cloudscat)[0])

#plot 2
model_name_26 = ("ACCESS1-3", "BCC-CSM1-1", "BCC-CSM1-1-m", "BNU-ESM", "CCSM4", "CanESM2", "FGOALS-s2", "GFDL-CM3","HadGEM2-ES", "inmcm4", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC-ESM", "MIROC5","MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P", "MRI-CGCM3", "NorESM1-M","ACCESS1-0", "CSIRO-Mk3-6-0","GFDL-ESM2G","GFDL-ESM2M","GISS-E2-H","GISS-E2-R")
from string import ascii_lowercase
from matplotlib.ticker import AutoMinorLocator
minorLocator = AutoMinorLocator(2)

ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.33,hspace=0.65)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='10', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='10', right='on', top='on')
    plt.scatter(100*Albedo_2d_saved[:,:,0,ind_sorted[i]],100*SW_cloud_2d_saved[:,:,0,ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("SWCF (\% K$^{-1}$)",fontsize=10)
    if i in range(22,26):
        plt.xlabel("SAF (\% K$^{-1}$)",fontsize=10)

    RR = stats.mstats.pearsonr(100*Albedo_2d_saved[:,:,0,ind_sorted[i]], 100*SW_cloud_2d_saved[:,:,0,ind_sorted[i]])[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=10, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),100*reg_intercept[ind_sorted[i]]+np.array([-20,30])*reg_coef[ind_sorted[i]],color="black")
    plt.ylim(-4.2,2.2)
    plt.xlim(-8,12)
    plt.text(11, -2.2, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=10)
    plt.text(11, -3.2, 's={:.2f}'.format(reg_coef[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=10)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_SWcloud_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 3
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(100*Albedo_2d_saved[:,:,0,ind_sorted[i]],100*SW_cloudamt_2d_saved[:,:,0,ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("SWCAF (\% K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("SAF (\% K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(100*Albedo_2d_saved[:,:,0,ind_sorted[i]], 100*SW_cloudamt_2d_saved[:,:,0,ind_sorted[i]])[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),100*reg_intercept_cloudamt[ind_sorted[i]]+np.array([-20,30])*reg_coef_cloudamt[ind_sorted[i]],color="black")
    plt.ylim(-4.2,2.2)
    plt.xlim(-8,12)
    plt.text(11, -2.2, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(11, -3.2, 's={:.2f}'.format(reg_coef_cloudamt[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_SWcloudamt_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)


#plot 4
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(100*Albedo_2d_saved[:,:,0,ind_sorted[i]],100*SW_cloudscat_2d_saved[:,:,0,ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("SWCSAF (\% K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("SAF (\% K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(Albedo_2d_saved[:,:,0,ind_sorted[i]], SW_cloudscat_2d_saved[:,:,0,ind_sorted[i]])[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),100*reg_intercept_cloudscat[ind_sorted[i]]+np.array([-20,30])*reg_coef_cloudscat[ind_sorted[i]],color="black")
    plt.ylim(-4.2,2.2)
    plt.xlim(-8,12)
    plt.text(11, -2.2, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(11, -3.2, 's={:.2f}'.format(reg_coef_cloudscat[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_SWcloudscat_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 4.5
fig, axes = plt.subplots(figsize=(7, 6.5))
ind_sorted = np.argsort(reg_coef)
x = reg_coef*Albedo_SO[0,:]*100
y = reg_intercept*100
reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
plt.plot([-2, 2], reg.intercept_ + [-2, 2] * reg.coef_[0], "k", linewidth='1')
plt.minorticks_on()
axes.tick_params(direction='out', length=12*3/4, width=1.5*3/4, colors='black', labelsize='18', right='on', top='on',pad=8)
axes.tick_params(direction='out', which='minor', length=9*3/4, width=0.5*3/4, colors='black', \
                     labelsize='18', right='on', bottom='on', top='on')
for j in range(26):
    plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1), horizontalalignment='center', verticalalignment='center', fontsize=18)
plt.ylabel("SAF-unrelated SWCF (\% K$^{-1}$)", fontsize=18)
plt.xlabel("SAF-related SWCF (\% K$^{-1}$)", fontsize=18)
plt.title("r=-0.64", fontsize=18, y=1.03)
print(stats.mstats.pearsonr(x, y))
print([reg.intercept_, reg.coef_[0][0]])
plt.xlim(-0.75, 0.75)
plt.ylim(-1.5, 0)
plt.savefig('scatterplot_SWCF_open_ocean_sea_ice.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)



# CC changes
f_tas_control         = Dataset(dir_path + '/tas_piControl_ts_26_models_monthly.nc', 'r')
f_tas_abrupt          = Dataset(dir_path + '/tas_abrupt_change_ts_26_models_monthly.nc', 'r')
f_cc_control          = Dataset(dir_path + '/clt_piControl_ts_26_models_monthly.nc', 'r')
f_cc_abrupt           = Dataset(dir_path + '/clt_abrupt_change_ts_26_models_monthly.nc', 'r')
CC_2d  = np.ma.zeros((90, 144, 5, 26))
for i in range(26):
  print(i)
  for imonth in range(5):
    if imonth==0:
        kk=list(range(12))
    elif imonth==1:
        kk=[0,1,11]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[0,1,2]
    elif imonth==2:
        kk=[2,3,4]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[3,4,5]
    elif imonth==3:
        kk=[5,6,7]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[6,7,8]
    elif imonth==4:
        kk=[8,9,10]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[9,10,11]
    tas_ts_control = np.ma.average(f_tas_control.variables['tas_ts_control'][:, :, i, :, :], 0)
    tas_ts_abrupt = np.ma.average(f_tas_abrupt.variables['tas_ts_abrupt'][:, :, i, :, :], 0)
    tas_ts_anomaly = tas_ts_abrupt - tas_ts_control
    del tas_ts_abrupt, tas_ts_control
    tas_global_anomaly = np.ma.average(np.ma.average(tas_ts_anomaly, 1, weight), 1)
    cc_ts_control = np.ma.average(f_cc_control.variables['clt_ts_control'][kk, :, i, :, :], 0)
    cc_ts_abrupt  = np.ma.average(f_cc_abrupt.variables['clt_ts_abrupt'][kk, :, i, :, :], 0)
    cc_ts_anomaly = cc_ts_abrupt - cc_ts_control
    del cc_ts_control, cc_ts_abrupt
    Data_length=150
    if model_name_26[i] == "IPSL-CM5A-MR":
        Data_length = 140
    if model_name_26[i] == "GFDL-ESM2M":
        Data_length = 119
# years 1-150 or 1-140 or 1-119
    XX=np.ma.ones((2,Data_length))
    XX[1,:]=tas_global_anomaly[:Data_length]
    YY=np.tensordot(XX,XX.T,1)
    ZZ0=np.tensordot(inv(YY),XX,1)
    ZZ = np.tensordot(ZZ0, cc_ts_anomaly[:Data_length, :, :], 1)
    CC_2d[:, :, imonth,i] = ZZ[1,:,:]
    del ZZ
CC_2d_saved  = CC_2d.copy()
CC_2d_saved[lat >= -60, :,:,:] = ma.masked
CC_2d_saved[sftlf >= 30, :,:] = ma.masked
reg_coef_CC = np.ma.zeros(26)
reg_intercept_CC = np.ma.zeros(26)
for i in range(26):
    x1 = Albedo_2d_saved[:, :, 0, i].flatten()
    y1 = CC_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_CC[i]=reg.coef_[0][0]
    reg_intercept_CC[i]=reg.intercept_[:]

#plot 5
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef_CC)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(Albedo_2d_saved[:,:,0,ind_sorted[i]],CC_2d_saved[:,:,0,ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("$\Delta$CC (\% K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("SAF (W m$^{-2}$ K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(Albedo_2d_saved[:,:,0,ind_sorted[i]], CC_2d_saved[:,:,0,ind_sorted[i]])[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),reg_intercept_CC[ind_sorted[i]]+np.array([-20,30])*reg_coef_CC[ind_sorted[i]],color="black")
    plt.ylim(-8,10)
    plt.xlim(-12,22)
    plt.text(20, 8-11, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(20, 5.2-11, 's={:.2f}'.format(reg_coef_CC[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_CC_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

# LCC changes
f_tas_control = Dataset(dir_path + '/tas_piControl_ts_26_models_monthly.nc', 'r')
f_tas_abrupt  = Dataset(dir_path + '/tas_abrupt_change_ts_26_models_monthly.nc', 'r')
f_lcc_control = Dataset(dir_path + '/low_cloud_cover_piControl_ts_25_models_monthly_max.nc', 'r')
f_lcc_abrupt  = Dataset(dir_path + '/low_cloud_cover_abrupt_change_ts_25_models_monthly_max.nc', 'r')
f_lcc_control_FGOALS_s2 = Dataset(dir_path + '/low_cloud_cover_piControl_ts_FGOALS-s2_monthly_max.nc', 'r')
f_lcc_abrupt_FGOALS_s2  = Dataset(dir_path + '/low_cloud_cover_abrupt_change_ts_FGOALS-s2_monthly_max.nc', 'r')
f_lcc_control_GISS_E2_R = Dataset(dir_path + '/low_cloud_cover_piControl_ts_GISS-E2-R_monthly_max.nc', 'r')
f_lcc_abrupt_GISS_E2_R  = Dataset(dir_path + '/low_cloud_cover_abrupt_change_ts_GISS-E2-R_monthly_max.nc', 'r')

LCC_2d  = np.ma.zeros((90, 144, 5, 26))
for i in range(26):
  print(i)
  for imonth in range(1):
    if imonth==0:
        kk=list(range(12))
    elif imonth==1:
        kk=[0,1,11]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[0,1,2]
    elif imonth==2:
        kk=[2,3,4]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[3,4,5]
    elif imonth==3:
        kk=[5,6,7]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[6,7,8]
    elif imonth==4:
        kk=[8,9,10]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[9,10,11]
    tas_ts_control = np.ma.average(f_tas_control.variables['tas_ts_control'][:, :, i, :, :], 0)
    tas_ts_abrupt = np.ma.average(f_tas_abrupt.variables['tas_ts_abrupt'][:, :, i, :, :], 0)
    tas_ts_anomaly = tas_ts_abrupt - tas_ts_control
    del tas_ts_abrupt, tas_ts_control
    tas_global_anomaly = np.ma.average(np.ma.average(tas_ts_anomaly, 1, weight), 1)
    if model_name_26[i]=="FGOALS-s2":
        lcc_ts_control = np.ma.average(f_lcc_control_FGOALS_s2.variables['cl_ts_control'][kk, :, :, :], 0)
        lcc_ts_abrupt  = np.ma.average(f_lcc_abrupt_FGOALS_s2.variables['cl_ts_abrupt'][kk, :, :, :], 0)
    elif model_name_26[i]=="GISS-E2-R":
        lcc_ts_control = np.ma.average(f_lcc_control_GISS_E2_R.variables['cl_ts_control'][kk, :, :, :], 0)
        lcc_ts_abrupt  = np.ma.average(f_lcc_abrupt_GISS_E2_R.variables['cl_ts_abrupt'][kk, :, :, :], 0)
    else:
        lcc_ts_control = np.ma.average(f_lcc_control.variables['cl_ts_control'][kk, :, i, :, :], 0)
        lcc_ts_abrupt  = np.ma.average(f_lcc_abrupt.variables['cl_ts_abrupt'][kk, :, i, :, :], 0)
    lcc_ts_anomaly = lcc_ts_abrupt - lcc_ts_control
    del lcc_ts_control, lcc_ts_abrupt
    Data_length=150
    if model_name_26[i] == "IPSL-CM5A-MR":
        Data_length = 140
    if model_name_26[i] == "GFDL-ESM2M":
        Data_length = 119
# years 1-150 or 1-140 or 1-119
    XX=np.ma.ones((2,Data_length))
    XX[1,:]=tas_global_anomaly[:Data_length]
    YY=np.tensordot(XX,XX.T,1)
    ZZ0=np.tensordot(inv(YY),XX,1)
    ZZ = np.tensordot(ZZ0, lcc_ts_anomaly[:Data_length, :, :], 1)
    LCC_2d[:, :, imonth,i] = ZZ[1,:,:]
    del ZZ

LCC_2d_saved  = LCC_2d.copy()
LCC_2d_saved[lat >= -60, :,:,:] = ma.masked
LCC_2d_saved[sftlf >= 30, :,:] = ma.masked
reg_coef_LCC = np.ma.zeros(26)
reg_intercept_LCC = np.ma.zeros(26)
for i in range(26):
    x1 = 100*Albedo_2d_saved[:, :, 0, i].flatten()
    y1 = LCC_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC[i]=reg.coef_[0][0]
    reg_intercept_LCC[i]=reg.intercept_[:]

#plot 6
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(100*Albedo_2d_saved[:,:,0,ind_sorted[i]],LCC_2d_saved[:,:,0,ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("$\Delta$LCC (\% K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("SAF (\% K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(Albedo_2d_saved[:,:,0,ind_sorted[i]], LCC_2d_saved[:,:,0,ind_sorted[i]])[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),reg_intercept_LCC[ind_sorted[i]]+np.array([-20,30])*reg_coef_LCC[ind_sorted[i]],color="black")
    plt.ylim(-10,10)
    plt.xlim(-8,12)
    plt.text(11, -4, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(11, -7.2, 's={:.2f}'.format(reg_coef_LCC[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_LCC_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)


# cloud liquid and ice water path changes
f_tas_control   = Dataset(dir_path + '/tas_piControl_ts_26_models_monthly.nc', 'r')
f_tas_abrupt    = Dataset(dir_path + '/tas_abrupt_change_ts_26_models_monthly.nc', 'r')
f_clwvi_control = Dataset(dir_path + '/clwvi_cl_removed_piControl_ts_26_models_monthly.nc', 'r')
f_clwvi_abrupt  = Dataset(dir_path + '/clwvi_cl_removed_abrupt_change_ts_26_models_monthly.nc', 'r')
f_clivi_control = Dataset(dir_path + '/clivi_cl_removed_piControl_ts_26_models_monthly.nc', 'r')
f_clivi_abrupt  = Dataset(dir_path + '/clivi_cl_removed_abrupt_change_ts_26_models_monthly.nc', 'r')
f_clwvi_control_HadGEM2_ES = Dataset(dir_path + '/clwvi_cl_removed_piControl_ts_HadGEM2-ES_monthly.nc', 'r')
f_clivi_control_HadGEM2_ES = Dataset(dir_path + '/clivi_cl_removed_piControl_ts_HadGEM2-ES_monthly.nc', 'r')
f_clwvi_abrupt_HadGEM2_ES  = Dataset(dir_path + '/clwvi_cl_removed_abrupt_change_ts_HadGEM2-ES_monthly.nc', 'r')
f_clivi_abrupt_HadGEM2_ES  = Dataset(dir_path + '/clivi_cl_removed_abrupt_change_ts_HadGEM2-ES_monthly.nc', 'r')
f_clwvi_control_MRI_CGCM3 = Dataset(dir_path + '/clwvi_cl_removed_piControl_ts_MRI_CGCM3_monthly.nc', 'r')
f_clivi_control_MRI_CGCM3 = Dataset(dir_path + '/clivi_cl_removed_piControl_ts_MRI_CGCM3_monthly.nc', 'r')

filename6 = "/Users/xinqu/Projects/gig2_backup/SAF_SWcld_feedback_correlation/data/rsdt_Amon_HadGEM2-ES_piControl_r1i1p1_185912-188411.nc"
f_rsdt = Dataset(filename6, 'r')
rsdt_mean_HadGEM2_ES = f_rsdt['rsdt'][0:12,:,:]
rsdt_mean = f_rsdt['rsdt'][1:13,:,:]
(lat_rsdt, lon_rsdt) = (f_rsdt.variables['lat'][:], f_rsdt.variables['lon'][:])
rsdt_mean_HadGEM2_ES_regridded = np.ma.zeros((12,150,90,144))
rsdt_mean_regridded = np.ma.zeros((12,150,90,144))
lon_2d, lat_2d = np.meshgrid(lon, lat)
for i in range(12):
    for j in range(150):
        rsdt_mean_HadGEM2_ES_regridded[i,j,:,:] = interp(rsdt_mean_HadGEM2_ES[i,:,:],lon_rsdt, lat_rsdt, lon_2d, lat_2d, False, False, 1)
        rsdt_mean_regridded[i, j, :, :] = interp(rsdt_mean[i, :, :], lon_rsdt, lat_rsdt, lon_2d, lat_2d, False, False,1)
LWP_2d  = np.ma.zeros((90, 144, 5, 26))
IWP_2d  = np.ma.zeros((90, 144, 5, 26))
for i in range(26):
  print(i)
  for imonth in range(1):
    if imonth==0:
        kk=list(range(12))
    elif imonth==1:
        kk=[0,1,11]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[0,1,2]
    elif imonth==2:
        kk=[2,3,4]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[3,4,5]
    elif imonth==3:
        kk=[5,6,7]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[6,7,8]
    elif imonth==4:
        kk=[8,9,10]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[9,10,11]
    tas_ts_control = np.ma.average(f_tas_control.variables['tas_ts_control'][:, :, i, :, :], 0)
    tas_ts_abrupt = np.ma.average(f_tas_abrupt.variables['tas_ts_abrupt'][:, :, i, :, :], 0)
    tas_ts_anomaly = tas_ts_abrupt - tas_ts_control
    del tas_ts_abrupt, tas_ts_control
    tas_global_anomaly = np.ma.average(np.ma.average(tas_ts_anomaly, 1, weight), 1)
    if model_name_26[i]=="HadGEM2-ES":
        clwvi_ts_control = np.ma.average(f_clwvi_control_HadGEM2_ES.variables['clwvi_cl_removed_ts_control'][kk, :, :, :]*1e5, axis=0)
        clwvi_ts_abrupt  = np.ma.average(f_clwvi_abrupt_HadGEM2_ES.variables['clwvi_cl_removed_ts_abrupt'][kk, :, :, :]*1e5, axis=0)
        clwvi_ts_anomaly = clwvi_ts_abrupt - clwvi_ts_control
        del clwvi_ts_control, clwvi_ts_abrupt
        clivi_ts_control = np.ma.average(f_clivi_control_HadGEM2_ES.variables['clivi_cl_removed_ts_control'][kk, :, :, :]*1e5, 0)
        clivi_ts_abrupt  = np.ma.average(f_clivi_abrupt_HadGEM2_ES.variables['clivi_cl_removed_ts_abrupt'][kk, :, :, :]*1e5, 0)
        clivi_ts_anomaly = clivi_ts_abrupt - clivi_ts_control
        del clivi_ts_control, clivi_ts_abrupt
    elif model_name_26[i]=="MRI-CGCM3":
        clwvi_ts_control = np.ma.average(f_clwvi_control_MRI_CGCM3.variables['clwvi_cl_removed_ts_control'][kk, :, :, :]*1e5, axis=0)
        clwvi_ts_abrupt  = np.ma.average(f_clwvi_abrupt.variables['clwvi_cl_removed_ts_abrupt'][kk, :, i, :, :]*1e5, axis=0)
        clwvi_ts_anomaly = clwvi_ts_abrupt - clwvi_ts_control
        del clwvi_ts_control, clwvi_ts_abrupt
        clivi_ts_control = np.ma.average(f_clivi_control_MRI_CGCM3.variables['clivi_cl_removed_ts_control'][kk, :, :, :]*1e5, 0)
        clivi_ts_abrupt  = np.ma.average(f_clivi_abrupt.variables['clivi_cl_removed_ts_abrupt'][kk, :, i, :, :]*1e5, 0)
        clivi_ts_anomaly = clivi_ts_abrupt - clivi_ts_control
        clwvi_ts_anomaly = clwvi_ts_anomaly - clivi_ts_anomaly
        del clivi_ts_control, clivi_ts_abrupt
    else:
        test  = np.ma.masked_all((12, 150, 90, 144))
        test1 = np.ma.masked_all((12, 150, 90, 144))
        test[:] = f_clwvi_control.variables['clwvi_cl_removed_ts_control'][kk, :, i, :, :] * 1e5
        test1[:] = f_clivi_control.variables['clivi_cl_removed_ts_control'][kk, :, i, :, :] * 1e5
        test_diff = test - test1
        if model_name_26[i] in ["CCSM4", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC-ESM","MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P"]:
            clwvi_ts_control = np.ma.average(test, axis=0)
        else:
            clwvi_ts_control = np.ma.average(test_diff, axis=0)

        test[:] = f_clwvi_abrupt.variables['clwvi_cl_removed_ts_abrupt'][kk, :, i, :, :] * 1e5
        test1[:] = f_clivi_abrupt.variables['clivi_cl_removed_ts_abrupt'][kk, :, i, :, :] * 1e5
        test_diff = test - test1
        if model_name_26[i] in ["CCSM4", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC-ESM","MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P"]:
            clwvi_ts_abrupt  = np.ma.average(test, axis=0)
        else:
            clwvi_ts_abrupt = np.ma.average(test_diff , axis=0)
        clwvi_ts_anomaly = clwvi_ts_abrupt - clwvi_ts_control
        del clwvi_ts_control, clwvi_ts_abrupt
        clivi_ts_control = np.ma.average(f_clivi_control.variables['clivi_cl_removed_ts_control'][kk, :, i, :, :]*1e5, 0)
        clivi_ts_abrupt  = np.ma.average(f_clivi_abrupt.variables['clivi_cl_removed_ts_abrupt'][kk, :, i, :, :]*1e5, 0)
        clivi_ts_anomaly = clivi_ts_abrupt - clivi_ts_control
        del clivi_ts_control, clivi_ts_abrupt
    Data_length=150
    if model_name_26[i] == "IPSL-CM5A-MR":
        Data_length = 140
    if model_name_26[i] == "GFDL-ESM2M":
        Data_length = 119
# years 1-150 or 1-140 or 1-119
    XX=np.ma.ones((2,Data_length))
    XX[1,:]=tas_global_anomaly[:Data_length]
    YY=np.tensordot(XX,XX.T,1)
    ZZ0=np.tensordot(inv(YY),XX,1)
    ZZ = np.tensordot(ZZ0, clwvi_ts_anomaly[:Data_length, :, :], 1)
    LWP_2d[:, :, imonth,i] = ZZ[1,:,:]
    del ZZ
    ZZ = np.tensordot(ZZ0, clivi_ts_anomaly[:Data_length, :, :], 1)
    IWP_2d[:, :, imonth,i] = ZZ[1,:,:]
    del ZZ

filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)
lon_len = len(lon)
meanLCC = np.ma.average(LCC_2d[:, :, 0, :],axis=2)
meanLCC_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanLCC_new[:,0:lon_len]=meanLCC[:,:]
meanLCC_new[:,lon_len]  =meanLCC[:,0]
meanLCC = meanLCC_new
del meanLCC_new

meanLWP = np.ma.average(LWP_2d[:, :, 0, :],axis=2)
meanLWP_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanLWP_new[:,0:lon_len]=meanLWP[:,:]
meanLWP_new[:,lon_len]  =meanLWP[:,0]
meanLWP = meanLWP_new
meanLWP[-1,:]=meanLWP[-2,:]
del meanLWP_new

lon_new = np.ones((lon_len+1))
lon_new[0:lon_len]=lon[:]
lon_new[lon_len]=lon[0]
lon=lon_new
del lon_new

lat[0]=-90
lat[-1]=90
lon, lat = np.meshgrid(lon,lat)

#SH
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.8, 9.6))
fig.subplots_adjust(wspace=0.1,hspace=0.3)
m1 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m1.fillcontinents(color='0.8')
    # draw parallels and meridians, but don't bother labelling them.
m1.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m1.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 1, 1, 1], dashes=[1, 3])
cmap1 = copy.deepcopy(plt.cm.jet)
cmap1.set_under((0.000000,0.0,0.500000))
im1 = m1.contourf(lon,lat,meanLCC,shading='flat',cmap=cmap1,latlon=True, vmin=-2, levels=[-1,-0.5, 0,0.5, 1, 1.5,], extend="both")
cb1 = m1.colorbar(im1,"bottom", size="5%", pad="8%", cmap=cmap1,label="\% K$^{-1}$")
axes[0].set_title('(a) $\Delta$LCC',y=1.08)

m2 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1])
#m2.drawcoastlines(linewidth=0.3)
m2.fillcontinents(color='0.8')
m2.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m2.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im2 = m2.contourf(lon, lat, meanLWP, shading='flat', cmap=cmap2, latlon=True, vmin=-4,levels=[5,7,9,11,13,15], extend="both")
cb2 = m2.colorbar(im2,"bottom", size="5%", pad="8%", label="g m$^{-2}$ K$^{-1}$")
axes[1].set_title('(b) $\Delta$LWP',y=1.08)
plt.savefig('LCC_LWP_changes_SH.pdf', papertype='letter', \
            orientation='landscape', bbox_inches='tight',pad_inches=0.5)


LWP_2d_saved  = LWP_2d.copy()
LWP_2d_saved[lat >= -60, : ,:, :] = ma.masked
LWP_2d_saved[sftlf >= 30, :, :] = ma.masked
reg_coef_LWP = np.ma.zeros(26)
reg_intercept_LWP = np.ma.zeros(26)
for i in range(26):
    x1 = 100*Albedo_2d_saved[:, :, 0, i].flatten()
    y1 = LWP_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP[i]=reg.coef_[0][0]
    reg_intercept_LWP[i]=reg.intercept_[:]

#plot 7
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(100*Albedo_2d_saved[:,:,0,ind_sorted[i]],LWP_2d_saved[:,:,0,ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("$\Delta$LWP (g m$^{-2}$ K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("SAF (\% K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(Albedo_2d_saved[:,:,0,ind_sorted[i]], LWP_2d_saved[:,:,0,ind_sorted[i]]*1e5)[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),reg_intercept_LWP[ind_sorted[i]]+np.array([-20,30])*reg_coef_LWP[ind_sorted[i]],color="black")
    plt.ylim(-40,80)
    plt.xlim(-8,12)
    plt.text(11, -7, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(11, -25, 's={:.2f}'.format(reg_coef_LWP[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_LWP_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)


#plot 8
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef_IWP*1e5)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(Albedo_2d_saved[:,:,0,ind_sorted[i]],IWP_2d_saved[:,:,0,ind_sorted[i]]*1e5, s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("$\Delta$IWP (g m$^{-2}$ K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("SAF (W m$^{-2}$ K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(Albedo_2d_saved[:,:,0,ind_sorted[i]], IWP_2d_saved[:,:,0,ind_sorted[i]]*1e5)[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),reg_intercept_IWP[ind_sorted[i]]*1e5+np.array([-20,30])*reg_coef_IWP[ind_sorted[i]]*1e5,color="black")
    plt.ylim(-30,50)
    plt.xlim(-12,22)
    plt.text(20, -8, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(20, -20, 's={:.2f}'.format(reg_coef_IWP[ind_sorted[i]]*1e5), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_Albedo_IWP_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 9
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.3)
reg_coef_overall = np.ma.zeros((26, 3))
reg_coef_overall[:, 0] = reg_coef
reg_coef_overall[:, 1] = reg_coef_cloudamt
reg_coef_overall[:, 2] = reg_coef_cloudscat
reg_coef_CC_overall = np.ma.zeros((26, 2))
reg_coef_CC_overall[:, 0] = reg_coef_LCC
reg_coef_CC_overall[:, 1] = reg_coef_LWP
reg_coef_LWP_overall = np.ma.zeros((26, 3))
reg_coef_LWP_overall[:, 0] = reg_coef_LWP
reg_coef_LWP_overall[:, 1] = reg_coef_IWP
reg_coef_LWP_overall[:, 2] = reg_coef_LWP+reg_coef_IWP
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=12 * 3 / 4, width=1.5 * 3 / 4, colors='black', labelsize='14', right='on',
                   top='off')
    ax.tick_params(axis='x', pad=10, labelsize='18')
    ax.tick_params(direction='out', which='minor', length=9*3/4, width=0.5*3/4, colors='black', \
                     labelsize='14', right='on', top='off',bottom='off')
    if i==0:
        plt.boxplot(reg_coef_overall,labels=['$\\frac{dSWCF}{dSAF}$','$\Large \\frac{dSWCAF}{dSAF}$','$\Large \\frac{dSWCSAF}{dSAF}$'],whis='range')
        plt.title("(a) Sensitivities of SWCF and its components to SAF", fontsize=14, y=1.02)
        plt.ylim(-0.6, 0.8)
        plt.plot(np.array([-20,30]),np.array([0,0]),color="black")
    elif i==1:
        plt.boxplot(reg_coef_CC_overall, labels=['$\Large \\frac{d\Delta LCC}{dSAF}$','$\Large \\frac{d\Delta LWP}{dSAF}$'],whis='range',widths=0.2)
        plt.title("(b) Sensitivities of $\Delta$LCC and $\Delta$LWP to SAF", fontsize=14, y=1.02)
        plt.ylim(-3.5, 5.5)
        plt.ylabel("", fontsize=14)
        plt.plot(np.array([-20, 30]), np.array([0, 0]), color="black")
        ax2 = ax.twinx()
        ax2.minorticks_on()
        ax2.tick_params(direction='out', length=12 * 3 / 4, width=1.5 * 3 / 4, colors='black', labelsize='14',right='on',top='off')
        ax2.tick_params(direction='out', which='minor', length=9 * 3 / 4, width=0.5 * 3 / 4, colors='black', labelsize='14', left='on', right='on',top='off', bottom='off')
        ax2.set_ylim(-3.5, 5.5)
        ax2.set_ylabel('g m$^{-2}$ \%$^{-1}$', fontsize=14)
        #plt.ylabel("g m$^{-2}$ W$^{-1}$ \%$^{1}$", fontsize=14)
plt.savefig('boxplot_SWCF_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 10
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8.6))
fig.subplots_adjust(wspace=0.4,hspace=0.4)
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=12*3/4, width=1.5*3/4, colors='black', labelsize='13', right='on', top='on',pad=8)
    ax.tick_params(direction='out', which='minor', length=9*3/4, width=0.5*3/4, colors='black', \
                     labelsize='13', right='on', bottom='on', top='on')
    if i == 0:
         x = reg_coef_LCC
         y = reg_coef_cloudamt
         reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
         plt.plot([-5, 5], reg.intercept_ + [-5, 5] * reg.coef_[0], "k", linewidth='1')
         for j in range(26):
            plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1), horizontalalignment='center', verticalalignment='center', fontsize=12)
         plt.ylabel("$\\frac{dSWCAF}{dSAF}$", fontsize=14)
         plt.xlabel("$\\frac{d\Delta LCC}{dSAF}$", fontsize=14)
         plt.title("(a) r=-0.82", fontsize=13, y=1.05)
         print(stats.mstats.pearsonr(x, y))
         print([reg.intercept_, reg.coef_[0][0]])
         plt.xlim(-1, 1.1)
         plt.ylim(-0.42, 0.2)
    else:
        x = reg_coef_LWP
        y = reg_coef_cloudscat
        reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        plt.plot([-5, 6], reg.intercept_ + [-5, 6] * reg.coef_[0], "k", linewidth='1')
        for j in range(26):
            plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j + 1), horizontalalignment='center',verticalalignment='center', fontsize=12)
        plt.ylabel("$\\frac{dSWCSAF}{dSAF}$", fontsize=14)
        plt.xlabel("$\\frac{d\Delta LWP}{dSAF}$ (g m$^{-2}$ \%$^{-1}$)", fontsize=14)
        plt.title("(b) r=-0.78", fontsize=13, y=1.05)
        print(stats.mstats.pearsonr(x, y))
        print([reg.intercept_, reg.coef_[0][0]])
        plt.xlim(-3.5, 5.2)
        plt.ylim(-0.2, 0.55)
plt.savefig('scatterplot_SWCF_LCC_LWP_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

# regression analysis on piControl LCC and LWP
f_ta_control_interannual    = Dataset(dir_path + '/ta_piControl_ts_26_models_monthly_interannual.nc', 'r')
f_tas_control_interannual   = Dataset(dir_path + '/tas_piControl_ts_26_models_monthly_interannual.nc', 'r')
f_clwvi_control_interannual = Dataset(dir_path + '/clwvi_cl_removed_piControl_ts_26_models_monthly_interannual.nc', 'r')
f_clivi_control_interannual = Dataset(dir_path + '/clivi_cl_removed_piControl_ts_26_models_monthly_interannual.nc', 'r')
f_clwvi_control_interannual_MRI_CGCM3 = Dataset(dir_path + '/clwvi_cl_removed_piControl_ts_MRI_CGCM3_monthly_interannual.nc', 'r')
f_clivi_control_interannual_MRI_CGCM3 = Dataset(dir_path + '/clivi_cl_removed_piControl_ts_MRI_CGCM3_monthly_interannual.nc', 'r')
f_lcc_control_interannual = Dataset(dir_path + '/low_cloud_cover_piControl_ts_26_models_monthly_max_interannual.nc', 'r')
LWP_2d_with_ts = np.ma.masked_all((90, 144, 26))
LWP_2d_with_is = np.ma.masked_all((90, 144, 26))
LCC_2d_with_ts = np.ma.masked_all((90, 144, 26))
LCC_2d_with_is = np.ma.masked_all((90, 144, 26))
for i in range(26):
  print(i)
  for imonth in range(1):
    if imonth==0:
        kk=list(range(12))
    elif imonth==1:
        kk=[0,1,11]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[0,1,2]
    elif imonth==2:
        kk=[2,3,4]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[3,4,5]
    elif imonth==3:
        kk=[5,6,7]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[6,7,8]
    elif imonth==4:
        kk=[8,9,10]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[9,10,11]
    tas_ts_control = np.ma.average(f_tas_control_interannual.variables['tas_ts_control'][kk, :, i, :, :], 0)
    ta_ts_control = np.ma.average(f_ta_control_interannual.variables['ta_ts_control'][kk, :, i, :, :], 0)
    LCC_ts_control = np.ma.average(f_lcc_control_interannual.variables['cl_ts_control'][kk, :, i, :, :], 0)
    for iyear in range(150):
        ta_ts_control[iyear, :, :] = np.where(ta_ts_control[iyear,:,:].mask == True, random.randint(150,300), ta_ts_control[iyear, :, :])
    if model_name_26[i]=="HadGEM2-ES":
        clwvi_ts_control = np.ma.average(f_clwvi_control_interannual.variables['clwvi_cl_removed_ts_control'][kk, :, i, :, :]*1e5, axis=0)
    elif model_name_26[i]=="MRI-CGCM3":
        clwvi_ts_control = np.ma.average(f_clwvi_control_interannual_MRI_CGCM3.variables['clwvi_cl_removed_ts_control'][kk, :, :, :] * 1e5, 0) - np.ma.average(f_clivi_control_interannual_MRI_CGCM3.variables['clivi_cl_removed_ts_control'][kk, :, :, :] * 1e5, 0)
    else:
        test  = np.ma.masked_all((12, 150, 90, 144))
        test1 = np.ma.masked_all((12, 150, 90, 144))
        test[:] = f_clwvi_control_interannual.variables['clwvi_cl_removed_ts_control'][kk, :, i, :, :] * 1e5
        test1[:] = f_clivi_control_interannual.variables['clivi_cl_removed_ts_control'][kk, :, i, :, :] * 1e5
        test_diff = test - test1
        if model_name_26[i] in ["CCSM4", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC-ESM","MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P"]:
            clwvi_ts_control = np.ma.average(test, axis=0)
        else:
            clwvi_ts_control = np.ma.average(test_diff, axis=0)
    Data_length=150
    if model_name_26[i] == "IPSL-CM5A-MR":
        Data_length = 140
    if model_name_26[i] == "GFDL-ESM2M":
        Data_length = 119
    XX = np.ma.ones((144, 30, 3, Data_length))

    XX[:, :, 1, :] = tas_ts_control[:Data_length, 2:32, :].swapaxes(0,2)
    XX[:, :, 2, :] = ta_ts_control[:Data_length, 2:32, :].swapaxes(0,2) - tas_ts_control[:Data_length, 2:32, :].swapaxes(0,2)
    XX1 = np.ma.ones((144, 30, Data_length, 3))
    XX1[:, :, :, 1] = tas_ts_control[:Data_length, 2:32, :].swapaxes(0,2)
    XX1[:, :, :, 2] = ta_ts_control[:Data_length, 2:32, :].swapaxes(0,2) - tas_ts_control[:Data_length, 2:32, :].swapaxes(0,2)
    YY=np.matmul(XX,XX1)
    ZZ0=np.matmul(inv(YY),XX)
    ZZ1 = np.ma.ones((144, 30, Data_length, 1))
    ZZ1[:, :, :, 0]=clwvi_ts_control[:Data_length, 2:32, :].swapaxes(0, 2)
    ZZ = np.matmul(ZZ0, ZZ1)
    LWP_2d_with_ts[2:32, :, i] = ZZ[:, :, 1, 0].swapaxes(0, 1)
    LWP_2d_with_is[2:32, :, i] = ZZ[:, :, 2, 0].swapaxes(0, 1)
    ZZ1 = np.ma.ones((144, 30, Data_length, 1))
    ZZ1[:, :, :, 0]=LCC_ts_control[:Data_length, 2:32, :].swapaxes(0, 2)
    ZZ = np.matmul(ZZ0, ZZ1)
    LCC_2d_with_ts[2:32, :, i] = ZZ[:, :, 1, 0].swapaxes(0, 1)
    LCC_2d_with_is[2:32, :, i] = ZZ[:, :, 2, 0].swapaxes(0, 1)

f_tas_control     = Dataset(dir_path + '/tas_piControl_ts_26_models_monthly.nc', 'r')
f_tas_abrupt      = Dataset(dir_path + '/tas_abrupt_change_ts_26_models_monthly.nc', 'r')
f_ta_control      = Dataset(dir_path + '/ta_piControl_ts_20_models_monthly.nc', 'r')
f_ta_abrupt       = Dataset(dir_path + '/ta850_abrupt_change_ts_20_models_monthly.nc', 'r')
f_hfls_control    = Dataset(dir_path + '/hfls_piControl_ts_20_models_monthly.nc', 'r')
f_hfls_abrupt     = Dataset(dir_path + '/hfls_abrupt_change_ts_20_models_monthly.nc', 'r')
f_hfss_control    = Dataset(dir_path + '/hfss_piControl_ts_20_models_monthly.nc', 'r')
f_hfss_abrupt     = Dataset(dir_path + '/hfss_abrupt_change_ts_20_models_monthly.nc', 'r')
f_ta_control_6    = Dataset(dir_path + '/ta_piControl_ts_6_models_monthly.nc', 'r')
f_ta_abrupt_6     = Dataset(dir_path + '/ta850_abrupt_change_ts_6_models_monthly.nc', 'r')
f_hfls_control_6  = Dataset(dir_path + '/hfls_piControl_ts_6_models_monthly.nc', 'r')
f_hfls_abrupt_6   = Dataset(dir_path + '/hfls_abrupt_change_ts_6_models_monthly.nc', 'r')
f_hfss_control_6  = Dataset(dir_path + '/hfss_piControl_ts_6_models_monthly.nc', 'r')
f_hfss_abrupt_6   = Dataset(dir_path + '/hfss_abrupt_change_ts_6_models_monthly.nc', 'r')
f_ta_control_FGOALS_s2    = Dataset(dir_path + '/ta_piControl_ts_FGOALS-s2_monthly.nc', 'r')
f_ta_abrupt_FGOALS_s2     = Dataset(dir_path + '/ta850_abrupt_change_ts_FGOALS-s2_monthly.nc', 'r')
f_hfls_control_FGOALS_s2  = Dataset(dir_path + '/hfls_piControl_ts_FGOALS-s2_monthly.nc', 'r')
f_hfls_abrupt_FGOALS_s2   = Dataset(dir_path + '/hfls_abrupt_change_ts_FGOALS-s2_monthly.nc', 'r')
f_hfss_control_FGOALS_s2  = Dataset(dir_path + '/hfss_piControl_ts_FGOALS-s2_monthly.nc', 'r')
f_hfss_abrupt_FGOALS_s2   = Dataset(dir_path + '/hfss_abrupt_change_ts_FGOALS-s2_monthly.nc', 'r')

tas_2d  = np.ma.zeros((90, 144, 5, 26))
hfss_2d = np.ma.zeros((90, 144, 5, 26))
hfls_2d = np.ma.zeros((90, 144, 5, 26))
ta_2d   = np.ma.zeros((90, 144, 5, 26))
for i in range(26):
  print(i)
  for imonth in range(1):
    if imonth==0:
        kk=list(range(12))
    elif imonth==1:
        kk=[0,1,11]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[0,1,2]
    elif imonth==2:
        kk=[2,3,4]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[3,4,5]
    elif imonth==3:
        kk=[5,6,7]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[6,7,8]
    elif imonth==4:
        kk=[8,9,10]
        if model_name_26[i]=="HadGEM2-ES":
            kk=[9,10,11]
    tas_ts_control = np.ma.average(f_tas_control.variables['tas_ts_control'][:, :, i, :, :], 0)
    tas_ts_abrupt = np.ma.average(f_tas_abrupt.variables['tas_ts_abrupt'][:, :, i, :, :], 0)
    tas_ts_anomaly = tas_ts_abrupt - tas_ts_control
    del tas_ts_abrupt, tas_ts_control
    tas_global_anomaly = np.ma.average(np.ma.average(tas_ts_anomaly, 1, weight), 1)
    if i<=19:
        if model_name_26[i]=="FGOALS-s2":
            hfss_ts_control = np.ma.average(f_hfss_control_FGOALS_s2.variables['hfss_ts_control'][kk, :, :, :], 0)
            hfss_ts_abrupt  = np.ma.average(f_hfss_abrupt_FGOALS_s2.variables['hfss_ts_abrupt'][kk, :, :, :], 0)
            hfss_ts_anomaly = hfss_ts_abrupt - hfss_ts_control
            del hfss_ts_control, hfss_ts_abrupt
            hfls_ts_control = np.ma.average(f_hfls_control_FGOALS_s2.variables['hfls_ts_control'][kk, :, :, :], 0)
            hfls_ts_abrupt  = np.ma.average(f_hfls_abrupt_FGOALS_s2.variables['hfls_ts_abrupt'][kk, :, :, :], 0)
            hfls_ts_anomaly = hfls_ts_abrupt - hfls_ts_control
            del hfls_ts_control, hfls_ts_abrupt
            ta_ts_control = np.ma.average(f_ta_control_FGOALS_s2.variables['ta_ts_control'][kk, :, :, :], 0)
            ta_ts_abrupt  = np.ma.average(f_ta_abrupt_FGOALS_s2.variables['ta_ts_abrupt'][kk, :, :, :], 0)
            ta_ts_anomaly = ta_ts_abrupt - ta_ts_control
            del ta_ts_control, ta_ts_abrupt
        else:
            hfss_ts_control = np.ma.average(f_hfss_control.variables['hfss_ts_control'][kk, :, i, :, :], 0)
            hfss_ts_abrupt  = np.ma.average(f_hfss_abrupt.variables['hfss_ts_abrupt'][kk, :, i, :, :], 0)
            hfss_ts_anomaly = hfss_ts_abrupt - hfss_ts_control
            del hfss_ts_control, hfss_ts_abrupt
            hfls_ts_control = np.ma.average(f_hfls_control.variables['hfls_ts_control'][kk, :, i, :, :], 0)
            hfls_ts_abrupt  = np.ma.average(f_hfls_abrupt.variables['hfls_ts_abrupt'][kk, :, i, :, :], 0)
            hfls_ts_anomaly = hfls_ts_abrupt - hfls_ts_control
            del hfls_ts_control, hfls_ts_abrupt
            ta_ts_control = np.ma.average(f_ta_control.variables['ta_ts_control'][kk, :, i, :, :], 0)
            ta_ts_abrupt  = np.ma.average(f_ta_abrupt.variables['ta_ts_abrupt'][kk, :, i, :, :], 0)
            ta_ts_anomaly = ta_ts_abrupt - ta_ts_control
            del ta_ts_control, ta_ts_abrupt
    else:
        hfss_ts_control = np.ma.average(f_hfss_control_6.variables['hfss_ts_control'][kk, :, i, :, :], 0)
        hfss_ts_abrupt  = np.ma.average(f_hfss_abrupt_6.variables['hfss_ts_abrupt'][kk, :, i, :, :], 0)
        hfss_ts_anomaly = hfss_ts_abrupt - hfss_ts_control
        del hfss_ts_control, hfss_ts_abrupt
        hfls_ts_control = np.ma.average(f_hfls_control_6.variables['hfls_ts_control'][kk, :, i, :, :], 0)
        hfls_ts_abrupt  = np.ma.average(f_hfls_abrupt_6.variables['hfls_ts_abrupt'][kk, :, i, :, :], 0)
        hfls_ts_anomaly = hfls_ts_abrupt - hfls_ts_control
        del hfls_ts_control, hfls_ts_abrupt
        ta_ts_control = np.ma.average(f_ta_control_6.variables['ta_ts_control'][kk, :, i, :, :], 0)
        ta_ts_abrupt  = np.ma.average(f_ta_abrupt_6.variables['ta_ts_abrupt'][kk, :, i, :, :], 0)
        ta_ts_anomaly = ta_ts_abrupt - ta_ts_control
        del ta_ts_control, ta_ts_abrupt
    Data_length=150
    if model_name_26[i] == "IPSL-CM5A-MR":
        Data_length = 140
    if model_name_26[i] == "GFDL-ESM2M":
        Data_length = 119
# years 1-150 or 1-140 or 1-119
    XX=np.ma.ones((2,Data_length))
    XX[1,:]=tas_global_anomaly[:Data_length]
    YY=np.tensordot(XX,XX.T,1)
    ZZ0=np.tensordot(inv(YY),XX,1)
    ZZ = np.tensordot(ZZ0, tas_ts_anomaly[:Data_length, :, :], 1)
    tas_2d[:, :, imonth,i] = ZZ[1, :, :]
    del ZZ
    ZZ = np.tensordot(ZZ0, hfss_ts_anomaly[:Data_length, :, :], 1)
    hfss_2d[:, :, imonth,i] = ZZ[1, :, :]
    del ZZ
    ZZ = np.tensordot(ZZ0, hfls_ts_anomaly[:Data_length, :, :], 1)
    hfls_2d[:, :, imonth,i] = ZZ[1, :, :]
    del ZZ
    ZZ = np.tensordot(ZZ0, ta_ts_anomaly[:Data_length, :, :], 1)
    ta_2d[:, :, imonth,i] = ZZ[1, :, :]
    del ZZ
# #   years 1-20
#     Data_length1=20
#     XX = np.ma.ones((2, Data_length1))
#     XX[1, :] = tas_global_anomaly[:Data_length1, i]
#     YY = np.tensordot(XX, XX.T, 1)
#     ZZ0 = np.tensordot(inv(YY), XX, 1)
#     ZZ = np.tensordot(ZZ0, lcc_ts_anomaly[:Data_length1, :, :], 1)
#     LCC_20_2d[:, :, imonth, 1, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, tas_ts_anomaly[:Data_length1, i, :, :], 1)
#     tas_20_2d[:, :, imonth, 1, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, hfss_ts_anomaly[:Data_length1, :, :], 1)
#     hfss_20_2d[:, :, imonth, 1, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, hfls_ts_anomaly[:Data_length1, :, :], 1)
#     hfls_20_2d[:, :, imonth, 1, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, ta_ts_anomaly[:Data_length1, :, :], 1)
#     ta_20_2d[:, :, imonth, 1, i] = ZZ[1, :, :]
#     del ZZ
# #   years 21-150 or 21-140
#     XX = np.ma.ones((2, Data_length-Data_length1))
#     XX[1, :] = tas_global_anomaly[Data_length1:Data_length, i]
#     YY = np.tensordot(XX, XX.T, 1)
#     ZZ0 = np.tensordot(inv(YY), XX, 1)
#     ZZ = np.tensordot(ZZ0, lcc_ts_anomaly[Data_length1:Data_length, :, :], 1)
#     LCC_20_2d[:, :, imonth, 2, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, tas_ts_anomaly[Data_length1:Data_length, i, :, :], 1)
#     tas_20_2d[:, :, imonth, 2, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, hfss_ts_anomaly[Data_length1:Data_length, :, :], 1)
#     hfss_20_2d[:, :, imonth, 2, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, hfls_ts_anomaly[Data_length1:Data_length, :, :], 1)
#     hfls_20_2d[:, :, imonth, 2, i] = ZZ[1, :, :]
#     del ZZ
#     ZZ = np.tensordot(ZZ0, ta_ts_anomaly[Data_length1:Data_length, :, :], 1)
#     ta_20_2d[:, :, imonth, 2, i] = ZZ[1, :, :]
#     del ZZ

filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()

f_landsea = Dataset(dir_path + '/sftlf_ncl.nc','r')
sftlf     = f_landsea.variables['sftlf'][:]
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)

tas_2d[(tas_2d<-100) | (tas_2d>100)]    = ma.masked
hfss_2d[(hfss_2d<-100) | (hfss_2d>100)] = ma.masked
hfls_2d[(hfls_2d<-100) | (hfls_2d>100)] = ma.masked
ta_2d[(ta_2d<-100) | (ta_2d>100)]       = ma.masked

tas_2d_saved   = tas_2d.copy()
hfss_2d_saved  = hfss_2d.copy()
hfls_2d_saved  = hfls_2d.copy()
ta_2d_saved    = ta_2d.copy()
LWP_2d_saved   = LWP_2d.copy()
LCC_2d_saved   = LCC_2d.copy()

tas_2d_saved[lat >= -60, :, :, :]  = ma.masked
hfss_2d_saved[lat >= -60, :, :, :] = ma.masked
hfls_2d_saved[lat >= -60, :, :, :] = ma.masked
ta_2d_saved[lat >= -60, :, :, :]   = ma.masked
tas_2d_saved[sftlf >= 30, :, :]    = ma.masked
hfss_2d_saved[sftlf >= 30, :, :]   = ma.masked
hfls_2d_saved[sftlf >= 30, :, :]   = ma.masked
ta_2d_saved[sftlf >= 30, :, :]     = ma.masked
LWP_2d_saved[lat >= -60, :, :, :] = ma.masked
LWP_2d_saved[sftlf >= 30, :, :] = ma.masked
LCC_2d_saved[lat >= -60, :, :, :] = ma.masked
LCC_2d_saved[sftlf >= 30, :, :] = ma.masked

reg_coef_LWP_interannual      = np.ma.zeros(26)
reg_intercept_LWP_interannual      = np.ma.zeros(26)
reg_coef_LWP_modeled      = np.ma.zeros(26)
reg_coef_LWP_modeled_tas      = np.ma.zeros(26)
reg_coef_LWP_modeled_tas_four_terms     = np.ma.zeros((26,4))
reg_coef_LWP_modeled_is      = np.ma.zeros(26)
reg_coef_LWP_modeled_is_four_terms     = np.ma.zeros((26,4))
reg_coef_LCC_interannual      = np.ma.zeros(26)
reg_intercept_LCC_interannual      = np.ma.zeros(26)
reg_coef_LCC_modeled      = np.ma.zeros(26)
reg_coef_LCC_modeled_tas      = np.ma.zeros(26)
reg_coef_LCC_modeled_tas_four_terms      = np.ma.zeros((26,4))
reg_coef_LCC_modeled_is      = np.ma.zeros(26)
reg_coef_LCC_modeled_is_four_terms     = np.ma.zeros((26,4))
cc_LCC_heuristic = []
cc_LWP_heuristic = []
for i in range(26):
    x1 = (LWP_2d_with_ts[:,:,i]*tas_2d_saved[:,:,0,i] + LWP_2d_with_is[:,:,i]*(ta_2d_saved[:,:,0,i] - tas_2d_saved[:,:,0,i])).flatten()
    y1 = LWP_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    cc_LWP_heuristic.append(stats.mstats.pearsonr(x2, y2)[0])
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_interannual[i] = reg.coef_[0][0]
    reg_intercept_LWP_interannual[i] = reg.intercept_
    y1 = (LWP_2d_with_ts[:, :, i] * tas_2d_saved[:, :, 0, i] + LWP_2d_with_is[:, :, i] * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    x1 = 100*Albedo_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled[i] = reg.coef_[0][0]
    y1 = (LWP_2d_with_ts[:, :, i] * tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled_tas[i] = reg.coef_[0][0]

    reg_coef_LWP_modeled_tas_four_terms[i, 0] = ma.mean(LWP_2d_with_ts[tas_2d_saved[:, :, 0, i].mask==False, i])
    reg_coef_LWP_modeled_tas_four_terms[i, 2] = ma.mean(tas_2d_saved[:, :, 0, i])

    y1 = (tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled_tas_four_terms[i, 1] = reg.coef_[0][0]

    y1 = (LWP_2d_with_ts[:, :, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled_tas_four_terms[i, 3] = reg.coef_[0][0]


    y1 = (LWP_2d_with_is[:, :, i] * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled_is[i] = reg.coef_[0][0]

    reg_coef_LWP_modeled_is_four_terms[i, 0] = ma.mean(LWP_2d_with_is[tas_2d_saved[:, :, 0, i].mask==False, i])
    reg_coef_LWP_modeled_is_four_terms[i, 2] = ma.mean(ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])

    y1 = (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled_is_four_terms[i, 1] = reg.coef_[0][0]

    y1 = (LWP_2d_with_is[:, :, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LWP_modeled_is_four_terms[i, 3] = reg.coef_[0][0]

    x1 = (LCC_2d_with_ts[:, :, i] * tas_2d_saved[:, :, 0, i] + LCC_2d_with_is[:, :, i] * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    y1 = LCC_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    cc_LCC_heuristic.append(stats.mstats.pearsonr(x2,y2)[0])
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_interannual[i] = reg.coef_[0][0]
    reg_intercept_LCC_interannual[i] = reg.intercept_
    y1 = (LCC_2d_with_ts[:, :, i] * tas_2d_saved[:, :, 0, i] + LCC_2d_with_is[:, :, i] * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    x1 = 100 * Albedo_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled[i] = reg.coef_[0][0]
    y1 = (LCC_2d_with_ts[:, :, i] * tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled_tas[i] = reg.coef_[0][0]

    reg_coef_LCC_modeled_tas_four_terms[i, 0] = ma.mean(LCC_2d_with_ts[tas_2d_saved[:, :, 0, i].mask==False, i])
    reg_coef_LCC_modeled_tas_four_terms[i, 2] = ma.mean(tas_2d_saved[:, :, 0, i])

    y1 = (tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled_tas_four_terms[i, 1] = reg.coef_[0][0]

    y1 = (LCC_2d_with_ts[:, :, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled_tas_four_terms[i, 3] = reg.coef_[0][0]

    y1 = (LCC_2d_with_is[:, :, i] * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled_is[i] = reg.coef_[0][0]

    reg_coef_LCC_modeled_is_four_terms[i, 0] = ma.mean(LCC_2d_with_is[tas_2d_saved[:, :, 0, i].mask == False, i])
    reg_coef_LCC_modeled_is_four_terms[i, 2] = ma.mean(ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])

    y1 = (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled_is_four_terms[i, 1] = reg.coef_[0][0]

    y1 = (LCC_2d_with_is[:, :, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_modeled_is_four_terms[i, 3] = reg.coef_[0][0]

# plot 11
from string import ascii_lowercase
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(8.2, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.7)
for i, ax in enumerate(axes.flatten()):
  if i<=25:
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(LWP_2d_with_ts[:,:,ind_sorted[i]]*tas_2d_saved[:,:,0,ind_sorted[i]] + LWP_2d_with_is[:,:,ind_sorted[i]]*(ta_2d_saved[:,:,0,ind_sorted[i]] - tas_2d_saved[:,:,0,ind_sorted[i]]),LWP_2d_saved[:, :, 0, ind_sorted[i]], s=2, color="black", edgecolors="black")
    if i in [0,4,8,12,16,20,24]:
        plt.ylabel("$\Delta$LWP (g m$^{-2}$ K$^{-1}$)",fontsize=8)
    if i in range(24,26):
        plt.xlabel("Modeled $\Delta$LWP (g m$^{-2}$ K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(LWP_2d_with_ts[:,:,ind_sorted[i]]*tas_2d_saved[:,:,0,ind_sorted[i]] + LWP_2d_with_is[:,:,ind_sorted[i]]*(ta_2d_saved[:,:,0,ind_sorted[i]] - tas_2d_saved[:,:,0,ind_sorted[i]]),LWP_2d_saved[:, :, 0, ind_sorted[i]])[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_26[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-40,80]),reg_intercept_LWP_interannual[ind_sorted[i]]+np.array([-40,80])*reg_coef_LWP_interannual[ind_sorted[i]],color="black")
    plt.ylim(-40,80)
    plt.xlim(-40,80)
    plt.text(72, -8, 'r={:.2f}'.format(RR), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
    plt.text(72, -27, 's={:.2f}'.format(reg_coef_LWP_interannual[ind_sorted[i]]), horizontalalignment='right', verticalalignment='center', multialignment='right',fontsize=8)
axes[6, 2].remove()
axes[6, 3].remove()
plt.savefig('scatterplot_LWP_simulated_modeled_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 12
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8.2, 8.2))
fig.subplots_adjust(wspace=0.50,hspace=0.6)
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=8*3/4, width=1.5*3/4, colors='black', labelsize='8', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='8', right='on', top='on')
    plt.scatter(Albedo_25_2d_saved[:,:,0,0,ind_sorted[i]],ta_20_2d_saved[:, :, 0, 0, ind_sorted[i]].flatten()-tas_20_2d_saved[:, :, 0, 0, ind_sorted[i]].flatten(), s=2, color="black", edgecolors="black")
    if i in [0,5,10,15,20]:
        plt.ylabel("IS changes (K K$^{-1}$)",fontsize=8)
    if i in range(20,25):
        plt.xlabel("SAF (W m$^{-2}$ K$^{-1}$)",fontsize=8)

    RR = stats.mstats.pearsonr(Albedo_25_2d_saved[:,:,0,0,ind_sorted[i]],ta_20_2d_saved[:, :, 0, 0, ind_sorted[i]].flatten()-tas_20_2d_saved[:, :, 0, 0, ind_sorted[i]].flatten())[0]
    plt.title("({0}) {1}".format(ascii_lowercase[i],model_name_28_reordered[ind_sorted[i]]),fontsize=8, y=1.06)
    print(RR)
    plt.plot(np.array([-20,30]),reg_intercept_IS[ind_sorted[i]]+np.array([-20,30])*reg_coef_IS[ind_sorted[i]],color="black")
    plt.ylim(-10,10)
    plt.xlim(-20,28)
    plt.text(-16.5, -5.5, 'r={:.2f}'.format(RR), horizontalalignment='left', verticalalignment='center', multialignment='left',fontsize=8)
    plt.text(-16.5, -8, 's={:.2f}'.format(reg_coef_IS[ind_sorted[i]]), horizontalalignment='left', verticalalignment='center', multialignment='left',fontsize=8)
plt.savefig('scatterplot_Albedo_IS_gridscale_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 13
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)
lon_len = len(lon)
lon_len = len(lon)
meanSAF = np.ma.average(LCC_2d_with_ts[:, :, :],axis=2)
meanSAF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSAF_new[:,0:lon_len]=meanSAF[:,:]
meanSAF_new[:,lon_len]  =meanSAF[:,0]
meanSAF = meanSAF_new
del meanSAF_new

meanSWCF = np.ma.average(LCC_2d_with_is[:, :, :],axis=2)
meanSWCF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSWCF_new[:,0:lon_len]=meanSWCF[:,:]
meanSWCF_new[:,lon_len]  =meanSWCF[:,0]
meanSWCF = meanSWCF_new
meanSWCF[-1,:]=meanSWCF[-2,:]
del meanSWCF_new

meanSWCAF = np.ma.average(LWP_2d_with_ts[:, :, :],axis=2)
meanSWCAF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSWCAF_new[:,0:lon_len]=meanSWCAF[:,:]
meanSWCAF_new[:,lon_len]  =meanSWCAF[:,0]
meanSWCAF = meanSWCAF_new
meanSWCAF[-1,:]=meanSWCAF[-2,:]
del meanSWCAF_new

meanSWCSAF = np.ma.average(LWP_2d_with_is[:, :, :],axis=2)
meanSWCSAF_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanSWCSAF_new[:,0:lon_len]=meanSWCSAF[:,:]
meanSWCSAF_new[:,lon_len]  =meanSWCSAF[:,0]
meanSWCSAF = meanSWCSAF_new
meanSWCSAF[-1,:]=meanSWCSAF[-2,:]
del meanSWCSAF_new

lon_new = np.ones((lon_len+1))
lon_new[0:lon_len]=lon[:]
lon_new[lon_len]=lon[0]
lon=lon_new
del lon_new

lat[0]=-90
lat[-1]=90
lon, lat = np.meshgrid(lon,lat)

#SH
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.8, 9.6))
fig.subplots_adjust(wspace=0.1,hspace=0.3)
m1 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0,0])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m1.fillcontinents(color='0.8')
    # draw parallels and meridians, but don't bother labelling them.
m1.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m1.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap1 = copy.deepcopy(plt.cm.jet)
cmap1.set_under((0.000000,0.0,0.500000))
im1 = m1.contourf(lon,lat,meanSAF,shading='flat',cmap=cmap1,latlon=True, vmin=-2, levels=[-1.5,-1,-0.5, 0,0.5, 1, 1.5], extend="both")
cb1 = m1.colorbar(im1,"bottom", size="5%", pad="8%", cmap=cmap1,label="\% K$^{-1}$")
axes[0,0].set_title('(a) $\\partial LCC/\\partial Ts$',y=1.08)

m2 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0,1])
#m2.drawcoastlines(linewidth=0.3)
m2.fillcontinents(color='0.8')
m2.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m2.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im2 = m2.contourf(lon, lat, meanSWCF, shading='flat', cmap=cmap2, latlon=True, vmin=-2,levels=[-1,-0.5, 0,0.5, 1, 1.5, 2.0, 2.5, 3.], extend="both")
cb2 = m2.colorbar(im2,"bottom", size="5%", pad="8%", label="\% K$^{-1}$")
axes[0,1].set_title('(b) $\\partial LCC/\\partial IS$',y=1.08)

m3 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1,0])
#m2.drawcoastlines(linewidth=0.3)
m3.fillcontinents(color='0.8')
m3.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m3.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im3 = m3.contourf(lon, lat, meanSWCAF, shading='flat', cmap=cmap2, latlon=True, vmin=-3,levels=[1, 2, 3, 4, 5, 6, 7, 8,9], extend="both")
cb3 = m3.colorbar(im3,"bottom", size="5%", pad="8%", label="g m$^{-2}$ K$^{-1}$")
axes[1,0].set_title('(c) $\\partial LWP/\\partial Ts$',y=1.08)

m4 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1,1])
#m2.drawcoastlines(linewidth=0.3)
m4.fillcontinents(color='0.8')
m4.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m4.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im4 = m4.contourf(lon, lat, meanSWCSAF, shading='flat', cmap=cmap2, latlon=True, vmin=-7,levels=[0, 2, 4, 6, 8, 10, 12, 14, 16], extend="both")
cb4 = m4.colorbar(im4,"bottom", size="5%", pad="8%", label="g m$^{-2}$ K$^{-1}$")
axes[1,1].set_title('(d) $\\partial LWP/\\partial IS$',y=1.08)
plt.savefig('LCC_LWP_piControl_SH.pdf', papertype='letter', \
            orientation='landscape', bbox_inches='tight',pad_inches=0.5)


#plot 14
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)
lon_len = len(lon)
meanLCC = np.ma.average(tas_2d[:, :, 0, :],axis=2)
meanLCC_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanLCC_new[:,0:lon_len]=meanLCC[:,:]
meanLCC_new[:,lon_len]  =meanLCC[:,0]
meanLCC = meanLCC_new
del meanLCC_new

meanLWP = np.ma.average(ta_2d[:, :, 0, :]-tas_2d[:, :, 0, :],axis=2)
meanLWP_new = np.ones((lat.shape[0],lon.shape[0]+1))
meanLWP_new[:,0:lon_len]=meanLWP[:,:]
meanLWP_new[:,lon_len]  =meanLWP[:,0]
meanLWP = meanLWP_new
meanLWP[-1,:]=meanLWP[-2,:]
del meanLWP_new

lon_new = np.ones((lon_len+1))
lon_new[0:lon_len]=lon[:]
lon_new[lon_len]=lon[0]
lon=lon_new
del lon_new

lat[0]=-90
lat[-1]=90
lon, lat = np.meshgrid(lon,lat)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.8, 9.6))
fig.subplots_adjust(wspace=0.1,hspace=0.3)
m1 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m1.fillcontinents(color='0.8')
    # draw parallels and meridians, but don't bother labelling them.
m1.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m1.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 1, 1, 1], dashes=[1, 3])
cmap1 = copy.deepcopy(plt.cm.jet)
cmap1.set_under((0.000000,0.0,0.500000))
im1 = m1.contourf(lon,lat,meanLCC,shading='flat',cmap=cmap1,latlon=True, vmin=-2, levels=[0.3, 0.6, 0.9,1.2, 1.5,1.8,2.1], extend="both")
cb1 = m1.colorbar(im1,"bottom", size="5%", pad="8%", cmap=cmap1,label="K K$^{-1}$")
axes[0].set_title('(a) $\Delta$Ts',y=1.08)

m2 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1])
#m2.drawcoastlines(linewidth=0.3)
m2.fillcontinents(color='0.8')
m2.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m2.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.50000))
im2 = m2.contourf(lon, lat, meanLWP, shading='flat', cmap=cmap2, latlon=True, vmin=-1.5,vmax=1,levels=[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4], extend="both")
cb2 = m2.colorbar(im2,"bottom", size="5%", pad="8%", label="K K$^{-1}$")
axes[1].set_title('(b) $\Delta$IS',y=1.08)
plt.savefig('tas_IS_changes_SH.pdf', papertype='letter', \
            orientation='landscape', bbox_inches='tight',pad_inches=0.5)

#plot 15
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
weight    = np.cos(3.14 * lat / 180.)
f_cl.close()
weight_2d = np.ones((90,144))
weight_2d = weight_2d * weight.reshape(-1,1)
lon_len = len(lon)

LCC_2d_heuristic = LCC_2d_with_ts*tas_2d[:,:,0,:] + LCC_2d_with_is*(ta_2d[:,:,0,:]-tas_2d[:,:,0,:])
meanLCC = np.ma.average(LCC_2d[:, :, 0, :],axis=2)
meanLCC_h = np.ma.average(LCC_2d_heuristic[:, :, :],axis=2)
meanLCC_new = np.ones((lat.shape[0],lon.shape[0]+1,2))
meanLCC_new[:,0:lon_len,0]=meanLCC[:,:]
meanLCC_new[:,lon_len,0]  =meanLCC[:,0]
meanLCC_new[:,0:lon_len,1]=meanLCC_h[:,:]
meanLCC_new[:,lon_len,1]  =meanLCC_h[:,0]
meanLCC = meanLCC_new
del meanLCC_new, meanLCC_h

LWP_2d_heuristic = LWP_2d_with_ts*tas_2d[:,:,0,:] + LWP_2d_with_is*(ta_2d[:,:,0,:]-tas_2d[:,:,0,:])
meanLWP = np.ma.average(LWP_2d[:, :, 0, :],axis=2)
meanLWP_h = np.ma.average(LWP_2d_heuristic[:, :, :],axis=2)
meanLWP_new = np.ones((lat.shape[0],lon.shape[0]+1,2))
meanLWP_new[:,0:lon_len,0]=meanLWP[:,:]
meanLWP_new[:,lon_len,0]  =meanLWP[:,0]
meanLWP_new[:,0:lon_len,1]=meanLWP_h[:,:]
meanLWP_new[:,lon_len,1]  =meanLWP_h[:,0]
meanLWP = meanLWP_new
meanLWP[-1,:,:]=meanLWP[-2,:,:]
del meanLWP_new, meanLWP_h

lon_new = np.ones((lon_len+1))
lon_new[0:lon_len]=lon[:]
lon_new[lon_len]=lon[0]
lon=lon_new
del lon_new

lat[0]=-90
lat[-1]=90
lon, lat = np.meshgrid(lon,lat)

#SH
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.8, 9.6))
fig.subplots_adjust(wspace=0.1,hspace=0.3)
m1 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0,0])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m1.fillcontinents(color='0.8')
    # draw parallels and meridians, but don't bother labelling them.
m1.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m1.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap1 = copy.deepcopy(plt.cm.jet)
cmap1.set_under((0.000000,0.0,0.500000))
im1 = m1.contourf(lon,lat,meanLCC[:,:,0],shading='flat',cmap=cmap1,latlon=True, vmin=-2, levels=[-1,-0.5, 0,0.5, 1, 1.5,], extend="both")
cb1 = m1.colorbar(im1,"bottom", size="5%", pad="8%", cmap=cmap1,label="\% K$^{-1}$")
axes[0,0].set_title('(a) GCM $\Delta$LCC',y=1.08)

m2 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[0,1])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m2.fillcontinents(color='0.8')
    # draw parallels and meridians, but don't bother labelling them.
m2.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m2.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000,0.0,0.500000))
im2 = m2.contourf(lon,lat,meanLCC[:,:,1],shading='flat',cmap=cmap2,latlon=True, vmin=-2, levels=[-1,-0.5, 0,0.5, 1, 1.5,], extend="both")
cb2 = m2.colorbar(im2,"bottom", size="5%", pad="8%", cmap=cmap1,label="\% K$^{-1}$")
axes[0,1].set_title('(b) Heuristic-model-predicted $\Delta\\widetilde{LCC}$',y=1.08)

m3 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1,0])
#m2.drawcoastlines(linewidth=0.3)
m3.fillcontinents(color='0.8')
m3.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m3.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap3 = copy.deepcopy(plt.cm.jet)
cmap3.set_under((0.000000, 0.0, 0.50000))
im3 = m3.contourf(lon, lat, meanLWP[:,:,0], shading='flat', cmap=cmap3, latlon=True, vmin=-4,levels=[5,7,9,11,13,15], extend="both")
cb3 = m3.colorbar(im3,"bottom", size="5%", pad="8%", label="g m$^{-2}$ K$^{-1}$")
axes[1,0].set_title('(c) GCM $\Delta$LWP',y=1.08)

m4 = Basemap(projection='spstere',boundinglat=-55,lon_0=270,resolution='l',ax=axes[1,1])
#m2.drawcoastlines(linewidth=0.3)
m4.fillcontinents(color='0.8')
m4.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m4.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap4 = copy.deepcopy(plt.cm.jet)
cmap4.set_under((0.000000, 0.0, 0.50000))
im4 = m4.contourf(lon, lat, meanLWP[:,:,1], shading='flat', cmap=cmap4, latlon=True, vmin=-4,levels=[5,7,9,11,13,15], extend="both")
cb4 = m4.colorbar(im4,"bottom", size="5%", pad="8%", label="g m$^{-2}$ K$^{-1}$")
axes[1,1].set_title('(d) Heuristic-model-predicted $\Delta\\widetilde{LWP}$',y=1.08)

plt.savefig('LCC_LWP_changes_SH.pdf', papertype='letter', \
            orientation='landscape', bbox_inches='tight',pad_inches=0.5)

# compute the correlation between GCM and heuristic-model-predicted LCC and LWP changes
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
meanLCC = np.ma.average(LCC_2d[:, :, 0, :],axis=2)
meanLCC_h = np.ma.average(LCC_2d_heuristic[:, :, :],axis=2)
meanLCC[lat >= -60, :]   = ma.masked
meanLCC[sftlf >= 30]    = ma.masked
meanLCC_h[lat >= -60, :] = ma.masked
meanLCC_h[sftlf >= 30]   = ma.masked
x1 = meanLCC.flatten()
y1 = meanLCC_h.flatten()
y2 = y1[(x1.mask == False) & (y1.mask == False)]
x2 = x1[(x1.mask == False) & (y1.mask == False)]
print(stats.mstats.pearsonr(x2,y2))
print(ma.corrcoef(x1,y1))

meanLCC = np.ma.average(LWP_2d[:, :, 0, :],axis=2)
meanLCC_h = np.ma.average(LWP_2d_heuristic[:, :, :],axis=2)
meanLCC[lat >= -60, :]   = ma.masked
meanLCC[sftlf >= 30]    = ma.masked
meanLCC_h[lat >= -60, :] = ma.masked
meanLCC_h[sftlf >= 30]   = ma.masked
x1 = meanLCC.flatten()
y1 = meanLCC_h.flatten()
y2 = y1[(x1.mask == False) & (y1.mask == False)]
x2 = x1[(x1.mask == False) & (y1.mask == False)]
print(stats.mstats.pearsonr(x2,y2))
print(ma.corrcoef(x1,y1))

#plot 16
reg_coef_overall = np.ma.zeros((26, 3))
reg_coef_overall[:, 0] = reg_coef
reg_coef_overall[:, 1] = reg_coef_cloudamt
reg_coef_overall[:, 2] = reg_coef_cloudscat
reg_coef_CC_overall = np.ma.zeros((26, 4))
reg_coef_CC_overall[:, 0] = reg_coef_LCC
reg_coef_CC_overall[:, 1] = reg_coef_LCC_modeled
reg_coef_CC_overall[:, 2] = reg_coef_LCC_modeled_tas
reg_coef_CC_overall[:, 3] = reg_coef_LCC_modeled_is
reg_coef_LWP_overall = np.ma.zeros((26, 4))
reg_coef_LWP_overall[:, 0] = reg_coef_LWP
reg_coef_LWP_overall[:, 1] = reg_coef_LWP_modeled
reg_coef_LWP_overall[:, 2] = reg_coef_LWP_modeled_tas
reg_coef_LWP_overall[:, 3] = reg_coef_LWP_modeled_is
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4.5, 11.2))
fig.subplots_adjust(wspace=0.35,hspace=0.45)
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=12 * 3 / 4, width=1.5 * 3 / 4, colors='black', labelsize='14', right='on',
                   top='off')
    ax.tick_params(axis='x', pad=2, labelsize='14')
    ax.tick_params(axis='y', pad=1, labelsize='14')
    ax.tick_params(direction='out', which='minor', length=9*3/4, width=0.5*3/4, colors='black', \
                     labelsize='14', right='on', top='off',bottom='off')
    if i==0:
        ax.tick_params(axis='x', pad=2, labelsize='14')
        plt.boxplot(reg_coef_overall,labels=['$\\frac{dSWCF}{dSAF}$','$\Large \\frac{dSWCAF}{dSAF}$','$\Large \\frac{dSWCSAF}{dSAF}$'],whis='range')
        plt.title("(a) Sensitivities of SWCF and its components to SAF", fontsize=14, y=1.04)
        plt.ylim(-0.6, 0.8)
        plt.plot(np.array([-20,30]),np.array([0,0]),color="black")
    elif i==1:
        ax.tick_params(axis='x', pad=2, labelsize='14')
        plt.boxplot(reg_coef_CC_overall, labels=['$\Large \\frac{d\Delta LCC}{dSAF}$','$\Large \\frac{d\Delta \\widetilde{LCC}}{dSAF}$','Ts term','IS term'],whis='range',widths=0.4)
        plt.title("(b) Sensitivities of $\Delta$LCC and $\Delta\\widetilde{LCC}$ to SAF", fontsize=14, y=1.02)
        plt.ylim(-1.2, 1.2)
        plt.ylabel("", fontsize=14)
        plt.plot(np.array([-20, 30]), np.array([0, 0]), color="black")
        # ax2 = ax.twinx()
        # ax2.minorticks_on()
        # ax2.tick_params(direction='out', length=12 * 3 / 4, width=1.5 * 3 / 4, colors='black', labelsize='14',right='on',top='off')
        # ax2.tick_params(direction='out', which='minor', length=9 * 3 / 4, width=0.5 * 3 / 4, colors='black', labelsize='14', left='on', right='on',top='off', bottom='off')
        # ax2.set_ylim(-3.5, 5.5)
        # ax2.set_ylabel('g m$^{-2}$ \%$^{-1}$', fontsize=14)
        #plt.ylabel("g m$^{-2}$ W$^{-1}$ \%$^{1}$", fontsize=14)
    else:
        ax.tick_params(axis='x', pad=2, labelsize='14')
        plt.boxplot(reg_coef_LWP_overall, labels=['$\Large \\frac{d\Delta LWP}{dSAF}$','$\Large \\frac{d\Delta \\widetilde{LWP}}{dSAF}$','Ts term','IS term'],whis='range',widths=0.4)
        plt.title("(c) Sensitivities of $\Delta$LWP and $\Delta\\widetilde{LWP}$ to SAF", fontsize=14, y=1.02)
        plt.ylim(-12, 12)
        plt.ylabel('g m$^{-2}$ \%$^{-1}$', fontsize=14)
        plt.plot(np.array([-20, 30]), np.array([0, 0]), color="black")
plt.savefig('boxplot_SWCF_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

#plot 17
ind_sorted = np.argsort(reg_coef)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8.6))
fig.subplots_adjust(wspace=0.4,hspace=0.4)
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=12*3/4, width=1.5*3/4, colors='black', labelsize='13', right='on', top='on', pad=8)
    ax.tick_params(direction='out', which='minor', length=9*3/4, width=0.5*3/4, colors='black', \
                     labelsize='13', right='on', bottom='on', top='on')
    if i == 0:
         x = reg_coef_LCC_modeled
         y = reg_coef_LCC
         reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
         plt.plot([-5, 5], reg.intercept_ + [-5, 5] * reg.coef_[0], "k", linewidth='1')
         for j in range(26):
            plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1), horizontalalignment='center', verticalalignment='center', fontsize=12)
         plt.ylabel("$\\frac{d\Delta LCC}{dSAF}$", fontsize=14)
         plt.xlabel("$\\frac{d\Delta \\widetilde{LCC}}{dSAF}$", fontsize=14)
         plt.title("(a) r=0.64", fontsize=13, y=1.05)
         print(stats.mstats.pearsonr(x, y))
         print([reg.intercept_, reg.coef_[0][0]])
         plt.xlim(-1, 1.2)
         plt.ylim(-1, 1.2)
    else:
        x = reg_coef_LWP_modeled
        y = reg_coef_LWP
        reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        plt.plot([-5, 7], reg.intercept_ + [-5, 7] * reg.coef_[0], "k", linewidth='1')
        for j in range(26):
            plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j + 1), horizontalalignment='center',verticalalignment='center', fontsize=12)
        plt.ylabel("$\\frac{d\Delta LWP}{dSAF}$ (g m$^{-2}$ \%$^{-1}$)", fontsize=14)
        plt.xlabel("$\\frac{d\Delta \\widetilde{LWP}}{dSAF}$ (g m$^{-2}$ \%$^{-1}$)", fontsize=14)
        plt.title("(b) r=0.77", fontsize=13, y=1.05)
        print(stats.mstats.pearsonr(x, y))
        print([reg.intercept_, reg.coef_[0][0]])
        plt.xlim(-4, 6)
        plt.ylim(-4, 6)
plt.savefig('scatterplot_LCC_LWP_modeled_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

# observational estimate of reg_coef_variables
filename = '/Volumes/Data_Server/GOCCP_v3/3D_CloudFraction330m_200606-201612_avg_CFMIP2_sat_3.1.2.nc'
f_cl = Dataset(filename, 'r')
(lat_clc, lon_clc, alt_mid) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:], f_cl.variables['alt_mid'][:])
clc = np.ma.masked_all((12,10,90,180))
for iyear in range(10):
    for imonth in range(12):
        print(imonth+iyear*12)
        if imonth+iyear*12 < 116:
            temp1 = f_cl.variables['clcalipso'][imonth + iyear * 12, :, :, :]
            temp1[alt_mid >= 3.5, :, :] = np.ma.masked
            clc[imonth, iyear, :, :] = ma.max(temp1,axis=0)
        if (imonth+iyear*12 > 116) & (imonth+iyear*12 <= 126):  # 201602 is missing
            temp1 = f_cl.variables['clcalipso'][imonth + iyear * 12 - 1, :, :, :]
            temp1[alt_mid >= 3.5, :, :] = np.ma.masked
            clc[imonth, iyear, :, :] = ma.max(temp1,axis=0)
clc_clim = np.ma.average(clc,0)
f_cc = Dataset(dir_path + '/clc_clim.nc', 'w')
f_cc.createDimension("time", 10)
f_cc.createDimension("lat", 90)
f_cc.createDimension("lon", 180)
f_cc.createVariable("time", "f4", ("time",))
f_cc.createVariable("lat", "f4", ("lat",))
f_cc.createVariable("lon", "f4", ("lon",))
f_cc.createVariable("clc_clim", "f4", ("time","lat", "lon",))
f_cc.variables['lat'][:] = lat_clc
f_cc.variables['lon'][:] = lon_clc
f_cc.variables['clc_clim'][:] = clc_clim
f_cc.close()

(clc_clim_shifted,lon_clc_new) = shiftgrid(1,clc_clim,lon_clc)

clc_clim_regridded = np.ma.masked_all((10,90,144))
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
lon_2d, lat_2d = np.meshgrid(lon, lat)
for i in range(10):
    clc_clim_regridded[i,:,:] = interp(clc_clim_shifted[i, :, :],lon_clc_new, lat_clc, lon_2d, lat_2d, False, False, 1)
f_cc = Dataset(dir_path + '/clc_clim_regridded.nc', 'w')
f_cc.createDimension("time", 10)
f_cc.createDimension("lat", 90)
f_cc.createDimension("lon", 144)
f_cc.createVariable("time", "f4", ("time",))
f_cc.createVariable("lat", "f4", ("lat",))
f_cc.createVariable("lon", "f4", ("lon",))
f_cc.createVariable("clc_clim", "f4", ("time","lat", "lon",))
f_cc.variables['lat'][:] = lat
f_cc.variables['lon'][:] = lon
f_cc.variables['clc_clim'][:] = clc_clim_regridded
f_cc.close()

filename_ta = '/Volumes/Data_Server/NCEP/air.mon.mean.presure.level_1948_01_2017_08.nc'
filename_tas = '/Volumes/Data_Server/NCEP/air.mon.mean_1948_01_2017_08.nc'
f_ta = Dataset(filename_ta, 'r')
f_tas = Dataset(filename_tas, 'r')
(lat_ta, lon_ta) = (f_ta.variables['lat'][::-1], f_ta.variables['lon'][:])
Ts_NCEP = np.ma.masked_all((12,10,73,144))
IS_NCEP = np.ma.masked_all((12,10,73,144))

for iyear in range(10):
    for imonth in range(12):
        print(imonth+iyear*12)
        if imonth+iyear*12 < 116:
            Ts_NCEP[imonth, iyear, :, :] = f_tas.variables['air'][imonth + (iyear+2006-1948) * 12 + 5, ::-1, :]
            IS_NCEP[imonth, iyear, :, :] = f_ta.variables['air'][imonth + (iyear + 2006 - 1948) * 12 + 5, 2, ::-1, :] - f_tas.variables['air'][imonth + (iyear + 2006 - 1948) * 12 + 5, ::-1, :]
        if (imonth+iyear*12 > 116) & (imonth+iyear*12 <= 126):  # 201602 is missing
            Ts_NCEP[imonth, iyear, :, :] = f_tas.variables['air'][imonth + (iyear+2006-1948) * 12 + 5, ::-1, :]
            IS_NCEP[imonth, iyear, :, :] = f_ta.variables['air'][imonth + (iyear + 2006 - 1948) * 12 + 5, 2, ::-1, :] - f_tas.variables['air'][imonth + (iyear + 2006 - 1948) * 12 + 5, ::-1, :]
Ts_NCEP_clim = np.ma.average(Ts_NCEP,0)
IS_NCEP_clim = np.ma.average(IS_NCEP,0)
f_cc = Dataset(dir_path + '/Ts_IS_NCEP_clim.nc', 'w')
f_cc.createDimension("time", 10)
f_cc.createDimension("lat", 73)
f_cc.createDimension("lon", 144)
f_cc.createVariable("time", "f4", ("time",))
f_cc.createVariable("lat", "f4", ("lat",))
f_cc.createVariable("lon", "f4", ("lon",))
f_cc.createVariable("Ts_NCEP_clim", "f4", ("time","lat", "lon",))
f_cc.createVariable("IS_NCEP_clim", "f4", ("time","lat", "lon",))
f_cc.variables['lat'][:] = lat_ta
f_cc.variables['lon'][:] = lon_ta
f_cc.variables['Ts_NCEP_clim'][:] = Ts_NCEP_clim
f_cc.variables['IS_NCEP_clim'][:] = IS_NCEP_clim
f_cc.close()

Ts_NCEP_clim_regridded = np.ma.masked_all((10,90,144))
IS_NCEP_clim_regridded = np.ma.masked_all((10,90,144))
filename = '/Users/xinqu/Projects/gig2_backup/sythesis_ECS/data/CMIP_forcing_feedback_APRP_2d_monthly/{0}_APRP_forcings_fdbks_v9_monres.nc'.format(model_name_26[0])
f_cl = Dataset(filename, 'r')
(lat, lon) = (f_cl.variables['latitude'][:], f_cl.variables['longitude'][:])
lon_2d, lat_2d = np.meshgrid(lon, lat)
for i in range(10):
    Ts_NCEP_clim_regridded[i, :, :] = interp(Ts_NCEP_clim[i, :, :],lon_ta, lat_ta, lon_2d, lat_2d, False, False, 1)
    IS_NCEP_clim_regridded[i, :, :] = interp(IS_NCEP_clim[i, :, :], lon_ta, lat_ta, lon_2d, lat_2d, False, False, 1)

clc_clim_regridded[:, lat >= -60, :] = ma.masked
Ts_NCEP_clim_regridded[:, lat >= -60, :] = ma.masked
IS_NCEP_clim_regridded[:, lat >= -60, :] = ma.masked
clc_clim_regridded[:, sftlf >= 30] = ma.masked
Ts_NCEP_clim_regridded[:, sftlf >= 30] = ma.masked
IS_NCEP_clim_regridded[:, sftlf >= 30] = ma.masked

XX = np.ma.ones((144, 30, 3, 9))
LCC_2d_with_ts_obs = np.ma.masked_all((90,144))
LCC_2d_with_is_obs = np.ma.masked_all((90,144))

XX[:, :, 1, :] = Ts_NCEP_clim_regridded[1:, 2:32, :].swapaxes(0, 2)
XX[:, :, 2, :] = IS_NCEP_clim_regridded[1:, 2:32, :].swapaxes(0, 2)
XX1 = np.ma.ones((144, 30, 9, 3))
XX1[:, :, :, 1] = Ts_NCEP_clim_regridded[1:, 2:32, :].swapaxes(0, 2)
XX1[:, :, :, 2] = IS_NCEP_clim_regridded[1:, 2:32, :].swapaxes(0, 2)
YY = np.matmul(XX, XX1)
ZZ0 = np.matmul(inv(YY), XX)
ZZ1 = np.ma.ones((144, 30, 9, 1))
ZZ1[:, :, :, 0] = 100*clc_clim_regridded[1:, 2:32, :].swapaxes(0, 2)
ZZ = np.matmul(ZZ0, ZZ1)
LCC_2d_with_ts_obs[2:32, :] = ZZ[:, :, 1, 0].swapaxes(0, 1)
LCC_2d_with_is_obs[2:32, :] = ZZ[:, :, 2, 0].swapaxes(0, 1)

f_cc = Dataset(dir_path + '/LCC_Ts_IS_slopes.nc', 'w')
f_cc.createDimension("lat", 90)
f_cc.createDimension("lon", 144)
f_cc.createVariable("lat", "f4", ("lat",))
f_cc.createVariable("lon", "f4", ("lon",))
f_cc.createVariable("Ts_NCEP_clim", "f4", ("lat", "lon",))
f_cc.createVariable("IS_NCEP_clim", "f4", ("lat", "lon",))
f_cc.variables['lat'][:] = lat
f_cc.variables['lon'][:] = lon
f_cc.variables['Ts_NCEP_clim'][:] = LCC_2d_with_ts_obs
f_cc.variables['IS_NCEP_clim'][:] = LCC_2d_with_is_obs
f_cc.close()


reg_coef_LCC_obs       = np.ma.zeros(26)
reg_coef_LCC_obs_tas   = np.ma.zeros(26)
reg_coef_LCC_obs_is    = np.ma.zeros(26)
for i in range(26):
    y1 = (LCC_2d_with_ts_obs * tas_2d_saved[:, :, 0, i] + LCC_2d_with_is_obs * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    x1 = 100 * Albedo_2d_saved[:, :, 0, i].flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_obs[i] = reg.coef_[0][0]
    y1 = (LCC_2d_with_ts_obs * tas_2d_saved[:, :, 0, i]).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_obs_tas[i] = reg.coef_[0][0]
    y1 = (LCC_2d_with_is_obs * (ta_2d_saved[:, :, 0, i] - tas_2d_saved[:, :, 0, i])).flatten()
    y2 = y1[(x1.mask == False) & (y1.mask == False)]
    x2 = x1[(x1.mask == False) & (y1.mask == False)]
    reg.fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
    reg_coef_LCC_obs_is[i] = reg.coef_[0][0]


# plot 18
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6., 8.6))
fig.subplots_adjust(wspace=0.4,hspace=0.3)
ind = np.arange(25)  # the x locations for the groups
width = 0.6
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=12*3/4, width=1.5*3/4, colors='black', labelsize='14', right='on', top='off')
    ax.tick_params(direction='out', which='minor', length=7*3/4, width=0.5*3/4, colors='black', \
                     labelsize='14', right='on', bottom='off')
#    ax.set_xticks((0,1,2,3))
#    ax.set_xticklabels(('DJF', 'MAM', 'JJA', 'SON'))
    if i == 0:
         ax.bar(ind+1, reg_coef_variables[0,0,ind_sorted], width, color='r',edgecolor='black',linewidth=0.7, capsize=4)
         plt.plot([-0.5, 25.8],[0,0], "k",linewidth='0.7')
         plt.plot([-0.5,25.8], [np.ma.average(reg_coef_variables[0,0,:]),np.ma.average(reg_coef_variables[0,0,:])], "k",linewidth='0.7')
         plt.ylabel("\% W$^{-1}$ m$^{2}$",fontsize=14)
         plt.title("(a) $\partial LCC/\partial F_{s}$",fontsize=14, y=1.02)
         plt.xlim(0.2,25.8)
         plt.ylim(-0.14,0.24)
         ax.set_xticks(np.linspace(1, 25, 25))
         ax.set_xticklabels(('1','','3','','5','','7','','9','','11','','13','','15','','17','','19','','21','','23','','25'))
    elif i == 1:
         ax.bar(ind+1, reg_coef_variables[1, 0, ind_sorted], width, color='r', edgecolor='black', linewidth=0.7,capsize=4)
         plt.plot([-0.5, 25.8], [0, 0], "k", linewidth='0.7')
         plt.plot([-0.5, 25.8], [np.ma.average(reg_coef_variables[1, 0, :]), np.ma.average(reg_coef_variables[1, 0, :])],"k", linewidth='0.7')
         plt.ylabel("\% K$^{-1}$", fontsize=14)
         plt.xlabel("model", fontsize=14)
         plt.title("(b) $\partial LCC/\partial IS$", fontsize=14, y=1.02)
         plt.xlim(0.2, 25.8)
         plt.ylim(-3.5, 1.8)
         ax.set_xticks(np.linspace(1, 25, 25))
         ax.set_xticklabels(('1', '', '3', '', '5', '', '7', '', '9', '', '11', '', '13', '', '15', '', '17', '', '19', '','21', '', '23', '', '25'))
plt.savefig('LCC_regression_slope_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

# plot 19
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(2.5, 8.6))
fig.subplots_adjust(wspace=0.4,hspace=0.5)
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.minorticks_on()
    ax.tick_params(direction='out', length=9*3/4, width=1.5*3/4, colors='black', labelsize='10', right='on', top='on')
    ax.tick_params(direction='out', which='minor', length=6*3/4, width=0.5*3/4, colors='black', \
                     labelsize='10', right='on', bottom='on', top='on')
#    ax.set_xticks((0,1,2,3))
#    ax.set_xticklabels(('DJF', 'MAM', 'JJA', 'SON'))
    if i == 0:
         x = reg_coef_variables[0,0,:]*reg_coef_SF
         y = reg_coef
         reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
         plt.plot([-0.3, 0.5],reg.intercept_+[-0.3, 0.5]*reg.coef_[0], "k",linewidth='0.7')
         for j in range(25):
             plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1),horizontalalignment='left', verticalalignment='top')
         plt.ylabel("d$[\Delta LCC/\Delta T_{g}]$/dSAF (\% W$^{-1}$ m$^{2}$)",fontsize=10)
         plt.xlabel("Contribution of F$_{s}$ (\% W$^{-1}$ m$^{2}$)", fontsize=10)
         plt.title("(a) r=-0.33, y=0.11-0.58x",fontsize=10, y=1.03)
         print(stats.mstats.pearsonr(x,y))
         print([reg.intercept_,reg.coef_[0][0]])
         plt.xlim(-0.30,0.45)
         plt.ylim(-0.30,0.45)
    elif i == 1:
         x = reg_coef_variables[1, 0, :] * reg_coef_IS
         y = reg_coef
         reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
         plt.plot([-0.3, 0.5], reg.intercept_ + [-0.3, 0.5] * reg.coef_[0], "k", linewidth='0.7')
         for j in range(25):
            plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1), horizontalalignment='left', verticalalignment='top')
         plt.ylabel("d$[\Delta LCC/\Delta T_{g}]$/dSAF (\% W$^{-1}$ m$^{2}$)", fontsize=10)
         plt.xlabel("Contribution of IS (\% W$^{-1}$ m$^{2}$)", fontsize=10)
         plt.title("(b) r=0.80, y=0.77x", fontsize=10, y=1.03)
         print(stats.mstats.pearsonr(x, y))
         print([reg.intercept_, reg.coef_[0][0]])
         plt.xlim(-0.30, 0.45)
         plt.ylim(-0.30, 0.45)
    elif i == 2:
         x = reg_coef_variables[0,0,:]*reg_coef_SF + reg_coef_variables[1, 0, :] * reg_coef_IS
         y = reg_coef_LCC
         reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
         plt.plot([-0.3, 0.5], reg.intercept_ + [-0.3, 0.5] * reg.coef_[0], "k", linewidth='0.7')
         for j in range(25):
            plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1), horizontalalignment='left', verticalalignment='top')
         plt.ylabel("d$[\Delta LCC/\Delta T_{g}]$/dSAF (\% W$^{-1}$ m$^{2}$)", fontsize=10)
         plt.xlabel("Contributions of F$_{s}$ + IS (\% W$^{-1}$ m$^{2}$)", fontsize=10)
         plt.title("(c) r=0.85, y=-0.07+1.11x", fontsize=10, y=1.03)
         print(stats.mstats.pearsonr(x, y))
         print([reg.intercept_, reg.coef_[0][0]])
         plt.xlim(-0.30, 0.45)
         plt.ylim(-0.30, 0.45)
plt.savefig('scatterplot_LCC_regression_model_SH.pdf', papertype='letter', orientation='landscape', \
            bbox_inches='tight',pad_inches=0.5)

# plot 20
fig, axes = plt.subplots(figsize=(5, 5))
plt.minorticks_on()
x = reg_coef_variables[1, 0, :]
y = reg_coef
reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
plt.plot([-4, 2], reg.intercept_ + [-4, 2] * reg.coef_[0], "k", linewidth='0.7')
for j in range(25):
    plt.text(x[ind_sorted[j]], y[ind_sorted[j]], str(j+1), horizontalalignment='center', verticalalignment='center', fontsize=14)
plt.ylabel("d$[\Delta LCC/\Delta T_{g}]$/dSAF (\% W$^{-1}$ m$^{2}$)", fontsize=14)
plt.xlabel("$\partial LCC/\partial IS$ (\% K$^{-1}$)", fontsize=14)
plt.title("r=-0.78", fontsize=14, y=1.04)
print(stats.mstats.pearsonr(x, y))
print([reg.intercept_, reg.coef_[0][0]])
plt.xlim(-3.2, 1.8)
plt.ylim(-0.30, 0.45)
axes.tick_params(direction='out', length=12, width=1.5, colors='black', labelsize='14', right='on',top='on')
axes.tick_params(direction='out', which='minor', length=7, width=0.5, colors='black', labelsize='14', right='on',top='on')
plt.savefig('scatterplot_LCC_IS_slope.pdf', papertype='letter', orientation='landscape', bbox_inches='tight',pad_inches=0.5)

# plot 21
#SH
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.8, 9.6))
fig.subplots_adjust(wspace=0.1,hspace=0.3)
m1 = Basemap(projection='spstere', boundinglat=-55, lon_0=270, resolution='l',ax=axes[0,0])
#    m.drawmapboundary(fill_color='0.3')
#m1.drawcoastlines(linewidth=0.3)
m1.fillcontinents(color='white')
# draw parallels and meridians, but don't bother labelling them.
m1.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m1.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap1 = copy.deepcopy(plt.cm.jet)
cmap1.set_under((0.000000, 0.833333, 1.000000))
im1 = m1.contourf(lon,lat,meantas,shading='flat',cmap=cmap1,latlon=True, vmin=-0.6,vmax=2.1, levels=[0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0], extend="both")
cb1=m1.colorbar(im1,"bottom", size="5%", pad="8%",label="K K$^{-1}$")
axes[0,0].set_title('(a) Changes in Ts',y=1.08)

m2 = Basemap(projection='spstere', boundinglat=-55, lon_0=270, resolution='l',ax=axes[0,1])
#    m.drawmapboundary(fill_color='0.3')
#m2.drawcoastlines(linewidth=0.3)
m2.fillcontinents(color='white')
# draw parallels and meridians, but don't bother labelling them.
m2.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m2.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap2 = copy.deepcopy(plt.cm.jet)
cmap2.set_under((0.000000, 0.0, 0.70000))
im2 = m2.contourf(lon, lat, meanhfss, shading='flat', cmap=cmap2, latlon=True, vmin=-3.4,vmax=3.1,levels=[-3, -2.0,-1, 0,1,2.0,3], extend="both")
cb2=m2.colorbar(im2,"bottom", size="5%", pad="8%", label="W m$^{-2}$ K$^{-1}$")
axes[0,1].set_title('(b) Changes in SH',y=1.08)

m3 = Basemap(projection='spstere', boundinglat=-55, lon_0=270, resolution='l',ax=axes[1,0])
#    m.drawmapboundary(fill_color='0.3')
#m3.drawcoastlines(linewidth=0.3)
m3.fillcontinents(color='white')
# draw parallels and meridians, but don't bother labelling them.
m3.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m3.drawmeridians(np.arange(-180., 181., 60.), labels=[1, 0, 1, 1], dashes=[1, 3])
cmap3 = copy.deepcopy(plt.cm.jet)
cmap3.set_under((0.000000, 0.0, 0.945633))
im3 = m3.contourf(lon, lat, meanhfls, shading='flat', cmap=cmap3, latlon=True, vmin=-3, vmax=4.1,
                 levels=[-2, -1, 0, 1.0, 2.0, 3.0,4], extend="both")
cb3=m3.colorbar(im3, "bottom", size="5%", pad="8%", label="W m$^{-2}$ K$^{-1}$")
axes[1,0].set_title('(c) Changes in LH', y=1.08)

m4 = Basemap(projection='spstere', boundinglat=-55, lon_0=270, resolution='l',ax=axes[1,1])
#    m.drawmapboundary(fill_color='0.3')
#m4.drawcoastlines(linewidth=0.3)
m4.fillcontinents(color='white')
# draw parallels and meridians, but don't bother labelling them.
m4.drawparallels(np.arange(-80., 81., 20.), dashes=[1, 3])
m4.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 1, 1, 1], dashes=[1, 3])
cmap4 = copy.deepcopy(plt.cm.jet)
cmap4.set_under((0.000000, 0.0, 0.70000))
cmap4.set_over((1.000000, 0.334786, 0.00000))
im4 = m4.contourf(lon, lat, meanIS, shading='flat', cmap=cmap4, latlon=True, vmin=-1.2, vmax=0.7,
                 levels=[-1, -0.8,-0.6, -0.4, -0.2, 0, 0.2], extend="both")
cb4=m4.colorbar(im4, "bottom", size="5%", pad="8%",label="K K$^{-1}$")
axes[1,1].set_title('(d) Changes in IS', y=1.08)
plt.savefig('tas_hfss_hfls_IS_mean_SH.pdf', papertype='letter', \
            orientation='landscape', bbox_inches='tight',pad_inches=0.5)
