# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:39:11 2020

@author: Nikos
"""

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

import numpy as np
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import h5py
from keras.models import load_model
from keras import backend as K 
import gc
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
import skimage.morphology
from skimage.exposure import histogram, equalize_hist
from skimage.filters import sobel
from scipy.stats import ttest_ind

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import ndimage as ndi
from skimage.feature import daisy
from skimage.transform import resize
from skimage.morphology import skeletonize, thin, watershed, medial_axis
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, remove_small_objects
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, threshold_local
from statsmodels.stats.proportion import proportion_confint
# import mahotas as mh
import skan
from skan import draw 
from matplotlib.colors import Normalize
from collections import OrderedDict
from scipy.stats import pearsonr
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# import sklearn
# sklearn.__version__
# Out[71]: '0.21.3'

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def dice2D(a,b):
    #https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
    #https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    intersection = np.sum(a[b==1])
    dice = (2*intersection)/(np.sum(a)+np.sum(b))
    if (np.sum(a)+np.sum(b))==0: #black/empty masks
        dice=1.0
    return(dice)

from scipy import stats

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

#%%
#load the data
IMHEIGHT = 768
IMWIDTH = 256

savepath = './data_'+str(IMHEIGHT)+'_'+str(IMWIDTH)+'_annotated/'

#df=pd.read_csv(savepath+'/metadata_qc.csv')
# df=pd.read_csv(savepath+'/metadata_qc_extra.csv')
X=np.load(savepath+'X.npy')#data
Y=np.load(savepath+'Y.npy')#masks

X=X[:,:,:,0:2]#drop the last black channel

df=pd.read_csv(savepath+'/metadata_qc_extra_med_onset2.csv')
study=df['case'].values
scan=df['scan'].values
#do quality control, only keep 115 out of 122 unique studies
#the 115 unique studies correspond to unique 205 scans (multiple scans for some patients)
qc_pass = df.quality_control.values=='pass'
study=study[qc_pass]
scan=scan[qc_pass]
X=X[qc_pass,:,:,:]
Y=Y[qc_pass,:,:,:]
study_unique=np.unique(study)


disease=df.disease.values#1 disease, 0 control
batch=df.batch.values
neuropathy=df.neuropathy.values
ascvd=df.ASCVD.values
bmi=df.BMI.values
age=df.age.values
gender=df.gender.values
splits=df.splits.values
medication=df.medication.values
case=df.case.values
diabetes_type=df.diabetes_type
gender_num=df.gender_num
years_since_onset=df.years_since_onset
nss=df.NSS
nds=df.NDS
hba1c=df.HbA1c

disease=disease[qc_pass]
batch=batch[qc_pass]
neuropathy=neuropathy[qc_pass]
ascvd=ascvd[qc_pass]
bmi=bmi[qc_pass]
age=age[qc_pass]
gender=gender[qc_pass]
splits=df.splits.values[qc_pass]
medication=medication[qc_pass]
case=case[qc_pass]
diabetes_type=diabetes_type[qc_pass]
gender_num=gender_num[qc_pass]
years_since_onset=years_since_onset[qc_pass]
nss=nss[qc_pass]
nds=nds[qc_pass]
hba1c=hba1c[qc_pass]

#convert medication to categorical
medication_cat=pd.Series(pd.Categorical(medication))
ix_pat080=166#np.where(case=='PAT080')#remove due to quality control

#%% load masks
div=8
drop_rate=0.5
aug=False

#after droping out patient 80 for metadata_qc_extra_med_onset.csv
Y_hat_skin=np.delete(np.load(savepath+'Y_hat_skin'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy'),ix_pat080,axis=0)
Y_hat_vasc=np.delete(np.load(savepath+'Y_hat_vasc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy'),ix_pat080,axis=0)
Y_hat_binary_skin=np.delete(np.load(savepath+'Y_hat_binary_skin'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy'),ix_pat080,axis=0)
Y_hat_binary_vasc=np.delete(np.load(savepath+'Y_hat_binary_vasc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy'),ix_pat080,axis=0)
Y_hat_binary_vasc_fine=np.delete(np.load(savepath+'Y_hat_binary_vasc_fine'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy'),ix_pat080,axis=0)
Y_hat_binary_thresholding_sato=np.delete(np.load(savepath+'Y_hat_binary_thresholding_sato'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy'),ix_pat080,axis=0)

#%%
Y_hat_both = np.zeros_like(Y)
Y_hat_both[Y_hat_binary_skin==True]=1
Y_hat_both[Y_hat_binary_vasc==True]=2

Y_hat_binary_thresholding_sato_skin=Y_hat_binary_thresholding_sato.copy()
Y_hat_binary_thresholding_sato_skin[Y_hat_binary_skin==False]=0
Y_hat_binary_thresholding_sato_vasc=Y_hat_binary_thresholding_sato.copy()
Y_hat_binary_thresholding_sato_vasc[Y_hat_binary_vasc==False]=0

#%%
i=0
plt.figure(figsize=(6,2))

plt.subplot(1,6,1)
plt.imshow(X[i,:].mean(axis=-1))
plt.xticks([])
plt.yticks([])
plt.title('Input')

plt.subplot(1,6,2)
plt.imshow(Y_hat_binary_thresholding_sato[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Mask')

plt.subplot(1,6,3)
plt.imshow(Y_hat_binary_thresholding_sato_skin[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Mask-Skin')

plt.subplot(1,6,4)
plt.imshow(Y_hat_binary_thresholding_sato_vasc[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Mask-Vasc')

plt.subplot(1,6,5)
plt.imshow(Y_hat_binary_skin[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Skin-Area')

plt.subplot(1,6,6)
plt.imshow(Y_hat_binary_vasc[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Vasc-Area')

#%% plot the RSOM image that corresponds to the vasculature and skin masks

#only segmented vessels
X_vasc=np.expand_dims(X.mean(axis=-1),axis=-1).copy()
X_vasc[Y_hat_binary_thresholding_sato_vasc==0]=0
#entire segmented area
X_vasc_region=np.expand_dims(X.mean(axis=-1),axis=-1).copy()
X_vasc_region[Y_hat_binary_vasc==0]=0

plt.figure()
plt.subplot(1,2,1)
plt.imshow(X_vasc[0,:,:,0])

#only segmented vessels
X_skin=np.expand_dims(X.mean(axis=-1),axis=-1).copy()
X_skin[Y_hat_binary_thresholding_sato_skin==0]=0
#entire segmented area
X_skin_region=np.expand_dims(X.mean(axis=-1),axis=-1).copy()
X_skin_region[Y_hat_binary_skin==0]=0

plt.subplot(1,2,2)
plt.imshow(X_skin[0,:,:,0])

#%% get the vascularity features

intensity_vasc=X_vasc.sum(-1).sum(-1).sum(-1)
area_vasc=Y_hat_binary_thresholding_sato_vasc.sum(-1).sum(-1).sum(-1)

intensity_vasc_region=X_vasc_region.sum(-1).sum(-1).sum(-1)
area_vasc_region=Y_hat_binary_vasc.sum(-1).sum(-1).sum(-1)

Xskeleton=np.zeros_like(Y_hat_binary_thresholding_sato_vasc)
for i in np.arange(len(X_vasc)):
    Xskeleton[i,:]=skeletonize(Y_hat_binary_thresholding_sato_vasc[i,:])

#calculate skeleton statistics for all images
branch_number = np.zeros(len(X))
branch_j2e_total = np.zeros(len(X))
branch_j2j_total = np.zeros(len(X))

for i in range(len(Y_hat_binary_skin)):#iterate over images
    print(i)
    skeleton=Xskeleton[i,:,:,0]
    try:
        branch_data = skan.summarize(skan.Skeleton(skeleton))
        branch_number[i]=len(branch_data['branch-distance'].values)
        branch_j2e_total[i] = np.sum(branch_data['branch-type'].values==1)
        branch_j2j_total[i] = np.sum(branch_data['branch-type'].values==2)
    except:
        print('problem with mask')

Xskeleton_vasc=Xskeleton

nbranch_vasc = branch_number
nbranch_j2e_vasc = branch_j2e_total
nbrach_j2j_vasc = branch_j2j_total

#calculate depth
depth_vasc=np.zeros(len(Y_hat_binary_thresholding_sato_vasc))

#for every column in Y_hat_vasc, find the deepest pixel of skin

for i in range(len(Y_hat_binary_thresholding_sato_vasc)):#iterate over images
    #plt.imshow(Y_hat_binary_vasc[i,:,:,0])
    skin_depth=[]
    for c in np.arange(Y.shape[2]):#iterate over columns
        col=Y_hat_binary_thresholding_sato_vasc[i,:,c,0]
        skin_pixels=np.where(Y_hat_binary_thresholding_sato_vasc[i,:,c,0]==1)[0]
        if len(skin_pixels>0):#the column might have no detected skin
            deepest_vasc_pixel=skin_pixels[-1]
            shallowest_vasc_pixel=skin_pixels[0]
            skin_depth.append(deepest_vasc_pixel-shallowest_vasc_pixel)
    depth_vasc[i]=np.median(skin_depth)
    if np.isnan(depth_vasc[i]):#no skin detected by the model
        depth_vasc[i]=0

#%% get the skin features

intensity_skin=X_skin.sum(-1).sum(-1).sum(-1)
area_skin=Y_hat_binary_thresholding_sato_skin.sum(-1).sum(-1).sum(-1)

intensity_skin_region=X_skin_region.sum(-1).sum(-1).sum(-1)#was the same as vasc before
area_skin_region=Y_hat_binary_skin.sum(-1).sum(-1).sum(-1)#was the same as vasc before

Xskeleton=np.zeros_like(Y_hat_binary_thresholding_sato_vasc)
for i in np.arange(len(X_vasc)):
    Xskeleton[i,:]=skeletonize(Y_hat_binary_thresholding_sato_skin[i,:])

#calculate skeleton statistics for all images
branch_number = np.zeros(len(X))
branch_j2e_total = np.zeros(len(X))
branch_j2j_total = np.zeros(len(X))

for i in range(len(Y_hat_binary_skin)):#iterate over images
    print(i)
    skeleton=Xskeleton[i,:,:,0]
    try:
        branch_data = skan.summarize(skan.Skeleton(skeleton))
        branch_number[i]=len(branch_data['branch-distance'].values)
        branch_j2e_total[i] = np.sum(branch_data['branch-type'].values==1)
        branch_j2j_total[i] = np.sum(branch_data['branch-type'].values==2)
    except:
        print('problem with mask')

Xskeleton_skin=Xskeleton

nbranch_skin = branch_number
nbranch_j2e_skin = branch_j2e_total
nbrach_j2j_skin = branch_j2j_total

#calculate depth
depth_skin=np.zeros(len(Y_hat_binary_thresholding_sato_skin))

#for every column in Y_hat_skin, find the deepest pixel of skin

for i in range(len(Y_hat_binary_thresholding_sato_skin)):#iterate over images
    #plt.imshow(Y_hat_binary_skin[i,:,:,0])
    skin_depth=[]
    for c in np.arange(Y.shape[2]):#iterate over columns
        col=Y_hat_binary_thresholding_sato_skin[i,:,c,0]
        skin_pixels=np.where(Y_hat_binary_thresholding_sato_skin[i,:,c,0]==1)[0]
        if len(skin_pixels>0):#the column might have no detected skin
            deepest_skin_pixel=skin_pixels[-1]
            shallowest_skin_pixel=skin_pixels[0]
            skin_depth.append(deepest_skin_pixel-shallowest_skin_pixel)
    depth_skin[i]=np.median(skin_depth)
    if np.isnan(depth_skin[i]):#no skin detected by the model
        depth_skin[i]=0

#%% put all features in an array

# epidermis and dermis features, including depth and region intensity and area
X_feat=np.concatenate((np.expand_dims(intensity_skin,axis=-1),
                        np.expand_dims(intensity_skin_region,axis=-1),
                        np.expand_dims(area_skin,axis=-1),
                        np.expand_dims(area_skin_region,axis=-1),
                        np.expand_dims(nbranch_skin,axis=-1),
                        np.expand_dims(nbranch_j2e_skin,axis=-1),
                        np.expand_dims(nbrach_j2j_skin,axis=-1),
                        np.expand_dims(depth_skin,axis=-1),
                        np.expand_dims(intensity_vasc,axis=-1),
                        np.expand_dims(intensity_vasc_region,axis=-1),
                        np.expand_dims(area_vasc,axis=-1),
                        np.expand_dims(area_vasc_region,axis=-1),
                        np.expand_dims(nbranch_vasc,axis=-1),
                        np.expand_dims(nbranch_j2e_vasc,axis=-1),
                        np.expand_dims(nbrach_j2j_vasc,axis=-1),
                        np.expand_dims(depth_vasc,axis=-1)),axis=-1)

# Only epidermis features, including depth and region intensity and area
X_feat_skin=np.concatenate((np.expand_dims(intensity_skin,axis=-1),
                        np.expand_dims(intensity_skin_region,axis=-1),
                        np.expand_dims(area_skin,axis=-1),
                        np.expand_dims(area_skin_region,axis=-1),
                        np.expand_dims(nbranch_skin,axis=-1),
                        np.expand_dims(nbranch_j2e_skin,axis=-1),
                        np.expand_dims(nbrach_j2j_skin,axis=-1),
                        np.expand_dims(depth_skin,axis=-1)),axis=-1)

# Only dermis features, including depth and region intensity and area
X_feat_vasc=np.concatenate((np.expand_dims(intensity_vasc,axis=-1),
                        np.expand_dims(intensity_vasc_region,axis=-1),
                        np.expand_dims(area_vasc,axis=-1),
                        np.expand_dims(area_vasc_region,axis=-1),
                        np.expand_dims(nbranch_vasc,axis=-1),
                        np.expand_dims(nbranch_j2e_vasc,axis=-1),
                        np.expand_dims(nbrach_j2j_vasc,axis=-1),
                        np.expand_dims(depth_vasc,axis=-1)),axis=-1)

#only region features
X_feat_region=np.concatenate((np.expand_dims(intensity_skin_region,axis=-1),
                        np.expand_dims(area_skin_region,axis=-1),
                        np.expand_dims(depth_skin,axis=-1),
                        np.expand_dims(intensity_vasc_region,axis=-1),
                        np.expand_dims(area_vasc_region,axis=-1),
                        np.expand_dims(depth_vasc,axis=-1)),axis=-1)

#%% do leave one out at study level
# disease_severe=((disease+neuropathy+ascvd)>=2).astype(int)

rfseed=42
# rfseed=2
N=len(study_unique)
P=X_feat.shape[1]
sample_weight=np.ones(N)
# sample_weight[disease==1]=0.01
rf_list=[]
rf_fimp=np.zeros((N,P))
y_pred=[]
y_pred_prob=[]
y_pred_study=np.zeros(N)
y_pred_prob_study=np.zeros(N,dtype='float')
disease_unique=np.zeros(N)
i=0
for s in study_unique:
    print(i+1,'of',len(study_unique),'study_unique',s)
    
    ix_tr=(study!=s)
    ix_ts=(study==s)
    
    
    X_tr=X_feat[ix_tr,:]
    y_tr=disease[ix_tr]
    # y_tr=disease_severe[ix_tr]
    X_ts=X_feat[ix_ts,:]
    y_ts=disease[ix_ts]
    # y_ts=disease_severe[ix_ts]
    if len(X_ts.shape) < 2:
        np.np.expand_dims(X_ts,axis=0)
    
    rf=RandomForestClassifier(n_estimators=100,max_depth=None,random_state=rfseed)#42
    rf.fit(X_tr,y_tr)
    
    #scan level
    y_pred_prob_study[i] = rf.predict_proba(X_ts)[:,1].mean()#get the score for the disease class
    y_pred_study[i]=int(y_pred_prob_study[i]>0.5)
    
    #image_level
    y_pred_prob=y_pred_prob+rf.predict_proba(X_ts)[:,1].tolist()
    y_pred=y_pred+rf.predict(X_ts).tolist()
    # y_pred[i]=rf.predict(X_ts).mean()>0.5#voting scheme
    rf_fimp[i,:]=rf.feature_importances_
    disease_unique[i]=disease[ix_ts][0]#pick one of the (same) true labels
    # disease_unique[i]=disease_severe[ix_ts][0]#pick one of the (same) true labels
    i=i+1
#convert to array
y_pred_prob=np.array(y_pred_prob)
y_pred=np.array(y_pred).astype('int')

#%% caclulate predictive performance metrics

#do at study level
y_ts=disease_unique
y_ts_hat=y_pred_study#defaul 0.5 threshold

cnf_matrix = confusion_matrix(y_ts, y_ts_hat)
print(cnf_matrix)
acc=(y_ts_hat==y_ts).sum()/len(y_ts_hat)
print('acc',np.round(acc,2))

TP=cnf_matrix[1,1]
TN=cnf_matrix[0,0]
FP=cnf_matrix[0,1]
FN=cnf_matrix[1,0]

sns=TP/(TP+FN)
spc=TN/(TN+FP)

print('')
print('Test results')
print('accuracy:',np.round(acc,2))
print('sensitivity:',np.round(sns,2))
print('specificity:',np.round(spc,2))


#%% leave one out for skin vasc and region features at the patient level

y_pred_prob_skin=np.zeros(len(disease_unique),dtype='float')-1
y_pred_prob_vasc=np.zeros(len(disease_unique),dtype='float')-1
y_pred_prob_region=np.zeros(len(disease_unique),dtype='float')-1
i=0
for s in study_unique:
    print(i+1,'of',len(study_unique),'study_unique',s)
    
    ix_tr=(study!=s)
    ix_ts=(study==s)
    
    y_tr=disease[ix_tr]
    y_ts=disease[ix_ts]

    #skin features
    X_tr=X_feat_skin[ix_tr,:]
    X_ts=X_feat_skin[ix_ts,:]
    if len(X_ts.shape) < 2:
        np.np.expand_dims(X_ts,axis=0)
    rf2=RandomForestClassifier(n_estimators=100,max_depth=None,random_state=rfseed)#42
    rf2.fit(X_tr,y_tr)
    y_pred_prob_skin[i] = rf2.predict_proba(X_ts)[:,1].mean()#get the score for the disease class
    
    #vasc features
    X_tr=X_feat_vasc[ix_tr,:]
    X_ts=X_feat_vasc[ix_ts,:]
    if len(X_ts.shape) < 2:
        np.np.expand_dims(X_ts,axis=0)
    rf2=RandomForestClassifier(n_estimators=100,max_depth=None,random_state=rfseed)#42
    rf2.fit(X_tr,y_tr)
    y_pred_prob_vasc[i] = rf2.predict_proba(X_ts)[:,1].mean()#get the score for the disease class
    
    #region features
    X_tr=X_feat_region[ix_tr,:]
    X_ts=X_feat_region[ix_ts,:]
    if len(X_ts.shape) < 2:
        np.np.expand_dims(X_ts,axis=0)
    rf2=RandomForestClassifier(n_estimators=100,max_depth=None,random_state=rfseed)#42
    rf2.fit(X_tr,y_tr)
    y_pred_prob_region[i] = rf2.predict_proba(X_ts)[:,1].mean()#get the score for the disease class
    i=i+1

#%% ROC curve mutliple models

from sklearn.metrics import roc_curve, auc

y_test=disease_unique#calculating at the scan level
y_pred_roc=y_pred_prob_study

plt.figure(figsize=(6,6))
lw = 3
#baseline
fpr, tpr, _ = roc_curve(y_test, y_pred_roc)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='black', lw=lw, label='Both AUC: %0.2f' % roc_auc,zorder=+1)
#epideris only
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_skin)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=plt.cm.Dark2(3), lw=lw, label='Epidermis AUC: %0.2f' % roc_auc,zorder=-1)
#dermis only
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_vasc)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=plt.cm.Dark2(0), lw=lw, label='Dermis AUC: %0.2f' % roc_auc,zorder=-1)
#unet only
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_region)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='silver', lw=lw, label='Both (UNET) AUC: %0.2f' % roc_auc,zorder=-1)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--',label='Random AUC: 0.5',zorder=-1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate',fontsize=20)
# plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('1 - Specificity',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.title('ROC curve',fontsize=22)
plt.xticks(np.arange(0,1.1,0.1),fontsize=14)
plt.yticks(np.arange(0,1.1,0.1),fontsize=14)
plt.scatter(1-spc,sns,marker='X',s=200,c='black',label='Selected threshold')
plt.legend(loc="lower right",fontsize=16)

plt.savefig('./example_figures/fig2_roc_auc_multiple_scanLevel.png',dpi=300,bbox_inches='tight')







