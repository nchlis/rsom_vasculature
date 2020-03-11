# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:01:15 2019

@author: Nikos
https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.neighbors import KNeighborsClassifier
from scipy import ndimage as ndi
from skimage.feature import daisy
from skimage.transform import resize
from skimage.morphology import skeletonize, thin, watershed, medial_axis
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, remove_small_objects
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, threshold_local
from scipy.ndimage import gaussian_filter

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

#%% load the data

IMHEIGHT = 768
IMWIDTH = 256

savepath = './data_'+str(IMHEIGHT)+'_'+str(IMWIDTH)+'_annotated/'

X=np.load(savepath+'X.npy')#data
Y=np.load(savepath+'Y.npy')#masks

X=X[:,:,:,0:2]#drop the last black channel

df=pd.read_csv(savepath+'/metadata_qc_extra.csv')#patient metadata
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

disease=disease[qc_pass]
batch=batch[qc_pass]
neuropathy=neuropathy[qc_pass]
ascvd=ascvd[qc_pass]
bmi=bmi[qc_pass]
age=age[qc_pass]
gender=gender[qc_pass]
splits=df.splits.values[qc_pass]

#%% load the different deep learning masks

div=8
drop_rate=0.5
aug=False

#Epidermis deep learning mask, raw
Y_hat_skin=np.load(savepath+'Y_hat_skin'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy')
#Dermis deep learning mask, raw
Y_hat_vasc=np.load(savepath+'Y_hat_vasc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy')
#Epidermis deep learning mask, after post-processing
Y_hat_binary_skin=np.load(savepath+'Y_hat_binary_skin'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy')
#Dermis deep learning mask, after post-processing
Y_hat_binary_vasc=np.load(savepath+'Y_hat_binary_vasc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy')

#%% Do local thresholding on original images

X_gray=np.expand_dims(X.mean(axis=-1),axis=-1)

X_gray_vasc=X_gray.copy()
X_gray_vasc=equalize_hist(X_gray_vasc)
X_gray_vasc[Y_hat_binary_vasc==0]=0#only keep the vasculature part of the image

Y_hat_binary_thresholding=np.zeros_like(Y_hat_vasc)

for i in range(Y.shape[0]):#iterate over images
    thresh = threshold_local(X_gray[i,:,:,0],block_size=101,method='gaussian')
    Y_hat_binary_thresholding[i,:,:,0] = X_gray[i,:,:,0]>thresh
    Y_hat_binary_thresholding[i,:,:,0] = remove_small_objects(X_gray[i,:,:,0]>thresh,min_size=200)

# np.save(savepath+'Y_hat_binary_thresholding'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug),Y_hat_binary_thresholding)

#%% Do local thresholding on original images after Sato preprocessing

X_gray=np.expand_dims(X.mean(axis=-1),axis=-1)

for i in range(len(X)):
    X_gray[i,:,:,0]=sato(X_gray[i,:,:,0],black_ridges=False)

X_gray_vasc=X_gray.copy()
X_gray_vasc=equalize_hist(X_gray_vasc)
X_gray_vasc[Y_hat_binary_vasc==0]=0#only keep the vasculature part of the image

Y_hat_binary_thresholding_sato=np.zeros_like(Y_hat_vasc)

for i in range(Y.shape[0]):#iterate over images
    thresh = threshold_local(X_gray[i,:,:,0],block_size=101,method='gaussian')
    Y_hat_binary_thresholding_sato[i,:,:,0] = X_gray[i,:,:,0]>thresh
    Y_hat_binary_thresholding_sato[i,:,:,0] = remove_small_objects(X_gray[i,:,:,0]>thresh,min_size=200)

# np.save(savepath+'Y_hat_binary_thresholding_sato'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug),Y_hat_binary_thresholding_sato)

#%%
i=1

plt.figure()

plt.subplot(1,4,1)
plt.imshow(X[i,:,:,:].mean(axis=-1))
plt.xticks([])
plt.yticks([])
plt.title('Input')


plt.subplot(1,4,2)
plt.imshow(X_gray[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Input+Sato.')

plt.subplot(1,4,3)
plt.imshow(Y_hat_binary_thresholding[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Seg.')

plt.subplot(1,4,4)
plt.imshow(Y_hat_binary_thresholding_sato[i,:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Seg+Sato.')

plt.savefig('./example_figures/local_thresholding_Sato'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.png',dpi=300,bbox_inches='tight')























