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

#df=pd.read_csv(savepath+'/metadata_qc.csv')
df=pd.read_csv(savepath+'/metadata_qc_extra.csv')
X=np.load(savepath+'X.npy')#data
Y=np.load(savepath+'Y.npy')#masks

X=X[:,:,:,0:2]#drop the last black channel

df=pd.read_csv(savepath+'/metadata_qc_extra.csv')
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

#%% load the raw deep learning masks in order to postprocess them

div=8
drop_rate=0.5
aug=False

#epidermis mask
Y_hat_skin=np.load(savepath+'Y_hat_skin'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy')
#dermis mask
Y_hat_vasc=np.load(savepath+'Y_hat_vasc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.npy')

#%% discretize original masks

threshold=0.5

#discretize epidermis mask
Y_hat=Y_hat_skin
Y_hat_binary = np.zeros_like(Y_hat)
Y_hat_binary[Y_hat<threshold]=0
Y_hat_binary[Y_hat>=threshold]=1
dice_skin = np.zeros(len(Y))

for i in range(len(Y)):
    dice_skin[i]=dice2D(Y[i,:,:,0]==1,Y_hat_binary[i,:,:,0])
Y_hat_binary_skin=Y_hat_binary

#discretize dermis mask
Y_hat=Y_hat_vasc
Y_hat_binary = np.zeros_like(Y_hat)
Y_hat_binary[Y_hat<threshold]=0
Y_hat_binary[Y_hat>=threshold]=1
dice_vasc = np.zeros(len(Y))

for i in range(len(Y)):
    dice_vasc[i]=dice2D(Y[i,:,:,0]==2,Y_hat_binary[i,:,:,0])
Y_hat_binary_vasc=Y_hat_binary

#both masks in one array
Y_hat_both = np.zeros_like(Y)
Y_hat_both[Y_hat_binary_skin==True]=1
Y_hat_both[Y_hat_binary_vasc==True]=2

#%% do post-processing - no dermis should be above the deepest epidermis pixel

#for every column in Y_hat_skin, find the deepest pixel of skin

for i in range(Y.shape[0]):#iterate over images
    #plt.imshow(Y_hat_binary_skin[i,:,:,0])
    for c in np.arange(Y.shape[2]):#iterate over columns
        col=Y_hat_binary_skin[i,:,c,0]
        #plt.plot(Y_hat_binary_skin[i,:,c,0])
        skin_pixels=np.where(Y_hat_binary_skin[i,:,c,0]==1)[0]
        if len(skin_pixels>0):#the column might have no detected skin
            deepest_skin_pixel=skin_pixels[-1]
            Y_hat_binary_vasc[i,:deepest_skin_pixel,c,0]=0

#% remove small objects

selem=np.ones((10,10))
#selem=None
for i in range(Y.shape[0]):#iterate over images
    #plt.imshow(Y_hat_binary_skin[i,:,:,0])
    Y_hat_binary_skin[i,:,:,0]=skimage.morphology.binary_opening(Y_hat_binary_skin[i,:,:,0],selem=selem)  
    Y_hat_binary_vasc[i,:,:,0]=skimage.morphology.binary_opening(Y_hat_binary_vasc[i,:,:,0],selem=selem)

#% fill holes

for i in range(Y.shape[0]):#iterate over images
    #plt.imshow(Y_hat_binary_skin[i,:,:,0])
    Y_hat_binary_skin[i,:,:,0]=ndi.binary_fill_holes(Y_hat_binary_skin[i,:,:,0])  
    Y_hat_binary_vasc[i,:,:,0]=ndi.binary_fill_holes(Y_hat_binary_vasc[i,:,:,0])

#% redo masks and dice after post-processing
dice_skin = np.zeros(len(Y))#epidermis dice
for i in range(len(Y)):
    dice_skin[i]=dice2D(Y[i,:,:,0]==1,Y_hat_binary_skin[i,:,:,0])

dice_vasc = np.zeros(len(Y))#dermis dice
for i in range(len(Y)):
    dice_vasc[i]=dice2D(Y[i,:,:,0]==2,Y_hat_binary_vasc[i,:,:,0])

#both masks in one array
Y_hat_both = np.zeros_like(Y)
Y_hat_both[Y_hat_binary_skin==True]=1
Y_hat_both[Y_hat_binary_vasc==True]=2

#%% create colormap and legend patches for plots

#https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar

cmap = plt.cm.magma  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
#cmaplist[0] = (.5, .5, .5, 1.0)

# create the new map
#cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 3)

# define the bins and normalize
#bounds = np.linspace(0, 20, 21)
bounds = np.linspace(0, 3, 4)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#background_patch = mpatches.Patch(color=cmap[0], label='Background')
skin_patch = mpatches.Patch(color=cmap(1), label='Epidermis')
vasc_patch = mpatches.Patch(color=cmap(2), label='Dermis')
#plt.legend(handles=[skin_patch,vasc_patch])
#plt.show()

X_plot=np.concatenate((X,np.zeros(X.shape[:3]+(1,))),axis=-1)

#%% plot all results for MSOT

vmin=0
vmax=1
#n_scans=len(Y_hat_vasc)
n_scans=10

fig, axes = plt.subplots(nrows=n_scans,ncols=3,figsize=(3*7,n_scans*7))
for i in range(n_scans):
    #for i in [0,1,2,3,4]:
    ax=axes[i,0]
    im=ax.imshow(X_plot[i,:,:,:],vmin=vmin,vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Image')
    #add_colorbar(im)
    ax.set_ylabel('study'+str(study[i])+'-scan'+str(scan[i]))
    
    ax=axes[i,1]
    im=ax.imshow(X_plot[i,:,:,:],vmin=vmin,vmax=vmax,alpha=0.9)
    im=ax.imshow(Y[i,:,:,0],vmin=vmin,vmax=2,alpha=0.7,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('True Mask')
    skin_patch = mpatches.Patch(color=cmap(1), label='Skin')
    vasc_patch = mpatches.Patch(color=cmap(2), label='Vasc.')
    ax.legend(handles=[skin_patch,vasc_patch],loc='lower left')
    #add_colorbar(im)
    
    ax=axes[i,2]
    im=ax.imshow(X_plot[i,:,:,:],vmin=vmin,vmax=vmax,alpha=0.9)
    im=ax.imshow(Y_hat_both[i,:,:,0],vmin=vmin,vmax=2,alpha=0.7,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Predicted Mask')
    skin_patch = mpatches.Patch(color=cmap(1), label='Skin ('+str(np.round(dice_skin[i],2))+')')
    vasc_patch = mpatches.Patch(color=cmap(2), label='Vasc. ('+str(np.round(dice_vasc[i],2))+')')
    ax.legend(handles=[skin_patch,vasc_patch],loc='lower left')
#    ax.set_title('Predicted Mask='+str(np.round(dice_skin[i],2)))
    #add_colorbar(im)
plt.savefig('./example_figures/masks_predicted_afterPostProc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.png',dpi=100,bbox_inches='tight')

#%% plot all results for MSOT

vmin=0
vmax=1
#n_scans=len(Y_hat_vasc)
n_scans=20
#ix_sorted=np.argsort(dice_skin+dice_vasc)[::-1]
ix_sorted=np.argsort(dice_vasc)[::-1]

fig, axes = plt.subplots(nrows=n_scans,ncols=3,figsize=(3*3.5,n_scans*7))
for i in range(n_scans):
    row=i
    i=ix_sorted[i]
    #for i in [0,1,2,3,4]:
    ax=axes[row,0]
    im=ax.imshow(X_plot[i,:,:,:],vmin=vmin,vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Image')
    #add_colorbar(im)
    ax.set_ylabel('study'+str(study[i])+'-scan'+str(scan[i]))
    
    ax=axes[row,1]
    im=ax.imshow(X_plot[i,:,:,:],vmin=vmin,vmax=vmax,alpha=0.9)
    im=ax.imshow(Y[i,:,:,0],vmin=vmin,vmax=2,alpha=0.7,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('True')
    skin_patch = mpatches.Patch(color=cmap(1), label='Skin')
    vasc_patch = mpatches.Patch(color=cmap(2), label='Vasc.')
    ax.legend(handles=[skin_patch,vasc_patch],loc='lower left')
    #add_colorbar(im)
    
    ax=axes[row,2]
    im=ax.imshow(X_plot[i,:,:,:],vmin=vmin,vmax=vmax,alpha=0.9)
    im=ax.imshow(Y_hat_both[i,:,:,0],vmin=vmin,vmax=2,alpha=0.7,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Predicted')
    skin_patch = mpatches.Patch(color=cmap(1), label='Skin ('+str(np.round(dice_skin[i],2))+')')
    vasc_patch = mpatches.Patch(color=cmap(2), label='Vasc. ('+str(np.round(dice_vasc[i],2))+')')
    ax.legend(handles=[skin_patch,vasc_patch],loc='lower left')
#    ax.set_title('Predicted Mask='+str(np.round(dice_skin[i],2)))
    #add_colorbar(im)
plt.savefig('./example_figures/masks_predicted_afterPostProc_diceSorted'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.png',dpi=100,bbox_inches='tight')

#%%
plt.figure(figsize=(5,5))
plt.boxplot([dice_skin,dice_vasc],labels=['skin','vasc'])
plt.xticks([1,2],['Epidermis','Dermis'],size=14)
plt.title('Dice, after post-processing',size=16)
plt.text(1+0.1,np.median(dice_skin),str(np.round(np.median(dice_skin),2)),size=14)
plt.text(2+0.1,np.median(dice_vasc),str(np.round(np.median(dice_vasc),2)),size=14)
plt.ylim(-0.1,1)
#plt.xticks([])
plt.savefig('./example_figures/dice_boxplot_afterPostProc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)+'.png',dpi=100,bbox_inches='tight')

#%% save the post-processed binary arrays
#np.save(savepath+'Y_hat_binary_skin'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug),Y_hat_binary_skin)
#np.save(savepath+'Y_hat_binary_vasc'+'_mcd_unet_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug),Y_hat_binary_vasc)















