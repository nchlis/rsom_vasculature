# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:04:13 2018

@author: N.Chlis
"""
#if used on a non-GUI server ######
#import matplotlib
#matplotlib.use('Agg')
###################################

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import h5py
import pandas as pd

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
#from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from keras.layers.merge import concatenate #Concatenate (capital C) not working 
#from keras.utils.vis_utils import plot_model
from keras.layers import Dropout

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import time
import skimage.transform
from sklearn.model_selection import train_test_split
from keras import backend as K 
import gc
import os

def rotateT(X,angle):
    #rotate image tensor, TF order, single channel
    X_rot = np.zeros_like(X)
    #repeat for every channel
    for ch in np.arange(X.shape[-1]):
        #print('channel',ch)
        #repeat for every image
        for i in np.arange(X.shape[0]):
            #print('image',i)
            X_rot[i,:,:,ch] = skimage.transform.rotate(X[i,:,:,ch],angle=angle,resize=False,preserve_range=True,mode='edge')
    return(X_rot)

def shiftT(X,dx,dy):
    #shift image tensor, TF order, single channel
    X_shift = np.zeros_like(X)
    #repeat for every image
    tform = skimage.transform.SimilarityTransform(translation=(dx, dy))
    for i in np.arange(X.shape[0]):
        #print('image',i)
        X_shift[i,:,:,:] = skimage.transform.warp(X[i,:,:,:],tform,mode='edge')
    return(X_shift)

#%% define the generator for training
# randomly flip, rotate and translate each input image

def aug_generator(X_raw=None,Y_raw=None,
                  batch_size=4,
                  flip_axes=['x','y'],
                  rotation_angles=[5,15],
                  translate_axes=['x','y'],
                  translate_percentages=[0,0.1]):
    
    batch_size=batch_size#recommended batch size    
    Ndatapoints = len(X_raw)
    #Naugmentations=4 #original + flip, rotation, noise_gaussian, noise_snp
    
    while(True):
        #print('start!')
        ix_randomized = np.random.choice(Ndatapoints,size=Ndatapoints,replace=False)
        ix_batches = np.array_split(ix_randomized,int(Ndatapoints/batch_size))
        for b in range(len(ix_batches)):
            #print('step',b,'of',len(ix_batches))
            ix_batch = ix_batches[b]
            current_batch_size=len(ix_batch)
            #print('size of current batch',current_batch_size)
            #print(ix_batch)
            X_batch = X_raw[ix_batch,:,:,:].copy()#.copy() to leave original unchanged
            Y_batch = Y_raw[ix_batch,:,:,:].copy()#.copy() to leave original unchanged
            
            #now do augmentation on images and masks
            #iterate over each image in the batch
            for img in range(current_batch_size):
                #print('current_image',img,': ',ix_batch[img])
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    #print('flipping',img)
                    flip_axis_selected = np.random.choice(flip_axes,1,replace=False)[0]
                    if flip_axis_selected == 'x':
                        flip_axis_selected = 1
                    else: # 'y'
                        flip_axis_selected = 0
                    #flip an axis
                    X_batch[img,:,:,:] = np.flip(X_batch[img,:,:,:],axis=flip_axis_selected)
                    Y_batch[img,:,:,:] = np.flip(Y_batch[img,:,:,:],axis=flip_axis_selected)
                    #print('Flip on axis',flip_axis_selected)
                
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    #print('rotating',img)
                    rotation_angle_selected = np.random.uniform(low=rotation_angles[0],high=rotation_angles[1],size=1)[0]
                    #rotate the image
                    X_batch[img,:,:,:] = rotateT(np.expand_dims(X_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
                    Y_batch[img,:,:,:] = rotateT(np.expand_dims(Y_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
                    #print('Rotate angle',rotation_angle_selected)
                    
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    #print('shifting',img)
                    #print(X_batch.shape)
                    dx=0
                    if 'x' in translate_axes:    
                        dx=np.random.uniform(low=translate_percentages[0],high=translate_percentages[1],size=1)[0]
                        dx=dx*X_batch.shape[1]
                    dy=0
                    if 'y' in translate_axes:    
                        dy=np.random.uniform(low=translate_percentages[0],high=translate_percentages[1],size=1)[0]
                        dy=dy*X_batch.shape[2]
                    #translate the image
                    #print('dx',dx)
                    #print('dy',dy)
                    X_batch[img,:,:,:] = shiftT(np.expand_dims(X_batch[img,:,:,:],axis=0),dx=dx,dy=dy)
                    Y_batch[img,:,:,:] = shiftT(np.expand_dims(Y_batch[img,:,:,:],axis=0),dx=dx,dy=dy)
                    
            Y_batch_skin=(Y_batch==1).astype('float')
            Y_batch_vasc=(Y_batch==2).astype('float')
            yield(X_batch,[Y_batch_skin,Y_batch_vasc])
            #print('step end after',b,'of',len(ix_batches))

#%%
#load the data
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
#do quality control, only keep 115 out of 122 unique studies
#the 115 unique studies correspond to unique 205 scans (multiple scans for some patients)
qc_pass = df.quality_control.values=='pass'
study=study[qc_pass]
X=X[qc_pass,:,:,:]
Y=Y[qc_pass,:,:,:]
study_unique=np.unique(study)

#%% do leave one patient out validation

start=0
#resume=True
resume=False
if resume==True:
    #there are 2 files per saved model in the trained_models folder
    start=int(len(os.listdir('./trained_models/'))/2)

for i in np.arange(start=start,stop=len(study_unique)):#Leave one study out
    print('*** Study',i+1,'of',len(study_unique),'***')

    s = study_unique[i]
    print('Study number',s)
    train_ix = study!=s
    test_ix = study==s
    
    X_tr=X[train_ix,:]
    Y_tr=Y[train_ix,:]
    
    #X_ts=X[test_ix,:]
    #Y_ts=Y[test_ix,:]
    
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.1, random_state=1)
    
    Y_tr_skin=(Y_tr==1).astype('float')
    Y_tr_vasc=(Y_tr==2).astype('float')    
    Y_val_skin=(Y_val==1).astype('float')
    Y_val_vasc=(Y_val==2).astype('float')
    #%% set-up the UNET model
    
    #model parameters
    bnorm_axis = -1
    #filter sizes of the original model
    nfilters = np.array([64, 128, 256, 512, 1024])
    drop_rate=0.5
    drop_train=False
    #downsize the UNET for this example.
    #the smaller network is faster to train
    #and produces excellent results on the dataset at hand
    div=8
    nfilters = (nfilters/div).astype('int')
    #aug=True
    aug=False
    
    #input
    input_tensor = Input(shape=X_tr.shape[1:], name='input_tensor')
    
    ####################################
    # encoder (contracting path)
    ####################################
    #encoder block 0
    e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(input_tensor)
    e0 = BatchNormalization(axis=bnorm_axis)(e0)
    e0 = Activation('relu')(e0)
    e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
    e0 = BatchNormalization(axis=bnorm_axis)(e0)
    e0 = Activation('relu')(e0)
    
    #encoder block 1
    e1 = MaxPooling2D((2, 2))(e0)
    e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
    e1 = BatchNormalization(axis=bnorm_axis)(e1)
    e1 = Activation('relu')(e1)
    e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
    e1 = BatchNormalization(axis=bnorm_axis)(e1)
    e1 = Activation('relu')(e1)
    
    #encoder block 2
    e2 = Dropout(drop_rate)(e1, training = drop_train)
    e2 = MaxPooling2D((2, 2))(e2)
    e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
    e2 = BatchNormalization(axis=bnorm_axis)(e2)
    e2 = Activation('relu')(e2)
    e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
    e2 = BatchNormalization(axis=bnorm_axis)(e2)
    e2 = Activation('relu')(e2)
    
    #encoder block 3
    e3 = Dropout(drop_rate)(e2, training = drop_train)
    e3 = MaxPooling2D((2, 2))(e3)
    e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
    e3 = BatchNormalization(axis=bnorm_axis)(e3)
    e3 = Activation('relu')(e3)
    e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
    e3 = BatchNormalization(axis=bnorm_axis)(e3)
    e3 = Activation('relu')(e3)
    
    #encoder block 4
    e4 = Dropout(drop_rate)(e3, training = drop_train)
    e4 = MaxPooling2D((2, 2))(e4)
    e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
    e4 = BatchNormalization(axis=bnorm_axis)(e4)
    e4 = Activation('relu')(e4)
    e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
    e4 = BatchNormalization(axis=bnorm_axis)(e4)
    e4 = Activation('relu')(e4)
    #e4 = MaxPooling2D((2, 2))(e4)
    
    ####################################
    # decoder (expansive path)
    ####################################
    
    #decoder block 3
    d3 = Dropout(drop_rate)(e4, training = drop_train)
    d3=UpSampling2D((2, 2),)(d3)
    d3=concatenate([e3,d3], axis=-1)#skip connection
    d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
    d3=BatchNormalization(axis=bnorm_axis)(d3)
    d3=Activation('relu')(d3)
    d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
    d3=BatchNormalization(axis=bnorm_axis)(d3)
    d3=Activation('relu')(d3)
    
    #decoder block 2
    d2 = Dropout(drop_rate)(d3, training = drop_train)
    d2=UpSampling2D((2, 2),)(d2)
    d2=concatenate([e2,d2], axis=-1)#skip connection
    d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
    d2=BatchNormalization(axis=bnorm_axis)(d2)
    d2=Activation('relu')(d2)
    d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
    d2=BatchNormalization(axis=bnorm_axis)(d2)
    d2=Activation('relu')(d2)
    
    #decoder block 1
    d1=UpSampling2D((2, 2),)(d2)
    d1=concatenate([e1,d1], axis=-1)#skip connection
    d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
    d1=BatchNormalization(axis=bnorm_axis)(d1)
    d1=Activation('relu')(d1)
    d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
    d1=BatchNormalization(axis=bnorm_axis)(d1)
    d1=Activation('relu')(d1)
    
    #decoder block 0
    d0=UpSampling2D((2, 2),)(d1)
    d0=concatenate([e0,d0], axis=-1)#skip connection
    d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
    d0=BatchNormalization(axis=bnorm_axis)(d0)
    d0=Activation('relu')(d0)
    d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
    d0=BatchNormalization(axis=bnorm_axis)(d0)
    d0=Activation('relu')(d0)
    
    #output
    #out_class = Dense(1)(d0)
    out_class_skin = Conv2D(1, (1, 1), padding='same')(d0)
    out_class_skin = Activation('sigmoid',name='output_skin')(out_class_skin)
    
    out_class_vasc = Conv2D(1, (1, 1), padding='same')(d0)
    out_class_vasc = Activation('sigmoid',name='output_vasc')(out_class_vasc)
    
    #create and compile the model
    model=Model(inputs=input_tensor,outputs=[out_class_skin,out_class_vasc])
    model.compile(loss={'output_skin':'binary_crossentropy',#epidermis region
                    'output_vasc':'binary_crossentropy'},#dermis region
              optimizer='adam')
    #%%
    print(model.summary())
    
    #%% train the model
    filepath = 'mcd_unet_testStudy'+str(s)+'_MSOT_div'+str(div)+'_drop_rate'+str(drop_rate)+'_aug'+str(aug)

    #save the model when val_loss improves during training
    checkpoint = ModelCheckpoint('./trained_models/'+filepath+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #save training progress in a .csv
    csvlog = CSVLogger('./trained_models/'+filepath+'_train_log.csv',append=True)
    #stop training if no improvement has been seen on val_loss for a while
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=40)
    batch_size=16
    
    #initialize the generator
    gen_train = aug_generator(X_tr,Y_tr,batch_size=batch_size,flip_axes=['x'])
    #split the array and see how many splits there are to determine #steps
    steps_per_epoch_tr = len(np.array_split(np.zeros(len(X_tr)),int(len(X_tr)/batch_size)))
    
    if aug==True:
        #actually do the training
        model.fit_generator(generator=gen_train,
                            steps_per_epoch=steps_per_epoch_tr,#the generator internally goes over the entire dataset in one iteration
                            validation_data=(X_val,[Y_val_skin,Y_val_vasc]),
                            epochs=200,
                            verbose=2,
                            initial_epoch=0,
                            callbacks=[checkpoint, csvlog, early_stopping])
    else:#no data augmentation
        model.fit(x=X_tr,y=[Y_tr_skin,Y_tr_vasc],
                            batch_size=batch_size,
                            validation_data=(X_val,[Y_val_skin,Y_val_vasc]),
                            epochs=200,
                            verbose=2,
                            initial_epoch=0,
                            callbacks=[checkpoint, csvlog, early_stopping])
    print('clearing Keras session...')
    del model
    K.clear_session()
    gc.collect()







