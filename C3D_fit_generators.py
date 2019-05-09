import numpy as np
import h5py
from keras.models import Model
import sys
sys.path.append('/home/atilocca/c3d')
#from c3d import C3D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Input,Flatten,Dropout,GlobalMaxPooling1D,GlobalMaxPooling3D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import StratifiedKFold,GroupKFold,GroupShuffleSplit
import math

def train_model(model,videoData,k,batch_size):
    with h5py.File(videoData, "r") as video_data:
         sample_count = int(video_data["y"].shape[0])
         sample_idxs = range(0, sample_count)
         seed = 8000
         groups = np.array(video_data['group'])
         #sample_idxs = np.random.permutation(sample_idxs)    
         kf = GroupShuffleSplit(n_splits=k,random_state =seed,test_size=0.20)
         for train_index, test_index in kf.split(sample_idxs ,groups=groups): #divide train and test in k different folds accroding to groups
              #divice once train and validation according to groups
              train, val = next(GroupShuffleSplit(random_state = seed,test_size=0.3).split(train_index, groups=groups[train_index]))
              train_idx = train_index[train]
              val_idx = train_index[val]
              training_sequence_generator = generate_training_sequences(batch_size=batch_size,
                                                                   video_data=video_data,
                                                                   training_sample_idxs=train_idx)
              val_sequence_generator = generate_val_sequences(batch_size=batch_size,
                                                                       video_data=video_data,
                                                                       test_sample_idxs=val_idx)
              model.fit_generator(generator=training_sequence_generator,
                             validation_data=val_sequence_generator,
                             steps_per_epoch=math.ceil(len(train_idx)//batch_size),
                             validation_steps=len(val_idx),
                             epochs=1,
                             verbose=2,
                             class_weight=None
                             
                            )
            #shuffle=True,  
            
              exit()   
           

def generate_training_sequences(batch_size, video_data, training_sample_idxs):
    """ Generates training sequences on demand
    """
    print ('generate training sequences') 
    while True:
        # generate sequences for training
        
        training_sample_count = len(training_sample_idxs)
        batches = int(training_sample_count/batch_size)
        remainder_samples = training_sample_count%batch_size
        print ('tr sample:', training_sample_count)
        
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            print (idx)
            if idx == batches - 1:
                batch_idxs = training_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = training_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
            batch_idxs = sorted(batch_idxs)
            #print video_data["X"][0][:200].shape
            #print video_data["X"][0].shape
            #add augmentation somehow
            X = video_data["X"][batch_idxs]#,1000:1016]#/255
            Y = video_data["y"][batch_idxs]
            yield (np.array(X), np.array(Y))

def generate_val_sequences(batch_size, video_data, test_sample_idxs):
    """ Generates validation sequences on demand
    """
    while True:
        
        test_sample_count = len(test_sample_idxs)
        batches = int(test_sample_count/batch_size)
        remainder_samples = test_sample_count%batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            if idx == batches - 1:
                batch_idxs = test_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = test_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
            batch_idxs = sorted(batch_idxs)

            X = video_data["X"][batch_idxs]#,1000:1016]
            Y = video_data["y"][batch_idxs]
            yield (np.array(X), np.array(Y))
 
def buildModel():
    if K.image_data_format() == 'channels_last':
        shape = (400,112,112,3)
    else:
        shape = (3,400,112,112)
    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0,1,1)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    model.add(GlobalMaxPooling3D(name='gmp'))
    model.add(Dense(1, activation='sigmoid'))
   
    
    '''
    nm = Sequential()
    #base_model = C3D(weights='sports1M')
    base_model = C3D(weights=None)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv1').output)
    for layer in model.layers:
        #layer.trainable = False
        nm.add(layer)
 
    nm.add(GlobalMaxPooling3D(name='gmp'))
    #nm.add(Dense(256, activation='relu',name='fc7'))
    model.add(Dense(1, activation='sigmoid'))
    '''
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    videoData="/home/atilocca/deceptionDetection/datasets/PS_datasetChunks.hdf5"
    folds= 10
    batch_size  = 2
    model = buildModel()
    train_model(model,videoData,folds,batch_size)
    exit()
    
    print (model.summary())
    train_model(model)
