import numpy as np
import h5py
from keras.models import Model
import sys
#from c3d import C3D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Input,Flatten,Dropout,GlobalMaxPooling1D,GlobalMaxPooling3D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import StratifiedKFold,GroupKFold,GroupShuffleSplit
import math
import keras
import pandas as pd

def train_model(model,dataFrame,k,batch_size):
    #with h5py.File(videoData, "r") as video_data:
         sample_count =  len(dataframe.index)
         #sample_count = 50
         
         sample_idxs = np.random.RandomState(seed=seed).permutation(sample_count)
         #groups = np.array(video_data['group'])
         groups =  dataframe['subject'].values        
         groups = groups[sample_idxs]
         
         kf = GroupShuffleSplit(n_splits=k,random_state =seed,test_size=0.20)
         foldCount = 0
         for train_index, test_index in kf.split(sample_idxs ,groups=groups): #divide train and test in k different folds accroding to groups       
             
              trainIDX = sample_idxs[train_index]
              testIDX = sample_idxs[test_index]

              #divide once train and validation according to groups
              train, val = next(GroupShuffleSplit(random_state = seed,test_size=0.2).split(trainIDX, groups=groups[train_index]))
              train_idx = trainIDX[train]
              val_idx = trainIDX[val]
              trainID = dataframe.iloc[(train_idx)]['ID'].values.tolist()
              valID = dataframe.iloc[(val_idx)]['ID'].values.tolist()
              testID = dataframe.iloc[(testIDX)]['ID'].values.tolist()
              '''
              print ('train')
              print (len(trainID))
              print (trainID)
              print ('valID')
              print (len(valID))
              print (valID)
              print ('testID')
              print (len(testID))
              print (testID)
              exit()
              '''
              '''
              training_sequence_generator = generate_training_sequences(batch_size=batch_size,
                                                                   video_data=video_data,
                                                                   training_sample_idxs=train_idx)
              val_sequence_generator = generate_val_sequences(batch_size=batch_size,
                                                                       video_data=video_data,
                                                                       test_sample_idxs=val_idx)
              test_sequence_generator = generate_test_sequences(batch_size=batch_size,
                                                                       video_data=video_data,
                                                                       test_sample_idxs=test_index)
             
              '''
              print('#train samples: ', len(trainID))
              print('#val samples: ', len(valID))
              print('#test samples: ', len(testID))
              #trainID = trainID[:100]
              #valID = valID[:16]
              #testID= testID[:20]
              training_sequence_generator = DataGenerator(list_IDs=trainID, batch_size=batch_size)
              val_sequence_generator = DataGenerator(list_IDs=valID, batch_size=batch_size)
              test_sequence_generator = DataGenerator(list_IDs=testID, batch_size=batch_size)
              
              
              es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
              checkpointer = ModelCheckpoint(filepath='/home/atilocca/deceptionDetection/weightsC3Dbaseline100epochs20PLR0p0001Perez_'+str(foldCount).zfill(3)+'.hdf5', verbose=1, save_best_only=True)
              history = model.fit_generator(generator=training_sequence_generator,
                             validation_data=val_sequence_generator,
                             steps_per_epoch=math.ceil(len(trainID)//batch_size),
                             validation_steps=len(valID),
                             epochs=100,
                             verbose=2,
                             class_weight=None,
                             callbacks=[es,checkpointer],
                             use_multiprocessing=True,
                             workers=6
                            )
            #shuffle=True,  
            
              model.load_weights('/home/atilocca/deceptionDetection/weightsC3Dbaseline100epochs20PLR0p0001Perez_'+str(foldCount).zfill(3)+'.hdf5') 
              scores = model.evaluate_generator(test_sequence_generator, steps = len(test_index))
              acc = history.history['acc']
              val_acc = history.history['val_acc']
              loss = history.history['loss']
              val_loss = history.history['val_loss']
              print ('train acc:')
              print (acc)
              print ('val_acc:')
              print (val_acc)
              print ('loss:')
              print (loss)
              print ('val_loss:')
              print (val_loss)
              print ('-----------------SCORES--------------')
              print ('test loss:')
              print (scores[0])
              print ('test acc:')
              print (scores[1])
              print ('-------------------------------------')
              foldCount +=1
'''
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
            x = np.array(X)/255.0
            yield (x.astype(np.float16), np.array(Y))
            

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
            x = np.array(X)/255.0
            yield (x.astype(np.float16), np.array(Y))
                                
def generate_test_sequences(batch_size, video_data, test_sample_idxs):
    """ Generates test sequences on demand
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
            x = np.array(X)/255.0
            yield (x.astype(np.float16), np.array(Y))
'''
 
def buildModel():
    if K.image_data_format() == 'channels_last':
        shape = (150,112,112,3)
    else:
        shape = (3,150,112,112)
    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', dilation_rate=2,padding='valid', name='conv1', input_shape=shape))
    #model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='pool1'))
    
    #model.add(Conv3D(128, 3, activation='relu', padding='valid', name='conv2'))
    #model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    #model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    #model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    #model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    #model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    #model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    #model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
    
    #model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    #model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    #model.add(ZeroPadding3D(padding=(0,1,1)))
    #model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(112,112,3), batch_size=2,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dim = dim

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        #return list_IDs_temp
        X, y = self.__data_generation(list_IDs_temp)



        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialization
        X = np.empty((self.batch_size, 150,self.dim[0],self.dim[1],self.dim[2]))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            #X[i,] = self.videoData["X"][ID,0:150]/255.0
            #dir file
            location = dataframe.loc[dataframe['ID']==ID]['filelocation'].values[0]
            X[i,] = np.load(location+ '.npz')['x']/255.0
            
            # Store class
            y[i] = dataframe.loc[dataframe['ID']==ID]['label'].values[0]

        return X, y

if __name__ == "__main__":
    videoData="/home/atilocca/deceptionDetection/datasets/PS_datasetChunks150Frames.hdf5"
    dataframe = pd.read_pickle('/home/atilocca/deceptionDetection/datasets/PS_DF_singleFile.pkl') 
    #dataframe = pd.read_pickle('/home/atilocca/deceptionDetection/datasets/perez_DF_singleFile.pkl') 
    #seed = 1000
    seed = 2222
    folds= 1
    batch_size  = 4
    model = buildModel()
    print (model.summary())
    train_model(model,dataframe,folds,batch_size)
    
    
  
