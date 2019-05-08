import numpy as np
import h5py
from keras.models import Model
import sys
sys.path.append('/home/atilocca/c3d')
from c3d import C3D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Input,Flatten,Dropout,GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers import TimeDistributed

#change test_ration with 2 parameter (train_ids[] and test_ids[])
def train_model(model, videoData="/home/atilocca/deceptionDetection/datasets/PS_datasetChunks.hdf5", test_ratio=0.3, batch_size=4):

    with h5py.File(videoData, "r") as video_data:
         sample_count = int(video_data["y"].shape[0])
         print sample_count
         #for now i get test and training like this then i should change according to groups and pass list of ids as parameter
         sample_idxs = range(0, sample_count)
         sample_idxs = np.random.permutation(sample_idxs)
         training_sample_idxs = sample_idxs[0:int((1-test_ratio)*sample_count)]
         test_sample_idxs = sample_idxs[int((1-test_ratio)*sample_count):]
         training_sequence_generator = generate_training_sequences(batch_size=batch_size,
                                                                   video_data=video_data,
                                                                   training_sample_idxs=training_sample_idxs)
         test_sequence_generator = generate_test_sequences(batch_size=batch_size,
                                                                       video_data=video_data,
                                                                       test_sample_idxs=test_sample_idxs)
         '''
         print training_sequence_generator
         #print next(training_sequence_generator).shape
         for item in training_sequence_generator:
             print(item[1].shape)     
         exit()
         '''
         print 'model.fit'
         #test generator used as validation for now, steps_per_epoch = len(train) or len(train)/batchsize?
         model.fit_generator(generator=training_sequence_generator,
                             validation_data=test_sequence_generator,
                             steps_per_epoch=len(training_sample_idxs)//batch_size,
                             validation_steps=len(test_sample_idxs),
                             epochs=1,
                             verbose=2,
                             class_weight=None
                             
                            )
            #shuffle=True,
def generate_training_sequences(batch_size, video_data, training_sample_idxs):
    """ Generates training sequences on demand
    """
    print 'generate training sequences' 
    while True:
        # generate sequences for training
        
        training_sample_count = len(training_sample_idxs)
        batches = int(training_sample_count/batch_size)
        remainder_samples = training_sample_count%batch_size
        print 'tr sample:', training_sample_count
        print 'batches:',batches
        
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in xrange(0, batches):
            print idx
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

def generate_test_sequences(batch_size, video_data, test_sample_idxs):
    """ Generates test sequences on demand
    """
    print 'generate test sequences' 
    while True:
        
        test_sample_count = len(test_sample_idxs)
        batches = int(test_sample_count/batch_size)
        remainder_samples = test_sample_count%batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in xrange(0, batches):
            if idx == batches - 1:
                batch_idxs = test_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = test_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
            batch_idxs = sorted(batch_idxs)

            X = video_data["X"][batch_idxs]#,1000:1016]
            Y = video_data["y"][batch_idxs]
            yield (np.array(X), np.array(Y))
            
def buildModel():
    nm = Sequential()
    #base_model = C3D(weights='sports1M')
    base_model = C3D(weights=None)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)
    for layer in model.layers:
        #layer.trainable = False
        nm.add(layer)
    '''
    nm.add(Dropout(0.5))
    nm.add(TimeDistributed(Dense(1024, activation='relu', name='fc7')))
    nm.add(TimeDistributed(Dense(512, activation='relu', name='fc8')))
    nm.add(GlobalMaxPooling1D()) 
    '''
    nm.add(Dense(1, activation='sigmoid'))
    nm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return nm

if __name__ == "__main__":
    model = buildModel()
    print model.summary()
    train_model(model)
