import numpy as np
import pandas as pd
import skvideo.io
import numpy as np
from c3d import C3D
from keras.models import Model
from sports1M_utils import preprocess_input, decode_predictions
import os
import sys
import cv2
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Input,Flatten,Dropout,GlobalMaxPooling1D,GlobalMaxPooling2D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, cross_val_predict,GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys, getopt
import matplotlib
import matplotlib.pyplot as plt
from keras.applications import VGG16
from scipy.misc import imresize
import gc
matplotlib.use('tkagg')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:")
    except getopt.GetoptError:
        print(' -i <dataset name (perez/ps)>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-i <dataset name (perez/ps)>')
            sys.exit()
        elif opt == "-i":
            return arg
def loadData(name):
    PIK = ''
    nameDS = ''
    X = []
    y = []
    groups = []
    if name == 'perez':
       dirLie = '/home/atilocca/Clips/Deceptive/'
       dirTruth = '/home/atilocca/Clips/Truthful/'
       dir = [dirTruth, dirLie]
       perezIdPeople = pd.read_csv('/home/atilocca/perezIdPeople.csv', sep=";", header=None)
       columnPerezID = ['ID', 'Role_trial_name']
       perezIdPeople.columns = columnPerezID
       dictPerez = {}
       for i in perezIdPeople['ID'].unique():
           dictPerez[i] = [perezIdPeople['Role_trial_name'][j] for j in perezIdPeople[perezIdPeople['ID'] == i].index]
       nameDS = name
    elif name =='PS':
       dir = ['/home/atilocca/DataSetPS_Aligned_cut_mp4/']
       nameDS = name
    else:
       print('dataset name is not valid')
       exit()
    nameDataFile = nameDS+'DivisionVideoIn'+str(frameDepth)+'Frames'
    if os.path.exists('./'+nameDataFile):  
       with open(nameDataFile, "rb") as input_file:
            data = pickle.load(input_file)
       #vedere dopo salvato come prendere i dati
       X =  data[0]
       y =  data[1]
       X = np.asarray(X)
       y = np.asarray(y)
       perm = np.random.permutation(y.shape[0])
       X = X[perm]
       y = y[perm]
       groups = []
       if nameDS == 'perez':
          groups =  data[2]
          groups = np.asarray(groups)
          groups = groups[perm]
   
       print ('dataset loaded')
    else:
       for i in dir:
           c = 0
           for filename in os.listdir(i):
               if c == 2:
                  break
               #print c
               #c = c+1
               print(filename)
               if nameDS =='perez':
                  groups.append(dictPerez[filename])
                  if 'lie' in filename:
                     y.append(1)
                  elif 'truth' in filename:
                     y.append(0)
               elif nameDS == 'PS':
                  if 'm' in filename[:-4]:
                     y.append(1)
                  elif 's' in filename[:-4]:
                     y.append(0)
               cap = cv2.VideoCapture(i + filename)  
               vid = []
               while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                       vid.append(frame)
                    else:
                       break
               cap.release()
               cv2.destroyAllWindows()
               gc.collect()
               #vid = np.asarray(vid)
               #prendo solo i primi 30 frame per testare il modello evitando memory error
               vid = vid[:300]
            
               reshape_frames = reshapeVid(vid)
               reshape_frames = np.asarray(reshape_frames)
               print reshape_frames.shape

               
               X.append(np.asarray(reshape_frames))

       X = np.asarray(X)
       y = np.asarray(y)
       groups = np.asarray(groups)
       print X.shape
       X, maxFrame = addPadding(X)
       print X.shape
       print X[0].shape
       print y.shape

       shape = (X.shape)
       model = buildModel(shape)
       history = model.fit(X, y, epochs=1000, verbose=1 )
       exit()
       data = [X, y, groups]
       with open(nameDataFile, "wb") as f:
            pickle.dump(data, f)
       X, maxFrame = addPadding(X)
       print ('added padding, frame numbers for video: ',maxFrame)
       splitTrainTestValidation(X,y,groups,nameDS)
       return X, y, groups

def addPadding(X):
    maxFrame = 0
    for i in X:
       if i.shape[0]>maxFrame:
          maxFrame = i.shape[0]
    print maxFrame
    print X.shape
    #exit()
    X = sequence.pad_sequences(X, maxlen=maxFrame)
    return X, maxFrame

def reshapeVid(frames):
    width = 60 #112
    height = 60 #112
    channel = 3
    dim = ( width,height)
    reshape_frames = np.zeros((len(frames), height, width, channel))
    #reshape_frames = np.zeros((frames.shape[0], 128, 171, frames.shape[3]))
    for i, img in enumerate(frames):
        #img = imresize(img, (128,171), 'bicubic')
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        reshape_frames[i,:,:,:] = resized
    #reshape_frames = reshape_frames[:,8:120,30:142,:]
    
    
    # resize image
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    reshape_frames = reshape_frames/255
    return reshape_frames
def splitTrainTestValidation(X,y,groups,nameDS):
    if nameDS == 'PS':
      kf = StratifiedKFold(n_splits=10,shuffle=True, random_state = seed)
      groups = np.zeros(y.shape)
    elif nameDS == 'perez':
      kf = GroupKFold(n_splits=10) 
    count = 0
    meanAccuracy = []
    for train_index, test_index in kf.split(X, y,groups=groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        count = count +1
        print ('---------------')
        print('train shape:')
        print X_train.shape
        print ('test shape:')
        print X_test.shape
        print ('---------------')
    
        '''
        #train validation
        k= 10 #fold
        numValidationSamples = len(X_train) // k
        totalAccuracy = []
        totalAccuracyVal = []
        totalLoss = []
        totalLossVal = []
        for fold in range(k):
            validationData = X_train[numValidationSamples * fold:numValidationSamples*(fold+1)]
            validationTarget =  y_train[numValidationSamples * fold:numValidationSamples*(fold+1)]
            trainingData = np.concatenate([X_train[:numValidationSamples * fold], X_train[numValidationSamples * (fold +1):]],axis=0)
            trainingTarget = np.concatenate([y_train[:numValidationSamples * fold], y_train[numValidationSamples * (fold +1):]],axis=0)
            print 'validation shape:', validationData.shape
            print 'training shape:',  trainingData.shape
        '''
        shape = (X_train.shape)      
        model = buildModel(shape)
        checkpointer = ModelCheckpoint(filepath='/home/atilocca/c3d/weightsC2D'+nameDS+str(count)+'.hdf5', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = model.fit(X_train, y_train, epochs=1000,validation_split=0.30,  verbose=1, callbacks=[es,checkpointer],batch_size=20 ) #callbacks=[es,checpointer]
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
      # totalLoss.append(loss)
      # totalLossVal.append(val_loss)
      # totalAccuracy.append(acc)
      # totalAccuracyVal.append(val_acc)
      # loss = np.mean(totalLoss, axis=0)
      # val_loss = np.mean(totalLossVal, axis=0)
      # acc = np.mean(totalAccuracy, axis=0)
      # val_acc = np.mean(totalAccuracyVal, axis=0)
        '''
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
        plt.clf()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        '''
        print ('result on test set:')
        model = buildModel((X_train.shape[1],X_train.shape[2]))
        model.load_weights('/home/atilocca/c3d/weightsC2D'+nameDS+str(count)+'.hdf5')   
 
        scores = model.evaluate(X_test,y_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        meanAccuracy.append(scores[1])
    print 'accuracy: '
    print meanAccuracy
    print 'mean:'
    print np.mean(meanAccuracy)

def buildModel(shape):
    frames = shape[1]
    h = shape[2]
    w = shape[3]
    channels= shape[4]
    
    inputVideo = Input(shape=(frames,h,w,channels))
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(h, w, channels))
    outputVideo = GlobalMaxPooling2D()(conv_base.output)
    cnn = Model(input=conv_base.input, output=outputVideo)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(inputVideo)
    encoded_sequence = LSTM(30)(encoded_frames)
    hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
    outputs = Dense(1, activation="sigmoid")(hidden_layer)
    model = Model([inputVideo], outputs)
    print model.summary()
    #exit()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    
    return model
    
if __name__ == "__main__":
    #extend with more params if needed
    nameDS = main(sys.argv[1:])
    #for reproducibilty
    seed = 35
    #dim frame chunks
    frameDepth = 16
    if nameDS == '':
       print('-i <dataset name (perez/PS)>')
       exit()
    loadData(nameDS)
    exit()
