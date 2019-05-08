import numpy as np
import pandas as pd
import numpy as np
import os
import sys
import cv2
import pickle
from keras.preprocessing import sequence
import sys, getopt
import matplotlib
import matplotlib.pyplot as plt
import gc
from sys import getsizeof
import h5py


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
    nameDS = name
    #X = []
    y = []
    groups = []
    ids = []
    df = pd.read_pickle('/home/atilocca/deceptionDetection/datasets/'+name+'_DF.pkl') #df with id-subject-label-nframe
    mFrame = 0 #max number of frames in all dataset
    if name == 'perez':
       dirLie = '/home/atilocca/Clips/Deceptive/'
       dirTruth = '/home/atilocca/Clips/Truthful/'
       dir = [dirTruth, dirLie]
       '''
       #create dict id video -> subject
       perezIdPeople = pd.read_csv('/home/atilocca/perezIdPeople.csv', sep=";", header=None)
       columnPerezID = ['ID', 'Role_trial_name']
       perezIdPeople.columns = columnPerezID
       dictPerez = {}
       for i in perezIdPeople['ID'].unique():
           dictPerez[i] = [perezIdPeople['Role_trial_name'][j] for j in perezIdPeople[perezIdPeople['ID'] == i].index]
       '''


    elif name =='PS':
       dir = ['/home/atilocca/DataSetPS_Aligned_cut_mp4/']
       nameDS = name
    else:
       print('dataset name is not valid')
       exit()
   
   
    videoCount = 0
    sampleNumber = df['ID'].count()
    #mFrame = df['nframe'].max()
    mframe = 400
    hdf5_store = h5py.File("/home/atilocca/deceptionDetection/datasets/"+nameDS+"_dataset.hdf5", "a")
    #X = hdf5_store.create_dataset("X", (sampleNumber,mFrame,width,height,channel), compression="gzip")
   
    for i in dir:
        for filename in os.listdir(i):           
            print(filename)
            cap = cv2.VideoCapture(i + filename) 
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid =  np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

            if nameDS =='perez':
               if 'lie' in filename:
                  y.append(1)
               elif 'truth' in filename:
                  y.append(0)
            elif nameDS == 'PS':
               if 'm' in filename[:-4]:
                  y.append(1)
               elif 's' in filename[:-4]:
                  y.append(0)

            subject = df.loc[df['ID'] == filename]['subject'].item()
            groups.append(subject)
            ids.append(filename)
            fc = 0 #framecount
            while (cap.isOpened()):
                 ret, frame = cap.read()
                 if ret:
                    vid[fc]=frame
                    fc = fc+1
                 else:
                    break

            cap.release()
            cv2.destroyAllWindows()
            gc.collect()

            reshape_frames = reshapeVid(vid)   
            print reshape_frames.shape

            #temp = np.expand_dims(reshape_frames, axis=0) 
            #temp = sequence.pad_sequences(temp, maxlen=mframe)
            #X[videoCount,]= temp[0,]
            videoCount = videoCount+1
            #print X.shape



    y = np.array(y)    
    group = np.asarray(groups)
    ids = np.asarray(ids)
    yHD = hdf5_store.create_dataset("y",data = y , compression="gzip")
    groupHD =  hdf5_store.create_dataset("group", data = group, compression="gzip")
    idsHD = hdf5_store.create_dataset("id", data = ids, compression="gzip")
    #print X.shape
    #print X.shape
    print  np.array(y)  
    print yHD.shape
    print groupHD.shape
    print idsHD.shape 
    print yHD
    print groupHD
    hdf5_store.close()
                             
def addPadding(X):
    maxFrame = 0
    for i in X:
       if i.shape[0]>maxFrame:
          maxFrame = i.shape[0]
    #exit()
    X = sequence.pad_sequences(X, maxlen=maxFrame)
    return X, maxFrame

def reshapeVid(frames):
    dim = ( width,height)
    reshape_frames = np.zeros((frames.shape[0], height, width, frames.shape[3]))
    for i, img in enumerate(frames):
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC )
        #resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        reshape_frames[i,:,:,:] = resized
    #reshape_frames = reshape_frames/255
    return reshape_frames

if __name__ == "__main__":
    #extend with more params if needed
    nameDS = main(sys.argv[1:])
    #for reproducibilty
    seed = 35
    #dim frame chunks
    frameDepth = 16
    width = 112
    height = 112
    channel = 3
    if nameDS == '':
       print('-i <dataset name (perez/PS)>')
       exit()
    loadData(nameDS)
    exit()
