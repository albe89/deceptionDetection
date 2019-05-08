import numpy as np
import pandas as pd
import numpy as np
import os
import sys
import getopt
import cv2
import pickle

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
        
def getDicts(name):
    PIK = ''
    nameDS = ''
    X = []
    y = []
    groups = []
    columns =['ID','subject','label','nframe']
    data=[]
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
    countVideo=0
    for i in dir:
        for filename in os.listdir(i):
            print(countVideo)
            print(filename)
            element=[]
            element.append(filename)
            if nameDS =='perez':
               personName = dictPerez[filename][0]
               element.append(personName)
               print personName
               groups.append(personName)
               if 'lie' in filename:
                  element.append(1)
               elif 'truth' in filename:
                  element.append(0)
            elif nameDS == 'PS':
               personName = 'subject_'+str(countVideo).zfill(3)
               element.append(personName)
               if 'm' in filename[:-4]:
                  element.append(1)
               elif 's' in filename[:-4]:
                  element.append(0)
            cap = cv2.VideoCapture(i + filename)  
            vid = []
            #useless just to avoid change code
            while (cap.isOpened()):
                 ret, frame = cap.read()
                 if ret:
                    vid.append(frame)
                 else:
                    break
            element.append(len(vid))
            cap.release()
            cv2.destroyAllWindows()
            countVideo= countVideo+1
            data.append(element)
    df = pd.DataFrame(data, columns = columns)
    df.to_pickle( '/home/atilocca/deceptionDecetion/datasets/'+nameDS+'_DF.pkl')
    print df   

        
        
if __name__ == "__main__":
    #extend with more params if needed
    nameDS = main(sys.argv[1:])
    #for reproducibilty
    seed = 35
    #dim frame chunks
    if nameDS == '':
       print('-i <dataset name (perez/PS)>')
       exit()
    getDicts(nameDS)
    exit()
