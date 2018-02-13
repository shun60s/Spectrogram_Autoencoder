
import os
import random
import numpy as np
import skimage.io
import skimage.draw
from skimage.transform import resize

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  scikit-image (0.13.1)



IN_DIR = 'spectrogram'
OUT_DIR = 'DataSet'

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


train_data = []
train_label = []


# name => (number)
number_dics = {}

with open('labels.txt') as fp:
    for line in fp:
        line = line.rstrip()
        cols = line.split()
        assert len(cols) == 2, ' Not expect input format'
        number = int(cols[0])
        name = cols[1]
        number_dics[name] = (number)


count=0
# random scan 
files=os.listdir(IN_DIR)
random.shuffle(files)
for f in files:
    # get head letter
    assert f[0:1].isdigit(), ' Not expect input file name'
    idx = int( f[0:1] )
    
    for name in number_dics:
        if number_dics[ name ] == idx:
            print (f)
            source = os.path.join(IN_DIR, f)
            image = skimage.img_as_float(skimage.io.imread(source)).astype(np.float32)
            ###train= np.reshape( image, (image.shape[0] * image.shape[1]))
            train= np.reshape( image, (1, image.shape[0] , image.shape[1]))
            label=np.int32(idx)
            train_data.append(train)
            train_label.append(label)
            #print train_data
            #print train_label
            count+=1

print ('count ', count)


# each data and label
np.save(os.path.join(OUT_DIR,'train_data.npy'), train_data)
np.save(os.path.join(OUT_DIR,'train_label.npy'), train_label)

#only data: train and test divided
#threshold = np.int32(train_data.shape[0]/10*9)
#train1 = train_data[0:threshold]
#test1  = train_data[threshold:]
#np.save(os.path.join(OUT_DIR,'train_data_only.npy'), train1)
#np.save(os.path.join(OUT_DIR,'test_data_only.npy'), test1)

