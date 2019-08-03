# where you installed caffe2. Probably '~/pytorch' or '~/src/pytorch'.
#! /usr/bin/env python
#download the pre-trained model by using 'python3 -m caffe2.python.models.download -i bvlc_alexnet'
# the pre-trained model in github
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
CAFFE2_ROOT = "./ALLNetWork/"
# assumes being a subdirectory of caffe2
CAFFE_MODELS = "./Documents/ALLNetWork/"
# if you have a mean file, place it in the same dir as the model

#%matplotlib inline
from caffe2.proto import caffe2_pb2
from caffe2.python import core,model_helper,brew,optimizer
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
import sys
from caffe2.python import core, workspace
print("Required modules imported.")
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)

gpu_no=0
IMAGE_LOCATION =  "./squeezenet/flower-631765_1280.jpg"
#form the testing model with original structure

# What model are we using? You should have already converted or downloaded one.
# format below is the model's:
# folder, INIT_NET, predict_net, mean, input image size
# you can switch the comments on MODEL to try out different model conversions
MODEL = 'bvlc_alexnet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227

# codes - these help decypher the output and source from a list from AlexNet's object codes to provide an result like "tabby cat" or "lemon" depending on what's in the picture you submit to the neural network.
# The list of output codes for the AlexNet models (also squeezenet)
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"
print ("Config set!")
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print(("Model's input shape is %dx%d") % (input_height, input_width))
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    pyplot.figure()
    pyplot.imshow(imgScaled)
    pyplot.axis('on')
    pyplot.title('Rescaled image')
    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled
print ("Functions set.")

# set paths and variables from model choice and prep image
CAFFE2_ROOT = os.path.expanduser(CAFFE2_ROOT)
CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
mean=128
'''MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])
if not os.path.exists(MEAN_FILE):
    mean = 128
else:
    mean = np.load(MEAN_FILE,allow_pickle=True).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]
    print ("mean was set to: ", mean)'''

# some models were trained with different image sizes, this helps you calibrate your image
INPUT_IMAGE_SIZE = MODEL[4]

# make sure all of the files are around...
if not os.path.exists(CAFFE2_ROOT):
    print("Houston, you may have a problem.")
#the location of pre-trained model
#INIT_NET='/home/pl/Documents/ALLNetWork/bvlc_alexnet/init_net.pb'
#PREDICT_NET='/home/pl/Documents/ALLNetWork/bvlc_alexnet/predict_net.pb'
INIT_NET = os.path.join(CAFFE_MODELS,MODEL[0], MODEL[1])
print ('INIT_NET = ', INIT_NET)
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0],MODEL[2])
print ('PREDICT_NET = ', PREDICT_NET)
if not os.path.exists(INIT_NET):
    print(INIT_NET + " not found!")
else:
    print ("Found ", INIT_NET, "...Now looking for", PREDICT_NET)
    if not os.path.exists(PREDICT_NET):
        print ("Caffe model file, " + PREDICT_NET + " was not found!")
    else:
        print ("All needed files found! Loading the model in the next block.")

# load and transform image
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print ("After crop: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Cropped')


# switch to CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)
pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))
# switch to BGR
img = img[(2, 1, 0), :, :]
mean=128
# remove mean for better results
img = img * 255 - mean

# add batch size
img = img[np.newaxis, :, :, :].astype(np.float32)
print ("NCHW: ", img.shape)
print('reset workspace & loading pre-trained model')
workspace.ResetWorkspace()
# initialize the alexnet
def load_alexnet(INIT_NET, PREDICT_NET, device_opts):
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())

    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.FeedBlob('data',img,device_opts)
        workspace.CreateNet(net_def.SerializeToString(),overwrite=True)

#initialize the squeezent
def load_squeezenet(INIT_NET, PREDICT_NET):
    init_net=caffe2_pb2.NetDef()
    with open(INIT_NET,'rb') as f:
        init_net = f.read()
        #workspace.CreateNet(init_net)
    predict_net=caffe2_pb2.NetDef()
    with open(PREDICT_NET,'rb') as f:
        predict_net = f.read()
        #workspace.CreateNet(predict_net)
    return init_net,predict_net
#initialize the googlenet
 
#init_net,predict_net=load_net_1(INIT_NET, PREDICT_NET)

load_alexnet(INIT_NET, PREDICT_NET, device_opts)
print('=======here======')
workspace.FeedBlob('data',img,device_opts)
results=workspace.RunNet('bvlc_alexnet', 1,allow_fail=True)
#results=workspace.RunNet('bvlc_alexnet', 1,allow_fail=True)
print('results',results)
print('Input: ones')
print('Output:' + str(workspace.FetchBlob("prob")))
print('Output class:' + str(np.argmax(workspace.FetchBlob("prob"))))
#print('workspace.RunNet',workspace.RunNet('test_net'))
'''p = workspace.Predictor(init_net, predict_net)

# run the net and return prediction
results = p.run({'data': img})

# turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)
print ("results shape: ", results.shape)
#workspace.FeedBlob('data',img)
results = np.asarray(results)
print ("results shape: ", results.shape)

results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i

print (index, " :: ", highest)

# lookup the code and return the result
# top 3 results
# sorted(arr, key=lambda x: x[1], reverse=True)[:3]

# now we can grab the code list
response = urlopen(codes)

# and lookup our result from the list
for line in str(response):
    code, result = line.partition(":")[::2]
    if (code.strip() == str(index)):
        print(result.strip()[1:-2])'''

boolHasBlob=workspace.HasBlob('data')
print('if blob is existed',boolHasBlob)
Blob=workspace.Blobs()
#print('parameter is stored in blob :',Blob)
with open('dataSizeSingle1AlexNet.txt','w') as f:
    #start_point =Blob.index("conv1")
    #new_Blob = Blob[start_point:]
    #print(start_point,new_Blob)
    for blob in Blob:
        currentLayer=workspace.FetchBlob(blob)
        #print('blob is ',blob)
        if type(currentLayer) is not bytes:
        #print('currentLayer',currentLayer)
        #f.write('the type of currentLayer %s '%type(currentLayer))
             f.write('The %s layer : %s\t'%(str(blob),str(currentLayer.shape)))
             f.write('get the size of parameter(MB) by getsizeof:%s \t'%str(sys.getsizeof(currentLayer)/(1024*1024)))
             #f.write('size of parameter(MB):%s \t'%str(currentLayer.itemsize*currentLayer.size/(1024*1024)))
             f.write('\n') 
'''with open('dataSizeSingle1AlexNet.txt','w') as f:
    start_point =Blob.index("conv1")
    new_Blob = Blob[start_point:]
    #print(start_point,new_Blob)
    for blob in new_Blob:
        currentLayer=workspace.FetchBlob(blob)
        print('currentLayer',currentLayer.shape)
        print('type of currentLayer',type(currentLayer))
       
        f.write('The %s layer : %s\t'%(str(blob),str(currentLayer.shape)))
        f.write('size of parameter(MB):%s \t'%str(currentLayer.itemsize*currentLayer.size/(1024*1024)))
        f.write('\n')  '''

#workspace.RunNet('testing_net', 1)
'''print('workspace.RunNet',workspace.RunNet('testing_net'))
Blob=workspace.Blobs()

print('Input: ones')
print('Output:' + str(workspace.FetchBlob("softmax")))
print('Output class:' + str(np.argmax(workspace.FetchBlob("softmax"))))'''






