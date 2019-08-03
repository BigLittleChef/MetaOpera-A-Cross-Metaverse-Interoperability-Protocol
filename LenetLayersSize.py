#!/usr/bin/env Python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
import numpy as np
import pickle as cPickle
import csv
#import pandas as pd
import skimage.io
import skimage.transform
import sys
from matplotlib import pyplot as plt
from caffe2.python import core,workspace,model_helper,brew,optimizer
from caffe2.python import net_printer
#import mobile_exporter
from caffe2.proto import caffe2_pb2
import utils
 
#CUDA means to use GPU
device_option = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
INIT_NET = './NeuralNetwork/init_net.pb'
PREDICT_NET = './NeuralNetwork/predict_net.pb'
gpu_no=0
def AddLeNetModel(model):
    with core.DeviceScope(device_option):
        conv1 = brew.conv(model,'data', 'conv1', 1, 20, 5,pad=1)

        pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
	
        conv2 = brew.conv(model, pool1, 'conv2', 20, 50, 5)
        pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
        fc3 = brew.fc(model, pool2, 'fc3', 50 * 4 * 4, 500)
        fc3 = brew.relu(model, fc3, fc3)
        pred = brew.fc(model, fc3, 'pred', 500, 10)
        softmax = brew.softmax(model, pred, 'softmax')
    return softmax
testing_model = model_helper.ModelHelper(name="testing_net", init_params=False)
testing_model.net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
testing_model.param_init_net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
#test_soft=AddLeNetModel(testing_model)
#workspace.RunNetOnce(testing_model.param_init_net)

def load_net(INIT_NET, PREDICT_NET, device_opts):
    init_def = caffe2_pb2.NetDef()
   
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())

    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

    with open('dataSize.txt','w+') as f_record:

        for blob in testing_model.GetParams():
            memorySize=workspace.FetchBlob(blob)
            #print('blob',blob)
            for i in range(1,2):
                f_record.write('The %s layer : %s\t'%(str(blob),str(memorySize.shape)))
                f_record.write('size of parameter(MB):%s \t'%str(memorySize.itemsize*memorySize.size/(1024*1024)))
                f_record.write('\n') 
#print(INIT_NET)
print('reset workspace & loading pre-trained model')
workspace.ResetWorkspace()


print("Load the trained model")
# generate a fake image with NCHW in float32

data = np.random.rand(1,1, 28, 28).astype(np.float32)

sizeOfData=sys.getsizeof(data)
#load data
IMAGE_LOCATION = "./images/num0.jpg"

INPUT_IMAGE_SIZE =28


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx: startx+cropx]
def rescale(img, input_height, input_width):
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        res = int(aspect*input_height)
        img_scaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        res = int(input_width/aspect)
        img_scaled = skimage.transform.resize(img, (res, input_height))
    if(aspect==1):
        img_scaled = skimage.transform.resize(img, (input_widht, input_height))
    return img_scaled
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print("Original Image Shape: {}".format(img.shape))
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image shape after rescaling: {}".format(img.shape))
plt.figure()
plt.imshow(img)
plt.title("after scaliing")
plt.savefig("./images/scaled.jpg", format='jpg')
#plt.show()

img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image shape after cropping: {}".format(img.shape))
plt.figure()
plt.imshow(img)
plt.title("croped ")
plt.savefig("./images/crop_center.jpg", format="jpg")
plt.show()

# switch to CHW(HWC->CHW)
img = img.swapaxes(1, 2).swapaxes(0, 1)
print("CHW Image Shape: {}".format(img.shape))

plt.figure()
for i in range(1):
    plt.subplot(1,1, i+1)
    plt.imshow(img[i])
    plt.axis("off")
    plt.title("RGB channels {}".format(i+1))
plt.savefig("./images/channels.jpg", format="jpg")
print(type(img))
# switch to BGR(RGB->BGR)
img = img[(2, 1, 0), :, :]
img=img[1,:,:]
mean=14
img = img * 15 - mean
img = img.reshape(1,1,28,28).astype(np.float32)
#[np.newaxis, :, :]
print("NCHW image: {}".format(img.shape))
print(" %s size is %s (MB)"%(type(sizeOfData),sizeOfData/(1024*1024)))
# feed the blob and run the net
# load the pre-trained deploy_model
load_net(INIT_NET, PREDICT_NET, device_opts=device_opts)

workspace.FeedBlob('data',img,device_opts)

#workspace.RunNet('testing_net', 1)
print('workspace.RunNet',workspace.RunNet('testing_net'))
Blob=workspace.Blobs()
with open('dataSizeSingle.txt','w') as f:

    for blob in Blob:
        currentLayer=workspace.FetchBlob(blob)
        f.write('The %s layer : %s\t'%(str(blob),str(currentLayer.shape)))
        f.write('size of parameter(MB):%s \t'%str(currentLayer.itemsize*currentLayer.size/(1024*1024)))
        f.write('\n') 
print('Input: ones')
print('Output:' + str(workspace.FetchBlob("softmax")))
print('Output class:' + str(np.argmax(workspace.FetchBlob("softmax"))))

#in batching the size is 8
#these picture from website, you can download by yourself, and then put them in your path
images = ["./images/num1-1.jpg","./images/num1-2.jpg", "./images/num1-3.jpg",
         "./images/num1-4.jpg", "./images/num1-5.jpg", "./images/num1-6.jpg",
         "./images/num1-7.jpg","./images/num1-8.jpg"]
NCHW_batch = np.zeros((len(images), 1, 28, 28))
print("Batch Shape: {}".format(NCHW_batch.shape))

for i, curr_img in enumerate(images):
    img = skimage.img_as_float(skimage.io.imread(curr_img)).astype(np.float32)
    img = rescale(img, 28, 28)
    img = crop_center(img, 28, 28)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = img[(2, 1, 0), :, :]
    img=img[1,:,:]
    img = img * 15 - mean
    img = img.reshape(1,1,28,28)
    NCHW_batch[i] = img
print("NCHW image: {}".format(NCHW_batch.shape))
NCHW_batch=NCHW_batch.astype(np.float32)
#workspace.RunNet('testing_net', 1)

workspace.FeedBlob('data',NCHW_batch,device_opts)
results=workspace.RunNet('testing_net')
print('results',results)
#=====================
#getBlobSizes=workspace.GetBlobSizeBytes
#getBlobNum=workspace.GetBlobNUMANode

Blobs=workspace.Blobs()
with open('dataSizeBatchingLenet.txt','w') as f:

    for blob in Blobs:
        currentLayer=workspace.FetchBlob(blob)
        f.write('The %s layer : %s\t'%(str(blob),str(currentLayer.shape)))
        f.write('get the size of parameter(MB) by getsizeof:%s \t'%str(sys.getsizeof(currentLayer)/(1024*1024)))
        f.write('\n') 
#print('getBlobsNumNode',getBlobNum)

for i in range(0,8):
    print("Results for: {}".format(images[i]))
    #workspace.RunNet('testing_net', 8)
    print('Input: batching form')
    Result_Matrix=workspace.FetchBlob("softmax")
    print('Output:' + str(Result_Matrix[i]))
    print('Output class:' + str(np.argmax(Result_Matrix[i],axis=None)))



