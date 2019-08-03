import numpy as np
import pickle as cPickle
import csv
#import pandas as pd
from matplotlib import pyplot as plt
from caffe2.python import core,workspace,model_helper,brew,optimizer
import mobile_exporter
from caffe2.proto import caffe2_pb2
import utils 
import sys
#load the train and test data
with open('./Downloads/digit-recognizer/train.csv') as f_train:
    raw_train = np.loadtxt(f_train,delimiter=',',skiprows=1)
    print('raw_train',raw_train.shape)
with open('./Downloads/digit-recognizer/test.csv') as f_test:
    test = np.loadtxt(f_test,delimiter=',',skiprows=1)
    print('test shape',test.shape)
train,val = np.split(raw_train,[int(0.8*raw_train.shape[0])])
print(train.shape,val.shape)
device_option = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)
#define the CNN model
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

def AddAccuracy(model, softmax):
    accuracy = brew.accuracy(model, [softmax, 'label'], "accuracy")
    return accuracy
def AddTrainingOperators(model, softmax):
    # Loss Calculation
    xent = model.LabelCrossEntropy([softmax, 'label'])
    loss = model.AveragedLoss(xent, "loss")
    # Calculating Accuracy
    AddAccuracy(model, softmax)
    # Add loss to gradient for backpropogation
    model.AddGradientOperators([loss])
    # Initializing the SGD the solver
    opt = optimizer.build_sgd(model, base_learning_rate=0.1, policy="step", stepsize=1, gamma=0.999)
Batch_Size = 128
workspace.ResetWorkspace()
training_model = model_helper.ModelHelper(name="training_net")
gpu_no=0
training_model.net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
training_model.param_init_net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
soft=AddLeNetModel(training_model)
AddTrainingOperators(training_model, soft)
workspace.RunNetOnce(training_model.param_init_net)
workspace.CreateNet(training_model.net,overwrite=True,input_blobs=['data','label'])
Snapshot_location='./Downloads/digit-recognizer/'
def save_snapshot(model,iter_no):
    d={}
    for blob in model.GetParams():
        d[blob]=workspace.FetchBlob(blob)
    cPickle.dump(d,open(Snapshot_location+str(iter_no),'wb'))
val_model = model_helper.ModelHelper(name="validation_net", init_params=False)
val_model.net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
val_model.param_init_net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
val_soft=AddLeNetModel(val_model)
AddAccuracy(val_model,val_soft)

workspace.RunNetOnce(val_model.param_init_net)
workspace.CreateNet(val_model.net,overwrite=True,input_blobs=['data','label'])

#
testing_model = model_helper.ModelHelper(name="testing_net", init_params=False)
testing_model.net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
testing_model.param_init_net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
test_soft=AddLeNetModel(testing_model)

workspace.RunNetOnce(testing_model.param_init_net)
workspace.CreateNet(testing_model.net,overwrite=True,input_blobs=['data'])

def check_val():
    accuracy = []
    start=0
    while start<val.shape[0]:
        l = val[start:start+Batch_Size,0].astype(np.int32)
        batch = val[start:start+Batch_Size,1:].reshape(l.shape[0],28,28)
        batch = batch[:,np.newaxis,...].astype(np.float32)
        batch = batch*float(1./256)
        workspace.FeedBlob("data", batch, device_option)
        workspace.FeedBlob("label", l, device_option)
        workspace.RunNet(val_model.net, num_iter=1)
        accuracy.append(workspace.FetchBlob('accuracy'))
        start+=l.shape[0]
    return np.mean(accuracy)

total_iterations = 501
Snapshot_interval=100
total_iterations = total_iterations * 64
#print(workspace.Blobs())
accuracy = []
val_accuracy = []
loss = []
lr = []
start=0
#start to train the
while start<total_iterations:
    l = train[start:start+Batch_Size,0].astype(np.int32) # labels for a given batch
    d=train[start:start+Batch_Size,1:].reshape(l.shape[0],28,28) # pixel values for each sample in the batch
    d=d[:,np.newaxis,...].astype(np.float32)
    d=d*float(1./256) # Scaling the pixel values for faster computation
    workspace.FeedBlob("data", d, device_option)
    workspace.FeedBlob("label", l, device_option)
    workspace.RunNet(training_model.net, num_iter=1)
    accuracy.append(workspace.FetchBlob('accuracy'))
    loss.append(workspace.FetchBlob('loss'))
    lr.append(workspace.FetchBlob('SgdOptimizer_0_lr_gpu0'))
#    lr.append(workspace.FetchBlob('conv1_b_lr'))
    if start%Snapshot_interval == 0:
        save_snapshot(training_model,start)
    val_accuracy.append(check_val())
    start+=Batch_Size
'''plt.plot(accuracy,'b',label='Training Set')
plt.plot(val_accuracy,'r',label='Validation Set')
plt.ylabel('Accuracy')
plt.xlabel('No. of Iterations')
plt.legend(loc=4)
plt.show()
plt.plot(loss,'b',label='Training Set')
plt.ylabel('Loss')
plt.xlabel('No. of Iterations')
plt.legend(loc=1)
plt.show()
lr = [-1*l for l in lr]
plt.plot(lr)
plt.show()'''
#print("training accuracy",accuracy)  
best = np.argmax(np.array(val_accuracy)[range(0,np.array(val_accuracy).shape[0],Snapshot_interval)])
best = best*Batch_Size*Snapshot_interval
params=cPickle.load(open(Snapshot_location+str(best),'rb'))
#print('params',params)
with open('records.txt','w') as f:
    #f.write('%s params'%str(params))
    for blob in params.keys():
        workspace.FeedBlob(blob,params[blob],device_option)
        f.write('In layer %s parameter size %s \n' %(str(blob),str((params[blob].itemsize*params[blob].size)/(1024*1024))))
        
    
 
#results=[['ImageId','Label']]
results=[]
start=0
while start<test.shape[0]:
    raw_batch = test[start:start+Batch_Size,:]
    labels = test[start:start+Batch_Size,0]

    batch = raw_batch.reshape(raw_batch.shape[0],28,28)
    batch = batch[:,np.newaxis,...].astype(np.float32)
    batch = batch*float(1./256)
    workspace.FeedBlob("data", batch, device_option)
    workspace.RunNet(testing_model.net, num_iter=1)
    res = np.argmax(workspace.FetchBlob('softmax'),axis=1)
    feat = workspace.FetchBlob('fc3')
    for r in range(raw_batch.shape[0]):
        results.append([start+r+1,res[r]])
    for r in range(raw_batch.shape[0]):
        temp=[]
        for i,j in enumerate(feat[r].tolist()):
            temp.append(str(i+1)+':'+str(j))
        results.append([int(labels[r])]+temp)
    start+=raw_batch.shape[0]

with open('results.csv', "w") as output:
    wr = csv.writer(output,delimiter=' ', lineterminator='\n')
    wr.writerows(results)
print("==================here=============")
# saving the trained model

INIT_NET = './init_net.pb'
PREDICT_NET = './predict_net.pb'
def SaveNet(INIT_NET, PREDICT_NET, workspace, model):
    init_net, predict_net = mobile_exporter.Export(
        workspace, model.net, model.params)
    
    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())
'''def save_net(INIT_NET, PREDICT_NET, model) :
    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    init_net = caffe2_pb2.NetDef()
    for param in model.params:
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill", [], [param],
                        arg=[utils.MakeArgument("shape", shape),
                        utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
    init_net.op.extend([core.CreateOperator("ConstantFill", [],
                        ["data"], shape=(1,28,28))])
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())'''
'''def load_net(INIT_NET, PREDICT_NET, device_opts):
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'r') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'r') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
    predict_net = core.Net(net_def)
    init_net = core.Net(init_def)
    return init_net, predict_net'''
def LoadNet(INIT_NET, PREDICT_NET, device_opts):
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'r') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())
    
    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'r') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0) # change to 'core.DeviceOption(caffe2_pb2.CUDA, 0)' for GPU processing
SaveNet(INIT_NET, PREDICT_NET,workspace,testing_model)  #test
print("model is saved successfully")


