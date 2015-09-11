# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import logging
import numpy as np

from neon import NervanaObject
from neon.backends import gen_backend
from neon.initializers import GlorotUniform, Uniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Deconv, Convolution, Dropout, Activation, Pooling, GeneralizedCost, BatchNorm
from neon.transforms import *
from neon.models import Model
from neon.data import DataIterator, load_cifar10
from matplotlib import pyplot as plt
from matplotlib import cm
from textwrap import wrap
import cPickle as pickle

logging.basicConfig(level=20)
logger = logging.getLogger()

ng = gen_backend(backend='gpu', rng_seed=0)
NervanaObject.be = ng

ng.bsz = ng.batch_size = 32 
ng.epsilon = 2**-23
num_epochs = 100 
np.random.seed(0)

(X_train, y_train), (X_test, y_test), nclass, lshape = load_cifar10()

# really 10 classes, pad to nearest power of 2 to match conv output
train = DataIterator(X_train, y_train, nclass=16, lshape=lshape)

init_uni = GlorotUniform()
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9)

layers = []

# dropout 20% here
# layers.append(Dropout(keep=.8))

layers.append(Convolution((3,3,96), init=init_uni))
layers.append(Activation(transform=Rectlin()))

layers.append(Conv((3,3,96), init=init_uni, pad=1))
layers.append(Activation(transform=Rectlin()))

layers.append(Conv((3,3,96), init=init_uni, pad=1, strides=2))
layers.append(Activation(transform=Rectlin()))
layers.append(Dropout(keep=.5))

layers.append(Conv((3,3,192), init=init_uni, pad=1))
layers.append(Activation(transform=Rectlin()))

layers.append(Conv((3,3,192), init=init_uni, pad=1))
layers.append(Activation(transform=Rectlin()))

layers.append(Conv((3,3,192), init=init_uni, pad=1, strides=2))
layers.append(Activation(transform=Rectlin()))
layers.append(Dropout(keep=.5))

layers.append(Conv((3,3,192), init=init_uni))
layers.append(Activation(transform=Rectlin()))

layers.append(Conv((1,1,192), init=init_uni))
layers.append(Activation(transform=Rectlin()))

layers.append(Conv((1,1,16), init=init_uni, activation=Rectlin()))
layers.append(Pooling(6, op="mean"))
layers.append(Activation(Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# Get the previously saved weights 
mlp.load_weights('allcnnweights.pkl')

test = DataIterator(X_test, y_test, nclass=10, lshape=(32,32,3))

def mapToColor(out):
    minOut = np.min(out)
    out -= minOut
    maxOut = np.max(out)
    if maxOut == 0:
        maxOut = 1
    out = out / maxOut
    return out

def plotFilters(row_range, col_range, total, we, out_shape, interpolate=True, title="figure"): 
    for i in range(row_range):
        for j in range(col_range):
            filtInd = i * row_range + j 
            if filtInd >= total:
                break
            plt.subplot2grid((row_range, col_range), (i, j))
            img = we[:,:,:,filtInd]
            img = np.reshape(img, out_shape, order='F')
            img = np.transpose(img, axes=(2,1,0))
            if interpolate:
                plt.imshow(img, interpolation="nearest") 
            else:
                plt.imshow(img)
            plt.axis("off")
            plt.clim(0, 1)
    plt.savefig(title + ".png", dpi=100)

def runDecFprop(act, weights, fs, st=1, pd=0): 
    undoConv = Deconv(fshape=fs, init=init_uni, strides=st, padding=pd)
    undoConv.W = weights
    activations = act 
    if not hasattr(activations, 'lshape'):
        activations.lshape = (act.shape[1], act.shape[2], act.shape[0])
    outputs = undoConv.fprop(activations)
    return outputs

def runActFprop(inputs):
    act = Activation(transform=Rectlin())
    outputs = act.fprop(inputs, inference=True)
    return outputs

def getActivation(model, dataset, layer_ind, batch_num=1):
    count = 0
    for x, t in dataset:
        model.fprop(x, inference=True)
        if count == 0:
            act = model.layers[layer_ind].outputs.asnumpyarray()
        else:
            act = np.append(act, model.layers[layer_ind].outputs.asnumpyarray(), 1)
        count += 1
        print(count)
        if count == batch_num:
            break
    return act

#acts = pickle.load(open("tempActivations.p", "rb"))

acts = getActivation(mlp, test, 9, 1)
acts = acts.reshape((192,225,32))
#acts = acts[4].reshape((96, 225, 80))
# Let's examine conv layer 4 
# After conv 4, my images should be 96 x 15 x 15, 1600 
# Let's feed in all images

#acts = acts[7].reshape((192, 225, 10000))

# Let's examine conv layer 5
#acts = acts[9].reshape((192, 225, 80)) 

# Get 4 max activation for feature maps from all images 

for featmap in range(5):
    fig = plt.figure(figsize=(9,9))
    for i in range(2):
        for j in range(2):
            plt.subplot2grid((2,2),(i,j))

            maxAct = np.max(acts[featmap,:,:], 0)

            maxIms = maxAct.argsort()[-4:][::-1]
            # The top 4 maxActs are:
            top4act = np.sort(maxAct)[-4:][::-1]

            maxLoc = np.argmax(acts[featmap,:,maxIms], 1)

            activations = np.zeros((192, 225, 32))
            #activations = np.zeros((192, 64, 32))

            # Create a batch 
            activations[featmap, maxLoc[i*2+j], :] = maxAct[maxIms[i*2+j]]

            activations = activations.reshape((192, 15, 15, 32))
            #activations = activations.reshape((192, 8, 8, 32))

            activations = NervanaObject.be.array(activations)

            outputs = runActFprop(activations)
            #outputs = runDecFprop(outputs, mlp.layers[11].W, (3,3,192), st=2, pd=1)
            #outputs = runActFprop(outputs)
            outputs = runDecFprop(outputs, mlp.layers[9].W, (3,3,192), pd=1) 
            outputs = runActFprop(outputs)
            outputs = runDecFprop(outputs, mlp.layers[7].W, (3,3,96), pd=1) 
            outputs = runActFprop(outputs)
            outputs = runDecFprop(outputs, mlp.layers[4].W, (3,3,96), st=2, pd=1)
            outputs = runActFprop(outputs)
            outputs = runDecFprop(outputs, mlp.layers[2].W, (3,3,96), st=1, pd=1)
            outputs = runActFprop(outputs)
            outputs = runDecFprop(outputs, mlp.layers[0].W, (3,3,3), st=1)
            outputs = outputs.asnumpyarray()
            img = mapToColor(outputs)
            img = img.reshape(3,31,31,32)

            img = np.transpose(img[:,:,:,0], axes=(2,1,0))
            plt.imshow(img, interpolation="nearest")

            plt.title("max Im " + str(maxIms[i*2+j]), fontsize=7)
            plt.axis("off")
            plt.clim(0,1)
    plt.savefig("conv5-max4-feat" + str(featmap+1) + ".png", dpi=100)

"""
maxAct = np.max(acts[49,:,:], 0)
maxIm = np.argmax(maxAct)
maxLoc = np.argmax(acts[49,:,maxIm])
activations = np.zeros((192, 225, 32))
# Create a batch 
for k in range(32):
    activations[49, maxLoc, k] = maxAct[maxIm]

activations = activations.reshape((192, 15, 15, 32))
activations = NervanaObject.be.array(activations)

outputs = runActFprop(activations)
outputs = runDecFprop(outputs, mlp.layers[7].W, (3,3,96), pd=1) 
outputs = runActFprop(outputs)
outputs = runDecFprop(outputs, mlp.layers[4].W, (3,3,96), st=2, pd=1)
outputs = runActFprop(outputs)
outputs = runDecFprop(outputs, mlp.layers[2].W, (3,3,96), st=1, pd=1)
outputs = runActFprop(outputs)
outputs = runDecFprop(outputs, mlp.layers[0].W, (3,3,3), st=1)
outputs = outputs.asnumpyarray()
img = mapToColor(outputs)
img = img.reshape(3,31,31,32)

img = np.transpose(img[:,:,:,0], axes=(2,1,0))
plt.imshow(img, interpolation="nearest")
plt.axis("off")
plt.clim(0,1)
plt.savefig("conv4_50.png", dpi=100)
"""
