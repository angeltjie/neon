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

from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Convolution, Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataIterator, load_cifar10
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
import cPickle as pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 128
num_epochs = args.epochs

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

(X_train, y_train), (X_test, y_test), nclass, lshape = load_cifar10(path=args.data_dir)

# really 10 classes, pad to nearest power of 2 to match conv output
train_set = DataIterator(X_train, y_train, nclass=16, lshape=lshape)
valid_set = DataIterator(X_test, y_test, nclass=16, lshape=lshape)

init_uni = GlorotUniform()
opt_gdm = GradientDescentMomentum(learning_rate=0.5,
                                  schedule=Schedule(step_config=[200, 250, 300],
                                                    change=0.1),
                                  momentum_coef=0.9, wdecay=.0001)

layers = []

layers.append(Dropout(keep=.8))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin()))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
layers.append(Dropout(keep=.5))

layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
layers.append(Dropout(keep=.5))

layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
layers.append(Conv((1 ,1, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
layers.append(Conv((1,1,16), init=init_uni, activation=Rectlin()))

layers.append(Pooling(6, op="avg"))
layers.append(Activation(Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# Get the previously saved weights 
mlp.load_weights('allcnnweights.pkl')

test = DataIterator(X_test, y_test, nclass=10, lshape=(3,32,32))

def mapToColor(out):
    minOut = np.min(out)
    out -= minOut
    maxOut = np.max(out)
    if maxOut == 0:
        maxOut = 1
    out = out / maxOut
    return out

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



class Visualize:
    def __init__(self, model, be, epoch_freq=1):
#        super(VisualizeCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.be = be

def _on_epoch_end(self, epoch):
    be = self.be
    saveAct = defaultdict(list) 

    # Loop over every layer to visualize
    for i in range(1, len(self.model.layers) + 1):
        count = 0
        layer_ind = len(self.model.layers) - i

        if not isinstance(self.model.layers[layer_ind], Convolution):
            continue 

        num_featmap = self.model.layers[layer_ind].convparams['K']
        act_height = self.model.layers[layer_ind].outputs.lshape[1]
        act_width = self.model.layers[layer_ind].outputs.lshape[2]
 
        # Loop to visualize every feature map
        for fm in range(num_featmap):
            # TODO: Change this into device array
            activation = np.zeros((num_featmap, act_height, act_width, be.bsz))
            activation[fm, act_height/2, act_width/2, :] = 1
            activation = be.array(activation)

            # Loop over the previous layers to perform deconv
            for l in self.model.layers[layer_ind::-1]:
    
                if isinstance(l, Convolution):
                    # the output shape from conv is the input shape into deconv
                    # p: conv output height, q: conv output width, k: conv number of output feature
                    # maps
                    shape = l.outputs.lshape
                    k, p, q = shape[0], shape[1], shape[2]
                    H, W, C = l.convparams['H'], l.convparams['W'], l.convparams['C']

                    out_shape = (C, H, W, be.bsz)
                    
                    # ReLU - note: the result of Rectlin() is on op-tree, so assign the values to
                    # activation
                    r = Rectlin()
                    activation[:] = r(activation)

                    # run deconv fprop (i.e. conv bprop)
                    out = be.empty(out_shape) 
                    l.be.bprop_conv(layer=l.nglayer, F=l.W, E=activation, grad_I=out)
                    activation = out

            saveAct[layer_ind].append(activation.asnumpyarray()[:,:,:,0])

        print(layer_ind)
    return saveAct

#getActivation(mlp, test,0) 
#v = Visualize(mlp, be)
#trial = _on_epoch_end(v,1)

#pickle.dump(trial, open("saveTrialWeights", "wb"))

trial = pickle.load( open("saveTrialWeights.p", "rb"))
img = trial[1][81]
img = mapToColor(img)
img = img.reshape(3,32,32)
img = np.transpose(img[:,:,:], axes=(2,1,0))
plt.imshow(img, interpolation="nearest")
plt.axis("off")
plt.clim(0,1)
plt.savefig("see-pic.png", dpi=100)
