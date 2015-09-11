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

from neon.callbacks.callbacks import Callback
from neon.layers import Convolution


# TODO: This does not actually take in any images right now. All that it does is generate 'fake'
# activations and send it back via deconv, so we get an idea of what the feature map looks like. We
# probably want to add in the image set later.  

class DeconvCallback(Callback):
    def __init__(self, callback_data, model, train_set, epoch_freq=1):
        super(DeconvCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.train_set = train_set
        self.callback_data = callback_data


    def on_train_begin(self, epochs):
        train_set = self.train_set
        self.H = train_set.H 
        self.W = train_set.W
        model = self.model
        layers = model.layers

        # TODO: Right now, the index into vdata is going to be 1 for 1st conv layer, 2 for 2nd conv
        # layer, etc. 
        curr_conv = 0
        for i in range(len(layers)):
            if isinstance(layers[i], Convolution):
                curr_conv += 1
                vdata = self.callback_data.create_dataset("deconv/layer/" + str(curr_conv), 
                                                          (3, self.H, self.W)) 

    def on_epoch_end(self, epoch):
        be = self.model.be
        model = self.model
        layers = model.layers

        # Loop over every layer to visualize
        for i in range(1, len(layers) + 1):
            count = 0
            layer_ind = len(layers) - i

            if not isinstance(layers[layer_ind], Convolution):
                continue 

            num_chn = layers[layer_ind].convparams['K']
            act_h = layers[layer_ind].outputs.lshape[1]
            act_w = layers[layer_ind].outputs.lshape[2]
 
            # Loop to visualize every feature map
            for chn in range(num_chn):
                activation = np.zeros((num_chn, act_h, act_w, be.bsz))
                activation[chn, act_h/2, act_w/2, :] = 1
                activation = NervanaObject.be.array(activation)

                # Loop over the previous layers to perform deconv
                for l in layers[layer_ind::-1]:
    
                    if isinstance(l, Convolution):
                        # the output shape from conv is the input shape into deconv
                        # p: conv output height, q: conv output width, k: conv number of output feature
                        # maps
                        shape = l.outputs.lshape
                        k, p, q = shape[0], shape[1], shape[2]
                        H, W, C = l.convparams['H'], l.convparams['W'], l.convparams['C']

                        out_shape = (C, H, W, be.bsz)
                    
                        # ReLU, then deconv-fprop
                        # The result of Rectlin() is an op-tree, so assign the values to activation
                        r = Rectlin()
                        activation[:] = r(activation)

                        out = be.empty(out_shape) 
                        l.be.bprop_conv(layer=l.nglayer, F=l.W, E=activation, grad_I=out)
                        activation = out

                self.callback_data["deconv/layer/"+str(layer_ind)] = activation.asnumpyarray()[:,:,:,0]


