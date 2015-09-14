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
import h5py
import numpy as np


def create_minibatch_x(minibatches, minibatch_markers, epoch_axis):
    """
    Helper function to build x axis for data captured per minibatch

    Arguments:
        minibatches (int): how many total minibatches
        minibatch_markers (int array): cumulative number of minibatches complete at a given epoch
        epoch_axis (bool): whether to render epoch or minibatch as the integer step in the x axis
    """
    if epoch_axis:
        x = np.zeros((minibatches,))
        last_e = 0
        for e_idx, e in enumerate(minibatch_markers):
            e_minibatches = e - last_e
            x[last_e:e] = e_idx + (np.arange(float(e_minibatches))/e_minibatches)
            last_e = e
    else:
        x = np.arange(minibatches)

    return x


def create_epoch_x(points, epoch_freq, minibatch_markers, epoch_axis):
    """
    Helper function to build x axis for points captured per epoch

    Arguments:
        points (int): how many data points need a corresponding x axis points
        epoch_freq (int): are points once an epoch or once every n epochs?
        minibatch_markers (int array): cumulative number of minibatches complete at a given epoch
        epoch_axis (bool): whether to render epoch or minibatch as the integer step in the x axis
    """

    if epoch_axis:
        x = np.zeros((points,))
        last_e = 0
        for e_idx, e in enumerate(minibatch_markers):
            e_minibatches = e - last_e
            if (e_idx + 1) % epoch_freq == 0:
                x[e_idx/epoch_freq] = e_idx + (e_minibatches - 1) / e_minibatches
            last_e = e
    else:
        x = minibatch_markers[(epoch_freq-1)::epoch_freq] - 1

    return x


def h5_cost_data(filename, epoch_axis=True):
    """
    Read cost data from hdf5 file. Generate x axis data for each cost line.

    Returns:
        list of tuples of (name, x data, y data)
    """
    ret = list()
    with h5py.File(filename, "r") as f:

        config, cost, time_markers = [f[x] for x in ['config', 'cost', 'time_markers']]
        total_epochs = config.attrs['total_epochs']
        total_minibatches = config.attrs['total_minibatches']
        minibatch_markers = time_markers['minibatch']

        for name, ydata in cost.iteritems():
            y = ydata[...]

            if ydata.attrs['time_markers'] == 'epoch_freq':
                y_epoch_freq = ydata.attrs['epoch_freq']
                assert len(y) == total_epochs / y_epoch_freq
                x = create_epoch_x(len(y), y_epoch_freq, minibatch_markers, epoch_axis)

            elif ydata.attrs['time_markers'] == 'minibatch':
                assert len(y) == total_minibatches
                x = create_minibatch_x(total_minibatches, minibatch_markers, epoch_axis)

            else:
                raise TypeError('Unsupported data format for h5_cost_data')

            ret.append((name, x, y))

    return ret

def length_max_ind(layers):
    currcount = 0
    count_in_layer = 0
    for l in layers:
        count_in_layer = 0
        for x in l:

            try:
                x = int(x)
                if x != 0:
                    count_in_layer += 1
                
            except:
                continue
    
        if count_in_layer > currcount:
            currcount += 1
    return currcount

def h5_deconv_data(filename, layer_ind) :
    """
    Read deconv visualization data from hdf5 file.

    Returns:
        list of tuples of (layer_name, fm_name, imgdata)
        list of the layer names
    """
    ret = list()
    with h5py.File(filename, "r") as f:

        deconv = f['deconv']
        layers = deconv.keys()

        len_max_ind = length_max_ind(layers)
        layer_key = "{0:0" + str(len_max_ind) + "}"
        layer_key = layer_key.format(layer_ind)
        layer_name = 'layer' + layer_key

        for fm_name, fm in deconv[layer_name].iteritems():
            imgdata = fm[...]
            ret.append((fm_name, imgdata))

    return ret 
