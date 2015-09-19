# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
import os
import sys
import logging
import h5py
from collections import deque
from neon import NervanaObject
from neon.util.persist import save_obj
from timeit import default_timer

from neon.layers import Convolution
import numpy as np
from neon.transforms.activation import Rectlin

logger = logging.getLogger(__name__)
import time


class Callbacks(NervanaObject):

    """
    Container class for storing and iterating over callbacks.

    Attributes:
        callbacks (list): Ordered set of Callback objects to be run.
    """

    def __init__(self, model, train_set, output_file=None, valid_set=None,
                 valid_freq=None, progress_bar=True):
        """
        Create a callbacks container with the default callbacks.

        Arguments:
            model (Model): the model object
            train_set (DataIterator): the training dataset
            output_file (string, optional): path to save callback data to
            valid_set (DataIterator, optional): the validation dataset to use
            valid_freq (int, optional): how often (in epochs) to run validation
            progress_bar (bool): control whether a progress bar callback is created.
                                 Defaults to True.
        """
        self.callbacks = list()
        self.epoch_marker = 0
        if output_file is None:
            self.callback_data = h5py.File("no_file", driver='core', backing_store=False)
        else:
            if os.path.isfile(output_file):
                logger.warn("Overwriting output file %s", output_file)
                os.remove(output_file)
            self.callback_data = h5py.File(output_file, "w")
        self.model = model
        self.train_set = train_set

        self.callbacks.append(TrainCostCallback(self.callback_data, self.model))

        if valid_set and valid_freq:
            self.callbacks.append(ValidationCallback(self.callback_data, self.model,
                                                     valid_set, valid_freq))
        if progress_bar:
            self.callbacks.append(ProgressBarCallback(self.callback_data, model, train_set))
        self.callbacks.append(TrainLoggerCallback(self.callback_data, model,
                                                  epoch_freq=1, minibatch_freq=None))

    def add_validation_callback(self, valid_set, epoch_freq):
        """
        Convenience function to create and add a Validation callback.

        Arguments:
            valid_set (DataIterator): the validation dataset to use
            epoch_freq (int): how often (in epochs) to run validation
        """
        # Insert before other callbacks since some depend on validation cost
        self.add_callback(ValidationCallback(self.callback_data, self.model,
                                             valid_set, epoch_freq),
                          insert_pos=0)

    def add_deconv_callback(self, train_set, valid_set, epoch_freq):
        """
        Convenience function to create and add a deconvolution callback. The data can be used for
        visualization.

        Arguments:
            train_set (DataIterator): the train dataset to use
            epoch_freq (int): how often (in epochs) to store deconvolution data.
            valid_set (DataIterator): the validation dataset to use
        """
        self.add_callback(DeconvCallback(self.callback_data, self.model,
                                         train_set, valid_set, epoch_freq))

    def add_serialize_callback(self, serialize_schedule, save_path, history=1):
        """
        Convenience function to create and add a model serialization callback.

        Arguments:
            serialize_schedule (Schedule): the serialization schedule to follow
            save_path (string): where to save the serialized data
            history (int): number of previous checkpoint files to retain
        """
        if save_path and serialize_schedule:
            # TODO can serialize be handled by regular data callback or should it be separate?
            self.callbacks.append(SerializeModelCallback(self.model,
                                                         save_path,
                                                         epoch_freq=serialize_schedule,
                                                         history=history))

    def add_save_best_state_callback(self, path):
        """
        Convenience function to create and add a save best state callback.

        Arguments:
            path (string): where to save the best model state.
        """
        self.callbacks.append(SaveBestStateCallback(self.callback_data, self.model, path))

    def add_early_stop_callback(self, stop_func):
        """
        Convenience function to create and add an early stopping callback.

        Arguments:
            stop_func (function): function to determine when to stop.
        """
        self.callbacks.append(EarlyStopCallback(self.callback_data, self.model, stop_func))

    def add_callback(self, callback, insert_pos=None):
        """
        Add a user supplied callback. Since callbacks are run serially and share data,
        order can matter.  If the default behavior (to append the callback) is not
        sufficient, insert position can be controlled.

        Arguments:
            callback (Callback): callback object to be registered
            insert_pos (int, optional): position in the list to insert the callback.
                                        Defaults to None, meaning append
        """
        if insert_pos is None:
            insert_pos = len(self.callbacks)
        self.callbacks.insert(insert_pos, callback)

    def on_train_begin(self, epochs):
        """
        Call all registered callbacks' on_train_begin functions
        """
        # data iterator wraps around to avoid partial minibatches
        # callbacks producing per-minibatch data need a way to preallocate buffers
        config = self.callback_data.create_group('config')
        total_minibatches = -((-self.train_set.ndata * epochs) // self.be.bsz)
        config.attrs['total_minibatches'] = total_minibatches
        config.attrs['total_epochs'] = epochs

        time_markers = self.callback_data.create_group("time_markers")
        time_markers.create_dataset("minibatch", (epochs,))

        for c in self.callbacks:
            c.on_train_begin(epochs)

    def on_train_end(self):
        """
        Call all registered callbacks' on_train_end functions
        """
        for c in self.callbacks:
            c.on_train_end()

        self.callback_data.close()

    def on_epoch_begin(self, epoch):
        """
        Call all registered callbacks' on_epoch_begin functions

        Arguments:
            epoch (int): index of epoch that is beginning
        """
        for c in self.callbacks:
            if c.should_fire(epoch, c.epoch_freq):
                c.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        """
        Call all registered callbacks' on_epoch_end functions

        Arguments:
            epoch (int): index of epoch that is ending
        """
        for c in self.callbacks:
            if c.should_fire(epoch, c.epoch_freq):
                c.on_epoch_end(epoch)

        self.epoch_marker += self.epoch_minibatches
        self.callback_data['time_markers/minibatch'][epoch] = self.epoch_marker
        self.callback_data['time_markers'].attrs['epochs_complete'] = epoch + 1
        self.callback_data['time_markers'].attrs['minibatches_complete'] = self.epoch_marker
        self.callback_data.flush()

    def on_minibatch_begin(self, epoch, minibatch):
        """
        Call all registered callbacks' on_minibatch_begin functions

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is beginning
        """
        for c in self.callbacks:
            if c.should_fire(minibatch, c.minibatch_freq):
                c.on_minibatch_begin(epoch, minibatch)

    def on_minibatch_end(self, epoch, minibatch):
        """
        Call all registered callbacks' on_minibatch_end functions

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is ending
        """
        for c in self.callbacks:
            if c.should_fire(minibatch, c.minibatch_freq):
                c.on_minibatch_end(epoch, minibatch)

        # keep track of the number of mb per epoch, since they vary
        self.epoch_minibatches = minibatch + 1


class Callback(NervanaObject):

    """
    Interface defining common callback functions.

    Implement a callback by subclassing Callback and overriding the necessary
    on_[train,epoch,minibatch]_[begin,end] functions.

    Callback functions provide time queues as arguments but derived callback
    classes must manage their own state
    """

    def __init__(self, epoch_freq=1, minibatch_freq=1):
        self.epoch_freq = epoch_freq
        self.minibatch_freq = minibatch_freq

    def on_train_begin(self, epochs):
        """
        Called when training is about to begin
        """
        pass

    def on_train_end(self):
        """
        Called when training is about to end
        """
        pass

    def on_epoch_begin(self, epoch):
        """
        Called when an epoch is about to begin

        Arguments:
            epoch (int): index of epoch that is beginning
        """
        pass

    def on_epoch_end(self, epoch):
        """
        Called when an epoch is about to end

        Arguments:
            epoch (int): index of epoch that is ending
        """
        pass

    def on_minibatch_begin(self, epoch, minibatch):
        """
        Called when a minibatch is about to begin

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is begininning
        """
        pass

    def on_minibatch_end(self, epoch, minibatch):
        """
        Called when a minibatch is about to end

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is ending
        """
        pass

    def should_fire(self, time, freq):
        """
        Helper function for determining if a callback should do work at a given
        interval.

        Arguments:
            time (int): current time, in an arbitrary unit
            freq (int, list, None): firing frequency, in multiples of the unit used
                                    for time, or a list of times, or None (never fire)
        """
        fire = False
        if isinstance(freq, int) and (time + 1) % freq == 0:
            fire = True
        elif isinstance(freq, list) and time in freq:
            fire = True
        return fire


class SerializeModelCallback(Callback):

    """
    Callback for serializing the state of the model.

    Arguments:
        model (Model): model object
        save_path (str): where to save the model dataset
        epoch_freq (int, optional): how often (in epochs) to serialize the
                                   model.  If not specified, we default to
                                   running every epoch.
        history (int, optional): number of checkpoint files to retain, newest
                                 files up to this count are retained.  filename
                                 for the check point files will be
                                 <save_path>_<epoch>.
    """

    def __init__(self, model, save_path, epoch_freq=1, history=1):
        super(SerializeModelCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.save_path = save_path
        self.history = history
        self.checkpoint_files = deque()

    def on_epoch_end(self, epoch):
        if self.history > 1:
            self.save_history(epoch)
        else:
            save_obj(self.model.serialize(keep_states=True), self.save_path)

    def save_history(self, epoch):
        # if history > 1, this function will save the last N checkpoints
        # where N is equal to self.history.  The files will have the form
        # of save_path with the epoch added to the filename before the ext

        if len(self.checkpoint_files) > self.history:
            # remove oldest checkpoint file when max count have been saved
            fn = self.checkpoint_files.popleft()
            try:
                os.remove(fn)
                logger.info('removed old checkpoint %s' % fn)
            except OSError:
                logger.warn('Could not delete old checkpoint file %s' % fn)

        path_split = os.path.splitext(self.save_path)
        save_path = '%s_%d%s' % (path_split[0], epoch, path_split[1])
        # add the current file to the deque
        self.checkpoint_files.append(save_path)
        save_obj(self.model.serialize(keep_states=True), save_path)


class TrainCostCallback(Callback):
    """
    Callback for computing average training cost periodically during training.

    Arguments:
        callback_data (HDF5 dataset): shared data between callbacks
        model (Model): model object

    """
    def __init__(self, callback_data, model):
        super(TrainCostCallback, self).__init__(epoch_freq=1)
        self.model = model
        self.callback_data = callback_data

    def on_train_begin(self, epochs):
        # preallocate space for the number of minibatches in the whole run
        points = self.callback_data['config'].attrs['total_minibatches']
        self.callback_data.create_dataset("cost/train", (points,))

        # clue in the data reader to use the 'minibatch' time_markers
        self.callback_data['cost/train'].attrs['time_markers'] = 'minibatch'

    def on_minibatch_end(self, epoch, minibatch):
        mb_complete = minibatch + 1
        mean_cost = float(self.model.total_cost.get() / mb_complete)
        prev_epoch_minibatches = 0
        if epoch > 0:
            prev_epoch_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]

        self.callback_data['cost/train'][prev_epoch_minibatches + minibatch] = mean_cost


class ValidationCallback(Callback):

    """
    Callback for processing the validation dataset periodically during training.

    Arguments:
        callback_data (HDF5 dataset): shared data between callbacks
        model (Model): model object
        valid_set (DataIterator): Validation dataset to process
        epoch_freq (int, optional): how often (in epochs) to log training info.
                                    Defaults to every 1 epoch.
        minibatch_freq (int, optional): how often (in minibatches) to log
                                        training info, or None to log only on
                                        epoch boundaries.  Defaults to None.
    """

    def __init__(self, callback_data, model, valid_set, epoch_freq=1):
        super(ValidationCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.valid_set = valid_set
        self.valid_cost = self.be.zeros((1, 1))
        self.callback_data = callback_data

    def on_train_begin(self, epochs):
        vdata = self.callback_data.create_dataset("cost/validation", (epochs/self.epoch_freq,))
        vdata.attrs['time_markers'] = 'epoch_freq'
        vdata.attrs['epoch_freq'] = self.epoch_freq
        self.callback_data.create_dataset("time/validation", (epochs/self.epoch_freq,))

    def on_epoch_end(self, epoch):
        model = self.model
        start_validation = default_timer()
        nprocessed = 0
        self.valid_cost[:] = 0
        self.valid_set.reset()
        for batch_index, (x, t) in enumerate(self.valid_set, 1):
            x = model.fprop(x, inference=True)
            bsz = min(self.valid_set.ndata - nprocessed, self.be.bsz)
            model.cost.get_cost(x, t)
            costbuf = model.cost.outputs[:, :bsz]
            nprocessed += bsz
            self.valid_cost[:] = self.valid_cost + self.be.sum(costbuf, axis=1)
            mean_cost = float(self.valid_cost.get() / nprocessed)

        end_validation = default_timer()
        self.callback_data["cost/validation"][epoch/self.epoch_freq] = mean_cost
        self.callback_data["time/validation"][epoch/self.epoch_freq] = (end_validation
                                                                        - start_validation)


def get_progress_string(tag, epoch, minibatch, nbatches, cost, time,
                        blockchar=u'\u2588'):
    """
    Generate a progress bar string.

    Arguments:
        tag (string): Label to print before the bar (i.e. Train, Valid, Test )
        epoch (int): current epoch to display
        minibatch (int): current minibatch to display
        nbatches (int): total number of minibatches, used to display relative progress
        cost (float): current cost value
        time (float): time elapsed so far in epoch
        blockchar (str, optional): character to display for each step of
                                   progress in the bar.  Defaults to u2588
                                   (solid block)
    """
    max_bar_width = 20
    bar_width = int(float(minibatch) / nbatches * max_bar_width)
    s = u'Epoch {:<3} [{} |{:<%s}| {:4}/{:<4} batches, {:.2f} cost, {:.2f}s]' % max_bar_width
    return s.format(epoch, tag, blockchar * bar_width, minibatch, nbatches, cost, time)


class ProgressBarCallback(Callback):

    """
    Callback providing a live updating console based progress bar.

    Arguments:
        model (Model): model object
        dataset (DataIterator): dataset object
    """

    def __init__(self, callback_data, model, dataset, epoch_freq=1,
                 minibatch_freq=1, update_thresh_s=0.1):
        super(ProgressBarCallback, self).__init__(epoch_freq=epoch_freq,
                                                  minibatch_freq=minibatch_freq)
        self.model = model
        self.dataset = dataset
        self.callback_data = callback_data
        self.update_thresh_s = update_thresh_s

    def on_epoch_begin(self, epoch):
        self.start_epoch = self.last_update = default_timer()
        self.nbatches = self.dataset.nbatches

    def on_minibatch_end(self, epoch, minibatch):
        now = default_timer()
        mb_complete = minibatch + 1
        if (now - self.last_update > self.update_thresh_s or
                mb_complete == self.nbatches):
            self.last_update = now
            prev_epoch_minibatches = 0
            if epoch > 0:
                prev_epoch_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]

            train_cost = self.callback_data['cost/train'][prev_epoch_minibatches + minibatch]
            progress_string = get_progress_string("Train", epoch, mb_complete,
                                                  self.nbatches, train_cost,
                                                  now - self.start_epoch)
            sys.stdout.write('\r')
            sys.stdout.write(progress_string.encode('utf-8'))
            sys.stdout.flush()

    def on_epoch_end(self, epoch):

        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]
                validation_time = self.callback_data['time/validation'][epoch/val_freq]
                progress_string = "[Validation %.2f cost, %.2fs]" % (validation_cost,
                                                                     validation_time)
                sys.stdout.write(progress_string.encode('utf-8'))
                sys.stdout.flush()

        sys.stdout.write('\n')


class TrainLoggerCallback(Callback):

    """
    Callback for logging training progress.

    Arguments:
        model (Model): model object

        epoch_freq (int, optional): how often (in epochs) to log training info.
                                    Defaults to every 1 epoch.
        minibatch_freq (int, optional): how often (in minibatches) to log
                                        training info, or None to log only on
                                        epoch boundaries.  Defaults to None.
    """

    def __init__(self, callback_data, model, epoch_freq=1, minibatch_freq=None):
        self.callback_data = callback_data
        self.model = model
        super(TrainLoggerCallback, self).__init__(epoch_freq=epoch_freq,
                                                  minibatch_freq=minibatch_freq)
        self.epoch_freq = epoch_freq
        self.minibatch_freq = minibatch_freq

    def on_minibatch_end(self, epoch, minibatch):
        prev_epoch_minibatches = 0
        if epoch > 0:
            prev_epoch_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]
        train_cost = self.callback_data['cost/train'][prev_epoch_minibatches + minibatch]
        logger.info("Epoch %d Minibatch %d complete. Train cost: %f", epoch, minibatch, train_cost)

    def on_epoch_end(self, epoch):
        log_str = "Epoch %d complete. Train Cost %f" % (epoch,
                                                        self.model.total_cost.get())
        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]
                log_str += ", Validation Cost %f" % (validation_cost)

        logger.info(log_str)


class SaveBestStateCallback(Callback):

    """
    Callback for saving the best model state so far.

    Arguments:
        callback_data
        model (Model): model object
        path (str): repeatedly write the best model parameters seen so far to the
                    filesystem path specified.
    """

    def __init__(self, callback_data, model, path):
        super(SaveBestStateCallback, self).__init__(epoch_freq=1)
        self.callback_data = callback_data
        self.model = model
        self.best_path = path
        self.best_cost = None

    def on_epoch_end(self, epoch):

        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]

                if validation_cost < self.best_cost or self.best_cost is None:
                    save_obj(self.model.serialize(keep_states=True), self.best_path)
                    self.best_cost = validation_cost


class EarlyStopCallback(Callback):

    """
    Callback for stopping training when a threshold has been triggered.

    Arguments:
        model (Model): model object
        callback_data:
        stop_func (Function): Takes a function that receives a tuple (State, Val[t])
                              of the current state and the validation error at this time
                              and returns a tuple (State', Bool) that returns the updated
                              state and an indication of whether to stop training.
    """

    def __init__(self, callback_data, model, stop_func):
        super(EarlyStopCallback, self).__init__(epoch_freq=1)
        self.callback_data = callback_data
        self.model = model
        self.stop_func = stop_func
        self.stop_state = None  # state needed for the stop func

    def on_epoch_end(self, epoch):
        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]

                self.stop_state, finished = self.stop_func(self.stop_state, validation_cost)

                if finished:
                    # should this just exit instead?
                    self.model.finished = True
                    logger.warn('Early stopping function has been triggered with mean_cost %f.'
                                % (validation_cost))


# TODO: This does not actually take in any images right now. All that it does is generate 'fake'
# activations and send it back via deconv, so we get an idea of what the feature map looks like. We
# probably want to add in the image set later.

class DeconvCallback(Callback):
    """
    Callback to store data after projecting activations back to pixel space using deconvolution.

    Arguments:
        model (Model): model object
        callback_data (HDF5 dataset): shared data between callbacks
        train_set (DataIterator): the training dataset
        epoch_freq (int): how often (in epochs) to store deconvolution data
    """
    def __init__(self, callback_data, model, train_set, valid_set, epoch_freq=1, history=1):
        super(DeconvCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.callback_data = callback_data
        self.history = history
        self.checkpoint_files = deque()

    def on_train_begin(self, epochs):
        H = self.train_set.lshape[1]
        W = self.train_set.lshape[2]
        layers = self.model.layers
        act_data = self.callback_data.create_group("deconv/act_data")
        img_data = self.callback_data.create_group("deconv/img_data")

        for i in range(len(layers)):
            if not isinstance(layers[i], Convolution):
                continue

            layer_name = "{0:04}".format(i)
            layer_data = act_data.create_group("layer_" + layer_name)
            num_fm = layers[i].convparams['K']

            for fm in range(num_fm):
                fm_name = "{0:04}".format(fm)
                fmap_data = layer_data.create_group("fmap_" + fm_name)
                fmap_data.create_dataset("plot", (3, H, W))
                fmap_data.create_dataset("max_act_val", (1,))
                fmap_data.create_dataset("img_ind", (2,), dtype='i32')
                fmap_data.create_dataset("fm_loc", (1,), dtype='i32')


    def get_activations(self):

        start = time.time()
        act_data = self.callback_data["deconv/act_data"]

        for lay in act_data.iterkeys():
            for fm in act_data[lay].iterkeys():
                act_data[lay][fm]["max_act_val"][...] = -1e8

        # For every image in the validation set

        self.valid_set.reset()
        for batch_ind, (x, t) in enumerate(self.valid_set, 0):
            imgs_temp_buf = x.get()
            self.get_layer_acts(x, batch_ind)

            self.store_images(imgs_temp_buf)

        end = time.time()
        print ("******* getting acts and storing images took", end-start)
        return

    def get_layer_acts(self, x, batch_ind):
        batch_size = self.be.bsz

        # Get the activation of each layer
        for lay_ind, la in enumerate(self.model.layers, 0):

            x = la.fprop(x, inference=True)

            if not isinstance(la, Convolution):
                continue

            layer_name = "{0:04}".format(lay_ind)

            layer_data = self.callback_data["deconv/act_data/layer_" + layer_name]

            num_fm, H, W = la.outputs.lshape

            all_acts = la.outputs.get().reshape((num_fm, H * W, batch_size))

            for fm in range(num_fm):
                fm_name = "fmap_" + "{0:04}".format(fm)
                max_act_val = layer_data[fm_name + "/max_act_val"]
                img_ind = layer_data[fm_name + "/img_ind"]
                fm_loc = layer_data[fm_name + "/fm_loc"]

                # This is all the activations of #batchsize images on one fm
                fm_acts = all_acts[fm, :, :]

                # TODO: maybe replace with np.argpartition to speed up

                # maximum activation by each image
                max_acts = np.sort(fm_acts, axis=0)[-1:][::-1][0]

                # If the current max activation on the fm is larger than the previously recorded
                # one, then replace it.

                # TODO: modify this to get k largest
                # TODO: just argsort once, and then index in to see if it is larger
                curr_fm_max_act = np.sort(max_acts)[-1:][::-1]

                if curr_fm_max_act > max_act_val:
                    max_act_val[...] = curr_fm_max_act

                    curr_img_ind = np.argsort(max_acts)[-1:][::-1]
                    img_ind[...] = (batch_ind, curr_img_ind)
                    fm_loc[...] = np.argmax(fm_acts[:, curr_img_ind])
        return

    def store_images(self, imgs_temp_buf):
        img_data_group = self.callback_data["deconv/img_data"]
        img_size = imgs_temp_buf.shape[0]

        act_data = self.callback_data["deconv/act_data"]
        imgs_to_keep = self.get_img_indices() 

        for batch_ind, ind in imgs_to_keep:
            key = "batch_" + str(batch_ind) + '_img_' + str(ind)
            if key not in img_data_group:
                img_data_group.create_dataset(key, (img_size,))
                img_data_group[key][...] = imgs_temp_buf[:,ind]
        return

    def get_img_indices(self):
        img_id = list()
        act_data = self.callback_data["deconv/act_data"]
        for lay in act_data.iterkeys():
            for fm in act_data[lay].iterkeys():
                batch_ind, img_ind = act_data[lay][fm]["img_ind"][...]
                img_id.append((batch_ind, img_ind))
        return img_id

    def visualize_layer(self, num_fm, act_size, layer_ind):
        be = self.model.be
        layer_name = "{0:04}".format(layer_ind)
        layer_data = self.callback_data["deconv/act_data/layer_" + layer_name]
        layers = self.model.layers

        # Loop to visualize every feature map
        for fm in range(num_fm):
            fm_name = "fmap_" + "{0:04}".format(fm)
            max_act_val = layer_data[fm_name + "/max_act_val"]
            fm_loc = layer_data[fm_name + "/fm_loc"]
            plot = layer_data[fm_name + "/plot"]

            activation = np.zeros((num_fm, act_size, be.bsz))

            # Set the max activation at the correct feature map location
            activation[fm, fm_loc, :] = max_act_val
            activation = be.array(activation)

            # Loop over the previous layers to perform deconv
            for l in layers[layer_ind::-1]:
                if isinstance(l, Convolution):
                    # output shape of deconv is the input shape of conv
                    H, W, C = l.convparams['H'], l.convparams['W'], l.convparams['C']
                    out_shape = (C, H, W, be.bsz)

                    r = Rectlin()
                    activation[:] = r(activation)

                    out = be.empty(out_shape)
                    l.be.bprop_conv(layer=l.nglayer, F=l.W, E=activation, grad_I=out)
                    activation = out
            plot[...] = activation.asnumpyarray()[:, :, :, 0]
        return

    def on_epoch_end(self, epoch):
        layers = self.model.layers

        # Get the activations
        self.get_activations()
        # Loop over every layer to visualize
        start = time.time()
        for i in range(1, len(layers) + 1):
            layer_ind = len(layers) - i

            if not isinstance(layers[layer_ind], Convolution):
                continue

            num_fm = layers[layer_ind].convparams['K']
            act_h = layers[layer_ind].outputs.lshape[1]
            act_w = layers[layer_ind].outputs.lshape[2]
            act_size = act_h * act_w

            self.visualize_layer(num_fm, act_size, layer_ind)
        
        return
