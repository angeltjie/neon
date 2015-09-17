.. ---------------------------------------------------------------------------
.. Copyright 2015 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
..  ---------------------------------------------------------------------------

Getting Started
===============

Installation
------------

External Dependencies
'''''''''''''''''''''

To install neon on a Linux or Mac OSX machine, please ensure you have recent
versions of the following system software (Ubuntu package names shown):

* ``python`` - We currently support python 2.7
* ``python-pip`` - Needed to install python dependencies.
* ``python-virtualenv`` - Needed to configure an isolated environment
* ``libhdf5-dev`` - (h5py) for callback hdf5 datasets
* ``libyaml-dev`` - (pyyaml) for YAML input file parsing
* ``libopencv-dev``, ``pkg-config`` - (imageset_decoder) optional requirement,
  used to perform decoding of image data

Though neon will run on a CPU, you'll get better performance by utilizing a
recent GPU (Maxwell based architecture).  This requires installation of the
`CUDA SDK and drivers <https://developer.nvidia.com/cuda-downloads>`_.


Virtualenv based Install
''''''''''''''''''''''''

A virtualenv based install is recommended as this will ensure a self-contained
environment in which to run and develop neon.  To setup neon in this manner
run the following commands:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make

The virtualenv will install all required files into the ``.venv`` directory.
To begin using it type the following:

.. code-block:: bash

    . .venv/bin/activate

You'll see your prompt change to highlight the venv, you can now run the neon
examples, or extend the code:

.. code-block:: bash

    cd examples
    ./mnist_mlp.py

When you have finished working on neon, you can deactivate the virtualenv via:

.. code-block:: bash

    deactivate

System-wide Install
'''''''''''''''''''

The virtualenv based install is recommended to ensure an isolated
environment. As an alternative, it is possible to install neon into
your sytem python path.  The process for doing so is:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make sysinstall


How to run a model
------------------
With the virtual environment activated, there are two ways to run
models through neon. The first is to use the neon executable and pass
it a model specified by a YAML file. The second way is to specify
the model directly through a python script. There are a number of
examples available in the ``examples`` directory. The one we focus on
here is an MLP trained on the MNIST dataset.

YAML based example
''''''''''''''''''

From the neon repository directory, type

.. code-block:: bash

    neon examples/mnist_mlp.yaml

On the first run, neon will download the MNIST dataset. It will crate
a ``~/nervana`` directory where the raw datasets are kept, and store
processed batches in the neon repository directory. Once that is done,
it will train a simple MLP model on the dataset and report
cross-entropy error after each epoch.

Python script example
'''''''''''''''''''''

The same model is avaiable as a python script that can be called
directly without using a YAML specification to create the model. To
run the script, type

.. code-block:: bash

    examples/mnist_mlp.py

This will run an identical MLP model and print the final
misclassification error after running for 10 epochs.


Simple MLP tutorial
-------------------

This example follows the model from a slightly simplified version of
:py:obj:`examples/mnist_mlp.py`.

The first step is to set up a logger and argument parser. The logging
module gives us control over printing messages to stdout or to file,
and controls verbosity of the output.
:py:class:`NeonArgparser<neon.util.argparser.NeonArgparser>` is used to
parse command line arguments, such as number of training epochs, how
often to run cross-validation, where to save the model, etc. It also
controls backend settings, such as running on GPU or CPU, which
datatype to use, and how rounding is performed. For a full list of
arguments, run ``neon --help`` and see :py:obj:`examples/mnist_mlp.py`
for an example of how custom arguments can easily be added.

.. code-block:: python

    import logging
    logger = logging.getLogger()

    # parse the command line arguments
    from neon.util.argparser import NeonArgparser
    parser = NeonArgparser()
    args = parser.parse_args()

Backend Setup
'''''''''''''
The backend is controlled via the ``-b`` command line argument, which can be
``gpu`` to select :py:class:`NervanaGPU<neon.backends.nervanagpu.NervanaGPU>`
or ``cpu`` to select :py:class:`NervanaCPU<neon.backends.nervanacpu.NervanaCPU>`
as the backend. By default, the GPU backend is used. On machines where no
compatible GPU is found, it will automatically fail back to CPU. The following
block of code sets up the backend.

.. code-block:: python

    from neon.backends import gen_backend
    be = gen_backend(backend=args.backend,
                     batch_size=128,
                     rng_seed=args.rng_seed,
                     device_id=args.device_id,
                     default_dtype=args.datatype,
                     stochastic_round=False)


The :py:func:`gen_backend` function will handle generating and
switch backends. When called repeatedly, it will clean up an
existing backend and generate a new one. If a GPU backend was
generated previously, then :py:func:`gen_backend` will destroy the
existing context and delete the backend object. See :doc:`backends`
for a list of all options that can be set when generating a backend.

The minibatch size for training is hard-coded to 128 images and
stochastic rounding (mainly useful for estimating models in 16 bit
precision) is disabled. The rng_seed argument can be used to specify a
fixed random seed, device_id controls which GPU to run on if multiple
GPUs are available, and the default_dtype can be used to specify a 32
or 16 bit data type.


Loading a Dataset
'''''''''''''''''

To load the MNIST dataset, the :py:func:`load_mnist` function is included
with the ``neon/data/loader.py`` utility. The data is set up on the
GPU as a :py:class:`DataIterator<neon.data.dataiterator.DataIterator>`, which
provides an interface to iterate over mini-batches after pre-loading them into
device memory.

.. code-block:: python

    from neon.data import DataIterator, load_mnist
    # split into train and tests sets
    (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
    # setup training set iterator
    train_set = DataIterator(X_train, y_train, nclass=nclass)
    # setup validation set iterator
    valid_set = DataIterator(X_test, y_test, nclass=nclass)


See :doc:`datasets`  to learn how to load the other datasets or add your own.

Weight Initialization
'''''''''''''''''''''

Neon supports initializing weight matrices with constant, uniform, Gaussian,
and automatically scaled uniform (Glorot initialization) distributed values.
This example uses :py:class:`Gaussian<neon.initializers.initializer.Gaussian>`
initialization with zero mean and 0.01 standard deviation.

.. code-block:: python

    from neon.initializers import Gaussian
    init_norm = Gaussian(loc=0.0, scale=0.01)

The weights will be initialized below when the layers are created.

Learning Rules
''''''''''''''

The examples uses :py:class:`Gradient Descent with Momentum<neon.optimizers.optimizer.GradientDescentMomentum>`
as the learning rule:

.. code-block:: python

    from neon.optimizers import GradientDescentMomentum
    optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9,
                                        stochastic_round=args.rounding)

If stochastic rounding is used, it is applied exclusively to weight updates, so
it is passed as a parameter to the optimizer.

Layers
''''''

The model is specified as a list of layer instances, which are defined
by a layer type and an activation function. This example uses affine
(i.e. fully-connected) layers with a rectified linear activation on
the hidden layer and a logistic activation on the output layer. We set
our final layer to have 10 units in order to match the number of
labels in the MNIST dataset.

.. code-block:: python

    from neon.layers import Affine
    from neon.transforms import Rectlin, Logistic

    layers = []
    layers.append(Affine(nout=100, init=init_norm, activation=Rectlin()))
    layers.append(Affine(nout=10, init=init_norm,
                         activation=Logistic(shortcut=True)))


Other layer types that are not used in this example include
convolution and pooling layers. They are described in :doc:`layers`. Weight
layers take an initializer for the weights, which we have defined
above.


Costs
'''''

The cost function is wrapped into a ``GeneralizedCost`` layer, which handles
the comparison of the cost function outputs with the labels provided with the
data set. The cost function passed into the cost layer is the cross-entropy
transform.

.. code-block:: python

    from neon.layers import GeneralizedCost
    from neon.transforms import CrossEntropyBinary
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())


Model
'''''

We generate a model using the layers created above, and instantiate a
set of standard callbacks to display a progress bar during training,
and to save the model to a file, if one is specified in the command
line arguments. We then train the model on the dataset set up as
``train_set``, using the optimizer and cost functions defined
above. The number of epochs to train for is also passed in through the
arguments.

.. code-block:: python

    # initialize model object
    from neon.models import Model
    mlp = Model(layers=layers)

    # setup standard fit callbacks
    from neon.callbacks.callbacks import Callbacks
    callbacks = Callbacks(mlp, train_set, output_file=args.output_file,
                          progress_bar=args.progress_bar)

    # run fit
    mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost,
            callbacks=callbacks)


Evaluation Metric
'''''''''''''''''

Finally, we can evaluate the performance of our model by examining its
misclassification rate on the test set.

.. code-block:: python

    from neon.transforms import  Misclassification
    print('Misclassification error = %.1f%%'
          % (mlp.eval(valid_set, metric=Misclassification())*100))


