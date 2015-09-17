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
.. ---------------------------------------------------------------------------

Datasets
========

MNIST
-----
MNIST is a dataset of handwritten digits, consisting of 60000 training samples and 10000 test
samples. Each image is 28x28.

MNIST can be fetched in the following manner:

.. code-block:: python

    (X_train, y_train), (X_test, y_test), nclass = load_mnist()


CIFAR10
-------
CIFAR10 is a dataset consisting of 50000 training samples and 10000 test samples. There are 10
categories and each sample is a 32x32 color image.

CIFAR10 can be fetched in the following manner:

.. code-block:: python

    (X_train, y_train), (X_test, y_test), nclass = load_cifar10()


ImageCaption
------------
This dataset uses precomputed CNN image features and caption sentences. It works
with the flickr8k, flickr30k, and COCO datasets and uses the VGG image features and
sentences from http://cs.stanford.edu/people/karpathy/deepimagesent/ which have been
converted to pkl format. These datasets have 5 reference sentences per image. For
each sentence, the dataset converts each word to its 1-hot representation so that each
input batch of sentences is of dimension (vocab_size, max_sentence_length * batch_size).

The image caption data can be fetched in the following manner:

.. code-block:: python

    # download dataset
    data_path = load_flickr8k()  # Other setnames are flickr30k and COCO

    # load data
    train_set = ImageCaption(path=data_path, max_images=-1)

Text
-----
For existing datasets, for example, Penn Treebank, Hutter Prize, and Shakespeare,
we have metadata built-in to retrieve from online sources and save the file locally.
And then, a text dataset object can be created from a local path. It allows users
to use their own local text files easily.
A text dataset can take a tokenizer to parse the file. Otherwise, the file will be parsed
on a character level. It also takes an vocabulary mapping as an input, so the same mapping
can be used for all training, testing and validation sets.

.. code-block:: python

    # download Penn Treebank
    train_path = load_text('ptb-train', path=args.data_dir)
    valid_path = load_text('ptb-valid', path=args.data_dir)

    # load data and parse on word-level
    train_set = Text(time_steps, train_path, tokenizer=str.split)
    valid_set = Text(time_steps, valid_path, vocab=train_set.vocab, tokenizer=str.split)

ImageNet
--------
The raw images need to be downloaded from ILSVRC as a tar file. A neon script can convert the raw
images into binaries. data_dir is where the processed batches will be stored, and image_dir is 
where the original tar files are saved.

.. code-block:: bash

    python neon/util/batch_writer.py  --data_dir /usr/local/data/tmp \
                                      --image_dir=/usr/local/data/I1K/imagenet_orig \
                                      --set_type=i1k


Then an ImgMaster will start a image server and client to feed images to the model.

.. code-block:: python

    train = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='train')


Add a new dataset
------------------

You can also add your own dataset, where the input and the labels are n-dimensional arrays. Here is
an example of what adding image data would look like (with random pixel and label values).

.. code-block:: python

    from neon.data import DataIterator

    """
    X is the input features and y is the labels.
    Here, we show how to load in 10000 images that each have height and width
    of 32, and 3 channels (R,G,B)
    The data in X has to be laid out as follows: (# examples, feature size)
    The labels y have the same dimensions as the number of examples (in the case
    of an autoencoder, we do not specify y).
    """

    X = np.random.rand(10000,3072)
    y = np.random.randint(1,11,10000)

    """
    We pass the data points and labels X, y to be loaded into the backend
    We set nclass to 10, for 10 possible labels
    We set lshape to (3,32,32), to represent the 32x32 image with 3 channels
    """

    train = DataIterator(X=X, y=y, nclass=10, lshape=(3,32,32))

Note: You can pass in any data, as long as it is specified as above. Image data must specify an
lshape - (number of input channels, input height, input width). The tensor layout is (M, N), where
M is the flattened lshape, and N is the batch size.
