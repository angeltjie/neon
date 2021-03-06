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
'''
Test of the activation functions
'''

from math import tanh as true_tanh
import numpy as np
from neon import NervanaObject
from neon.transforms import Identity, Rectlin, Softmax, Tanh, Logistic


def compare_tensors(func, inputs, outputs, deriv=False, tol=0.):
    be = NervanaObject.be
    temp = be.empty(outputs.shape)
    dtypeu = np.float32
    if deriv is True:
        temp[:] = func.bprop(be.array(dtypeu(inputs)))
    else:
        temp[:] = func(be.array(dtypeu(inputs)))
    cond = np.sum(np.abs(temp.get() - outputs) <= tol)
    assert cond == np.prod(outputs.shape)

"""Identity
"""


def test_identity(backend_default):
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.array([0, 1, -2]).reshape((3, 1))
    compare_tensors(Identity(), inputs, outputs)


def test_identity_derivative(backend_default):
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.ones((1, 1))
    compare_tensors(Identity(), inputs, outputs, deriv=True)

"""Rectified Linear unit
"""


def test_rectlin_positives(backend_default):
    inputs = np.array([1, 3, 2]).reshape((3, 1))
    outputs = np.array([1, 3, 2]).reshape((3, 1))
    compare_tensors(Rectlin(), inputs, outputs)


def test_rectlin_negatives(backend_default):
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0, 0], [0, 0]])
    compare_tensors(Rectlin(), inputs, outputs)


def test_rectlin_mixed(backend_default):
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [0, 9]])
    compare_tensors(Rectlin(), inputs, outputs)


def test_rectlin_derivative_positives(backend_default):
    inputs = np.array([1, 3, 2]).reshape((3, 1))
    outputs = np.array([1, 1, 1]).reshape((3, 1))
    compare_tensors(Rectlin(), inputs, outputs, deriv=True)


def test_rectlin_derivative_negatives(backend_default):
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0, 0], [0, 0]])
    compare_tensors(Rectlin(), inputs, outputs, deriv=True)


def test_rectlin_derivative_mixed(backend_default):
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0], [0, 1]])
    compare_tensors(Rectlin(), inputs, outputs, deriv=True)

"""Softmax
"""


def test_softmax(backend_default):
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.exp(inputs - 1) / np.sum(np.exp(inputs - 1))
    compare_tensors(Softmax(), inputs, outputs, tol=1e-7)


def test_softmax_derivative(backend_default):
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.ones((1, 1))  # shortcut only
    compare_tensors(Softmax(), inputs, outputs, deriv=True)


"""Tanh
"""


def test_tanh(backend_default):
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.array(
        [true_tanh(0), true_tanh(1), true_tanh(-2)]).reshape((3, 1))
    compare_tensors(Tanh(), inputs, outputs, tol=1e-7)


def test_tanh_derivative(backend_default):
    inputs = np.array(
        [true_tanh(0), true_tanh(1), true_tanh(-2)]).reshape((3, 1))
    # bprop is on the output
    outputs = np.array([1 - true_tanh(0) ** 2,
                        1 - true_tanh(1) ** 2,
                        1 - true_tanh(-2) ** 2]).reshape((3, 1))
    compare_tensors(Tanh(), inputs, outputs, deriv=True, tol=1e-7)

"""Logistic
"""


def test_logistic(backend_default):
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    compare_tensors(Logistic(), inputs, outputs, tol=1e-7)


def test_logistic_derivative(backend_default):
        # bprop is on the output
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    inputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = inputs * (1.0 - inputs)
    compare_tensors(Logistic(shortcut=False),
                    inputs, outputs, deriv=True, tol=1e-7)
