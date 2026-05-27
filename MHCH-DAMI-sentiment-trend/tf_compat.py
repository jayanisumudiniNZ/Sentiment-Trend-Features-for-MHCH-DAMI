"""TF 1.x compatibility shim for TensorFlow 2.x (Path A extension).

Import this module *before* any other TF-dependent module to disable eager
execution and re-expose removed TF 1.x symbols so legacy code runs unmodified.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --- Random seed ---
if not hasattr(tf, 'set_random_seed'):
    tf.set_random_seed = tf.compat.v1.set_random_seed

# --- Core graph-mode symbols ---
if not hasattr(tf, 'variable_scope'):
    tf.variable_scope = tf.compat.v1.variable_scope
if not hasattr(tf, 'get_variable'):
    tf.get_variable = tf.compat.v1.get_variable
if not hasattr(tf, 'AUTO_REUSE'):
    tf.AUTO_REUSE = tf.compat.v1.AUTO_REUSE
if not hasattr(tf, 'placeholder'):
    tf.placeholder = tf.compat.v1.placeholder

# --- Casting helpers ---
if not hasattr(tf, 'to_float'):
    tf.to_float = lambda x, name='ToFloat': tf.cast(x, tf.float32, name=name)
if not hasattr(tf, 'to_int32'):
    tf.to_int32 = lambda x, name='ToInt32': tf.cast(x, tf.int32, name=name)

# --- tf.layers ---
if not hasattr(tf, 'layers') or not hasattr(tf.layers, 'dense'):
    tf.layers = tf.compat.v1.layers

# --- tf.GPUOptions ---
if not hasattr(tf, 'GPUOptions'):
    tf.GPUOptions = tf.compat.v1.GPUOptions

# --- tf.nn functions removed in TF 2.x ---
if not hasattr(tf.nn, 'bidirectional_dynamic_rnn'):
    tf.nn.bidirectional_dynamic_rnn = tf.compat.v1.nn.bidirectional_dynamic_rnn
if not hasattr(tf.nn, 'dynamic_rnn'):
    tf.nn.dynamic_rnn = tf.compat.v1.nn.dynamic_rnn

# --- tf.contrib.rnn / tf.contrib.layers ---
if not hasattr(tf, 'contrib'):
    class _ContribRNN:
        BasicLSTMCell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell
        LSTMCell = tf.compat.v1.nn.rnn_cell.LSTMCell
        GRUCell = tf.compat.v1.nn.rnn_cell.GRUCell
        DropoutWrapper = tf.compat.v1.nn.rnn_cell.DropoutWrapper
        MultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell

    class _ContribLayers:
        @staticmethod
        def xavier_initializer():
            return tf.initializers.GlorotUniform()

    class _Contrib:
        rnn = _ContribRNN()
        layers = _ContribLayers()

    tf.contrib = _Contrib()

# --- tf.linalg aliases removed from top-level in TF 2.x ---
if not hasattr(tf, 'matrix_band_part'):
    tf.matrix_band_part = tf.linalg.band_part

# --- Global variables ---
if not hasattr(tf, 'all_variables'):
    tf.all_variables = tf.compat.v1.global_variables

# Replace this module with the patched tf so `import tf_compat as tf` works
sys.modules[__name__] = tf
