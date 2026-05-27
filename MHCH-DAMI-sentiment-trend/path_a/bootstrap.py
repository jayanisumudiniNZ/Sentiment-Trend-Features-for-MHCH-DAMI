#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Link Path A extension to upstream MHCH-DAMI (main branch).

Set MHCH_DAMI_ROOT to override the default sibling path ../MHCH-DAMI
"""

import os
import sys
import importlib.util

_EXTENSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MHCH_DAMI_ROOT = None
_INITIALIZED = False


def get_extension_root():
    return _EXTENSION_ROOT


def get_mhch_dami_root():
    global _MHCH_DAMI_ROOT
    if _MHCH_DAMI_ROOT is None:
        _MHCH_DAMI_ROOT = os.path.abspath(
            os.environ.get(
                'MHCH_DAMI_ROOT',
                os.path.join(_EXTENSION_ROOT, '..', 'MHCH-DAMI'),
            ))
    if not os.path.isdir(_MHCH_DAMI_ROOT):
        raise FileNotFoundError(
            'MHCH-DAMI not found at {}. Clone upstream or set MHCH_DAMI_ROOT.'.format(
                _MHCH_DAMI_ROOT))
    return _MHCH_DAMI_ROOT


def get_data_dir():
    return os.path.join(get_mhch_dami_root(), 'data')


def get_mhch_config_dir():
    return os.path.join(get_mhch_dami_root(), 'config')


def get_extension_config_dir():
    return os.path.join(_EXTENSION_ROOT, 'config')


def get_weights_dir():
    return os.path.join(_EXTENSION_ROOT, 'weights')


def get_logs_dir():
    return os.path.join(_EXTENSION_ROOT, 'networks', 'logs')


def get_results_dir():
    return os.path.join(_EXTENSION_ROOT, 'results')


def _install_tf_compat_shim():
    """Expose extension tf_compat before MHCH modules import tensorflow."""
    compat_path = os.path.join(_EXTENSION_ROOT, 'tf_compat.py')
    if not os.path.isfile(compat_path):
        return
    spec = importlib.util.spec_from_file_location('tf_compat', compat_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if 'tf_compat' not in sys.modules:
        sys.modules['tf_compat'] = module


def init_environment():
    """Insert MHCH-DAMI on sys.path and register tf_compat. Idempotent."""
    global _INITIALIZED
    if _INITIALIZED:
        return get_mhch_dami_root(), get_extension_root()

    ext = get_extension_root()
    mhch = get_mhch_dami_root()

    if ext not in sys.path:
        sys.path.insert(0, ext)
    if mhch not in sys.path:
        sys.path.insert(0, mhch)

    _install_tf_compat_shim()
    _INITIALIZED = True
    return mhch, ext


def patch_network_artifact_dirs(network):
    """Store checkpoints and logs in the extension project, not upstream."""
    ext = get_extension_root()
    weights_base = os.path.join(get_weights_dir(), network.data_name)
    os.makedirs(weights_base, exist_ok=True)
    network.save_dir = os.path.join(weights_base, network.model_name + os.sep)
    os.makedirs(network.save_dir, exist_ok=True)
    import tf_compat as tf
    network.saver = tf.compat.v1.train.Saver()
