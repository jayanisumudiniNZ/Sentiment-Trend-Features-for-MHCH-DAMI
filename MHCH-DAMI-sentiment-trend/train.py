#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train DAMI with Path A sentiment-trend features (extension entry point)."""

import argparse
import logging
import os
import pickle as pkl
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from path_a.bootstrap import (
    get_data_dir,
    get_extension_config_dir,
    get_logs_dir,
    get_mhch_config_dir,
    init_environment,
    patch_network_artifact_dirs,
)
from path_a.integrations.dami import DAMI
from path_a.integrations.data_loader import Data_loader

init_environment()

import tf_compat as tf

random_seed = 7
tf.set_random_seed(random_seed)


def main():
    parser = argparse.ArgumentParser('MHCH-DAMI Path A — train')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--data_name', default='clothing')
    parser.add_argument('--model_name', default='dami')
    parser.add_argument('--memory', default='0')
    parser.add_argument('--log_path', default=None)
    parser.add_argument('--suffix', default='.128.trend')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--ways', default='dami')
    parser.add_argument('--trend_features', default='full',
                        help='baseline|full|pol_only|slope3_only|slope5_only|slope7_only|volatility5_only')
    args = parser.parse_args()

    log_dir = args.log_path or get_logs_dir()
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('PathA-Train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' ')[:3])
    log_path = os.path.join(
        log_dir,
        '{}.{}.train{}.{}.{}.log'.format(
            args.model_name, args.data_name, args.suffix, args.mode, args.ways, now_time),
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    tf_logger = logging.getLogger('Tensorflow')
    tf_logger.setLevel(logging.INFO)
    tf_logger.addHandler(fh)
    tf_logger.addHandler(logging.StreamHandler())

    data_config_path = os.path.join(get_mhch_config_dir(), 'data', 'config.' + args.data_name + '.json')
    model_config_path = os.path.join(get_extension_config_dir(), 'model', 'config.' + args.model_name + '.json')
    if not os.path.isfile(model_config_path):
        model_config_path = os.path.join(get_mhch_config_dir(), 'model', 'config.' + args.model_name + '.json')

    data_loader = Data_loader(data_name=args.data_name)
    data_config = data_loader.load_config(data_config_path)
    model_config = data_loader.load_config(model_config_path)
    if args.trend_features:
        model_config['trend_features'] = args.trend_features
    data_loader.trend_features = model_config.get('trend_features', 'full')

    logger.info('MHCH-DAMI data: %s', get_data_dir())
    logger.info('Args: %s', args)
    logger.info('Model config: %s', model_config)

    vocab_path = os.path.join(get_data_dir(), args.data_name, 'vocab.pkl')
    with open(vocab_path, 'rb') as fp:
        vocab = pkl.load(fp)

    network = DAMI(memory=float(args.memory), vocab=vocab, config_dict=model_config)
    network.set_nb_words(min(vocab.size(), data_config['nb_words']) + 1)
    network.set_data_name(args.data_name)
    network.set_name(args.model_name + args.suffix + 'train')
    network.set_from_model_config(model_config)
    network.set_from_data_config(data_config)
    network.build_graph()
    patch_network_artifact_dirs(network)

    if args.ways != 'dami':
        raise ValueError('Path A requires --ways dami')
    network.train(
        data_generator=data_loader.data_generator_m,
        keep_prob=model_config['keep_prob'],
        epochs=model_config['epochs'],
        data_name=args.data_name,
        mode=args.mode,
        batch_size=model_config['batch_size'],
        nb_classes=data_config['nb_classes'],
        shuffle=model_config['shuffle'],
        is_val=model_config['is_val'],
        is_test=model_config['is_test'],
        save_best=model_config['save_best'],
    )
    print('DONE! Logs:', log_path)


if __name__ == '__main__':
    main()
