#!/usr/bin/env python3
"""
Official implementation demo script for:
"Meta-Learning with Variational Semantic Memory for Word Sense Disambiguation"

This script provides a convenient way to train and test the VSM model
with various configurations and analysis options.
"""


import logging
import os
import random
import warnings
from argparse import ArgumentParser

import coloredlogs
import torch
import yaml

from datetime import datetime
from datasets import utils
from models.vsm_network import VSMNetwork

logger = logging.getLogger('VSMDemoLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_file):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['base_path'] = os.path.dirname(os.path.abspath(__file__))
    config['stamp'] = str(datetime.now()).replace(':', '-').replace(' ', '_')
    return config


def print_vsm_info():
    print("=" * 80)
    print("VSM (Variational Semantic Memory) for Word Sense Disambiguation")
    print("=" * 80)
    print("VSM :")
    print("1. Variational Encoder: Encode input text to a probability distribution in the latent space")
    print("2. Semantic Memory Module: Maintain learnable semantic prototypes and access them through attention mechanisms")
    print("3. Variational Decoder: Decode the enhanced latent representation to output features")
    print("4. Meta-learning framework: Combine few-shot learning and prototype networks for word sense disambiguation")
    print()
    print("Key Features:")
    print("- Uncertainty modeling based on variational inference")
    print("- Dynamic semantic memory update mechanism")
    print("- Multi-level regularization losses")
    print("- Interpretability of attention weights")
    print("=" * 80)


def main():
    print_vsm_info()
    

    parser = ArgumentParser(description='VSM model training demo')
    parser.add_argument('--config', dest='config_file', type=str, 
                       default='config/wsd/vsm_net/vsm_glove_4.yaml',
                       help='Configuration file path')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs for training')
    parser.add_argument('--analyze_attention', action='store_true', 
                       help='Analyze the attention pattern of semantic memory')
    parser.add_argument('--demo_mode', action='store_true', 
                       help='Demo mode (use less data for quick testing)')
    args = parser.parse_args()


    config = load_config(args.config_file)
    config['multi_gpu'] = args.multi_gpu
    
    if args.demo_mode:
        logger.info('Running in demo mode')
        config['num_train_episodes']['wsd'] = 100
        config['num_val_episodes']['wsd'] = 20
        config['num_test_episodes']['wsd'] = 20
        config['num_meta_epochs'] = 5
        config['early_stopping'] = 2
    
    logger.info('Using configuration: {}'.format(config))

    torch.manual_seed(42)
    random.seed(42)

    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(config['base_path'], 'analysis'), exist_ok=True)

    wsd_base_path = os.path.join(config['base_path'], '../data/semcor_meta/')
    wsd_train_path = os.path.join(wsd_base_path, 'meta_train_' + str(config['num_shots']['wsd']) + '-' +
                                  str(config['num_test_samples']['wsd']))
    wsd_val_path = os.path.join(wsd_base_path, 'meta_val_' + str(config['num_shots']['wsd']) + '-' +
                                str(config['num_test_samples']['wsd']))
    wsd_test_path = os.path.join(wsd_base_path, 'meta_test_' + str(config['num_shots']['wsd']) + '-' +
                                 str(config['num_test_samples']['wsd']))

    logger.info('Generating WSD episodes')
    try:
        wsd_train_episodes = utils.generate_wsd_episodes(
            dir=wsd_train_path,
            n_episodes=config['num_train_episodes']['wsd'],
            n_support_examples=config['num_shots']['wsd'],
            n_query_examples=config['num_test_samples']['wsd'],
            task='wsd',
            meta_train=True
        )
        
        wsd_val_episodes = utils.generate_wsd_episodes(
            dir=wsd_val_path,
            n_episodes=config['num_val_episodes']['wsd'],
            n_support_examples=config['num_shots']['wsd'],
            n_query_examples=config['num_test_samples']['wsd'],
            task='wsd',
            meta_train=False
        )
        
        wsd_test_episodes = utils.generate_wsd_episodes(
            dir=wsd_test_path,
            n_episodes=config['num_test_episodes']['wsd'],
            n_support_examples=config['num_shots']['wsd'],
            n_query_examples=config['num_test_samples']['wsd'],
            task='wsd',
            meta_train=False
        )
        
        logger.info('Successfully generated {} training episodes, {} validation episodes, {} test episodes'.format(
            len(wsd_train_episodes), len(wsd_val_episodes), len(wsd_test_episodes)
        ))
        
    except Exception as e:
        logger.error('Failed to generate episodes: {}'.format(e))
        logger.info('Possible reasons: data path does not exist or data format problem')
        logger.info('Please ensure that the data is prepared according to the instructions in README.md')
        return

    logger.info('Initializing VSM network')
    vsm_network = VSMNetwork(config)
    

    logger.info('VSM model parameter statistics:')
    total_params = sum(p.numel() for p in vsm_network.vsm_model.parameters())
    trainable_params = sum(p.numel() for p in vsm_network.vsm_model.parameters() if p.requires_grad)
    logger.info('Total number of parameters: {:,}'.format(total_params))
    logger.info('Trainable number of parameters: {:,}'.format(trainable_params))
    

    logger.info('VSM hyperparameters:')
    logger.info('- Latent dimension: {}'.format(config['vsm_params']['latent_dim']))
    logger.info('- Memory size: {}'.format(config['vsm_params']['memory_size']))
    logger.info('- KL loss weight: {}'.format(config['vsm_params']['beta']))
    logger.info('- VSM loss weight: {}'.format(config['vsm_weight']))

    try:

        logger.info('Starting VSM meta-learning training')
        best_f1 = vsm_network.training(wsd_train_episodes, wsd_val_episodes)
        logger.info('Meta-learning training completed, best F1 score: {:.4f}'.format(best_f1))

        logger.info('Starting VSM meta-learning testing')
        test_f1 = vsm_network.testing(wsd_test_episodes)
        logger.info('Meta-learning testing completed, test F1 score: {:.4f}'.format(test_f1))

        if args.analyze_attention:
            logger.info('Analyzing the attention pattern of semantic memory')
            attention_data = vsm_network.analyze_memory_attention(wsd_test_episodes)
            logger.info('Attention analysis completed, data saved')

        # 保存最终模型
        vsm_network.save_model()
        
        # 总结
        logger.info('=' * 60)
        logger.info('VSM training summary:')
        logger.info('Best F1 on validation set: {:.4f}'.format(best_f1))
        logger.info('Test F1: {:.4f}'.format(test_f1))
        logger.info('Model saved to saved_models directory')
        if args.analyze_attention:
            logger.info('Attention analysis results saved to analysis directory')
        logger.info('=' * 60)
        
    except Exception as e:
        logger.error('Error occurred during training: {}'.format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 