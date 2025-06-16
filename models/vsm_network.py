import coloredlogs
import logging
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.seq_vsm import SeqVSMNetwork

logger = logging.getLogger('VSMNetworkLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
tensorboard_writer = SummaryWriter(log_dir='runs/VSMNet')


class VSMNetwork:

    
    def __init__(self, config):
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.updates = config['num_updates']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        self.stopping_threshold = config.get('stopping_threshold', 1e-3)

        if 'seq' in config['meta_model']:
            self.vsm_model = SeqVSMNetwork(config)

        logger.info('VSM network instantiated')

    def training(self, train_episodes, val_episodes):

        best_loss = float('inf')
        best_f1 = 0
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'VSMNet-{}.h5'.format(self.stamp))
        logger.info('Model name: VSMNet-{}.h5'.format(self.stamp))
        
        for epoch in range(self.meta_epochs):
            logger.info('Starting epoch {}'.format(epoch + 1))
            
            losses, accuracies, precisions, recalls, f1s = self.vsm_model(train_episodes, self.updates)
            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)

            logger.info('Meta train epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(
                            epoch + 1, avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1
                        ))

            tensorboard_writer.add_scalar('Loss/train', avg_loss, global_step=epoch + 1)
            tensorboard_writer.add_scalar('Accuracy/train', avg_accuracy, global_step=epoch + 1)
            tensorboard_writer.add_scalar('F1/train', avg_f1, global_step=epoch + 1)

            losses, accuracies, precisions, recalls, f1s = self.vsm_model(val_episodes, self.updates, testing=True)

            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)

            logger.info('Meta val epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(
                            epoch + 1, avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1
                        ))

            tensorboard_writer.add_scalar('Loss/val', avg_loss, global_step=epoch + 1)
            tensorboard_writer.add_scalar('Accuracy/val', avg_accuracy, global_step=epoch + 1)
            tensorboard_writer.add_scalar('F1/val', avg_f1, global_step=epoch + 1)

            if avg_f1 > best_f1 + self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                torch.save({
                    'feature_extractor': self.vsm_model.feature_extractor.state_dict(),
                    'vsm': self.vsm_model.vsm.state_dict()
                }, model_path)
                logger.info('Saving the model since the F1 improved')
                logger.info('')
            else:
                patience += 1
                logger.info('F1 did not improve')
                logger.info('')
                if patience == self.early_stopping:
                    break

            for name, param in self.vsm_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    tensorboard_writer.add_histogram('Params/' + name, param.data.view(-1),
                                                     global_step=epoch + 1)
                    tensorboard_writer.add_histogram('Grads/' + name, param.grad.data.view(-1),
                                                     global_step=epoch + 1)
            
            self._log_vsm_metrics(epoch + 1)

        checkpoint = torch.load(model_path)
        self.vsm_model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.vsm_model.vsm.load_state_dict(checkpoint['vsm'])
        
        return best_f1

    def testing(self, test_episodes):

        logger.info('---------- VSM testing starts here ----------')
        episode_accuracies, episode_precisions, episode_recalls, episode_f1s = [], [], [], []
        
        for episode in test_episodes:
            _, accuracy, precision, recall, f1_score = self.vsm_model([episode], updates=self.updates, testing=True)
            accuracy, precision, recall, f1_score = accuracy[0], precision[0], recall[0], f1_score[0]

            episode_accuracies.append(accuracy)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episode_f1s.append(f1_score)

        avg_accuracy = np.mean(episode_accuracies)
        avg_precision = np.mean(episode_precisions)
        avg_recall = np.mean(episode_recalls)
        avg_f1 = np.mean(episode_f1s)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(avg_accuracy, avg_precision, avg_recall, avg_f1))
        
        # 记录测试结果到tensorboard
        tensorboard_writer.add_scalar('Test/Accuracy', avg_accuracy, global_step=0)
        tensorboard_writer.add_scalar('Test/Precision', avg_precision, global_step=0)
        tensorboard_writer.add_scalar('Test/Recall', avg_recall, global_step=0)
        tensorboard_writer.add_scalar('Test/F1', avg_f1, global_step=0)
        
        return avg_f1

    def _log_vsm_metrics(self, step):

        with torch.no_grad():
            memory = self.vsm_model.vsm.memory.memory.cpu().numpy()
            
            memory_norms = np.linalg.norm(memory, axis=1)
            tensorboard_writer.add_histogram('VSM/Memory_Norms', memory_norms, global_step=step)
            tensorboard_writer.add_scalar('VSM/Memory_Norm_Mean', memory_norms.mean(), global_step=step)
            tensorboard_writer.add_scalar('VSM/Memory_Norm_Std', memory_norms.std(), global_step=step)
            
            normalized_memory = memory / (np.linalg.norm(memory, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = np.dot(normalized_memory, normalized_memory.T)
            
            mask = np.eye(similarity_matrix.shape[0], dtype=bool)
            off_diagonal_similarities = similarity_matrix[~mask]
            
            tensorboard_writer.add_histogram('VSM/Memory_Similarities', off_diagonal_similarities, global_step=step)
            tensorboard_writer.add_scalar('VSM/Memory_Similarity_Mean', off_diagonal_similarities.mean(), global_step=step)
            tensorboard_writer.add_scalar('VSM/Memory_Similarity_Std', off_diagonal_similarities.std(), global_step=step)

    def analyze_memory_attention(self, test_episodes):

        logger.info('---------- Analyzing memory attention patterns ----------')
        
        attention_patterns = []
        word_tasks = []
        
        for episode in test_episodes[:10]:
            for batch_x, batch_len, batch_y in episode.support_loader:
                attention_weights = self.vsm_model.get_memory_attention(batch_x, batch_len, batch_y)
                attention_patterns.append(attention_weights.cpu().numpy())
                word_tasks.append(episode.task_id)
                break
        
        attention_data = {
            'patterns': attention_patterns,
            'tasks': word_tasks
        }
        
        analysis_path = os.path.join(self.base_path, 'analysis', 'vsm_attention_patterns.npy')
        os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
        np.save(analysis_path, attention_data)
        
        logger.info('Attention patterns saved to {}'.format(analysis_path))
        
        avg_attention = np.mean([pattern.mean(axis=(0, 1)) for pattern in attention_patterns], axis=0)
        
        for i, att_val in enumerate(avg_attention):
            tensorboard_writer.add_scalar(f'VSM/Memory_Attention_Slot_{i}', att_val, global_step=0)
        
        return attention_data

    def save_model(self, path=None):

        if path is None:
            path = os.path.join(self.base_path, 'saved_models', 'VSMNet-final-{}.h5'.format(self.stamp))
        
        torch.save({
            'feature_extractor': self.vsm_model.feature_extractor.state_dict(),
            'vsm': self.vsm_model.vsm.state_dict(),
            'config': {
                'vectors': self.vsm_model.vectors,
                'device': str(self.vsm_model.device)
            }
        }, path)
        
        logger.info('Model saved to {}'.format(path))

    def load_model(self, path):

        checkpoint = torch.load(path, map_location=self.vsm_model.device)
        self.vsm_model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.vsm_model.vsm.load_state_dict(checkpoint['vsm'])
        
        logger.info('Model loaded from {}'.format(path)) 