import torchtext
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids

from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch
from transformers import BertTokenizer, AdamW, get_constant_schedule_with_warmup

from models import utils
from models.base_models import RNNSequenceModel, MLPModel, BERTSequenceModel
from models.vsm_models import VSMModel
from models.utils import make_prediction

logger = logging.getLogger('VSMLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SeqVSMNetwork(nn.Module):
    
    def __init__(self, config):
        super(SeqVSMNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)
        self.vsm_weight = config.get('vsm_weight', 1.0)

        if 'seq' in config['learner_model']:
            self.feature_extractor = RNNSequenceModel(config['learner_params'])
        elif 'mlp' in config['learner_model']:
            self.feature_extractor = MLPModel(config['learner_params'])
        elif 'bert' in config['learner_model']:
            self.feature_extractor = BERTSequenceModel(config['learner_params'])

        vsm_params = config['learner_params'].copy()
        vsm_params.update(config.get('vsm_params', {}))
        self.vsm = VSMModel(vsm_params)

        self.num_outputs = config['learner_params']['num_outputs']
        self.vectors = config.get('vectors', 'glove')

        if self.vectors == 'elmo':
            self.elmo = Elmo(
                options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                num_output_representations=1,
                dropout=0,
                requires_grad=False
            )
        elif self.vectors == 'glove':
            self.glove = torchtext.vocab.GloVe(name='840B', dim=300)
        elif self.vectors == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.loss_fn = {}
        for task in config['learner_params']['num_outputs']:
            self.loss_fn[task] = nn.CrossEntropyLoss(ignore_index=-1)

        if config.get('trained_learner', False):
            self.feature_extractor.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)

        if self.vectors == 'elmo':
            self.elmo.to(self.device)

        self.initialize_optimizer_scheduler()

    def initialize_optimizer_scheduler(self):
        learner_params = [p for p in self.feature_extractor.parameters() if p.requires_grad] + \
                        [p for p in self.vsm.parameters() if p.requires_grad]
        
        if isinstance(self.feature_extractor, BERTSequenceModel):
            self.optimizer = AdamW(learner_params, lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)
        else:
            self.optimizer = optim.Adam(learner_params, lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
            if self.vectors == 'elmo':
                char_ids = batch_to_ids(batch_x)
                char_ids = char_ids.to(self.device)
                batch_x = self.elmo(char_ids)['elmo_representations'][0]
            elif self.vectors == 'glove':
                max_batch_len = max(batch_len)
                vec_batch_x = torch.ones((len(batch_x), max_batch_len, 300))
                for i, sent in enumerate(batch_x):
                    sent_emb = self.glove.get_vecs_by_tokens(sent, lower_case_backup=True)
                    vec_batch_x[i, :len(sent_emb)] = sent_emb
                batch_x = vec_batch_x.to(self.device)
            elif self.vectors == 'bert':
                max_batch_len = max(batch_len) + 2
                input_ids = torch.zeros((len(batch_x), max_batch_len)).long()
                for i, sent in enumerate(batch_x):
                    sent_token_ids = self.bert_tokenizer.encode(sent, add_special_tokens=True)
                    input_ids[i, :len(sent_token_ids)] = torch.tensor(sent_token_ids)
                batch_x = input_ids.to(self.device)

        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def compute_prototypes(self, features, labels, n_classes):
        batch_size, seq_len, feature_dim = features.shape
        features_flat = features.view(-1, feature_dim)
        labels_flat = labels.view(-1)
        
        prototypes = torch.zeros(n_classes, feature_dim).to(self.device)
        
        for c in range(n_classes):
            mask = (labels_flat == c)
            if mask.sum() > 0:
                class_features = features_flat[mask]
                prototypes[c] = class_features.mean(dim=0)
        
        return prototypes

    def compute_distances(self, query_features, prototypes):
        batch_size, seq_len, feature_dim = query_features.shape
        n_classes = prototypes.shape[0]
        
        query_flat = query_features.view(-1, feature_dim)  # [batch_size * seq_len, feature_dim]

        distances = torch.cdist(query_flat, prototypes)  # [batch_size * seq_len, n_classes]
        distances = distances.view(batch_size, seq_len, n_classes)
        
        similarities = -distances
        return similarities

    def forward(self, episodes, updates=1, testing=False):

        query_losses, query_accuracies, query_precisions, query_recalls, query_f1s = [], [], [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):
            if not testing:
                self.train()
            else:
                self.eval()

            support_features_list = []
            support_labels_list = []
            
            for batch_x, batch_len, batch_y in episode.support_loader:
                batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                
                base_features = self.feature_extractor(batch_x, batch_len)
                
                vsm_features, vsm_losses = self.vsm(base_features, batch_len, compute_loss=not testing)
                
                support_features_list.append(vsm_features)
                support_labels_list.append(batch_y)

            support_features = torch.cat(support_features_list, dim=0)
            support_labels = torch.cat(support_labels_list, dim=0)

            prototypes = self.compute_prototypes(support_features, support_labels, episode.n_classes)

            all_predictions, all_labels = [], []
            query_loss = 0.0
            total_vsm_loss = 0.0
            n_batches = 0

            for batch_x, batch_len, batch_y in episode.query_loader:
                batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                
                base_features = self.feature_extractor(batch_x, batch_len)
                
                vsm_features, vsm_losses = self.vsm(base_features, batch_len, compute_loss=not testing)
                
                logits = self.compute_distances(vsm_features, prototypes)
                
                logits_flat = logits.view(-1, episode.n_classes)
                labels_flat = batch_y.view(-1)
                
                classification_loss = self.loss_fn[episode.base_task](logits_flat, labels_flat)
                
                total_loss = classification_loss
                if not testing and vsm_losses:
                    total_vsm_loss += vsm_losses['total_loss']
                    total_loss = total_loss + self.vsm_weight * vsm_losses['total_loss']

                if not testing:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                query_loss += classification_loss.item()
                n_batches += 1

                relevant_indices = torch.nonzero(labels_flat != -1).view(-1).detach()
                if len(relevant_indices) > 0:
                    pred = make_prediction(logits_flat[relevant_indices].detach()).cpu()
                    all_predictions.extend(pred)
                    all_labels.extend(labels_flat[relevant_indices].cpu())

            query_loss /= n_batches
            total_vsm_loss /= n_batches

            accuracy, precision, recall, f1_score = utils.calculate_metrics(
                all_predictions, all_labels, binary=False
            )

            logger.info('Episode {}/{}, task {} [{}]: Loss = {:.5f}, VSM Loss = {:.5f}, '
                       'accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, F1 score = {:.5f}'.format(
                episode_id + 1, n_episodes, episode.task_id, 
                'test' if testing else 'train', query_loss, total_vsm_loss,
                accuracy, precision, recall, f1_score
            ))

            query_losses.append(query_loss)
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1_score)

            if not testing:
                self.lr_scheduler.step()

        return query_losses, query_accuracies, query_precisions, query_recalls, query_f1s

    def get_memory_attention(self, batch_x, batch_len, batch_y):
    
        self.eval()
        with torch.no_grad():
            batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
            base_features = self.feature_extractor(batch_x, batch_len)
            attention_weights = self.vsm.get_attention_weights(base_features, batch_len)
            return attention_weights 