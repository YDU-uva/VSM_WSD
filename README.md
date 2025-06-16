# Meta-Learning with Variational Semantic Memory for Word Sense Disambiguation

This is the official implementation of the paper "Meta-Learning with Variational Semantic Memory for Word Sense Disambiguation". https://arxiv.org/pdf/2106.02960



## Requirements

The code is implemented in Python 3.7+ and PyTorch. Install the required dependencies:

```bash
pip install torch transformers allennlp torchtext coloredlogs tensorboard numpy pyyaml
```

## Data Preparation

Place your WSD dataset in the following structure:
```
../data/semcor_meta/
├── meta_train_4-4/
├── meta_val_4-4/
└── meta_test_4-4/
```

## Quick Start

### Training with Demo Mode (Recommended for First Run)

```bash
# Train with GloVe embeddings (faster)
python train_vsm_demo.py --config config/wsd/vsm_net/vsm_glove_4.yaml --demo_mode

# Train with BERT embeddings
python train_vsm_demo.py --config config/wsd/vsm_net/vsm_bert_4.yaml --demo_mode
```

### Full Training

```bash
# Train VSM model with GloVe embeddings
python train_wsd.py --config config/wsd/vsm_net/vsm_glove_4.yaml

# Train VSM model with BERT embeddings  
python train_wsd.py --config config/wsd/vsm_net/vsm_bert_4.yaml
```

### Using the Original Training Script

You can also use the main training script:

```bash
python train_wsd.py --config config/wsd/vsm_net/vsm_glove_4.yaml
```

## Configuration

The model hyperparameters can be configured in the YAML files:

- `config/wsd/vsm_net/vsm_glove_4.yaml`: Configuration for GloVe embeddings
- `config/wsd/vsm_net/vsm_bert_4.yaml`: Configuration for BERT embeddings

### Key Parameters

- `latent_dim`: Dimensionality of the latent space (default: 128)
- `memory_size`: Number of memory slots (default: 512)  
- `beta`: Weight for KL divergence loss (default: 1.0)
- `vsm_weight`: Weight for VSM loss in total loss (default: 0.5)

## Monitoring Training

### TensorBoard Visualization

```bash
tensorboard --logdir runs/VSMNet
```

This provides visualization of:
- Training/validation loss and metrics
- VSM-specific metrics (KL loss, memory statistics)
- Parameter and gradient distributions
- Attention weight analysis

### Model Analysis

To analyze semantic memory attention patterns:

```bash
python train_vsm_demo.py --config config/wsd/vsm_net/vsm_glove_4.yaml --analyze_attention
```

## Testing the Implementation

Run the test script to verify the model implementation:

```bash
python test_vsm.py
```

## File Structure

```
models/
├── vsm_models.py          # Core VSM model implementation
├── seq_vsm.py            # Sequential VSM network
├── vsm_network.py        # Main VSM network interface
└── ...

config/wsd/vsm_net/
├── vsm_glove_4.yaml      # GloVe configuration
└── vsm_bert_4.yaml       # BERT configuration

train_vsm_demo.py         # Demo training script
train_wsd.py             # Main training script
test_vsm.py              # Model testing script
```

## Expected Results

The VSM model should achieve competitive performance on few-shot word sense disambiguation tasks. Training logs and model checkpoints will be saved in the `saved_models/` directory.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use `--demo_mode` for smaller datasets
2. **Data not found**: Ensure the data is prepared according to the directory structure above
3. **Import errors**: Install all required dependencies listed in Requirements

### Debug Mode

Use demo mode for quick testing with reduced data:

```bash
python train_vsm_demo.py --demo_mode
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{du2021meta,
  title={Meta-learning with variational semantic memory for word sense disambiguation},
  author={Du, Yingjun and Holla, Nithin and Zhen, Xiantong and Snoek, Cees GM and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2106.02960},
  year={2021}
}
```
