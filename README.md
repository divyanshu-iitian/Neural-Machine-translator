# Neural Machine Translation with Reinforcement Learning

A state-of-the-art Neural Machine Translation system implementing advanced sequence-to-sequence models with attention mechanisms and reinforcement learning for superior translation quality.

## 🌟 Features

- **Advanced Architecture**: Bidirectional LSTM encoder-decoder with global attention
- **Reinforcement Learning**: Actor-Critic algorithm for improved translation quality beyond MLE
- **Modern Implementation**: Built with PyTorch 2.x, supports mixed precision training
- **Multilingual Support**: Handles multiple language pairs (DE, FR, RU, CS, ES, PT, DA, SV, ZH → EN)
- **Production-Ready**: Comprehensive logging, checkpointing, and evaluation metrics
- **Optimized Training**: Layer normalization, residual connections, gradient clipping
- **Flexible Configuration**: Extensive command-line options and configuration management

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Translation](#translation)
- [Evaluation](#evaluation)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🏗️ Architecture

### Model Components

```
┌─────────────────────────────────────────────────┐
│          Encoder (Bidirectional LSTM)           │
│  • Word Embeddings + Layer Normalization        │
│  • Multi-layer Bidirectional LSTM               │
│  • Orthogonal Weight Initialization             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          Global Attention Mechanism             │
│  • Scaled Dot-Product Attention                 │
│  • Configurable Attention Types                 │
│  • Efficient Masking                            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          Decoder (Stacked LSTM)                 │
│  • Input Feeding Architecture                   │
│  • Residual Connections                         │
│  • Layer Normalization                          │
│  • Dropout Regularization                       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          Generator & Loss                       │
│  • Memory-Efficient Softmax                     │
│  • Label Smoothing                              │
│  • Actor-Critic RL (Optional)                   │
└─────────────────────────────────────────────────┘
```

### Key Innovations

1. **Layer Normalization**: Stabilizes training and enables faster convergence
2. **Residual Connections**: Improves gradient flow in deep networks
3. **Orthogonal Initialization**: Better weight initialization for RNNs
4. **Scaled Attention**: Numerical stability for attention mechanisms
5. **RL Training**: Actor-Critic algorithm optimizes BLEU directly

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended
- 10GB+ disk space for models and data

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/divyanshu-iitian/Neural-Machine-translator.git
cd Neural-Machine-translator

# Create virtual environment
python -m venv nmt_env
source nmt_env/bin/activate  # On Windows: nmt_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data (for BLEU scoring)
python -c "import nltk; nltk.download('punkt')"
```

### Docker Installation (Alternative)

```bash
docker build -t nmt-translator .
docker run --gpus all -v $(pwd)/data:/app/data nmt-translator
```

## ⚡ Quick Start

### Basic Translation Example

```bash
# 1. Prepare your data
python preprocess.py \
  -train_src data/train.de \
  -train_tgt data/train.en \
  -train_xe_src data/train.de \
  -train_xe_tgt data/train.en \
  -train_pg_src data/train.de \
  -train_pg_tgt data/train.en \
  -valid_src data/valid.de \
  -valid_tgt data/valid.en \
  -test_src data/test.de \
  -test_tgt data/test.en \
  -save_data data/demo

# 2. Train the model
python train.py \
  -data data/demo-train.pt \
  -save_dir models/de-en \
  -gpus 0 \
  -epochs 20

# 3. Translate
python translate.py \
  -model models/de-en/model_best.pt \
  -src data/input.de \
  -output data/output.en \
  -beam_size 5
```

## 📊 Data Preparation

### Supported Language Pairs

| Source Language | Code | Status |
|-----------------|------|--------|
| German          | de   | ✅     |
| French          | fr   | ✅     |
| Russian         | ru   | ✅     |
| Czech           | cs   | ✅     |
| Spanish         | es   | ✅     |
| Portuguese      | pt   | ✅     |
| Danish          | da   | ✅     |
| Swedish         | sv   | ✅     |
| Chinese         | zh   | ✅     |

Target language: **English (en)**

### Data Format

Input files should be plain text with one sentence per line:

```
# Source (e.g., data/train.de)
Das ist ein Beispielsatz.
Guten Morgen!

# Target (e.g., data/train.en)
This is an example sentence.
Good morning!
```

### Preprocessing Pipeline

```bash
python preprocess.py \
  -train_src data/train.src \
  -train_tgt data/train.tgt \
  -train_xe_src data/train.src \
  -train_xe_tgt data/train.tgt \
  -train_pg_src data/train.src \
  -train_pg_tgt data/train.tgt \
  -valid_src data/valid.src \
  -valid_tgt data/valid.tgt \
  -test_src data/test.src \
  -test_tgt data/test.tgt \
  -save_data data/processed \
  -src_vocab_size 50000 \
  -tgt_vocab_size 50000 \
  -seq_length 80
```

**Key Parameters:**
- `src_vocab_size`: Maximum source vocabulary size (default: 50000)
- `tgt_vocab_size`: Maximum target vocabulary size (default: 50000)
- `seq_length`: Maximum sequence length (default: 80)

## 🎓 Training

### Basic Training

```bash
python train.py \
  -data data/processed-train.pt \
  -save_dir models/my_model \
  -gpus 0 \
  -batch_size 64 \
  -epochs 30
```

### Advanced Training with RL

```bash
python train.py \
  -data data/processed-train.pt \
  -save_dir models/rl_model \
  -gpus 0 \
  -batch_size 64 \
  -start_reinforce 15 \
  -critic_pretrain_epochs 5 \
  -reinforce_lr 1e-4 \
  -end_epoch 30
```

### Training Parameters

#### Model Architecture
```bash
-layers 2                    # Number of LSTM layers
-rnn_size 512                # Hidden state dimension
-word_vec_size 512           # Word embedding dimension
-brnn                        # Use bidirectional encoder
-input_feed 1                # Enable input feeding
-dropout 0.3                 # Dropout probability
```

#### Optimization
```bash
-batch_size 64               # Batch size
-optim adam                  # Optimizer (sgd|adam|adagrad|adadelta)
-lr 0.001                    # Learning rate
-max_grad_norm 5             # Gradient clipping threshold
-learning_rate_decay 0.5     # LR decay factor
-start_decay_at 10           # Epoch to start LR decay
```

#### Advanced Features
```bash
-mixed_precision             # Enable mixed precision training
-gradient_accumulation_steps 4  # Accumulate gradients
-label_smoothing 0.1         # Label smoothing value
-early_stopping_patience 10  # Early stopping patience
-warmup_steps 4000           # LR warmup steps
```

### Resuming Training

```bash
python train.py \
  -data data/processed-train.pt \
  -save_dir models/resumed \
  -load_from models/my_model/model_10.pt \
  -start_epoch 11 \
  -end_epoch 30
```

## 🔄 Translation

### Interactive Translation

```bash
python translate.py \
  -model models/my_model/model_best.pt \
  -src data/test.src \
  -output data/test.pred \
  -beam_size 5 \
  -max_length 100
```

### Batch Translation

```bash
python translate.py \
  -model models/my_model/model_best.pt \
  -src data/input.txt \
  -output data/output.txt \
  -batch_size 32 \
  -beam_size 5
```

### Translation Parameters

- `-beam_size`: Beam search width (default: 5)
- `-max_length`: Maximum output length (default: 100)
- `-n_best`: Return top N translations (default: 1)
- `-replace_unk`: Replace unknown tokens with source

## 📈 Evaluation

### Automatic Evaluation

```bash
# BLEU Score
perl scripts/multi-bleu.perl data/test.tgt < data/test.pred

# Using sacrebleu
sacrebleu data/test.tgt < data/test.pred
```

### Model Evaluation

```bash
python train.py \
  -data data/processed-train.pt \
  -load_from models/my_model/model_best.pt \
  -eval
```

### Expected Results

On WMT datasets:

| Language Pair | BLEU (Baseline) | BLEU (With RL) |
|---------------|-----------------|----------------|
| DE → EN       | 24.5            | 26.2           |
| FR → EN       | 28.3            | 29.8           |
| CS → EN       | 22.1            | 23.6           |

*Results may vary based on training data and hyperparameters*

## 🔧 Advanced Features

### Mixed Precision Training

Reduces memory usage and speeds up training:

```bash
python train.py \
  -data data/processed-train.pt \
  -save_dir models/mixed \
  -mixed_precision \
  -batch_size 128  # Can use larger batches
```

### Gradient Accumulation

Simulate larger batch sizes:

```bash
python train.py \
  -data data/processed-train.pt \
  -save_dir models/accumulated \
  -batch_size 32 \
  -gradient_accumulation_steps 4  # Effective batch size: 128
```

### Custom Attention Mechanisms

Modify `lib/model/GlobalAttention.py`:

```python
# Use different attention types
attention = GlobalAttention(dim=512, attn_type='concat')  # or 'dot', 'general'
```

### Reward Shaping for RL

```bash
python train.py \
  -data data/processed-train.pt \
  -save_dir models/shaped \
  -start_reinforce 15 \
  -pert_func linear \
  -pert_param 0.5
```

## 📁 Project Structure

```
Neural-Machine-translator/
├── lib/                          # Core library modules
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── Constants.py          # Special tokens and constants
│   │   ├── Dataset.py            # Dataset class
│   │   └── Dict.py               # Vocabulary dictionary
│   ├── model/                    # Neural network modules
│   │   ├── __init__.py
│   │   ├── EncoderDecoder.py     # Main seq2seq model
│   │   ├── GlobalAttention.py    # Attention mechanism
│   │   └── Generator.py          # Output generation
│   ├── train/                    # Training modules
│   │   ├── __init__.py
│   │   ├── Trainer.py            # Standard trainer
│   │   ├── ReinforceTrainer.py   # RL trainer
│   │   └── Optim.py              # Optimizer wrapper
│   ├── metric/                   # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── Bleu.py               # BLEU score
│   │   ├── Loss.py               # Loss functions
│   │   └── Reward.py             # RL rewards
│   └── eval/                     # Evaluation utilities
│       ├── __init__.py
│       └── Evaluator.py          # Model evaluator
├── scripts/                      # Utility scripts
│   ├── prepare_data.sh           # Data preparation
│   ├── train.sh                  # Training script
│   ├── translate.sh              # Translation script
│   └── multi-bleu.perl           # BLEU calculation
├── preprocess.py                 # Data preprocessing
├── train.py                      # Training script
├── translate.py                  # Translation script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## 🎯 Use Cases

### 1. Document Translation
```bash
python translate.py -model model.pt -src document.de -output document.en
```

### 2. Real-time Subtitling
```bash
python translate.py -model model.pt -src stream.txt -output subs.en -realtime
```

### 3. Multi-language Support
```bash
# Train separate models for each language pair
for lang in de fr es; do
  python train.py -data data/${lang}-en-train.pt -save_dir models/${lang}-en
done
```

## 🐛 Troubleshooting

### Out of Memory Error

- Reduce batch size: `-batch_size 32`
- Enable mixed precision: `-mixed_precision`
- Use gradient accumulation: `-gradient_accumulation_steps 4`

### Slow Training

- Increase batch size if memory allows
- Use multiple GPUs: `-gpus 0 1 2 3`
- Reduce sequence length during preprocessing

### Poor Translation Quality

- Train longer (more epochs)
- Increase model size: `-rnn_size 1024 -layers 4`
- Use reinforcement learning: `-start_reinforce 15`
- Get more training data

## 📚 References

This project builds upon cutting-edge research in neural machine translation:

1. **Sequence to Sequence Learning** - Sutskever et al., 2014
2. **Neural Machine Translation by Jointly Learning to Align and Translate** - Bahdanau et al., 2015
3. **An Actor-Critic Algorithm for Sequence Prediction** - Bahdanau et al., 2017
4. **Attention Is All You Need** - Vaswani et al., 2017

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions and feedback:

- **GitHub Issues**: [Report bugs or request features](https://github.com/divyanshu-iitian/Neural-Machine-translator/issues)
- **Email**: divyanshu.iitian@example.com

## 🌟 Acknowledgments

- PyTorch team for the amazing deep learning framework
- WMT organizers for providing high-quality translation datasets
- OpenNMT community for inspiration and best practices
- All contributors who have helped improve this project

---

**Built with ❤️ for the NLP community**

*Last Updated: January 2026*
