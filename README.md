# BPE-ELMo: Contextual Financial Embedding Pipeline

An implementation of a custom **BPE-Tokenized ELMo (Embeddings from Language Models)** network trained from scratch on financial sentiment text. This project demonstrates how bidirectional LSTMs capture context-dependent word meanings, bypassing the limitations of traditional static embedding models.

The repository includes a comparative evaluation framework that benchmarks this architecture against an industry-standard, authenticated language model (**ProsusAI/FinBERT**).


## Key Features

* **Custom BPE Tokenizer:** Built using the Hugging Face `tokenizers` library to break text into highly frequent subwords, optimizing vocabulary size for smaller domain-specific datasets.
* **Deep BiLSTM Architecture:** A multi-layer bidirectional LSTM network that processes left-to-right and right-to-left contexts simultaneously.
* **Smart Checkpoint Loading:** Automatic saving and loading mechanisms (`financial_elmo.pt`) to prevent unnecessary CPU retraining cycles.
* **Interactive Verification Suite:** A command-line program to evaluate custom user-defined sentences and compute mathematical contextual cosine similarity scores side-by-side with FinBERT.


## 📂 Repository Structure

* `train_elmo.py`: Handles raw financial text compilation, Byte-Pair Encoding (BPE) vocabulary construction, PyTorch data batching, and the unsupervised language modeling training loop.
* `verify_elmo.py`: A modular pipeline that loads the trained weights, executes inference on real-time user-provided inputs, and runs the comparative analysis against industry baselines.

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have Python installed along with the required deep learning dependencies:

```bash
pip install torch pandas tokenizers transformers
