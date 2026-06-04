# BPE-ELMo: Contextual Financial Embedding Pipeline

An implementation of a custom **BPE-Tokenized ELMo (Embeddings from Language Models)** network trained from scratch on financial sentiment text. This project demonstrates how bidirectional LSTMs capture context-dependent word meanings, bypassing the limitations of traditional static embedding models.

The repository includes a comparative evaluation framework that benchmarks this architecture against an industry-standard, authenticated language model (**ProsusAI/FinBERT**).


## Key Features

* **Custom BPE Tokenizer:** Built using the Hugging Face `tokenizers` library to break text into highly frequent subwords, optimizing vocabulary size for smaller domain-specific datasets.
* **Deep BiLSTM Architecture:** A 2-layer bidirectional LSTM network that processes left-to-right and right-to-left contexts simultaneously.
* **Smart Checkpoint Loading:** Automatic saving and loading mechanisms (`financial_elmo.pt`) to prevent unnecessary CPU retraining cycles.
* **Interactive Verification Suite:** A command-line program to evaluate custom user-defined sentences and compute mathematical contextual cosine similarity scores side-by-side with FinBERT.


## Repository Structure

* `train_elmo.py`: Handles raw financial text compilation, Byte-Pair Encoding (BPE) vocabulary construction, PyTorch data batching, and the unsupervised language modeling training loop.
* `verify_elmo.py`: A modular pipeline that loads the trained weights, executes inference on real-time user-provided inputs, and runs the comparative analysis against industry baselines.

---

## Installation & Setup

### Prerequisites
Ensure you have Python installed along with the required deep learning dependencies:

```bash
pip install torch
pip install pandas
pip install tokenizers
pip install transformes
```
## How to Run

### 1. Train the Model (Unsupervised Language Modeling)
Execute the training script to compile the BPE vocabulary and train the bidirectional LSTM network's hidden layer states:
```bash
python train_elmo.py
```
### 2. Run Interactive Live Verification
Execute the training script to compile the BPE vocabulary and train the bidirectional LSTM network's hidden layer states:
```bash
python verify_elmo.py
```

## Verification Strategy
### Verification Workflow

When running `verify_elmo.py`, you can test your model's context-tracking capability by inputting any target word along with two diverse sentences. 

#### Scenario A: Testing Different Contexts (Target Word: `bank`)
1. **Sentence 1:** `"The river bank was muddy and full of clay"`
2. **Sentence 2:** `"The central bank raised the interest rates"`

Both models (Our BPE-ELMo and FinBERT) will calculate low cosine similarity scores (typically between **0.1** and **0.5**), demonstrating that the bidirectional LSTMs successfully shifted the word's vector representation based on the surrounding words.

#### Scenario B: Testing Similar Contexts (Target Word: `bank`)
1. **Sentence 1:** `"The central bank decided to lower interest rates"`
2. **Sentence 2:** `"The commercial bank updated its prime loan policies"`

Both models will output high cosine similarity scores (typically between **0.7** and **0.95**), proving that the network maps both environments to a shared semantic financial space.


### Comparative Evaluation Matrix
The terminal output produces a side-by-side comparison with **FinBERT**, a state-of-the-art transformer-based financial model.
