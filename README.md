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

pip install transformers






BPE-ELMo: Contextual Financial Embedding PipelineAn implementation of a custom BPE-Tokenized ELMo (Embeddings from Language Models) network trained from scratch on financial sentiment text. This project demonstrates how bidirectional LSTMs capture dynamic, context-dependent word meanings, bypassing the limitations of traditional static embedding models like Word2Vec.The repository includes a live comparative evaluation framework that benchmarks this architecture against an industry-standard, authenticated language model (ProsusAI/FinBERT).🚀 Key FeaturesCustom BPE Tokenizer: Built using the Hugging Face tokenizers library to break text into highly frequent subwords, optimizing vocabulary size for smaller domain-specific datasets (e.g., ~3,000 sentences).Deep BiLSTM Architecture: A multi-layer bidirectional LSTM network that processes left-to-right and right-to-left contexts simultaneously.Smart Checkpoint Saving: Automatic saving and loading mechanisms (financial_elmo.pt) to prevent unnecessary CPU retraining cycles.Interactive Verification Suite: A command-line program to evaluate custom user-defined sentences and compute mathematical contextual cosine similarity scores side-by-side with FinBERT.📂 Repository Structuretrain_elmo.py: Handles raw financial text compilation, Byte-Pair Encoding (BPE) vocabulary construction, PyTorch data batching, and the unsupervised language modeling training loop.verify_elmo.py: A modular pipeline that loads the trained weights, executes inference on real-time user-provided inputs, and runs the comparative analysis against industry baselines.🛠️ Installation & SetupPrerequisitesEnsure you have Python installed along with the required deep learning dependencies:pip install torch pandas tokenizers transformers
💻 How to Run1. Train the Model (Unsupervised Language Modeling)Execute the training script to compile the BPE vocabulary and train the bidirectional LSTM network's hidden layer states:python train_elmo.py
Note: If a pre-trained financial_elmo.pt checkpoint file is detected in the working directory, this training phase will automatically skip, loading the saved weights instantly to save time.2. Run Interactive Live VerificationLaunch the testing suite to input custom sentences and observe real-time context vector generation:python verify_elmo.py
📊 Evaluation & Verification StrategyThe Limitations of Static EmbeddingsTraditional static embedding algorithms (such as Word2Vec or GloVe) map a word to a single, static multi-dimensional vector. Consequently, the word "bank" receives the exact same mathematical representation regardless of whether it is used in a geographic context ("river bank") or a monetary context ("investment bank").$$Embedding(\text{"river bank"}) = Embedding(\text{"investment bank"}) \implies \text{Similarity} = 1.0 \text{ (100\%)}$$Our BPE-ELMo model resolves this limitation by extracting and combining representations across distinct neural network layers:Layer 0: Static subword token embeddings (morphological features).Layer 1 & 2: Deep bidirectional sequential tracking (syntactic and semantic context).Interactive Verification WorkflowWhen running verify_elmo.py, you can test your model's context-tracking capability by inputting any target word along with two diverse sentences.Scenario A: Testing Different Contexts (Target Word: bank)Sentence 1: "The river bank was muddy and full of clay"Sentence 2: "The central bank raised the interest rates"Both models (Our BPE-ELMo and FinBERT) will calculate low cosine similarity scores (typically between 0.1 and 0.5), demonstrating that the bidirectional LSTMs successfully shifted the word's vector representation based on the surrounding words.Scenario B: Testing Similar Contexts (Target Word: bank)Sentence 1: "The central bank decided to lower interest rates"Sentence 2: "The commercial bank updated its prime loan policies"Both models will output high cosine similarity scores (typically between 0.7 and 0.95), proving that the network maps both environments to a shared semantic financial space.Comparative Evaluation MatrixThe terminal output produces a side-by-side comparison with FinBERT, a state-of-the-art transformer-based financial model:=======================================================
             VERIFICATION AUDIT ANALYSIS               
=======================================================
Target Token Word Under Review : 'bank'
Sentence 1 Vector Source     : 'The river bank is muddy'
Sentence 2 Vector Source     : 'The central bank raised interest rates'
-------------------------------------------------------
YOUR BPE-ELMo MODEL Cosine Similarity   : 0.3842
INDUSTRY FINBERT MODEL Cosine Similarity : 0.4512
=======================================================
By presenting this side-by-side comparison, this project demonstrates that our custom-trained 2-layer biLSTM network, despite being trained on a lean local dataset, closely mirrors the relative trends of complex industrial models.📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
