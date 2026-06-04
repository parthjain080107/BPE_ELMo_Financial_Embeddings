import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# CONFIGURATION & HYPERPARAMETERS

CSV_PATH = r"C:\Users\Dell\OneDrive - iitr.ac.in\Desktop\BPE\bpe_dataset.csv"
MODEL_SAVE_PATH = "financial_elmo.pt"
TOKENIZER_SAVE_PATH = "financial_bpe_tokenizer.json"

TEXT_COLUMN = "Text"   
LABEL_COLUMN = "Label" 

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 16
EPOCHS = 20
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


# PYTORCH DATASET SETUP

class FinancialDataset(Dataset):
    def __init__(self, data, text_col, label_col, tokenizer, label_map):
        self.data = data
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.label_map = label_map
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = str(self.data[idx][self.text_col])
        label_str = str(self.data[idx][self.label_col]).lower().strip() 
        
        encoded = self.tokenizer.encode(text)
        tokens = [self.tokenizer.token_to_id("[SOS]")] + encoded.ids + [self.tokenizer.token_to_id("[EOS]")]
        
        return torch.tensor(tokens), torch.tensor(self.label_map[label_str])

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=1) # 1 is [PAD]
    return padded_sequences, torch.stack(labels)


# BPE-ELMo ARCHITECTURE

class BPE_ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(BPE_ELMo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1) # 1 is [PAD]
        
        self.forward_lstms = nn.ModuleList([
            nn.LSTM(embedding_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True)
            for i in range(num_layers)
        ])
        
        self.backward_lstms = nn.ModuleList([
            nn.LSTM(embedding_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True)
            for i in range(num_layers)
        ])
        
        self.forward_lm_head = nn.Linear(hidden_dim, vocab_size)
        self.backward_lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        forward_states = [embeds]
        backward_states = [embeds]
        
        f_out = embeds
        for lstm in self.forward_lstms:
            f_out, _ = lstm(f_out)
            forward_states.append(f_out)
            
        b_embeds = torch.flip(embeds, dims=[1])
        b_out = b_embeds
        for lstm in self.backward_lstms:
            b_out, _ = lstm(b_out)
            backward_states.append(torch.flip(b_out, dims=[1]))
            
        return forward_states, backward_states


# CONTEXTUAL EMBEDDING EXTRACTION FUNCTION

def extract_elmo_vectors(sentence_text, tokenizer_obj, elmo_model, layer_weights=[0.20, 0.40, 0.40]):
    encoded = tokenizer_obj.encode(str(sentence_text))
    token_ids = torch.tensor([[tokenizer_obj.token_to_id("[SOS]")] + encoded.ids + [tokenizer_obj.token_to_id("[EOS]")]])
    
    with torch.no_grad():
        f_states, b_states = elmo_model(token_ids)
    
    layer0 = f_states[0]
    layer1 = torch.cat([f_states[1], b_states[1]], dim=-1)
    layer2 = torch.cat([f_states[2], b_states[2]], dim=-1)
    
    padding_size = (layer1.shape[-1] - layer0.shape[-1])
    layer0_padded = nn.functional.pad(layer0, (0, padding_size))
    
    final_contextual_embeddings = (
        (layer_weights[0] * layer0_padded) + 
        (layer_weights[1] * layer1) + 
        (layer_weights[2] * layer2)
    )
    return final_contextual_embeddings, encoded.tokens


# Execution block for training setup
if __name__ == "__main__":
    print("Loading Local Dataset")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Successfully loaded {len(df)} rows from: {CSV_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {CSV_PATH}.")
        exit()

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    dataset_records = df.to_dict('records')

    with open("raw_financial_text.txt", "w", encoding="utf-8") as f:
        for text in df[TEXT_COLUMN]:
            f.write(str(text) + "\n")

    print("\n Training BPE Tokenizer on the financial text corpus")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace() 
    trainer = BpeTrainer(vocab_size=1500, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    tokenizer.train(files=["raw_financial_text.txt"], trainer=trainer)
    tokenizer.save(TOKENIZER_SAVE_PATH)
    
    PAD_IDX = tokenizer.token_to_id("[PAD]")
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Tokenizer saved. Vocabulary Size: {VOCAB_SIZE}")

    dataset = FinancialDataset(dataset_records, TEXT_COLUMN, LABEL_COLUMN, tokenizer, LABEL_MAP)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = BPE_ELMo(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    
    # TRAINING LOOP
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n[CHECKPOINT DETECTED] Loading weights from '{MODEL_SAVE_PATH}'")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Skipping training phase.")
    else:
        print("\n Executing Language Model Training Loop. This may take several minutes")
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_idx, (batch_x, _) in enumerate(dataloader): 
                optimizer.zero_grad()
                f_states, b_states = model(batch_x)
                
                f_predictions = model.forward_lm_head(f_states[-1])
                b_predictions = model.backward_lm_head(b_states[-1])
                
                f_loss = criterion(f_predictions[:, :-1, :].reshape(-1, VOCAB_SIZE), batch_x[:, 1:].reshape(-1))
                b_loss = criterion(b_predictions[:, 1:, :].reshape(-1, VOCAB_SIZE), batch_x[:, :-1].reshape(-1))
                
                loss = f_loss + b_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                    
            print(f"=== Epoch {epoch+1}/{EPOCHS} Completed | Avg Loss: {total_loss/len(dataloader):.4f} ===")
        
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\nModel architecture parameters safely transfered to '{MODEL_SAVE_PATH}'.")