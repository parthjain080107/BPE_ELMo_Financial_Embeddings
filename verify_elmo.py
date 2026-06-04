import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Hugging Face imports
from transformers import AutoTokenizer as HfTokenizer
from transformers import AutoModel as HfModel

# import directly from training script 
from train_elmo import BPE_ELMo, extract_elmo_vectors, EMBEDDING_DIM, HIDDEN_DIM, TOKENIZER_SAVE_PATH, MODEL_SAVE_PATH

def find_hf_word_index(tokens, target_word):
    for idx, t in enumerate(tokens):
        cleaned_token = t.lower().replace("##", "").replace("Ġ", "")
        if cleaned_token == target_word.lower():
            return idx
    return -1

def find_my_word_index(tokens, target_word):
    for idx, t in enumerate(tokens):
        if t.lower() == target_word.lower():
            return idx + 1 # Offset by 1 to bypass the [SOS] token space
    return -1

if __name__ == "__main__":
    if not os.path.exists(TOKENIZER_SAVE_PATH) or not os.path.exists(MODEL_SAVE_PATH):
        print("Verification Failure: Weight states or configuration schemas missing. Run train_elmo.py first.")
        exit()

    # Loading trained model states
    tokenizer = Tokenizer.from_file(TOKENIZER_SAVE_PATH)
    VOCAB_SIZE = tokenizer.get_vocab_size()
    
    model = BPE_ELMo(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print("BPE-ELMo Model Weights Loaded.")

    # Load baseline verified weights
    print("Synchronizing baseline industry platform weights (FinBERT)")
    try:
        hf_model_name = "ProsusAI/finbert"
        hf_tokenizer = HfTokenizer.from_pretrained(hf_model_name)
        hf_model = HfModel.from_pretrained(hf_model_name)
        hf_model.eval()
        print("Baseline models successfully initialized.")
    except Exception as e:
        print(f"HuggingFace Initialization Error: {e}. Check network or run pip install transformers")
        exit()

    # Interactive Testing
    print("Provide custom data parameters below to perform real-time testing.\n")

    # Interactive Inputs
    target_word = input("1. Enter a Target Word to evaluate context : ").strip()  #e.g., bank, rate, stock
    sentence_1  = input(f"2. Enter Sentence 1 containing the word '{target_word}': ").strip()
    sentence_2  = input(f"3. Enter Sentence 2 containing the word '{target_word}': ").strip()

    # Process BPE-ELMo Model Contextual Embeddings
    emb_1, tokens_1 = extract_elmo_vectors(sentence_1, tokenizer, model)
    emb_2, tokens_2 = extract_elmo_vectors(sentence_2, tokenizer, model)

    idx_my_1 = find_my_word_index(tokens_1, target_word)
    idx_my_2 = find_my_word_index(tokens_2, target_word)

    # Process Hugging Face Model Contextual Embeddings
    hf_inputs_1 = hf_tokenizer(sentence_1, return_tensors="pt")
    hf_inputs_2 = hf_tokenizer(sentence_2, return_tensors="pt")

    with torch.no_grad():
        hf_out_1 = hf_model(**hf_inputs_1).last_hidden_state
        hf_out_2 = hf_model(**hf_inputs_2).last_hidden_state

    hf_tokens_1 = hf_tokenizer.convert_ids_to_tokens(hf_inputs_1["input_ids"][0])
    hf_tokens_2 = hf_tokenizer.convert_ids_to_tokens(hf_inputs_2["input_ids"][0])

    idx_hf_1 = find_hf_word_index(hf_tokens_1, target_word)
    idx_hf_2 = find_hf_word_index(hf_tokens_2, target_word)

    # Validation & Error Safeguards
    if idx_my_1 == -1 or idx_my_2 == -1:
        print(f"\n[Token Error] '{target_word}' could not be cleanly isolated in your model's tokenizer arrays.")
        print(f"Sentence 1 Tokens: {tokens_1}\nSentence 2 Tokens: {tokens_2}")
        exit()

    if idx_hf_1 == -1 or idx_hf_2 == -1:
        print(f"\n[Token Error] '{target_word}' could not be isolated in FinBERT's token layout structure.")
        print(f"FinBERT S1 Tokens: {hf_tokens_1}\nFinBERT S2 Tokens: {hf_tokens_2}")
        exit()

    # Vector Extraction
    my_vec_1 = emb_1[0, idx_my_1, :]
    my_vec_2 = emb_2[0, idx_my_2, :]

    hf_vec_1 = hf_out_1[0, idx_hf_1, :]
    hf_vec_2 = hf_out_2[0, idx_hf_2, :]

    # Calculation Step
    my_similarity = F.cosine_similarity(my_vec_1.unsqueeze(0), my_vec_2.unsqueeze(0)).item()
    hf_similarity = F.cosine_similarity(hf_vec_1.unsqueeze(0), hf_vec_2.unsqueeze(0)).item()

    # Final Comparative Verification Output
    print("\n")
    print(f"Target Token Word Under Review : '{target_word}'")
    print(f"Sentence 1 Vector Source     : '{sentence_1}'")
    print(f"Sentence 2 Vector Source     : '{sentence_2}'")
    print("\n")
    print(f"BPE-ELMo MODEL Cosine Similarity   : {my_similarity:.4f}")
    print(f"INDUSTRY FINBERT MODEL Cosine Similarity : {hf_similarity:.4f}")