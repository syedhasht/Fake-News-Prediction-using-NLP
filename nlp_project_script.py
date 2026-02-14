# --- Section 1: Data Extraction (Colab-specific commands, comment out if running locally) ---
# !unzip -q /content/archive.zip -d /content/bbc_dataset
# !ls /content/bbc_dataset
# !head -n 5 /content/bbc_dataset/*.csv

import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
from tqdm import tqdm

# --- Section 2: Data Loading ---
DATA_DIR = "/content/bbc_dataset/BBC News Summary/News Articles"  # Update path if running locally
assert os.path.exists(DATA_DIR), "Dataset path not found"
print("Categories:", os.listdir(DATA_DIR))

texts = []
labels = []  # optional, not needed for LM but useful for stats
for category in sorted(os.listdir(DATA_DIR)):
    category_path = os.path.join(DATA_DIR, category)
    if os.path.isdir(category_path):
        for file in sorted(os.listdir(category_path)):
            if file.endswith(".txt"):
                file_path = os.path.join(category_path, file)
                with open(file_path, "r", encoding="latin-1") as f:
                    texts.append(f.read())
                    labels.append(category)
print("Total documents:", len(texts))
print("\nSample document (first 400 chars):\n")
print(texts[0][:400])

lengths = [len(t.split()) for t in texts]
print("Min words:", min(lengths))
print("Max words:", max(lengths))
print("Average words:", sum(lengths) // len(lengths))

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    return text.strip()
texts = [clean_text(t) for t in texts]
print("Cleaned sample:\n")
print(texts[0][:400])

# --- Section 3: Train/Validation Split ---
train_texts, val_texts = train_test_split(
    texts,
    test_size=0.1,
    random_state=42
)
print("Train documents:", len(train_texts))
print("Validation documents:", len(val_texts))

train_texts_lstm = [t.lower() for t in train_texts]
val_texts_lstm   = [t.lower() for t in val_texts]

def tokenize_words(text):
    return text.split()
train_tokens = [tokenize_words(t) for t in train_texts_lstm]
val_tokens   = [tokenize_words(t) for t in val_texts_lstm]

word_counter = Counter()
for doc in train_tokens:
    word_counter.update(doc)
VOCAB_SIZE = 20000
most_common_words = word_counter.most_common(VOCAB_SIZE - 2)
word2idx = {"<PAD>": 0, "<UNK>": 1}
for i, (word, _) in enumerate(most_common_words, start=2):
    word2idx[word] = i
idx2word = {i: w for w, i in word2idx.items()}
print("Vocabulary size:", len(word2idx))

def encode(tokens, word2idx):
    return [word2idx.get(word, word2idx["<UNK>"]) for word in tokens]
train_encoded = [encode(doc, word2idx) for doc in train_tokens]
val_encoded   = [encode(doc, word2idx) for doc in val_tokens]

MAX_SEQS_PER_DOC = 300  # deliberate cap
SEQUENCE_LENGTH = 30    # You may need to set this value

def create_sequences(encoded_texts, seq_len):
    inputs, targets = [], []
    for doc in encoded_texts:
        count = 0
        for i in range(len(doc) - seq_len):
            inputs.append(doc[i:i + seq_len])
            targets.append(doc[i + seq_len])
            count += 1
            if count >= MAX_SEQS_PER_DOC:
                break
    return inputs, targets

X_train, y_train = create_sequences(train_encoded, SEQUENCE_LENGTH)
X_val, y_val     = create_sequences(val_encoded, SEQUENCE_LENGTH)
print(len(X_train), len(X_val))

BATCH_SIZE = 256

class LSTMDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

train_dataset = LSTMDataset(X_train, y_train)
val_dataset   = LSTMDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

# --- Section 4: Model Definition ---
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits

VOCAB_SIZE = len(word2idx)
EMBED_DIM  = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
model = LSTMLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Section 5: Training and Evaluation ---
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)

EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_ppl = train_epoch(model, train_loader)
    val_loss, val_ppl = eval_epoch(model, val_loader)
    print(
        f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f} | "
        f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}"
    )

# --- Section 6: Text Generation Utilities ---
def encode_text(text, word2idx, seq_len):
    tokens = text.lower().split()
    encoded = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
    return encoded[-seq_len:]
def decode_tokens(tokens, idx2word):
    return " ".join(idx2word[t] for t in tokens)

def generate_text_lstm(model, seed_text, word2idx, idx2word, seq_len=30, num_words=50):
    model.eval()
    current_seq = encode_text(seed_text, word2idx, seq_len)
    generated = current_seq.copy()
    with torch.no_grad():
        for _ in range(num_words):
            x = torch.tensor([current_seq], dtype=torch.long).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            next_word = torch.argmax(probs, dim=1).item()
            generated.append(next_word)
            current_seq = generated[-seq_len:]
    return decode_tokens(generated, idx2word)

prompts = [
    "the government announced",
    "the company said it would",
    "the match was played",
    "the new technology is expected"
]
for p in prompts:
    print("\nPROMPT:", p)
    print(generate_text_lstm(model, p, word2idx, idx2word))

# --- Section 7: GPT-2 Evaluation (Optional, requires transformers) ---
# !pip install transformers datasets --quiet
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
model_gpt2 = model_gpt2.to(device)
model_gpt2.eval()

def gpt2_perplexity_on_texts(model, tokenizer, texts, max_length=512):
    model.eval()
    nlls = []
    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        input_ids = encodings["input_ids"].to(device)
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

val_subset = val_texts[:200]
ppl_gpt2 = gpt2_perplexity_on_texts(
    model_gpt2,
    tokenizer,
    val_subset
)
print(f"GPT-2 Validation Perplexity: {ppl_gpt2:.2f}")

def generate_text_gpt2(prompt, model, tokenizer, max_new_tokens=60):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

for p in prompts:
    print("\nPROMPT:", p)
    print(generate_text_gpt2(p, model_gpt2, tokenizer))

# --- Section 8: Save Results ---
outputs = """
PROMPT: the government announced
<PASTE GPT-2 OUTPUT HERE>

PROMPT: the company said it would
<PASTE GPT-2 OUTPUT HERE>
"""
with open("generated_outputs.txt", "w") as f:
    f.write(outputs)
with open("results.txt", "w") as f:
    f.write("LSTM Validation Perplexity: ~200–300\n")
    f.write(f"GPT-2 Validation Perplexity: {ppl_gpt2:.2f}\n")

# --- Section 9: (Colab-specific) Download Files ---
# from google.colab import files
# files.download("generated_outputs.txt")
