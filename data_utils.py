import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial
# -----------------------------

# Data loading & preprocessing (PyTorch + Hugging Face datasets)
def load_imdb():
    ds=load_dataset("imdb")
    return ds['train'],ds['test']

# For LSTM/GRU models we'll build our own tokenizer (simple) or use HF's basic tokenizer + build vocab
from collections import Counter
import re

def simple_tokenize(text):
    # lower, basic punctuation split
    text = text.lower()
    tokens =  re.findall(r'\b\w+\b', text)
    return tokens

def build_vocab(texts, vocab_size=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    # keep most common
    vocab = {'<PAD>':0, '<UNK>':1}
    for i,(tok,cnt) in enumerate(counter.most_common(vocab_size)):
        if cnt < min_freq:
            break
        vocab[tok] = len(vocab)
    return vocab

def encode_text(text,vocab,max_len=200):
    tokens = simple_tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens][:max_len]
    # pad
    if len(ids)<max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids
def collate_batch(batch, vocab, max_len=200):
    texts = [b['text'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    ids = torch.tensor([encode_text(t, vocab, max_len) for t in texts], dtype=torch.long)
    return ids, labels

# For BERT, we'll use AutoTokenizer
def get_bert_tokenizer(model_name="bert-base-uncased", max_len=256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def bert_encode_batch(batch, tokenizer, max_len=256):
    texts = batch['text']
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    labels = torch.tensor(batch['label'], dtype=torch.long)
    return enc, labels

# Models
# 4.1 Baseline: Embedding + Bidirectional LSTM (PyTorch)

import torch.nn as nn

class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, rnn_hidden=128, rnn_type='lstm', n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, rnn_hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers>1 else 0)
        else:
            self.rnn = nn.GRU(embed_dim, rnn_hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers>1 else 0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*rnn_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        # x: [B, T]
        emb = self.embedding(x)                # [B, T, E]
        out, _ = self.rnn(emb)                 # [B, T, 2H]
        # pool across time: use mean or last; last is ambiguous with bidirectional, so use mean/max
        pooled = out.mean(dim=1)               # [B, 2H]
        logits = self.fc(pooled)
        return logits
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, mask=None):
        # encoder_outputs: [B, T, H]
        # compute score per timestep
        scores = self.attn(encoder_outputs).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)          # [B, T]
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, H]
        return context, weights

class BiRNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, rnn_hidden=128, rnn_type='lstm', n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.LSTM if self.rnn_type=='lstm' else nn.GRU
        self.rnn = rnn_cls(embed_dim, rnn_hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers>1 else 0)
        self.attn = Attention(2*rnn_hidden)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*rnn_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, mask=None):
        emb = self.embedding(x)              # [B, T, E]
        out, _ = self.rnn(emb)               # [B, T, 2H]
        # attention
        context, weights = self.attn(out, mask)  # [B, 2H], [B, T]
        logits = self.fc(context)
        return logits, weights

# file: train_utils.py
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(dataloader, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad()
        out = model(xb)
        if isinstance(out, tuple):  # model returns (logits, weights)
            logits = out[0]
        else:
            logits = out
        loss = criterion(logits, yb)
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for xb, yb in tqdm(dataloader, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        p = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
        pred = (p >= 0.5).astype(int)
        ys.extend(yb.cpu().numpy().tolist())
        preds.extend(pred.tolist())
        probs.extend(p.tolist())
    acc = accuracy_score(ys, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average='binary')
    try:
        auc = roc_auc_score(ys, probs)
    except:
        auc = float('nan')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

# Simple training loop
def fit(model, train_loader, val_loader, epochs, optim, criterion, device, patience=3):
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f}")
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_f1
