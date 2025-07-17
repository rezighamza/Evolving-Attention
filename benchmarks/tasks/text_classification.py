# benchmarks/tasks/text_classification.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter


# --- Manual Tokenizer and Vocab Builder (No torchtext needed) ---

def simple_tokenizer(text: str) -> list:
    """A basic tokenizer that splits on whitespace and lowercases."""
    return text.lower().split()


def build_vocab(data_iterator, min_freq=5):
    """Builds a vocabulary manually from an iterator."""
    word_counter = Counter()
    for text in data_iterator:
        word_counter.update(simple_tokenizer(text))

    # Create a vocab dictionary, starting with special tokens
    # <unk> is for unknown words, <pad> is for padding sentences to the same length
    vocab = {"<unk>": 0, "<pad>": 1}
    idx = 2
    for word, count in word_counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


# --- Data Loading and Benchmarking ---

def get_imdb_dataloaders(batch_size=16):
    """
    Loads IMDB data using the Hugging Face `datasets` library and
    our manual vocab/tokenizer tools.
    """
    print("Loading IMDB dataset from Hugging Face hub...")
    imdb_dataset = load_dataset("imdb")

    # Build the vocabulary from the training text
    print("Building vocabulary from training data...")
    train_texts = (example['text'] for example in imdb_dataset['train'])
    vocab = build_vocab(train_texts)
    unk_idx = vocab["<unk>"]
    pad_idx = vocab["<pad>"]

    # Define pipelines to convert text/labels to tensors
    text_pipeline = lambda x: [vocab.get(word, unk_idx) for word in simple_tokenizer(x)]
    label_pipeline = lambda x: float(x)  # Label is already 0 or 1

    def collate_batch(batch):
        label_list, text_list = [], []
        for example in batch:
            label_list.append(label_pipeline(example['label']))
            processed_text = torch.tensor(text_pipeline(example['text']), dtype=torch.int64)
            text_list.append(processed_text)

        padded_text = nn.utils.rnn.pad_sequence(text_list, padding_value=pad_idx, batch_first=False)
        return torch.tensor(label_list, dtype=torch.float32), padded_text

    train_dataloader = DataLoader(imdb_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(imdb_dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    print("IMDB Dataloaders created successfully.")
    return train_dataloader, test_dataloader, len(vocab)


def run_text_benchmark(model, epochs=5, lr=5e-5, device='cuda'):
    model.to(device)
    train_loader, test_loader, _ = get_imdb_dataloaders()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Text Training]")
        for labels, text in pbar:
            labels, text = labels.to(device), text.to(device)
            optimizer.zero_grad()

            output = model(text)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            acc = ((torch.sigmoid(output) > 0.5) == (labels > 0.5)).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            pbar.set_postfix(loss=loss.item(), acc=acc.item())

        model.eval()
        total_eval_acc = 0
        with torch.no_grad():
            for labels, text in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Text Validation]"):
                labels, text = labels.to(device), text.to(device)
                output = model(text)
                acc = ((torch.sigmoid(output) > 0.5) == (labels > 0.5)).float().mean()
                total_eval_acc += acc.item()

        final_acc = total_eval_acc / len(test_loader)
        print(f"Epoch {epoch + 1} Summary: Train Acc: {total_acc / len(train_loader):.4f}, Val Acc: {final_acc:.4f}")

    return final_acc * 100