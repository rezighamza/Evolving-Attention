import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from tqdm import tqdm


def get_cifar100_dataloaders(batch_size=128):
    """ Gets CIFAR-100 dataloaders with standard augmentations. """
    # Normalization values for CIFAR-100
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])

    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def run_benchmark(model, epochs=20, lr=1e-3, device='cuda'):
    """ Runs a full training and evaluation benchmark for a given model. """
    model.to(device)
    train_loader, test_loader = get_cifar100_dataloaders()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1} Summary: Train Loss: {train_loss / len(train_loader):.4f}, Val Accuracy: {accuracy:.2f}%")

    return accuracy