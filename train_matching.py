"""
Training script for face matching/verification model.
This trains a binary classifier on face embeddings to predict if two faces belong to the same person.
Alternatively, this can be used to fine-tune similarity thresholds.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import os
from PIL import Image
import numpy as np
import random

class FacePairDataset(Dataset):
    def __init__(self, root_dir, transform=None, pairs_per_class=10):
        """
        Args:
            root_dir: Directory with subfolders for each identity
            transform: Image transforms
            pairs_per_class: Number of positive/negative pairs per class
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pairs_per_class = pairs_per_class
        self.classes = os.listdir(root_dir)
        self.pairs = []

        # Create positive pairs (same person)
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            if os.path.isdir(cls_folder):
                images = [os.path.join(cls, img) for img in os.listdir(cls_folder)
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(images) >= 2:
                    for _ in range(pairs_per_class):
                        img1, img2 = random.sample(images, 2)
                        self.pairs.append((img1, img2, 1))  # 1 for same person

        # Create negative pairs (different persons)
        all_images = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            if os.path.isdir(cls_folder):
                images = [os.path.join(cls, img) for img in os.listdir(cls_folder)
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.extend(images)

        for _ in range(len(self.pairs)):  # Same number as positive pairs
            img1, img2 = random.sample(all_images, 2)
            # Ensure they are from different classes
            cls1 = img1.split(os.sep)[0]
            cls2 = img2.split(os.sep)[0]
            if cls1 != cls2:
                self.pairs.append((img1, img2, 0))  # 0 for different persons

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(os.path.join(self.root_dir, img1_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.root_dir, img2_path)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

class FaceVerificationModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceVerificationModel, self).__init__()
        self.embedding_net = InceptionResnetV1(pretrained='vggface2')
        # Freeze embedding layers
        for param in self.embedding_net.parameters():
            param.requires_grad = False

        # Verification head
        self.verification_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        combined = torch.cat((emb1, emb2), dim=1)
        output = self.verification_head(combined)
        return output

def train_matching_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    """Train face verification model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset and dataloader
    dataset = FacePairDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = FaceVerificationModel().to(device)
    model.train()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.verification_head.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (img1, img2, labels) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f'matching_model_epoch_{epoch+1}.pth')

    print("Training completed!")

def fine_tune_threshold(data_dir, model_path=None):
    """Fine-tune similarity threshold using validation data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained embedding model
    embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FacePairDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    similarities = []
    labels = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)

            emb1 = embedding_model(img1)
            emb2 = embedding_model(img2)

            # Normalize embeddings
            emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
            emb2 = emb2 / emb2.norm(dim=1, keepdim=True)

            # Cosine similarity
            sim = torch.sum(emb1 * emb2, dim=1)
            similarities.extend(sim.cpu().numpy())
            labels.extend(label.numpy())

    # Find optimal threshold
    similarities = np.array(similarities)
    labels = np.array(labels)

    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in np.arange(0.1, 0.9, 0.01):
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.3f}, Accuracy: {best_accuracy:.4f}")
    return best_threshold

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/face_dataset"

    # Train verification model
    # train_matching_model(data_dir)

    # Or fine-tune threshold
    optimal_threshold = fine_tune_threshold(data_dir)
    print(f"Use this threshold in config.py: SIMILARITY_THRESHOLD = {optimal_threshold}")
