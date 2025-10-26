"""
Training script for FaceNet embedding model using triplet loss.
Requires a large labeled face dataset (e.g., VGGFace2) with identities.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import os
from PIL import Image
import random
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with subfolders for each identity.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(cls, img_name))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def triplet_loss(anchor, positive, negative, margin=0.2):
    """Triplet loss for FaceNet training."""
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))
    return loss

def create_triplets(dataset, batch_size):
    """Create triplets for training."""
    triplets = []
    labels = list(set(dataset.labels))
    for _ in range(batch_size):
        # Select anchor
        anchor_class = random.choice(labels)
        anchor_indices = [i for i, label in enumerate(dataset.labels) if label == anchor_class]
        anchor_idx = random.choice(anchor_indices)

        # Select positive (same class)
        positive_idx = random.choice([idx for idx in anchor_indices if idx != anchor_idx])

        # Select negative (different class)
        negative_class = random.choice([cls for cls in labels if cls != anchor_class])
        negative_indices = [i for i, label in enumerate(dataset.labels) if label == negative_class]
        negative_idx = random.choice(negative_indices)

        triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets

def train_facenet(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    """Train FaceNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset and dataloader
    dataset = FaceDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)

            # Create triplets
            triplets = create_triplets(dataset, len(images))

            # Forward pass
            embeddings = model(images)

            # Compute triplet loss
            loss = 0.0
            for anchor_idx, pos_idx, neg_idx in triplets:
                anchor = embeddings[anchor_idx]
                positive = embeddings[pos_idx]
                negative = embeddings[neg_idx]
                loss += triplet_loss(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))

            loss /= len(triplets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f'facenet_epoch_{epoch+1}.pth')

    print("Training completed!")

if __name__ == "__main__":
    # Example usage
    # Assume data_dir has subfolders for each identity with face images
    data_dir = "path/to/face_dataset"  # e.g., VGGFace2 or custom dataset
    train_facenet(data_dir)
