import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json
# -------------------------
# 1. Create a custom dataset
# -------------------------
class FaceDataset(Dataset):
    def __init__(self, csv_file, mtcnn, transform=None):
        """
        csv_file: Path to CSV containing columns 'gt' and 'image_path'
        mtcnn: An instance of MTCNN to detect and crop faces.
        transform: Optional transform to be applied on the face tensor.
        """
        self.df = pd.read_csv(csv_file)
        self.mtcnn = mtcnn
        self.transform = transform
        # Sorted list of unique persons and mapping to indices
        self.persons = sorted(self.df['gt'].unique())
        self.person2idx = {p: idx for idx, p in enumerate(self.persons)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        person = row['gt']
        label = self.person2idx[person]
        
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy tensor and label in case of error
            return torch.zeros(3, 160, 160), label
        
        # Use MTCNN to detect and crop the face
        face = self.mtcnn(img)
        if face is None:
            # If no face is detected, return a tensor of zeros.
            face = torch.zeros(3, 160, 160)
        if self.transform:
            face = self.transform(face)
        return face, label

def main():
    # -------------------------
    # 2. Prepare training data and dataloader
    # -------------------------
    mtcnn_train = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True)
    train_dataset = FaceDataset('trainset.csv', mtcnn_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    num_classes = len(train_dataset.persons)
    print(f"Number of classes: {num_classes}")
    
    # -------------------------
    # 3. Set up the model for fine-tuning
    # -------------------------
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Finetuning
    epochs = 10  # Adjust the number of epochs as needed
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for faces, labels in train_loader:
            faces, labels = faces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(faces)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * faces.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), 'finetuned_facenet.pth')
    print("Model saved as finetuned_facenet.pth")

    # Save label mapping
    with open('label_mapping.json', 'w') as f:
        json.dump(train_dataset.person2idx, f)
    print("Label mapping saved as label_mapping.json")


if __name__ == '__main__':
    main()
