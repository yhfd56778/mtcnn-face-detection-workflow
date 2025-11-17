"""
===============================================================================
This script is provided strictly for EDUCATIONAL and NON-COMMERCIAL use only.
Do NOT use any part of this code for production, commercial deployment, or
redistribution without proper authorization or licensing from the original
dataset providers (e.g., CelebA, FaceNet authors, etc.).
===============================================================================

Modification Meta
- Script Title : CelebA Face Cropping & Identification (MTCNN + ResNet18)
- Version      : 1.0.1
- Modified On  : wcv17624,2025-09-05
- Changes      :
    * Added version metadata/header (this block).
    * Minor variable/comment tidy-up; switched to explicit Exception in try/except.
    * No functional changes to logic, I/O paths, training/eval, or outputs.
===============================================================================
"""

# This script is for educational and non-commercial use only.

# Import necessary libraries
import os
import gdown
import zipfile
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Define URL for CelebA aligned face images
celeba_url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
output_zip = 'img_align_celeba.zip'

# Download and unzip the CelebA dataset if not already present
if not os.path.exists('img_align_celeba'):
    gdown.download(celeba_url, output_zip, quiet=False)
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall('img_align_celeba')

# Define URL for identity mapping file
identity_url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pQ2dXSHMzVUhXY0E'

# Download the identity file
gdown.download(identity_url, 'identity_CelebA.txt', quiet=False)

# Load the identity file to create a mapping from filename to identity ID
identity_map = {}
with open('identity_CelebA.txt', 'r') as f:
    for line in f:
        filename, identity = line.strip().split()
        identity_map[filename] = identity

# Initialize MTCNN model for face detection and alignment
mtcnn = MTCNN(image_size=160, margin=20, device='cuda' if torch.cuda.is_available() else 'cpu')

# Define input and output directories
input_dir = 'img_align_celeba/img_align_celeba'
output_root = 'celebA_cropped_by_id'
os.makedirs(output_root, exist_ok=True)

# Read the image filenames from the dataset directory (limit to 5000 for demo)
image_list = sorted(os.listdir(input_dir))[:5000]

# Loop through each image to perform face detection and save the cropped result
for filename in image_list:
    input_path = os.path.join(input_dir, filename)
    identity = identity_map.get(filename)  # same behavior as get(..., None)
    if identity is None:
        continue
    try:
        img = Image.open(input_path).convert('RGB')  # Load image and ensure it's in RGB format
        face = mtcnn(img)  # Use MTCNN to detect and crop the face
        if face is not None:
            face_img = transforms.ToPILImage()(face)  # Convert tensor back to image
            id_folder = os.path.join(output_root, identity)  # Create folder by ID
            os.makedirs(id_folder, exist_ok=True)
            save_path = os.path.join(id_folder, filename)  # Define save path
            face_img.save(save_path)  # Save cropped face image
    except Exception:
        continue  # Skip any images that raise errors

# Define transformation for loading cropped dataset
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize all images to 160x160
    transforms.ToTensor()           # Convert image to tensor
])

# Load the cropped face dataset using ImageFolder (structured by ID subfolders)
dataset = datasets.ImageFolder(output_root, transform=transform)

# Split dataset into 80% train and 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

# Use DataLoader to feed data in batches
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Load a pre-trained ResNet18 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)

# Replace the final classification layer to match number of identity classes
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)  # Move model to GPU if available

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Train for one epoch (example only, not full training)
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)               # Forward pass
    loss = criterion(outputs, labels)     # Compute loss
    optimizer.zero_grad()                 # Reset gradients
    loss.backward()                       # Backpropagation
    optimizer.step()                      # Update weights

# Evaluate model on test dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print test accuracy
accuracy = 100 * correct / total
print("Test Accuracy:", accuracy)
