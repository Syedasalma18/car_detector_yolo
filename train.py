import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
train_dir = "Cars Dataset/train"
test_dir = "Cars Dataset/test"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets & Dataloaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Label map (save for later)
label_map = {v: k for k, v in train_dataset.class_to_idx.items()}
os.makedirs("car_model_classifier", exist_ok=True)
with open("car_model_classifier/labels.json", "w") as f:
    import json
    json.dump(label_map, f)

# Model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_map))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "car_model_classifier/resnet_car_model.pth")
print("✅ Model trained and saved!")
