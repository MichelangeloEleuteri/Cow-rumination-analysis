import os
BASE_DIR = os.getcwd()
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Caricamento immagini positive e negative
pos_paths = list(Path(os.path.join(BASE_DIR, 'dynamic_images_positive')).rglob('*.jpg'))
neg_paths = list(Path(os.path.join(BASE_DIR, 'dynamic_images_negative')).rglob('*.jpg'))

rows = [{'path': str(p), 'label': 1} for p in pos_paths] + \
       [{'path': str(p), 'label': 0} for p in neg_paths]
df = pd.DataFrame(rows)

# ======== 1b) Calcolo pesi di classe per la loss ========
neg_count = (df.label == 0).sum()
pos_count = (df.label == 1).sum()
total = neg_count + pos_count
class_weights = torch.tensor([ total/neg_count, total/pos_count ], dtype=torch.float).to(device)
print(f"Class weights (neg, pos): {class_weights.tolist()}")

# 2) Split train/val/test con stratificazione
df_train, df_tmp = train_test_split(df, test_size=0.3, stratify=df.label, random_state=42)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp.label, random_state=42)

# Salvataggio csv (opzionale)
df_train.to_csv('train_simple.csv', index=False)
df_val.to_csv('val_simple.csv', index=False)
df_test.to_csv('test_simple.csv', index=False)

print("Distribuzione dati:")
for name, subset in [('train', df_train), ('val', df_val), ('test', df_test)]:
    print(f"{name}: pos={(subset.label==1).sum()}, neg={(subset.label==0).sum()}, totale={len(subset)}")

# 3) Trasformazioni
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.9,1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 4) Dataset e DataLoader
class DynChewDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.long)
        return img, label

train_ds = DynChewDataset(df_train, transform=train_transforms)
val_ds   = DynChewDataset(df_val,   transform=val_transforms)
test_ds  = DynChewDataset(df_test,  transform=val_transforms)

# Se vuoi usare oversampling della classe minoritaria:
train_labels = df_train.label.values
class_sample_counts = [ (train_labels==0).sum(), (train_labels==1).sum() ]
weights = [1.0/count for count in class_sample_counts]
samples_weight = [ weights[label] for label in train_labels ]
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

# 5) Modello VGG16 fine-tuning
model = models.vgg16(pretrained=True)
for param in model.features.parameters():  # congela backbone
    param.requires_grad = False
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
model = model.to(device)

# 6) Ottimizzatore, scheduler e loss con pesi
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=20)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 7) Training loop con early stopping
best_val_loss = float('inf')
patience, max_patience = 0, 7
num_epochs = 20

for epoch in range(num_epochs):
    if epoch == 5:
        for param in model.features.parameters():
            param.requires_grad = True
        print("-- Unfroze backbone layers --")

    # ——— Training ———
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_ds)

    # ——— Validation ———
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item() * imgs.size(0)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    val_loss /= len(val_ds)
    report = classification_report(all_labels, all_preds, target_names=['neg','pos'], output_dict=True)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Val Acc: {report['accuracy']:.4f} | F1-pos: {report['pos']['f1-score']:.4f}")

    # Early stopping & save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), 'best_vgg_model.pth')
    else:
        patience += 1
        if patience >= max_patience:
            print("Early stopping triggered.")
            break

# 8) Testing finale
model.load_state_dict(torch.load('best_vgg_model.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("Test Results:")
print(classification_report(all_labels, all_preds, target_names=['neg','pos']))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
