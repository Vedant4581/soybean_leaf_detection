import torch
import torch.nn as nn
import torch.optim as optim
from models.efficientnetv2 import EfficientNetV2L
from utils.data_loader import get_data_loaders 
from datetime import datetime
import os

def log_message(message, log_file="logs/training.log"):
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2L(num_classes=3).to(device)
    model.load_pretrained('checkpoints/efficientnetv2_l_imagenet.pth') #Name according to the name you save .pth file in the checkpoints folder

    train_loader, val_loader, _ = get_data_loaders("data", batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                val_outputs = model(val_imgs)
                loss = criterion(val_outputs, val_labels)

                val_loss += loss.item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total 

        # Logging and printing
        log_msg = (f"Epoch [{epoch+1}/10] - "
                   f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(log_msg)
        log_message(log_msg)

    torch.save(model.state_dict(), "models/efficientnetv2_l.pth")
    log_message("Training complete. Model saved to models/efficientnetv2_l.pth")

if __name__ == '__main__':
    train()
    # callbacks = [
    # tf.keras.callbacks.ModelCheckpoint(
    #     filepath='weights.weights.h5', 
    #     save_best_only=True,
    #     save_weights_only=True,
    #     monitor='val_loss'
    # ),
    # tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     patience=4,
    #     restore_best_weights=True
    # ),
    # ]
