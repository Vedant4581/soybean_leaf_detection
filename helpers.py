import torch
import os
from datetime import datetime
from config import LOG_FILE, LOGS_DIR

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

def log_message(message):
    # Log messages with timestamp to training.log
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

def save_checkpoint(model, epoch, optimizer, loss, filepath):
    # Save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    log_message(f"Checkpoint saved at {filepath} (Epoch {epoch})")

def load_checkpoint(model, optimizer, filepath):
    # Load model checkpoint
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    log_message(f"Checkpoint loaded from {filepath} (Epoch {epoch})")
    return model, optimizer, epoch, loss
