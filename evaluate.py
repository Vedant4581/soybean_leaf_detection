import torch
from models.efficientnetv2 import EfficientNetV2L
from utils.data_loader import get_data_loaders

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2L(num_classes=3)
    model.load_state_dict(torch.load("models/efficientnetv2_l.pth"))
    model = model.to(device).eval()

    _, _, test_loader = get_data_loaders("data", batch_size=32)

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    evaluate()
