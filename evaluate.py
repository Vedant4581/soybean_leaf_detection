import torch
from utils.data_loader import get_data_loaders

def get_model(model_name, num_classes):
    if model_name == 'efficientnetv2l':
        from models.efficientnetv2 import EfficientNetV2L
        model = EfficientNetV2L(num_classes=num_classes)
        path = "models/efficientnetv2_l.pth"
    elif model_name == 'efficientnetb0':
        from models.efficientnet_b0 import EfficientNetB0
        model = EfficientNetB0(num_classes=num_classes)
        path = "models/efficientnet-b0.pth"
    elif model_name == 'efficientnetb1':
        from models.efficientnet_b1 import EfficientNetB1
        model = EfficientNetB1(num_classes=num_classes)
        path = "models/efficientnet_b1.pth"
    elif model_name == 'efficientnetb2':
        from models.efficientnet_b2 import EfficientNetB2
        model = EfficientNetB2(num_classes=num_classes)
        path = "models/efficientnet_b2.pth"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model, path

def evaluate():
    model_name = input("Enter model name (efficientnetv2l, efficientnetb0, efficientnetb1, efficientnetb2): ").strip().lower()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_path = get_model(model_name, num_classes=3)
    model.load_state_dict(torch.load(model_path))
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

    print(f"[{model_name}] Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    evaluate()
