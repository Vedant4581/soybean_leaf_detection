import torch
from models.efficientnetv2 import EfficientNetV2L #We currently chose V2 model you may replace with any other model
from utils.data_loader import get_data_loaders
import matplotlib.pyplot as plt

# Display a single image
def imshow(img, title=None):
    img = img.cpu().numpy().transpose((1, 2, 0))  # CHW to HWC
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2L(num_classes=3)
    model.load_state_dict(torch.load("models/efficientnetv2_l.pth", map_location=device))
    model = model.to(device).eval()

    _, _, test_loader = get_data_loaders("prediction", batch_size=32)

    class_names = ['bacterial_blight', 'healthy', 'rust']

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            for i in range(imgs.size(0)):
                class_id = labels[i].item()

                plt.figure(figsize=(3, 3))
                imshow(
                    imgs[i],
                    title=f"Pred: {class_names[predicted[i]]} | True: {class_names[class_id]}"
                )
                plt.show()

if __name__ == '__main__':
    evaluate()