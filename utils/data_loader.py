import tensorflow as tf
from tensorflow.keras import layers
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def effnet_preprocess(x):
    """Preprocess input images manually for EfficientNet."""
    x = tf.image.convert_image_dtype(x, tf.float32)  # Scale [0, 255] → [0, 1]
    
    # ImageNet mean and std (Torch mode)
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    
    x = (x - mean) / std 
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    return x


def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    test_data = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# tr,va,te=get_data_loaders(r"D:\soybean_leaf_detection\data",32)

# import matplotlib.pyplot as plt

# # Display a few images and their corresponding class labels from the DataLoader
# def show_data(data_loader, num_items=5):
#     data_iter = iter(data_loader)  # Create an iterator for the DataLoader
#     images, labels = next(data_iter)  # Get the next batch
    
#     # Display num_items images and their classes
#     for i in range(num_items):
#         image = images[i].permute(1, 2, 0)  # Rearrange dimensions for displaying (CHW to HWC)
#         plt.imshow(image.numpy())
#         plt.title(f"Class: {labels[i]}")
#         plt.axis('off')
#         plt.show()

# # Show 5 items from the training DataLoader
# show_data(tr)
