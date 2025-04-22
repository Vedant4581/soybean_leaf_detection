
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
