# Soybean Leaf Disease Detection using EfficientNet Architectures

This project focuses on **multi-class image classification** of soybean leaf diseases using **EfficientNet architectures (V2, B0, B1, and B2)**. The model classifies images into three categories:

- **Healthy**
- **Bacterial Blight**
- **Rust**

## 📌 Objective

To develop a deep learning-based system for automatic classification of soybean leaf diseases to assist in early and accurate detection, thereby supporting farmers and agronomists in disease management.

---

## 🧠 Model Architectures Used

We employed the following variants of the EfficientNet architecture:

- [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- [EfficientNet-B0](https://arxiv.org/abs/1905.11946)
- [EfficientNet-B1](https://arxiv.org/abs/1905.11946)
- [EfficientNet-B2](https://arxiv.org/abs/1905.11946)

These models were chosen for their excellent trade-off between accuracy and computational efficiency.

---

## 📁 Dataset

The dataset consists of labeled images of soybean leaves grouped into the following three classes:

- `healthy`
- `bacterial blight`
- `rust`

The dataset was split into training, validation, and testing subsets.

---

## 🚀 How to Run

### 1. Download the Code files
### 2. Place the dataset in the data folder
We got the dataset from the research journal [Soybean disease identification using original field images and transfer learning with convolutional neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0168169922007578?via%3Dihub)
[Dataset Link](https://datadryad.org/dataset/doi:10.5061/dryad.41ns1rnj3)
We then split the data into train, validation and test in the ratio 0.7 : 0.15 : 0.15 and put it into the data folder.

### 3. Load the pretrained models in checkpoint folder
In the checkpoints folder keep the pretrained models by imagenet for weight initialization:
You can download the models from the links below:
[EfficienNet-V2](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_l_cifar100.pth)
[EfficienNet-B0](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth)
[EfficienNet-B1](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth)
[EfficienNet-B2](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth)

### 4. Train the Model
Choose the model to train. Then accordingly run the file train_{v2/b0/b1/b2}.py file. The final model will be saved in models folder.

### 5. Evaluate the Model
Evaluate the model. You can do so by running the evaluate.py file.

### 6. Prediction
For running the model on new images. Place the images in the prediction/test folder in the appropriate class. Then run the predict.py file. It will show the predicted class along with the true class.

# 📊 Evaluation Metrics
Each model is evaluated using:
Accuracy

# 📈 Results

Model	Accuracy (%)
EfficientNet-V2	95.09
EfficientNet-B0	92.86
EfficientNet-B1	87.05
EfficientNet-B2	86.61

(The above ones are testing accuracies.)
# ✍️ Team Members
Yannam Yeswanth Reddy (220001082)
Vedant Upadhyay (220002081)
