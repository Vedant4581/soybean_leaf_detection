import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10

# Classes for Soybean Disease Detection (modify as needed)
NUM_CLASSES = 3  # Adjust to the number of disease classes you have

# Pretrained weights path
PRETRAINED_WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "efficientnetv2_l_imagenet.pth")

# Log file
LOG_FILE = os.path.join(LOGS_DIR, "training.log")
