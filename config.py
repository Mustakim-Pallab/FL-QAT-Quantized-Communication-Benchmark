import random

import numpy as np
import torch

DATASETS = [
    {
        "name": "Brain Tumor",
        "train_dir": "/content/drive/MyDrive/datasets_for_fl/brain_tumor/train",
        "test_dir": "/content/drive/MyDrive/datasets_for_fl/brain_tumor/test",
    },
    {
        "name": "Fundus Diabetic Retinopathy",
        "train_dir": "/content/drive/MyDrive/datasets_for_fl/fundus_diabetic_retinopathy/train",
        "test_dir": "/content/drive/MyDrive/datasets_for_fl/fundus_diabetic_retinopathy/test",
    },
]

MODEL_NAMES = [
    "mobilenet_v2",
    "resnet18",
    "densenet121",
    "alexnet",
    "vit_b_16",
]

NUM_CLIENTS = 2
DIRICHLET_BETA = 0.8
GLOBAL_ROUNDS = 3
LOCAL_EPOCHS = 2
BATCH_SIZE = 32

VANILLA_LR = 1e-4
ADAPTER_LR = 1e-3
ADAPTER_BACKBONE_LR = 5e-5
WEIGHT_DECAY = 1e-4

ADAPTER_HIDDEN_DIM = 512
ADAPTER_DROPOUT = 0.1
ADAPTER_TYPE = "attention"
ADAPTER_ATTENTION_HEADS = 8
ADAPTER_ATTENTION_TOKENS = 8
ADAPTER_TRAIN_BACKBONE_TAIL = False
ADAPTER_TAIL_BLOCKS = {
    "mobilenet_v2": 2,
    "resnet18": 1,
    "densenet121": 1,
    "alexnet": 1,
    "vit_b_16": 2,
}
USE_CLASS_BALANCED_LOSS = True
USE_PRETRAINED = True
QUANTIZE_DOWNLINK = True
QUANTIZED_WARMUP_ROUNDS = 1
USE_ERROR_FEEDBACK = True
KEEP_SMALL_TENSORS_FP32 = True
FEDPROX_MU = 0.01
GRAD_CLIP_NORM = 1.0
SEED = 42

BRAIN_TUMOR_GLOBAL_ROUNDS = 5
BRAIN_TUMOR_LOCAL_EPOCHS = 3
BRAIN_TUMOR_QUANTIZED_WARMUP_ROUNDS = 2
BRAIN_TUMOR_VANILLA_LR = 5e-5
BRAIN_TUMOR_ADAPTER_LR = 5e-4
BRAIN_TUMOR_ADAPTER_BACKBONE_LR = 1e-5
BRAIN_TUMOR_TUNE_ADAPTER_TAIL = True

CURRENT_DATASET_NAME = None
CURRENT_MODEL_NAME = None


def is_brain_tumor_dataset(dataset_name=None):
    dataset_name = CURRENT_DATASET_NAME if dataset_name is None else dataset_name
    return dataset_name == "Brain Tumor"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
