from enum import Enum


class ModelName(Enum):
    DENSENET201 = "densenet201"
    RESNET50 = "resnet50"
    VGG16 = "vgg16"


class Device(Enum):
    CPU = "cpu"
    GPU = "cuda"
