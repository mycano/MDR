import torchvision.models as models

MODEL = [
    # the input can be (batch, 3, 224, 224)
    models.alexnet(pretrained=False),
    models.resnet18(pretrained=False),
    models.resnet34(pretrained=False),
    models.resnet50(pretrained=False),
    models.resnet101(pretrained=False),
    models.resnet152(pretrained=False),
    models.vgg11(pretrained=False),
    models.vgg13(pretrained=False),
    models.vgg16(pretrained=False),
    models.vgg19(pretrained=False)
]

MODEL_NAME = [
    "AlexNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101",
    "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19"
]


def byte2mb(size):
    """
    8 bit(位) = 1 Byte(字节)
    1024 Byte(字节) = 1KB
    1024 KB = 1MB
    1024 MB = 1GB
    """
    return float(size)/(1024*1024)