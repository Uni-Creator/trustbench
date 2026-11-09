"""Model adapters and model architecture definitions for TrustBench."""

from trustbench.models.base import ModelAdapter
from trustbench.models.jax import JAXModelAdapter
from trustbench.models.keras import KerasModelAdapter
from trustbench.models.onnx import ONNXModelAdapter
from trustbench.models.sklearn import SklearnModelAdapter

__all__ = [
    "JAXModelAdapter",
    "KerasModelAdapter",
    "ModelAdapter",
    "ONNXModelAdapter",
    "PyTorchModelAdapter",
    "SklearnModelAdapter",
]

# Mapping of known PyTorch CIFAR architectures to their submodule names
_ARCH_MODULES: dict[str, str] = {
    "DenseNet121": "densenet",
    "DenseNet161": "densenet",
    "DenseNet169": "densenet",
    "DenseNet201": "densenet",
    "DLA": "dla",
    "DPN26": "dpn",
    "DPN92": "dpn",
    "EfficientNetB0": "efficientnet",
    "GoogLeNet": "googlenet",
    "LeNet": "lenet",
    "MobileNet": "mobilenet",
    "MobileNetV2": "mobilenetv2",
    "PNASNetA": "pnasnet",
    "PNASNetB": "pnasnet",
    "PreActResNet18": "preact_resnet",
    "PreActResNet34": "preact_resnet",
    "PreActResNet50": "preact_resnet",
    "PreActResNet101": "preact_resnet",
    "PreActResNet152": "preact_resnet",
    "RegNetX_200MF": "regnet",
    "RegNetX_400MF": "regnet",
    "RegNetY_400MF": "regnet",
    "ResNet18": "resnet",
    "ResNet34": "resnet",
    "ResNet50": "resnet",
    "ResNet101": "resnet",
    "ResNet152": "resnet",
    "ResNeXt29_2x64d": "resnext",
    "ResNeXt29_4x64d": "resnext",
    "ResNeXt29_8x64d": "resnext",
    "ResNeXt29_32x4d": "resnext",
    "SENet18": "senet",
    "ShuffleNetG2": "shufflenet",
    "ShuffleNetG3": "shufflenet",
    "ShuffleNetV2": "shufflenetv2",
    "SimpleDLA": "dla_simple",
    "VGG": "vgg",
    "VGG11": "vgg",
    "VGG13": "vgg",
    "VGG16": "vgg",
    "VGG19": "vgg",
}


def __getattr__(name: str):
    if name == "PyTorchModelAdapter":
        from trustbench.models.pytorch import PyTorchModelAdapter

        return PyTorchModelAdapter
    if name in _ARCH_MODULES:
        import importlib

        mod = importlib.import_module(f"trustbench.models.{_ARCH_MODULES[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
