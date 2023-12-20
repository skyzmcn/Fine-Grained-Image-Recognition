import torch.nn as nn
from torchvision import models

WEIGHTS_DICT = {
    "vgg11": "VGG11_Weights",
    "vgg11_bn": "VGG11_BN_Weights",
    "vgg13": "VGG13_Weights",
    "vgg13_bn": "VGG13_BN_Weights",
    "vgg16": "VGG16_Weights",
    "vgg16_bn": "VGG16_BN_Weights",
    "vgg19": "VGG19_Weights",
    "vgg19_bn": "VGG19_BN_Weights",
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
    "alexnet": "AlexNet_Weights",
    "googlenet": "GoogLeNet_Weights",
}

FEATURE_DIM_DICT = {
    "vgg11": 512 * 7 * 7,
    "vgg11_bn": 512 * 7 * 7,
    "vgg13": 512 * 7 * 7,
    "vgg13_bn": 512 * 7 * 7,
    "vgg16": 512 * 7 * 7,
    "vgg16_bn": 512 * 7 * 7,
    "vgg19": 512 * 7 * 7,
    "vgg19_bn": 512 * 7 * 7,
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "alexnet": 256,
    "googlenet": 1024,
}


class BaseModel(nn.Module):
    def __init__(self, model_name="vgg16", num_class=200, pretrained=False, train_method="all", init_weight=True):
        super().__init__()
        weights = getattr(getattr(models, WEIGHTS_DICT[model_name]), "DEFAULT") if pretrained else None
        if model_name[0:3] == "vgg":
            self.features = getattr(models, model_name)(weights=weights).features
            self.features[-1] = nn.AvgPool2d(kernel_size=1, stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif model_name[0:6] == "resnet":
            self.features = nn.Sequential(*list((getattr(models, model_name)(weights=weights).children())))[0:-2]
            self.maxpool = nn.MaxPool2d(kernel_size=7, stride=7)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(FEATURE_DIM_DICT[model_name]),
            # nn.Dropout(0.5),
            nn.Linear(FEATURE_DIM_DICT[model_name], 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )

        if init_weight:
            if pretrained:
                nn.init.normal_(self.classifier.weight, 0, 0.01)
                nn.init.constant_(self.classifier.bias, 0)
            else:
                self.init_weights()
        else:
            pass

        if train_method == "features_part":
            for parameter in self.classifier.parameters():
                parameter.requires_grad = False
        elif train_method == "classifier_part":
            for parameter in self.features.parameters():
                parameter.requires_grad = False
        else:
            pass

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
            else:
                pass

    def forward(self, x):
        feat = self.features(x)
        out = self.maxpool(feat)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return feat, out


if __name__ == '__main__':
    model = BaseModel("vgg16_bn", 200, False, "all", False)
    print(model)
