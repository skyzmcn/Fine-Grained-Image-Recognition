## Mutual-Channel Loss for Fine-Grained Image Classification

Reimplementation of MCLoss on CUB_200_2011 dataset. 

The code of mcloss is integrated into a class.

#### Requirements:

- python 3
- PyTorch 2.0.0
- CUDA 11.8
- torchvision

#### Results:

- The experiment is conducted with 2 RTX 2080Ti GPUs, and the batchsize is set to 32.
- Trained from scratch:
  - Init_lr: 0.1 for all
  - lr_scheduler: MultiStepLR (total-300, milestones-[150, 225], lr_gamma-0.1)

- Uses cropped image set

| Model     | cnums | cgroups | p    | alpha | img_size | feat_dim | Acc   |
| --------- | ----- | ------- | ---- | ----- | -------- | -------- | ----- |
| ResNet18  | [3]   | [200]   | 0.5  | 1.5   | 224\*224 | 512      | 74.71 |
| ResNet34  | [3]   | [200]   | 0.5  | 1.5   | 224\*224 | 512      | 74.61 |
| ResNet50  | [3]   | [200]   | 0.5  | 1.5   | 224\*224 | 2048     | 74.37 |
| ResNet101 | [3]   | [200]   | 0.5  | 1.5   | 224\*224 | 2048     | 75.21 |
| ResNet152 | [3]   | [200]   | 0.5  | 1.5   | 224\*224 | 2048     | 76.04 |

| Model     | cnums | cgroups | loss    | lr(lr0,γ) | img_size | Acc   |
| --------- | ----- | ------- | ------- | --------- | -------- | ----- |
| ResNet152 | [3]   | [200]   | MC-Loss | 0.1-0.1   | 224\*224 | 76.04 |
| ResNet152 | [3]   | [200]   | MC-Loss | 0.1-0.1   | 448\*448 | 68.65 |
| ResNet152 | [3]   | [200]   | MC-Loss | 0.1-0.1   | 600\*600 | 61.84 |
| ResNet152 | [3]   | [200]   | MC-Loss | 0.1-0.2   | 224\*224 | 75.63 |
| ResNet152 | [3]   | [200]   | MC-Loss | 0.1-0.05  | 224\*224 | 75.54 |
| ResNet152 | [3]   | [200]   | MC-Loss | 0.1-0.01  | 224\*224 | 72.47 |

- Uses uncropped image set
  | ResNet152  |[3]|[200]|MC-Loss|0.1-0.1| 224\*224  | 65.33 ||
  | ResNet152  |[3]|[200]|MC-Loss|0.1-0.1| 448\*448  | 66.42 ||
  | ResNet152  |[3]|[200]|MC-Loss|0.1-0.1| 600\*600  | 67.92 ||

| Base Model | Data Augmentation | Size |   Method   | Max_accuracy |
| :--------: | :---------------: | :--: | :--------: | :----------: |
|   VGG11    |        ——         | 224  |     ——     |    41.62     |
|   VGG11    |       Flip        | 224  |     ——     |    55.52     |
|   VGG11    |       Crop        | 224  |     ——     |    53.47     |
|   VGG11    |     Flip+Crop     | 224  |     ——     |    63.10     |
|   VGG11    |  Flip+Crop+Bbox   | 224  |     ——     |    72.17     |
|   VGG13    |        ——         | 224  |     ——     |    66.13     |
|   VGG16    |        ——         | 224  |     ——     |    64.13     |
|   VGG19    |        ——         | 224  |     ——     |    57.42     |
|  RESNET18  |  Flip+Crop+Bbox   | 224  |     ——     |    74.71     |
|  RESNET18  |  Flip+Crop+Bbox   | 448  |     ——     |    67.67     |
|  RESNET34  |  Flip+Crop+Bbox   | 224  |     ——     |    74.61     |
|  RESNET50  |  Flip+Crop+Bbox   | 224  |     ——     |    74.37     |
| RESNET101  |  Flip+Crop+Bbox   | 224  |     ——     |    75.21     |
| RESNET152  |  Flip+Crop+Bbox   | 224  |     ——     |    76.04     |
| RESNET152  |  Flip+Crop+Bbox   | 448  |     ——     |    68.65     |
| RESNET152  |  Flip+Crop+Bbox   | 600  |     ——     |    61.84     |
|  RESNET18  |     Flip+Crop     | 224  |     ——     |    59.89     |
| RESNET152  |     Flip+Crop     | 224  |     ——     |    63.33     |
| RESNET152  |     Flip+Crop     | 448  |     ——     |    64.42     |
| RESNET152  |     Flip+Crop     | 600  |     ——     |    65.92     |
|   VGG16    |     Flip+Crop     | 224  |    BCNN    |    74.81     |
|   VGG16    |     Flip+Crop     | 448  |    BCNN    |    82.51     |
|   VGG19    |     Flip+Crop     | 448  | Pretrained |    81.53     |
|  RESNET50  |     Flip+Crop     | 448  | Pretrained |    86.97     |
|  RESNET50  |  Flip+Crop+Bbox   | 224  | Pretrained |   Running    |


***PS: the first feat dimension should equal to np.dot(cnums, cgroups) and sum(cgroups) should equal to num_class. ***


#### Reference:

- The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification (TIP 2020) [DOI](https://doi.org/10.1109/TIP.2020.2973812)

- Official code: https://github.com/PRIS-CV/Mutual-Channel-Loss





