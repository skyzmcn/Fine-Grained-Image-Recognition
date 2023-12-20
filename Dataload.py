import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import dataset


def default_loader(path):
    return Image.open(path).convert("RGB")


def gray_loader(path):
    return Image.open(path).convert("L").convert("RGB")


class DataSet(dataset.Dataset):
    def __init__(self, mode, size=224, gray=False):
        self.img_pwd = "./CUB_200_2011/images"
        self.img = []
        self.bbox = []
        self.label = []
        with open("./CUB_200_2011/{}.txt".format(mode), "r") as f:
            for i in f.readlines():
                parts = i.strip().split(",")
                img = parts[0]
                label = parts[1]
                bbox = [int(float(coord)) for coord in parts[2:]]
                self.img.append(img)
                self.bbox.append(bbox)
                self.label.append(label)

        if gray:
            self.loader = gray_loader
        else:
            self.loader = default_loader

        if size == 224:
            if mode == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        else:
            if mode == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((600, 600)),
                    transforms.RandomCrop((size, size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((600, 600)),
                    transforms.CenterCrop((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, item):
        img_path = self.img[item]
        bbox = self.bbox[item]
        label = self.label[item]
        img = self.loader(os.path.join(self.img_pwd, img_path))
        if bbox:
            img = img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
        img = self.transform(img)

        return img, int(label)

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    data = DataSet("train_bbox", 224, False)
    data.__getitem__(1000)
    data.__len__()
