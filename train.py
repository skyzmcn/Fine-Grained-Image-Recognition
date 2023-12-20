from torch.utils.data import DataLoader
from model import BaseModel
import time
import numpy as np
import random
from torch.optim import lr_scheduler
from torch.backends import cudnn
from Dataload import DataSet
import os
import torch
from multiprocessing import cpu_count
import torch.nn as nn
from Loss import MCLoss


class Args:
    def __init__(self):
        self.model_name = 'resnet18'
        self.savepath = './Test/'
        self.eps = 3
        self.alpha = 1.5
        self.lambda_ = 10
        self.p = 0.5
        self.num_class = 200
        self.lr = 0.1
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.scheduler = 'multi'
        self.loss = "mcloss"
        self.use_gpu = True
        self.lr_step = 30
        self.lr_gamma = 0.1
        self.total_epoch = 300
        self.batch_size = 32
        self.num_workers = 4
        self.img_size = 224
        self.seed = None
        self.gpu_id = 0
        self.multi_gpus = False
        self.pretrained = False
        self.bbox = False
        self.gray = False
        self.init_weight = False
        self.train_method = "all"
        self.cnums = [3]
        self.cgroups = [200]

# 输出维度表，供设置cnums和cgroups参考使用，下一个版本将字典化
# feat_dim = {
#     "vgg11": 512,
#     "vgg11_bn": 512,
#     "vgg13": 512,
#     "vgg13_bn": 512,
#     "vgg16": 512,
#     "vgg16_bn": 512,
#     "vgg19": 512,
#     "vgg19_bn": 512,
#     "resnet18": 512,
#     "resnet34": 512,
#     "resnet50": 2048,
#     "resnet101": 2048,
#     "resnet152": 2048,
#     "alexnet": 256,
#     "googlenet": 1024,
# }


args = Args()
args.model_name = "resnet50"
args.savepath = './Test/'
args.eps = 3
args.alpha = 1.5
args.lambda_ = 10
args.p = 0.4
args.num_class = 200
args.lr = 0.005
args.weight_decay = 5e-4
args.momentum = 0.9
args.scheduler = 'multi'
args.loss = "mcloss"
args.lr_step = 30
args.lr_gamma = 0.8
args.total_epoch = 300
args.batch_size = 32
args.num_workers = cpu_count()
args.img_size = 224
args.seed = None
args.gpu_id = 0
args.multi_gpus = False
args.pretrained = True
args.bbox = True
args.gray = False
args.init_weight = False
args.train_method = "all"
args.cnums = [10, 11]
args.cgroups = [152, 48]


def train():
    model.train()

    epoch_loss = 0
    correct = 0.
    total = 0.
    t1 = time.time()
    for idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.long().to(device)
        feat, out = model(data)

        loss = criterion(out, labels) + args.alpha * mcloss(feat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        total += data.size(0)
        _, pred = torch.max(out, 1)
        correct += pred.eq(labels).sum().item()

    acc = correct / total
    loss = epoch_loss / total

    print(f'loss:{loss:.4f} acc@1:{acc:.4f} time:{time.time() - t1:.2f}s', end=' --> ')
    return {'loss': loss, 'acc': acc}


def test(epoch):
    model.eval()

    epoch_loss = 0
    correct = 0.
    total = 0.
    with torch.no_grad():
        for idx, (data, labels) in enumerate(testloader):
            data, labels = data.to(device), labels.long().to(device)
            _, out = model(data)

            loss = criterion(out, labels)

            epoch_loss += loss.item() * data.size(0)
            total += data.size(0)
            _, pred = torch.max(out, 1)
            correct += pred.eq(labels).sum().item()

        acc = correct / total
        loss = epoch_loss / total

        print(f'test loss:{loss:.4f} acc@1:{acc:.4f}', end=' ')

    global best_acc, best_epoch
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        if isinstance(model, nn.parallel.distributed.DistributedDataParallel):
            state = {
                'net': model.module.state_dict(),
                'acc': acc,
                'epoch': epoch
            }
        else:
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch
            }
        torch.save(state, os.path.join(savepath, 'ckpt.pth'))
        print('*')
    else:
        print()

    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, loss, acc))

    return {'loss': loss, 'acc': acc}


def plot(d, mode='train', best_acc_=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.suptitle('%s_curve' % mode)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    epochs = len(d['acc'])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), d['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), d['acc'], label='acc')
    if best_acc_ is not None:
        plt.scatter(best_acc_[0], best_acc_[1], c='r')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(savepath, '%s.jpg' % mode), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0.
    use_gpu = False

    if args.seed is not None:
        print('use random seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    if torch.cuda.is_available():
        use_gpu = True
        cudnn.benchmark = True

    # loss
    if args.loss == 'celoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'mcloss':
        criterion = torch.nn.CrossEntropyLoss()
        mcloss = MCLoss(num_classes=args.num_class, cnums=args.cnums, cgroups=args.cgroups, p=args.p, lambda_=args.lambda_)
    else:
        pass

    # dataloader
    if args.bbox:
        train_set = DataSet("train_bbox", args.img_size, args.gray)
        test_set = DataSet("test_bbox", args.img_size, args.gray)
    else:
        train_set = DataSet("train", args.img_size, args.gray)
        test_set = DataSet("test", args.img_size, args.gray)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    # model
    model = BaseModel(args.model_name, args.num_class, args.pretrained,
                      args.train_method, args.init_weight)

    if torch.cuda.device_count() > 1 and args.multi_gpus:
        print('we will use multi-gpus.')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.distributed.init_process_group(backend="ncc", init_method='tcp://localhost:23456', rank=0, world_size=1)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        device = ('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

    # optim
    if args.pretrained:
        optimizer = torch.optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': args.lr},
             {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': 10 * args.lr}],
            lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}],
            lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    print('init_learn_rate={}, weight_decay={}, momentum={}'.format(args.lr, args.weight_decay, args.momentum))

    if args.scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=args.lr_gamma, last_epoch=-1)

    # savepath
    savepath = os.path.join(args.savepath, args.model_name)
    savepath = savepath + '_' + args.loss + '_' + str(args.img_size) + '_' + args.scheduler

    if args.seed is not None:
        savepath = savepath + '_s' + str(args.seed)

    if not args.pretrained:
        savepath = savepath + '_' + str(args.eps)

    print('savepath:', savepath)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(os.path.join(savepath, 'setting.txt'), 'w')as f:
        for k, v in vars(args).items():
            f.write('{}:{}\n'.format(k, v))

    f = open(os.path.join(savepath, 'log.txt'), 'w')
    f.close()

    total = args.total_epoch
    start = time.time()

    train_info = {'loss': [], 'acc': []}
    test_info = {'loss': [], 'acc': []}

    for epoch in range(total):
        print('epoch[{:>3}/{:>3}]'.format(epoch, total), end=' ')
        d_train = train()
        scheduler.step()
        d_test = test(epoch)

        for k in train_info.keys():
            train_info[k].append(d_train[k])
            test_info[k].append(d_test[k])

        plot(train_info, mode='train')
        plot(test_info, mode='test', best_acc_=[best_epoch, best_acc])

    end = time.time()
    print('total time:{}m{:.2f}s'.format((end - start) // 60, (end - start) % 60))
    print('best_epoch:', best_epoch)
    print('best_acc:', best_acc)
    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('# best_acc:{:.4f}, best_epoch:{}'.format(best_acc, best_epoch))
