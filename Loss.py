import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=200, feat_dim=1, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        """
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        """
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = nn.functional.normalize(W, p=2, dim=1)

        x = nn.functional.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class AngularSoftmaxWithLoss(nn.Module):
    """"""

    def __init__(self, gamma=0):
        super(AngularSoftmaxWithLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.iter += 1
        target = target.view(-1, 1)

        index = input[0].data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = Variable(index.byte())

        # Tricks
        # output(θyi) = (lambda * cos(θyi) + (-1) ** k * cos(m * θyi) - 2 * k)) / (1 + lambda)
        #             = cos(θyi) - cos(θyi) / (1 + lambda) + Phi(θyi) / (1 + lambda)
        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = input[0] * 1.0
        output[index] -= input[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input[1][index] * 1.0 / (1 + self.lamb)

        # softmax loss
        logit = nn.functional.log_softmax(output)
        logit = logit.gather(1, target).view(-1)
        pt = logit.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss


class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats=1,  # 输入数据x的维度
                 n_classes=200,  # 分类数目
                 m=0.35,  # margin的大小，一般不用变动
                 s=30  # 放大尺度因子，论文推荐使用30):
                 ):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda:
            lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss, costh_m_s


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        # print('Focal Loss:')
        # print('    Alpha = {}'.format(self.alpha))
        # print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """

        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = nn.functional.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1,
                                             labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class MCLoss(nn.Module):
    def __init__(self, num_classes=200, cnums=[10, 11], cgroups=[152, 48], p=0.4, lambda_=10):
        super().__init__()
        if isinstance(cnums, int): cnums = [cnums]
        elif isinstance(cnums, tuple): cnums = list(cnums)
        assert isinstance(cnums, list), print("Error: cnums should be int or a list of int, not {}".format(type(cnums)))
        assert sum(cgroups) == num_classes, print("Error: num_classes != cgroups.")

        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, feat, targets):
        n, c, h, w = feat.size()
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[:i + 1]))
        # L_div branch
        feature = feat
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(nn.functional.softmax(feature[:, sp[i - 1]:sp[i]].view(n, -1, h * w), dim=2).view(n, -1, h, w)) # Softmax

        l_div = 0.
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = nn.functional.max_pool2d(features.view(n, -1, h * w), kernel_size=(self.cnums[i], 1), stride=(self.cnums[i], 1))
            l_div = l_div + (1.0 - torch.mean(torch.sum(features, dim=2)) / (self.cnums[i] * 1.0))

        # L_dis branch
        mask = self._gen_mask(self.cnums, self.cgroups, self.p).expand_as(feat)
        if feat.is_cuda: mask = mask.cuda()

        feature = mask * feat  # CWA
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(feature[:, sp[i - 1]:sp[i]])

        dis_branch = []
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = nn.functional.max_pool2d(features.view(n, -1, h * w), kernel_size=(self.cnums[i], 1), stride=(self.cnums[i], 1))
            dis_branch.append(features)

        dis_branch = torch.cat(dis_branch, dim=1).view(n, -1, h, w)  # CCMP
        dis_branch = self.avgpool(dis_branch).view(n, -1)  # GAP

        l_dis = self.celoss(dis_branch, targets)

        return l_dis + self.lambda_ * l_div

    def _gen_mask(self, cnums, cgroups, p):
        """
        :param cnums:
        :param cgroups:
        :param p: float, probability of random deactivation
        """
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(-1,)
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
            drop_idx = np.stack(drop_idx, axis=0).reshape(-1,)
            foo[drop_idx] = 0.
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar

