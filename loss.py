import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicAMSoftmax(nn.Module):
    def __init__(self, n_class, m=0.2, s=30.0, c=1.0, dim=192):
        super(HyperbolicAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.c = c

        self.class_centers = nn.Parameter(torch.randn(n_class, dim) * 1e-3)

        self.ce = nn.CrossEntropyLoss(reduction='none')

        print(f"Initialized Hyperbolic-AMSoftmax with m={m}, s={s}, curvature={c}")

    def proj_to_ball(self, x, eps=1e-5):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        max_norm = (1.0 - eps) / (self.c ** 0.5)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def poincare_distance(self, x, y, eps=1e-5):
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(max=1 - eps)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(max=1 - eps)
        diff = x - y
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)

        num = 2 * diff_norm.pow(2)
        denom = (1 - x_norm.pow(2)) * (1 - y_norm.pow(2))
        return torch.acosh(1 + num / denom.clamp_min(eps))


    def forward(self, x, label):
        x_hyp = self.proj_to_ball(x)
        class_centers_hyp = self.proj_to_ball(self.class_centers)

        B, D = x_hyp.shape
        C = class_centers_hyp.shape[0]

        x_exp = x_hyp.unsqueeze(1).expand(B, C, D)
        c_exp = class_centers_hyp.unsqueeze(0).expand(B, C, D)

        dist = self.poincare_distance(x_exp, c_exp).squeeze(-1)
        positive_dist = dist.gather(1, label.view(-1, 1)).squeeze(1).detach()

        margin = torch.zeros_like(dist)
        margin.scatter_(1, label.view(-1, 1), self.m)
        dist_m = dist + margin

        logits = -self.s * dist_m
        loss = self.ce(logits, label)

        pred = torch.argmax(-dist, dim=1)
        acc = pred.eq(label).sum().item() / float(label.size(0))

        return loss, torch.softmax(logits, dim=-1), acc, positive_dist


class Softmax(nn.Module):
    def __init__(self, n_class):
        super(Softmax, self).__init__()
        self.fc = nn.Linear(192, n_class)
        self.criertion = nn.CrossEntropyLoss(reduction='none')
        print('Use the cross-entropy loss function')

    def forward(self, x, label=None):
        output = self.fc(x)
        loss = self.criertion(output, label.to(torch.long))

        with torch.no_grad():
            acc = torch.argmax(output, dim=1).eq(label).sum().item() / float(label.size(0))
        return loss, output, acc


class SoftmaxScale(nn.Module):
    def __init__(self, n_class):
        super(SoftmaxScale, self).__init__()
        self.fc = nn.Linear(192, n_class)
        self.s = 30
        self.criertion = nn.CrossEntropyLoss(reduction='none')
        print('Use the cross-entropy loss function')

    def forward(self, x, label=None):
        output = self.fc(x)
        output = self.s * output
        loss = self.criertion(output, label.to(torch.long))

        with torch.no_grad():
            acc = torch.argmax(output, dim=1).eq(label).sum().item() / float(label.size(0))
        return loss, output, acc


class AMSoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.W = torch.nn.Parameter(torch.randn(192, n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(reduction='none')
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None, gate=None):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()

        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.to(costh.device)

        costh_m = costh - delt_costh
        logits = self.s * costh_m

        loss = self.ce(logits, label.to(torch.long))

        with torch.no_grad():
            output = torch.mm(x, self.W)
            acc = torch.argmax(output, dim=1).eq(label).sum().item() / float(label.size(0))
        return loss, output, acc


class AAMSoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        super(AAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        with torch.no_grad():
            output = F.linear(x, self.weight)
            acc = torch.argmax(output, dim=1).eq(label).sum().item() / float(label.size(0))
        return loss, output, acc


class RAMSoftmax(nn.Module):
    def __init__(self, n_class, m=0.2, s=30, **kwargs):
        super(RAMSoftmax, self).__init__()

        self.m = m
        self.s = s
        self.W = torch.nn.Parameter(torch.randn(192, n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised RAM-Softmax m=%.3f s=%.3f'%(self.m, self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == 192

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m

        if costh_m_s.is_cuda: label_view=label_view.cuda()
        delt_costh_m_s = costh_m_s.gather(1, label_view).repeat(1,costh_m_s.size()[1])

        costh_m_s_reduct = costh_m_s - delt_costh_m_s

        costh_relu = torch.where(costh_m_s_reduct < 0.0, torch.zeros_like(costh_m_s), costh_m_s)
        loss    = self.ce(costh_relu, label)

        with torch.no_grad():
            output = torch.mm(x, self.W)
            acc = torch.argmax(output, dim=1).eq(label).sum().item() / float(label.size(0))
        return loss, output, acc


if __name__ == '__main__':
    pass

