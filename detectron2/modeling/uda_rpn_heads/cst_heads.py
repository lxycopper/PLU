# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from .ResNet import resnet18,resnet34,resnet50
from typing import Tuple

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
  
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class TsallisEntropy(nn.Module):
    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        
        pred = F.softmax(logits / self.temperature, dim=1) 
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  
        
        sum_dim = torch.sum(pred * entropy_weight, dim = 0).unsqueeze(dim=0)
      
        return 1 / (self.alpha - 1) * torch.sum((1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim = -1)))

class CST_Net(nn.Module):
    def __init__(self,device):
        super(CST_Net, self).__init__()
        self.net = resnet50(pretrained=False,num_classes=[2],device=device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1,3,128,128)
        predictions,f = self.net(x)
        return predictions,f

class CST_Module(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(CST_Module, self).__init__()
        self.cfg = cfg.clone()
        self.input_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM # representation_size
        self.uda_net = CST_Net(torch.device(cfg.MODEL.DEVICE)) # 要保存的模型
        self.uda_auto_labelling = cfg.MODEL.ROI_HEADS.UDA_AUTO_LABELLING
        
        self.num_classes = 2
        self.trade_off = [0.08, 2.0,0.5]
        self.threshold = 0.97
        self.temperature = 3.0
        self.alpha = 1.9
        self.TsallisEntropy = TsallisEntropy(self.temperature,self.alpha)

        self.batch_size = 4

    def forward(self, x_source, source_cls_labels = None, t_weak=None, t_strong=None):
        if not self.uda_auto_labelling:
            x = torch.cat((x_source, t_weak),dim=0)
            y,f = self.uda_net(x)
            y_t_u,_ = self.uda_net(t_strong)

            f_s, f_t = f.chunk(2, dim=0)
            y_s, y_t = y.chunk(2, dim=0)

            max_prob, pred_u = torch.max(F.softmax(y_t,dim=-1), dim=-1)
            Lu = (F.cross_entropy(y_t_u, pred_u,reduction='none') * max_prob.ge(self.threshold).float().detach()).mean()

            # compute cst
            target_data_train_r = f_t


            target_data_train_r = target_data_train_r / (torch.norm(target_data_train_r, dim = -1).reshape(target_data_train_r.shape[0], 1))
            target_data_test_r = f_s
            target_data_test_r = target_data_test_r / (torch.norm(target_data_test_r, dim = -1).reshape(target_data_test_r.shape[0], 1))
            target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
            target_kernel_r = target_gram_r
            test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
            test_kernel_r = test_gram_r

            target_train_label_r = torch.nn.functional.one_hot(pred_u, self.num_classes) - 1 / float(self.num_classes)

            #print('shape',f_t.shape,y_t.shape,test_kernel_r.shape,target_kernel_r.shape,target_train_label_r.shape)
            batch_size = target_kernel_r.shape[0]
            sum_kernel = target_kernel_r + 0.001 * torch.eye(batch_size).cuda()
            #print('sum',sum_kernel.shape)
            target_test_pred_r = test_kernel_r.mm(torch.inverse(sum_kernel)).mm(target_train_label_r)

            reverse_loss = torch.nn.MSELoss()(target_test_pred_r, 
            torch.nn.functional.one_hot(source_cls_labels, self.num_classes) - 1 / float(self.num_classes)) 

            cls_loss = F.cross_entropy(y_s, source_cls_labels)
            transfer_loss = self.TsallisEntropy(y_t)

            if Lu != 0:
                loss = cls_loss + transfer_loss * self.trade_off[0] + reverse_loss * self.trade_off[1] + Lu * self.trade_off[2]
            else: 
                loss = cls_loss + transfer_loss * self.trade_off[0] + reverse_loss * self.trade_off[1]
            return loss
            
        cls_logits,_ = self.uda_net(x_source)
        cls_logits = F.softmax(cls_logits,dim=-1)
        cls_out = torch.argmax(cls_logits,dim=-1)
        return cls_out, cls_logits

def build_cst_heads(cfg):
    #if cfg.MODEL.OWOD_ON:
    return CST_Module(cfg)
    #return []