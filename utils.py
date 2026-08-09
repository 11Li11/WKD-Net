import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
            epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


class RainfallLoss(nn.Module):
    def __init__(self, omega_t, alpha, beta, c):
        super(RainfallLoss, self).__init__()
        self.omega_t = omega_t
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def forward(self, pred, target):
        w0 = 0.57
        loss_overal1 = torch.sum(((pred >= target)).float() * (1 - w0) * abs(pred - target))
        loss_overal2 = torch.sum(((pred < target)).float() * w0 * abs(pred - target))
        wi = self.alpha * torch.exp(target)
        loss_greater = torch.sum(
            ((pred >= target) & (target >= 0.7)).float() * (1 - self.omega_t) * wi * abs(pred - target))
        loss_less = torch.sum(((pred < target) & (target >= 0.7)).float() * self.omega_t * wi * abs(pred - target))
        num_sample = target.numel()
        total_loss = (loss_overal1 + loss_overal2) / num_sample + (loss_greater + loss_less) / num_sample
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


# class RainfallLoss(nn.Module):
#     def __init__(self, omega_t=0.6, alpha=2.0, threshold=0.7, spatial_weight=0.2):
#         """
#         :param omega_t: 漏报(欠估计)极值时的权重 (默认0.6，宁可多报不可漏报)
#         :param alpha: 极值指数放大的系数
#         :param threshold: 强降水/强回波的阈值 (假设输入已归一化到 [0, 1])
#         :param spatial_weight: 空间容错(缓解双重惩罚)的损失权重
#         """
#         super(RainfallLoss, self).__init__()
#         self.omega_t = omega_t
#         self.alpha = alpha
#         self.w0 = 0.57  # 基础非对称权重 (漏报惩罚 0.57 > 空报惩罚 0.43)
#         self.threshold = threshold
#         self.spatial_weight = spatial_weight
#
#     def forward(self, pred, target):
#         # 提前计算绝对误差，加速计算
#         diff = pred - target
#         abs_diff = torch.abs(diff)
#
#         # ==========================================
#         # 1. 基础全局非对称损失 (Global Asymmetric L1)
#         # ==========================================
#         # pred >= target (空报)，权重为 1-w0 = 0.43
#         # pred < target  (漏报)，权重为 w0 = 0.57
#         weight_overall = torch.where(diff >= 0, 1 - self.w0, self.w0)
#         loss_overall = weight_overall * abs_diff
#
#         # 基础损失除以总像素，计算平均值
#         mean_base_loss = loss_overall.sum() / target.numel()
#
#         # ==========================================
#         # 2. 强降水专属极值损失 (Extreme Rainfall Penalty)
#         # ==========================================
#         # 极值权重指数放大 (前提：确保 target 在 [0, 1] 之间，否则 exp() 会溢出！)
#         wi = self.alpha * torch.exp(target)
#
#         # 极值非对称权重
#         weight_heavy = torch.where(diff >= 0, 1 - self.omega_t, self.omega_t)
#
#         # 找出强降水区域的掩码 (Mask)
#         heavy_mask = (target >= self.threshold).float()
#
#         # 计算强降水损失
#         loss_heavy = heavy_mask * weight_heavy * wi * abs_diff
#
#         # 【关键修复】：强降水损失必须除以【强降水的实际像素数】，而不是总像素！
#         # 加 1e-6 是为了防止整张图都是晴天时除以 0 导致 Loss 变成 NaN
#         heavy_pixels_count = heavy_mask.sum() + 1e-6
#         mean_heavy_loss = loss_heavy.sum() / heavy_pixels_count
#
#         # ==========================================
#         # 3. 空间容错损失 (Spatial Tolerance / Fix Double Penalty)
#         # ==========================================
#         # 允许预测的降水云团有 1~2 个像素的位移偏差，这在气象学上非常关键
#         pooled_pred = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
#         pooled_target = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
#         mean_spatial_loss = torch.mean(torch.abs(pooled_pred - pooled_target))
#
#         # ==========================================
#         # 总损失融合
#         # ==========================================
#         # 如果当前批次没有强降水，mean_heavy_loss 自动接近 0，不影响训练
#         total_loss = mean_base_loss + mean_heavy_loss + (self.spatial_weight * mean_spatial_loss)
#
#         return total_loss


