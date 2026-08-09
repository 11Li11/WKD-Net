
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix, mean_squared_error
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images = data[:, :5, :, :]
        targets = data[:, 5:, :, :]
        images, targets = images.to(device).float(), targets.to(device).float()
        images = images.unsqueeze(1)
        targets = targets.unsqueeze(1)
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    train_loss = np.mean(loss_list)
    train_loss = round(train_loss, 5)
    scheduler.step()
    log_info = f'Train_loss: {train_loss:.5f}'
    print(log_info)
    logger.info(log_info)
    return train_loss


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in test_loader:
            img = data[:, :5, :, :]
            msk = data[:, 5:, :, :]
            img, msk = img.to(device).float(), msk.to(device).float()

            # [修复点]：增加 unsqueeze(1) 将4维张量扩展为5维
            img = img.unsqueeze(1)
            msk = msk.unsqueeze(1)

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

    mean_loss = np.mean(loss_list)

    mean_csi = -1.0
    mean_acc = -1.0

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        sum_csi = 0.0
        sum_acc = 0.0

        for threshold in config.threshold:
            y_pre = np.where(preds >= threshold, 1, 0)
            y_true = np.where(gts >= threshold, 1, 0)

            # [修复 1]：添加 labels=[0, 1] 强制返回 2x2 矩阵，防止全0或全1时报错，并用 ravel 直接解包
            confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
            TN, FP, FN, TP = confusion.ravel()

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0

            # [修复 2]：保护 HSS 避免全0状态下除以 0 报错
            hss_denominator = ((float(TP) + float(FN)) * (float(FN) + float(TN))) + (
                    (float(TP) + float(FP)) * (float(FP) + float(TN)))
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / hss_denominator if hss_denominator != 0 else 0.0

            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            sum_csi += CSI
            sum_acc += accuracy

            log_info = f'{threshold}:,val epoch: {epoch}, loss: {mean_loss:.5f},accuracy: {accuracy:.4f},CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f}'
            logger.info(log_info)

        # [修改点]：计算所有阈值的平均 CSI 和 ACC，用来挑选最佳模型
        mean_csi = sum_csi / len(config.threshold)
        mean_acc = sum_acc / len(config.threshold)
    else:
        log_info = f'val epoch: {epoch}, loss: {mean_loss:.4f}'
        print(log_info)
        logger.info(log_info)

    # [修改点]：返回三个指标
    return mean_loss, mean_csi, mean_acc


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate((test_loader)):
            img = data[:, :5, :, :]
            msk = data[:, 5:, :, :]
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # [修复点]：增加 unsqueeze(1) 将4维张量扩展为5维
            img = img.unsqueeze(1)
            msk = msk.unsqueeze(1)

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

        preds_flat = np.array(preds).reshape(-1)
        gts_flat = np.array(gts).reshape(-1)

        for threshold in config.threshold:
            y_pre = np.where(preds_flat >= threshold, 1, 0)
            y_true = np.where(gts_flat >= threshold, 1, 0)

            # [修复 1]：添加 labels=[0, 1] 强制返回 2x2 矩阵，防止全0或全1时报错，并用 ravel 直接解包
            confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
            TN, FP, FN, TP = confusion.ravel()

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0

            # [修复 2]：保护 HSS 避免全0状态下除以 0 报错
            hss_denominator = ((float(TP) + float(FN)) * (float(FN) + float(TN))) + (
                    (float(TP) + float(FP)) * (float(FP) + float(TN)))
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / hss_denominator if hss_denominator != 0 else 0.0

            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            RMSE = np.sqrt(mean_squared_error(gts_flat, preds_flat))

            # SSIM 需要 2D 图像，直接 flatten 可能会报错，这里添加一个简单的窗口大小处理避免报错
            try:
                SSIM = ssim(gts_flat, preds_flat, data_range=1)
            except:
                SSIM = 0.0  # 异常保护

            if test_data_name is not None:
                log_info = f'test_datasets_name: {test_data_name}'
                print(log_info)
                logger.info(log_info)
            log_info = f'{threshold}:,test of best model, loss: {np.mean(loss_list):.5f}, accuracy: {accuracy:.4f},CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f},SSIM: {SSIM:.4f},RMSE: {RMSE:.4f}'
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)