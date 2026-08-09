from torch.cuda.amp import autocast, GradScaler
import h5py
from torch.utils.data import DataLoader
from models.exprecast import exPreCast
from engine import *
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from configs.config_setting import setting_config
import warnings

warnings.filterwarnings("ignore")

def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    print('#----------Preparing dataset----------#')

    print('#----------Preparing dataset----------#')
    with h5py.File('/home/user/D_Disk/libo/Net_3D/merged_data.h5', 'r') as hf:
        data = hf['vil'][:]

    num_samples = int(data.shape[0])
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    group_size = 8

    num_train = int(train_ratio * (num_samples - group_size + 1))
    num_val = int(val_ratio * (num_samples - group_size + 1))
    num_test = (num_samples - group_size + 1) - num_train - num_val
    groups = [data[i:i + group_size] for i in range(0, num_samples - group_size)]

    train_groups = groups[:num_train]
    valid_groups = groups[num_train:num_val + num_train]
    test_groups = groups[num_train + num_val:]

    train_groups = torch.tensor(np.array(train_groups))
    valid_groups = torch.tensor(np.array(valid_groups))
    test_groups = torch.tensor(np.array(test_groups))

    print(test_groups.shape, 'test_groups')

    train_loader = DataLoader(train_groups, config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_groups, config.batch_size, drop_last=True)
    test_loader = DataLoader(test_groups, config.batch_size, drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = exPreCast(
        input_frames=model_cfg['input_frames'],
        output_frames=model_cfg['predicted_frames'],
    )
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    # [修改点]：设置三个独立的最优记录器
    print('#----------Set other params----------#')
    min_loss = 99999.0
    max_csi = -1.0
    max_acc = -1.0

    start_epoch = 1

    total_params = sum(p.numel() for p in model.parameters())
    print("Number of Model Parameters:", total_params)
    print('#----------Training----------#')

    loss1_list = []

    for epoch in tqdm(range(start_epoch, config.epochs + 1)):
        torch.cuda.empty_cache()

        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer,
            scheduler, epoch, logger, config, scaler=scaler
        )
        loss1_list.append(train_loss)

        # [修改点]：接收三个返回值
        val_loss, val_csi, val_acc = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config
        )

        print(f'Epoch: {epoch} | val_loss: {val_loss:.4f} | val_csi: {val_csi:.4f} | val_acc: {val_acc:.4f}')

        # [修改点]：1. 依据最低 Loss 保存模型
        if val_loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_loss.pth'))
            min_loss = val_loss

        # [修改点]：2. 依据最高 CSI 保存模型 (仅在执行了验证指标计算的 epoch 更新)
        if val_csi > max_csi and val_csi != -1.0:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_csi.pth'))
            max_csi = val_csi

        # [修改点]：3. 依据最高 ACC 保存模型 (仅在执行了验证指标计算的 epoch 更新)
        if val_acc > max_acc and val_acc != -1.0:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_acc.pth'))
            max_acc = val_acc

        # 常规间隔保存
        if epoch % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'Mamba_UNet_latest.pth'))

    # ==========================================
    # [修改点]：测试阶段分别验证三个最优模型
    # ==========================================
    print('\n#----------Testing----------#')

    best_models = {
        'Best Loss Model': 'best_loss.pth',
        'Best CSI Model': 'best_csi.pth',
        'Best ACC Model': 'best_acc.pth'
    }

    for model_name, file_name in best_models.items():
        weight_path = os.path.join(checkpoint_dir, file_name)
        if os.path.exists(weight_path):
            print(f'\n========== Evaluating {model_name} ==========')
            logger.info(f'\n========== Evaluating {model_name} ==========')

            # 加载对应的权重
            best_weight = torch.load(weight_path, map_location=torch.device('cpu'))
            model.module.load_state_dict(best_weight)

            test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                test_data_name=model_name
            )

    return loss1_list

if __name__ == '__main__':
    config = setting_config
    loss1_list = main(config)