import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import wandb
import torch.optim as optim
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet
from models.loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels

from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--log_dir', default='logs/log')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
parser.add_argument('--vggt_feat', type=str, default=None, help='where to extract vggt feature, used in vggt forward pass') # None | "interp" | "pos"
parser.add_argument('--vggt_fuse', type=str, default=None, help='how to use vggt feature')  # replace | fuse | None
parser.add_argument('--proj_name', type=str, default='grasp_vggt', help='name for wandb project')
cfgs = parser.parse_args()
cfgs.exp_name = (
    f"vggt_f{cfgs.vggt_fuse if cfgs.vggt_fuse is not None else 'n'}_"
    f"p{cfgs.vggt_feat if cfgs.vggt_feat is not None else 'n'}"
)
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def namespace_to_dict(ns):
    if isinstance(ns, (SimpleNamespace, argparse.Namespace)):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# ------------------- load train data ------------------
grasp_labels = load_grasp_labels(cfgs.dataset_root)
ret_vggt = True if cfgs.vggt_fuse is not None else False # if ret_vggt True then also return image preprocssed for vggt
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True, vggt=ret_vggt)
print('train dataset length: ', len(TRAIN_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('train dataloader length: ', len(TRAIN_DATALOADER))

# ------------------- load test data ------------------
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='test_seen',  # or 'test_similar'
                               num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, 
                               load_label=True, augment=False, vggt=ret_vggt)
print('test dataset length: ', len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,      
                             num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('test dataloader length: ', len(TEST_DATALOADER))

# ----------------- model and optimizer ---------------
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True, vggt=cfgs.vggt_fuse)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))

# ------ load VGGT model 
device = "cuda" if torch.cuda.is_available() else "cpu" 
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_vggt_feat(batch_data_label):
    # ----------- inputs -----------
    images = batch_data_label['image'].unsqueeze(1)  # [B, 1, 3, H, W]
    uv = batch_data_label['uv']         # [B, 15000, 2] 
    
    # ----------- Normalize (u, v) → (-1, 1) ------------
    H_img, W_img = 720, 1280
    u, v = uv[..., 0], uv[..., 1]  # [B, N]
    u_norm = (u / (W_img - 1)) * 2 - 1
    v_norm = (v / (H_img - 1)) * 2 - 1

    grid = torch.stack([u_norm, v_norm], dim=-1)  # [B, N, 2]
    grid = grid.unsqueeze(1)                      # [B, 1, N, 2]
    
    # ----------- VGGT forward -----------
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = vggt_model(images, sg_feat = cfgs.vggt_feat) # 'interp': after interploation, 'pos': after adding pos embedding
            sg_feature = predictions['sg_feature']

    # ------------ Sample VGGT features -------------
    sampled_sg_feat = F.grid_sample(
        sg_feature.squeeze(1),   # [B, C, Hf, Wf]
        grid,                    # [B, 1, N, 2]
        mode='bilinear',
        align_corners=True
    ) # [B, 128, 1, 15000]
    sampled_sg_feat = sampled_sg_feat.squeeze(2).permute(0, 2, 1) # [B, 15000, 128]
    batch_data_label['sg_features'] = sampled_sg_feat  
    

def train_one_epoch(epoch_cnt=None):
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 40
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            elif 'meta' not in key:
                batch_data_label[key] = batch_data_label[key].to(device)

        # ----------- get vggt features -----------
        if cfgs.vggt_fuse is not None:
            get_vggt_feat(batch_data_label)
        
        # ------------ model training  ------------ 
        end_points = net(batch_data_label)
        loss, end_points = get_loss(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))

            train_metrics = {f'train/{key}': stat_dict[key] / batch_interval for key in stat_dict}
            wandb.log(train_metrics, step=epoch_cnt * len(TRAIN_DATALOADER) + batch_idx)
            
            for key in sorted(stat_dict.keys()):
                log_string('Train mean %s: %f' % (key, train_metrics[f'train/{key}']))
                stat_dict[key] = 0


def evaluate_one_epoch(epoch_cnt=None):
    stat_dict = {}
    net.eval()

    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print(f'Eval batch: {batch_idx}')

        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            elif 'meta' not in key:
                batch_data_label[key] = batch_data_label[key].to(device)
        
        # ----------- get vggt features -----------
        if cfgs.vggt_fuse is not None:
            get_vggt_feat(batch_data_label)

        with torch.no_grad():
            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points)

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    eval_metrics = { f'val/{key}': stat_dict[key] / float(batch_idx + 1) for key in stat_dict }
    wandb.log(eval_metrics, step=(epoch_cnt + 1) * len(TRAIN_DATALOADER))
    log_string('------------- evaluation -------------')
    for key in sorted(eval_metrics.keys()):
        log_string('mean %s: %f' % (key, eval_metrics[key]))
        stat_dict[key] = 0
    return stat_dict['loss/overall_loss'] / float(batch_idx + 1)


def train(start_epoch):
    global EPOCH_CNT
    wandb.login()
    os.environ["WANDB_MODE"] = "online"
    wandb.init(project=cfgs.proj_name, config=namespace_to_dict(cfgs), name=cfgs.exp_name)
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch(EPOCH_CNT)
        
        val_loss = evaluate_one_epoch(EPOCH_CNT)

        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
