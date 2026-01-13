import numpy as np
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_1 import *
from torch.autograd import Variable
from IPython.display import clear_output
# from UNetFormer_DINO import UNetFormer2 as MFNet
# from UNetFormer_test3 import UNetFormer as MFNet
from model.build_model.Build_MultiSenseSeg import Build_MultiSenseSeg
# from loss_test import CombinedLoss
import math
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    random.seed(seed)                 # 控制 ISPRS_dataset 里用到的 random.randint / random.random
    np.random.seed(seed)              # 以后如果有 numpy 随机也会受控
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 相关：尽量走确定性实现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = 666
set_seed(SEED)
# 新增：TensorBoard 日志
if MODE=="Train":
    if DATASET == "Potsdam":
        writer = SummaryWriter(log_dir="./runs/multisenseseg_p2v")
    elif DATASET == "Hunan":
        writer = SummaryWriter(log_dir="./runs/multisenseseg_h")
    else:
        writer = SummaryWriter(log_dir="./runs/multisenseseg_v2p")

IN_CHANS = (3, 1)
# net = Build_MultiSenseSeg(
#     n_classes=N_CLASSES,
#     in_chans=IN_CHANS,
#     # 以下参数为默认值，可根据显存大小和精度需求调整
#     decoder_chans=64,   # 解码器通道数
#     aux=False
# )
# net = Build_MultiSenseSeg(
#     n_classes=N_CLASSES,
#     in_chans=IN_CHANS,
#     # 以下参数为默认值，可根据显存大小和精度需求调整
#     decoder_chans=512,   # 解码器通道数
#     patch_size=4,        # Patch Embedding 大小
#     embed_dim=96,        # 基础 Embedding 维度
#     depths=(2, 2, 8, 2), # 每个 Stage 的深度
#     num_heads=(3, 6, 12, 24), # Attention Head 数量
#     aux=False
# )
# ---------------------------------------------------------
# 配置参数：MultiSenseSeg-2-B (Base)
# ---------------------------------------------------------
# 1. 数据集相关 (需要根据你的实际数据修改)

IN_CHANS = (3, 1)      # 输入通道 (例如 RGB=3, DSM=1)。"-2-" 代表双模态，所以这里长度必须为2

# 2. 模型核心配置 (对应 Base 版本参数)
# Modal dim = 32
HEAD_OUT_CHANS = 32
# embed dim = 96
EMBED_DIM = 96
# decoder dim = 512
DECODER_CHANS = 512
# depths = {2, 2, 8, 2}
DEPTHS = (2, 2, 8, 2)
# heads = {3, 6, 12, 24}
NUM_HEADS = (3, 6, 12, 24)
# group dim = 8
GROUP_DIM = 8

# ---------------------------------------------------------
# 实例化模型
# ---------------------------------------------------------
net = Build_MultiSenseSeg(
    n_classes=N_CLASSES,
    in_chans=IN_CHANS,
    
    # --- Base Version Configs ---
    head_out_chans=HEAD_OUT_CHANS, # 对应 Modal dim
    embed_dim=EMBED_DIM,           # 对应 embed dim
    decoder_chans=DECODER_CHANS,   # 对应 decoder dim
    depths=DEPTHS,                 # 对应 depths
    num_heads=NUM_HEADS,           # 对应 heads
    group_dim=GROUP_DIM,           # 对应 group dim
    
    # --- 其他默认推荐配置 (保持原论文设置) ---
    patch_size=4,
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_ratio=1.5,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    norm_layer='BN',
    act_layer=nn.GELU,
    use_faster=False,  # Base 版本使用包含 Attention 的混合架构，不要设为 True
    aux=False           # 训练时开启辅助头
)
net.cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0

# # 统计整个图像编码器的参数数量
# for name, param in net.image_encoder.named_parameters():
#     params1 += param.nelement()

# # 统计图像编码器中可训练参数的数量（LoRA参数）
# for name, param in net.image_encoder.named_parameters():
#     if param.requires_grad:
#         params2 += param.nelement()

# print('ImgEncoder:   ', params1)
# print('Lora:         ', params2)
# print('Others:       ', params - params1)

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# params = 0
# for name, param in net.sam.prompt_encoder.named_parameters():
#     params += param.nelement()
# print('prompt_encoder: ', params)

# params = 0
# for name, param in net.sam.mask_decoder.named_parameters():
#     params += param.nelement()
# print('mask_decoder: ', params)

# print(net)

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
# train_loader = torch.utils.data.DataLoader(train_set,
#     batch_size=BATCH_SIZE,
#     shuffle=False,               # dataset 内部已随机抽样
#     num_workers=8,               # 先从 8 试，按 CPU 核数可提高到 12/16
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=4,
#     drop_last=True)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

# base_lr = 1e-3
# optimizer = torch.optim.AdamW(
#     net.parameters(),
#     lr=base_lr,
#     betas=(0.9, 0.999),
#     weight_decay=0.01
# )

# # Cosine + Warmup
# warmup_epochs = 5
# def warmup_lambda(epoch):
#     if epoch < warmup_epochs:
#         return float(epoch) / float(max(1, warmup_epochs))
#     return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
# # 训练脚本里（替换你现有的 optimizer 构造）
# def is_lora_param(n):  # 名字里包含 lora_A/lora_B 的就是 LoRA
#     return ('lora_A' in n) or ('lora_B' in n)

# lora_params, head_params = [], []
# for n, p in net.named_parameters():
#     if not p.requires_grad:
#         continue
#     if is_lora_param(n):
#         lora_params.append(p)
#     else:
#         head_params.append(p)

# # 建议 AdamW，LoRA 稍微更大的 lr
# optimizer = torch.optim.AdamW(
#     [
#         {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-2},
#         {'params': lora_params, 'lr': 5e-4, 'weight_decay': 0.0},  # LoRA 分支更大学习率，且通常不做 wd
#     ],
#     betas=(0.9, 0.999),
# )

# # 调度器建议：Poly 或 Cosine + Warmup（二选一）
from torch.optim.lr_scheduler import LambdaLR
# epochs = epochs  # 你已有
# warmup = 5
# def cosine_warmup(e):
#     if e < warmup:
#         return (e + 1) / warmup
#     progress = (e - warmup) / max(1, (epochs - warmup))
#     return 0.5 * (1 + math.cos(math.pi * progress))

# scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup)
# ======================================================
# 1. 参数分组
# ======================================================
def is_lora_param(n):
    return ('lora_A' in n) or ('lora_B' in n)

def is_geo_param(n):
    """识别几何先验生成器的参数"""
    return 'geo_gen' in n

lora_params, geo_params, norm_params, head_params = [], [], [], []

for n, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if is_lora_param(n):
        lora_params.append(p)
    elif is_geo_param(n):
        geo_params.append(p)  # 新增：几何先验参数单独分组
    elif any(nd in n for nd in ['norm', 'bn', 'bias', 'ln', 'LayerNorm']):
        norm_params.append(p)
    else:
        head_params.append(p)

# ======================================================
# 2. 优化器（为几何先验参数单独设置参数组）
# ======================================================
optimizer = optim.AdamW(
    [
        # 几何先验参数：高学习率，无权重衰减
        {'params': geo_params, 'lr': 5e-4, 'weight_decay': 0.0, 'name': 'geo_params'},
        # 头部/Decoder参数：标准学习率，标准权重衰减
        {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-2},
        # Norm/Bias参数：标准学习率，无权重衰减
        {'params': norm_params, 'lr': 1e-4, 'weight_decay': 0.0},
        # LoRA参数：中等学习率，无权重衰减
        {'params': lora_params, 'lr': 3e-4, 'weight_decay': 0.0},
    ],
    betas=(0.9, 0.999)
)

# ======================================================
# 3. 调度器：Cosine + Warmup 改进版
# ======================================================
total_epochs = epochs
warmup_epochs = max(5, int(total_epochs * 0.1))  # 前10%热身

def cosine_warmup(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    # 平滑尾部衰减
    return 0.5 * (1 + math.cos(math.pi * progress)) ** 1.2

scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup)


def test(net, test_ids, logger=None, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    if TEST_DATASET == 'Potsdam':
        # Potsdam 原始读取逻辑
        test_images = (1 / 255 * np.asarray(io.imread(TEST_DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    elif TEST_DATASET == 'Vaihingen':
        # Vaihingen 原始读取逻辑
        test_images = (1 / 255 * np.asarray(io.imread(TEST_DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    else:
        # Hunan 或其他
        test_images = (1 / 255 * np.asarray(io.imread(TEST_DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    
    # --- [修改] 使用 TEST_DSM_FOLDER, TEST_LABEL_FOLDER ---
    test_dsms = (np.asarray(io.imread(TEST_DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(TEST_LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    
    if TEST_DATASET == 'Hunan':
        eroded_labels = ((np.asarray(io.imread(TEST_ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(TEST_ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    net.eval() # 确保进入评估模式
    
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            
            # --- 🚀 核心优化 START: 将 DSM 归一化移出内层循环 ---
            # 原代码在这里每一轮 batch 都会重新计算全图 min/max，极其耗时。
            # 现在改为每张图只计算一次。
            min_d = np.min(dsm)
            max_d = np.max(dsm)
            if DATASET == 'Hunan':
                dsm = (dsm - min_d) / (max_d - min_d + 1e-8)
            else:
                dsm = (dsm - min_d) / (max_d - min_d)
            # --- 🚀 核心优化 END ---

            pred = np.zeros(img.shape[:2] + (N_CLASSES,), dtype=np.float32)

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            
            # 内层循环：滑窗预测
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                
                # 1. Image patches
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                # 优化：移除过时的 Variable，使用 non_blocking 加速传输
                image_patches = torch.from_numpy(image_patches).cuda(non_blocking=True)

                # 2. DSM patches (此时直接从已归一化的 dsm 中切片即可)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).cuda(non_blocking=True)

                # Do the inference
                if len(dsm_patches.shape) == 3:
                    dsm_patches = dsm_patches.unsqueeze(1)
                outs = net([image_patches, dsm_patches])
                outs = outs.detach().cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                
                del outs

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
    
    if DATASET == 'Hunan':
        accuracy = metrics_loveda(np.concatenate([p.ravel() for p in all_preds]),
                                np.concatenate([p.ravel() for p in all_gts]).ravel(), logger=logger)
    else:
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel(), logger=logger)
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


from torch.amp import autocast, GradScaler
scaler = GradScaler()
USE_FP16 = False
max_grad_norm = 5.0
def train(net, optimizer, epochs, scheduler=None, logger=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.00
    # criterion = CombinedLoss(ignore_index=255, use_aux=False).cuda()
    
    for e in range(1, epochs + 1):
        
        net.train()
        epoch_loss = 0.0
        epoch_aux_loss = 0.0 # [新增] 用于记录辅助任务的累计 Loss
        start_time = time.time()
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=USE_FP16):
                # forward 返回分割结果和加权后的辅助 loss
                # print(f"data:{data.shape}")
                # print(f"dsm:{dsm.shape}")
                if len(dsm.shape) == 3:
                    dsm = dsm.unsqueeze(1)
                output = net([data, dsm])
                
                # [注意] 如果模型内部已经乘过权重(如 aux_loss * 0.2)
                # 这里 λ 应设为 1.0，否则会双重衰减 (0.2 * 0.2 = 0.04)
                λ = 0.0 
                
                loss = loss_calc(output.float(), target, weights)
            
            # loss = CrossEntropy2d(output, target, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # 使用 .item() 获取 python float 数值，避免 tensor 累加导致的显存泄漏
            epoch_loss += loss.item()
            # epoch_aux_loss += aux_loss.item() # [新增] 累加当前 batch 的辅助 loss
            
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                
                # [修改] 打印信息中增加 Aux Loss
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            
            # 计算 Epoch 平均 Loss
            epoch_loss /= len(train_loader)
            avg_aux_loss = epoch_aux_loss / len(train_loader) # [新增] 计算平均 Aux Loss
            
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size, logger=logger)
            net.train()
            
            # [新增] 写入 TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, e)
            writer.add_scalar("Loss/aux", avg_aux_loss, e) # 记录辅助损失曲线
            writer.add_scalar("mIoU/val", MIoU, e)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], e)
            
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), './resultsv2p_multisenseseg/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './resultsp2v_multisenseseg/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Hunan':
                    torch.save(net.state_dict(), './resultsh_multisenseseg/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
                
    writer.close()
    print('MIoU_best: ', MIoU_best)

if MODE == 'Train':
    if DATASET == "Vaihingen":
        out_dir = './resultsv2p_multisenseseg'
    elif DATASET == "Potsdam":
        out_dir = './resultsp2v_multisenseseg'
    else:
        out_dir = './resultsh_multisenseseg'
    logger = setup_logger(out_dir, name="train")
    train(net, optimizer, epochs, scheduler, logger=logger, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        # out_dir = './resultsp_test27/p2v'
        # # out_dir = './resultsp_multisenseseg7/v'
        # logger = setup_logger(out_dir, name="p2v_test")

        # # ckpt = '/home/csf1/Documents/DINOv3/testbase/resultsp_multisenseseg7/UNetformer_epoch31_0.8760012626633433'
        # # ckpt = '/home/csf1/Documents/DINOv3/testbase/resultsp_test27/UNetformer_epoch11_0.872122894260379'
        # ckpt = '/home/csf1/Documents/DINOv3/testbase/resultsp_test27/UNetformer_epoch5_0.8602161403639101'
        # logger.info("Loading checkpoint: %s (strict=False)", ckpt)
        # net.load_state_dict(torch.load(ckpt), strict=False)

        # net.eval()
        # logger.info("Start testing... stride=%d all=%s #test_ids=%d", 32, True, len(test_ids))

        # MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32, logger=logger)
        # logger.info("MIoU: %.6f", float(MIoU))

        # logger.info("Saving predictions to %s", out_dir)
        # for p, id_ in zip(all_preds, test_ids):
        #     img = convert_to_color(p)
        #     save_path = os.path.join(out_dir, f'inference_UNetFormer_huge_tile_{id_}.png')
        #     io.imsave(save_path, img)

        # logger.info("Done. Saved %d prediction images.", len(all_preds))
        # 1. 设置路径
        ckpt_root = '/home/csf1/Documents/DINOv3/testbase/resultsp_test27/'
        out_dir = './resultsp_test27/p2v'  # 建议修改输出目录名，表明这是最好的结果
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 2. 初始化 Logger
        # 注意：如果 setup_logger 是追加模式，所有日志都会在里面；如果是覆盖模式，建议在这里初始化一次
        logger = setup_logger(out_dir, name="p2v_batch_test")

        # 3. 获取所有 ckpt 文件
        # 假设文件名包含 'UNetformer'，或者你可以直接列出所有文件
        ckpt_files = [f for f in os.listdir(ckpt_root) if os.path.isfile(os.path.join(ckpt_root, f)) and 'UNetformer' in f]
        # 按文件名排序(可选，方便观察)
        ckpt_files.sort()

        logger.info(f"Found {len(ckpt_files)} checkpoints in {ckpt_root}")

        # 4. 初始化最佳记录变量
        best_miou = 0.0
        best_ckpt_name = ""

        # 5. 开始循环测试
        for ckpt_name in ckpt_files:
            ckpt_path = os.path.join(ckpt_root, ckpt_name)
            
            logger.info("=" * 30)
            logger.info("Processing checkpoint: %s", ckpt_name)
            
            try:
                # 加载权重
                # 注意：这里加了 try-except 防止某个坏文件中断整个测试
                net.load_state_dict(torch.load(ckpt_path), strict=False)
            except Exception as e:
                logger.error(f"Failed to load {ckpt_name}: {e}")
                continue

            net.eval()
            
            # 运行测试
            # 注意：stride 和 all 参数保持您原有的设置
            try:
                # 假设 test 函数返回 (MIoU, all_preds, all_gts)
                curr_miou, all_preds, all_gts = test(net, test_ids, all=True, stride=32, logger=logger)
                curr_miou = float(curr_miou)
            except Exception as e:
                logger.error(f"Error during testing {ckpt_name}: {e}")
                continue

            logger.info(f"Checkpoint: {ckpt_name} | Current MIoU: {curr_miou:.6f}")

            # 6. 比较并保存最佳结果
            if curr_miou > best_miou:
                logger.info(f"🔥 New Best MIoU found! ({curr_miou:.6f} > {best_miou:.6f})")
                logger.info(f"Updating records and saving prediction images to {out_dir}...")
                
                best_miou = curr_miou
                best_ckpt_name = ckpt_name
                
                # 保存图片 (只保存当前最好的，覆盖之前的)
                for p, id_ in zip(all_preds, test_ids):
                    img = convert_to_color(p)
                    save_path = os.path.join(out_dir, f'inference_UNetFormer_huge_tile_{id_}.png')
                    io.imsave(save_path, img)
                    
                logger.info("Images saved.")
            else:
                logger.info(f"Current MIoU ({curr_miou:.6f}) is lower than best ({best_miou:.6f}). Skipping image save.")

        # 7. 结束总结
        logger.info("=" * 30)
        logger.info("All testing finished.")
        logger.info(f"🏆 Best Checkpoint: {best_ckpt_name}")
        logger.info(f"🏆 Best MIoU: {best_miou:.6f}")
        logger.info(f"Best prediction images are saved in: {out_dir}")

    elif DATASET == 'Potsdam':
        # out_dir = './resultsp_multisenseseg7/p'
        out_dir = './resultsv_test27/v2p'
        logger = setup_logger(out_dir, name="p_test")

        ckpt = '/home/csf1/Documents/DINOv3/testbase/resultsv_test27/UNetformer_epoch13_0.8530910776153316'
        # ckpt = '/home/csf1/Documents/DINOv3/resultsv/resultsv_multisenseseg7/UNetformer_epoch21_0.859179331058526'
        logger.info("Loading checkpoint: %s (strict=False)", ckpt)
        net.load_state_dict(torch.load(ckpt), strict=False)

        net.eval()
        logger.info("Start testing... stride=%d all=%s #test_ids=%d", 32, True, len(test_ids))

        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32, logger=logger)
        logger.info("MIoU: %.6f", float(MIoU))

        logger.info("Saving predictions to %s", out_dir)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            save_path = os.path.join(out_dir, f'inference_UNetFormer_huge_tile_{id_}.png')
            io.imsave(save_path, img)

        logger.info("Done. Saved %d prediction images.", len(all_preds))

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/UNetformer_epoch23_0.5009448279555474'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsh/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)