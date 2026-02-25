"""
Task Distiller for Continual Learning
用于在持续学习场景中对单个任务的数据进行蒸馏
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import copy

from networks import ConvNet
from distiller.reparam_module import ReparamModule


class ParamDiffAug:
    """Differentiable augmentation parameters"""
    def __init__(self):
        self.aug_mode = 'S'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.latestseed = -1
        self.batchmode = False


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    """Apply differentiable augmentation"""
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True
    param.latestseed = seed
    
    if strategy == 'None' or strategy == 'none' or not strategy:
        return x
    
    if strategy:
        if param.aug_mode == 'M':
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        x = x.contiguous()
    return x


# Augmentation functions
def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:
        randb[:] = randb[0]
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


class TaskDistiller:
    """
    用于持续学习场景的任务蒸馏器
    对单个任务的数据进行蒸馏，生成合成数据用于后续replay
    """

    def __init__(self,
                 channel=3,
                 num_classes=10,
                 im_size=(32, 32),
                 ipc=10,
                 device='cuda',
                 iteration=1000,
                 lr_img=1000,
                 lr_lr=1e-5,
                 lr_teacher=0.01,
                 expert_epochs=3,
                 syn_steps=20,
                 max_start_epoch=25,
                 batch_train=256,
                 dsa_strategy='color_crop_cutout_flip_scale_rotate',
                 pix_init='real',
                 model_name='ConvNet'):
        """
        Args:
            channel: 图像通道数
            num_classes: 当前任务的类别数
            im_size: 图像尺寸
            ipc: images per class，每个类别的蒸馏图像数
            device: 计算设备
            iteration: 蒸馏迭代次数
            lr_img: 合成图像的学习率
            lr_lr: 学习率的学习率
            lr_teacher: teacher网络的学习率
            expert_epochs: expert trajectory的epoch数
            syn_steps: 每次迭代在合成数据上的步数
            max_start_epoch: 最大起始epoch
            batch_train: 训练batch大小
            dsa_strategy: 数据增强策略
            pix_init: 像素初始化方式 ('real' or 'noise')
            model_name: 模型名称
        """
        self.channel = channel
        self.num_classes = num_classes
        self.im_size = im_size
        self.ipc = ipc
        self.device = device
        self.iteration = iteration
        self.lr_img = lr_img
        self.lr_lr = lr_lr
        self.lr_teacher = lr_teacher
        self.expert_epochs = expert_epochs
        self.syn_steps = syn_steps
        self.max_start_epoch = max_start_epoch
        self.batch_train = batch_train
        self.dsa_strategy = dsa_strategy
        self.pix_init = pix_init
        self.model_name = model_name

        self.dsa_param = ParamDiffAug()

        self.meta_init_params = None

    def set_meta_init_params(self, params):
        """
        设置从dd_lamaml_cifar传入的meta-update后的参数
        这些参数将用于初始化expert trajectory中的teacher网络

        Args:
            params: 模型参数的state_dict或参数列表
        """
        if params is not None:
            # 深拷贝参数，避免后续修改影响
            if isinstance(params, dict):
                self.meta_init_params = {k: v.detach().cpu().clone() for k, v in params.items()}
            else:
                # 如果是参数列表
                self.meta_init_params = [p.detach().cpu().clone() for p in params]
        else:
            self.meta_init_params = None

    def _normalize_distilled_images(self, images, clip_std=7.0):
        """
        对蒸馏后的图像进行归一化处理，输出范围为 [-1, 1]

        处理策略 (与 main.py 中 predistilled_norm='standard' 一致):
        1. 先进行 clipping，去除超出 mean ± clip_std * std 的极端值
        2. 标准化到 mean=0, std=1
        3. 使用 tanh 映射到 [-1, 1]

        Args:
            images: 输入图像 [N, C, H, W]
            clip_std: clipping 阈值 (以原始标准差为单位)，默认 7.0

        Returns:
            归一化后的图像，范围 [-1, 1]
        """
        images_normalized = images.clone()

        # 计算当前统计量
        current_mean = images.mean()
        current_std = images.std()

        # Step 1: 先 clipping 去除极端值
        if clip_std is not None and current_std > 1e-8:
            clip_min = current_mean - clip_std * current_std
            clip_max = current_mean + clip_std * current_std
            images_normalized = torch.clamp(images_normalized, min=clip_min, max=clip_max)
            # 重新计算 clipping 后的统计量
            current_mean = images_normalized.mean()
            current_std = images_normalized.std()

        # Step 2: 标准化到 mean=0, std=1
        if current_std > 1e-8:
            images_normalized = (images_normalized - current_mean) / current_std

        # Step 3: 使用 tanh 映射到 [-1, 1]
        images_normalized = torch.tanh(images_normalized)

        return images_normalized

    def _get_network(self):
        """创建网络"""
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        net = ConvNet(
            channel=self.channel,
            num_classes=self.num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=self.im_size
        )
        return net.to(self.device)

    def _generate_expert_trajectory(self, images, labels, num_experts=1, train_epochs=50):
        """
        生成expert trajectories用于蒸馏

        Args:
            images: 训练图像 [N, C, H, W]
            labels: 标签 [N]
            num_experts: expert数量
            train_epochs: 训练epoch数

        Returns:
            trajectories: list of expert trajectories
        """
        trajectories = []

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(images, labels)
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_train, shuffle=True, num_workers=0
        )

        criterion = nn.CrossEntropyLoss().to(self.device)

        for _ in range(num_experts):
            teacher_net = self._get_network()

            # 如果有meta_init_params，使用它来初始化teacher_net
            if self.meta_init_params is not None:
                self._load_meta_init_params(teacher_net)

            teacher_net.train()

            lr = self.lr_teacher
            optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0, weight_decay=0)

            timestamps = []
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            lr_schedule = [train_epochs // 2 + 1]

            for e in range(train_epochs):
                for img, lab in trainloader:
                    img, lab = img.to(self.device), lab.to(self.device)

                    # Apply augmentation
                    img = DiffAugment(img, self.dsa_strategy, param=self.dsa_param)

                    optimizer.zero_grad()
                    output = teacher_net(img)
                    loss = criterion(output, lab)
                    loss.backward()
                    optimizer.step()

                timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

                if e in lr_schedule:
                    lr *= 0.1
                    optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0, weight_decay=0)
                    optimizer.zero_grad()

            trajectories.append(timestamps)
            if len(trajectories) % 10 == 0:
                print(f'Generated {len(trajectories)} expert trajectories...')
            del teacher_net

        return trajectories

    def _load_meta_init_params(self, teacher_net):
        """
        将meta_init_params加载到teacher_net中
        需要处理dd_lamaml_cifar的Learner参数格式与ConvNet参数格式的映射

        Args:
            teacher_net: ConvNet网络实例
        """
        if self.meta_init_params is None:
            return

        teacher_state_dict = teacher_net.state_dict()
        teacher_param_names = list(teacher_state_dict.keys())

        if isinstance(self.meta_init_params, list):

            meta_params = self.meta_init_params
            param_idx = 0

            for name in teacher_param_names:
                if param_idx >= len(meta_params):
                    break

                teacher_param = teacher_state_dict[name]
                meta_param = meta_params[param_idx]

                # 检查形状是否匹配
                if teacher_param.shape == meta_param.shape:
                    teacher_state_dict[name] = meta_param.clone().to(self.device)
                    param_idx += 1
                else:
                    # 形状不匹配，跳过（可能是分类器层，类别数不同）
                    print(f"Shape mismatch for {name}: teacher {teacher_param.shape} vs meta {meta_param.shape}, skipping")

            teacher_net.load_state_dict(teacher_state_dict)
            print(f"Loaded {param_idx} parameters from meta_init_params to teacher_net")

        elif isinstance(self.meta_init_params, dict):
            loaded_count = 0
            for name, param in self.meta_init_params.items():
                if name in teacher_state_dict:
                    if teacher_state_dict[name].shape == param.shape:
                        teacher_state_dict[name] = param.clone().to(self.device)
                        loaded_count += 1

            teacher_net.load_state_dict(teacher_state_dict)
            print(f"Loaded {loaded_count} parameters from meta_init_params dict to teacher_net")

    def distill(self, images, labels, task_classes=None, verbose=True):
        """
        对任务数据进行蒸馏

        Args:
            images: 输入图像 [N, C, H, W]，假设范围为 [-1, 1]
            labels: 标签 [N]
            task_classes: 当前任务的类别列表，如果为None则自动推断
            verbose: 是否打印进度

        Returns:
            dict: {'images': 蒸馏图像 (范围 [-1, 1]), 'labels': 对应标签}
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        if verbose:
            print(f'Input images range: [{images.min():.3f}, {images.max():.3f}]')

        # 推断当前任务的类别
        if task_classes is None:
            task_classes = torch.unique(labels).tolist()

        num_classes = len(task_classes)

        # 创建类别映射 (原始标签 -> 连续索引)
        class_map = {c: i for i, c in enumerate(task_classes)}
        class_map_inv = {i: c for i, c in enumerate(task_classes)}

        # 按类别组织数据
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels.cpu().tolist()):
            if lab in class_map:
                indices_class[class_map[lab]].append(i)

        def get_images(c, n):
            """从类别c中随机获取n张图像，如果样本不足则重复采样"""
            class_indices = indices_class[c]
            if len(class_indices) == 0:
                return None
            if len(class_indices) >= n:
                idx_shuffle = np.random.permutation(class_indices)[:n]
            else:
                # 样本不足时，重复采样
                idx_shuffle = np.random.choice(class_indices, size=n, replace=True)
            return images[idx_shuffle]

        # 初始化合成数据
        label_syn = torch.tensor(
            [np.ones(self.ipc, dtype=np.int64) * i for i in range(num_classes)],
            dtype=torch.long, device=self.device
        ).view(-1)

        image_syn = torch.randn(
            size=(num_classes * self.ipc, self.channel, self.im_size[0], self.im_size[1]),
            dtype=torch.float, device=self.device
        )

        # 使用真实图像初始化
        if self.pix_init == 'real':
            for c in range(num_classes):
                real_imgs = get_images(c, self.ipc)
                if real_imgs is not None:
                    image_syn.data[c * self.ipc:(c + 1) * self.ipc] = real_imgs.detach().data

        image_syn = image_syn.detach().requires_grad_(True)
        syn_lr = torch.tensor(self.lr_teacher, device=self.device).requires_grad_(True)

        optimizer_img = torch.optim.SGD([image_syn], lr=self.lr_img, momentum=0.5)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=self.lr_lr, momentum=0.5)
        optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(self.device)

        # 生成expert trajectories
        if verbose:
            print(f'Generating expert trajectories...')

        # 使用映射后的标签进行训练
        mapped_labels = torch.tensor([class_map[l.item()] for l in labels], device=self.device)
        # trajectories = self._generate_expert_trajectory(
        #     images, mapped_labels,
        #     num_experts=1,
        #     train_epochs=self.max_start_epoch + self.expert_epochs + 1
        # )
        trajectories = self._generate_expert_trajectory(
            images, mapped_labels,
            num_experts=100,
            train_epochs=30
        )
        buffer = trajectories

        if verbose:
            print(f'Starting distillation for {self.iteration} iterations...')

        # 蒸馏循环
        for it in tqdm(range(self.iteration), disable=not verbose):
            # 随机选择一个expert trajectory
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]

            # 随机选择起始epoch，确保不越界
            max_start = max(1, len(expert_trajectory) - self.expert_epochs - 1)
            start_epoch = np.random.randint(0, min(self.max_start_epoch, max_start))
            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[min(start_epoch + self.expert_epochs, len(expert_trajectory) - 1)]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            # 创建student网络
            student_net = self._get_network()
            student_net = ReparamModule(student_net)
            student_net.train()

            num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

            batch_syn = num_classes * self.ipc
            indices_chunks = []
            for step in range(self.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(image_syn))
                    indices_chunks = list(torch.split(indices, batch_syn))

                these_indices = indices_chunks.pop()
                x = image_syn[these_indices]
                this_y = label_syn[these_indices]

                # 应用数据增强
                x = DiffAugment(x, self.dsa_strategy, param=self.dsa_param)

                forward_params = student_params[-1]
                output = student_net(x, flat_param=forward_params)
                ce_loss = criterion(output, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            # 计算参数匹配损失
            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss /= num_params
            param_dist /= num_params

            param_loss = param_loss / (param_dist + 1e-8)

            grand_loss = param_loss

            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()
            grand_loss.backward()
            optimizer_img.step()
            optimizer_lr.step()

            # 清理
            for _ in student_params:
                del _
            del student_net

            if verbose and it % 10 == 0:
                print(f'Iter {it}, Loss: {grand_loss.item():.4f}')

        # 将标签映射回原始类别
        final_labels = torch.tensor([class_map_inv[l.item()] for l in label_syn], device=self.device)

        # 对合成图像进行归一化处理，输出范围为 [-1, 1]
        if verbose:
            print(f'Before normalization - Range: [{image_syn.min():.3f}, {image_syn.max():.3f}]')
            print(f'Before normalization - Mean: {image_syn.mean():.3f}, Std: {image_syn.std():.3f}')

        # 归一化到 [-1, 1]，与 main.py 中 predistilled_norm='standard' 处理一致
        image_syn_output = self._normalize_distilled_images(image_syn.detach(), clip_std=7.0)

        if verbose:
            print(f'After normalization - Range: [{image_syn_output.min():.3f}, {image_syn_output.max():.3f}]')
            print(f'After normalization - Mean: {image_syn_output.mean():.3f}, Std: {image_syn_output.std():.3f}')

        return {
            'images': image_syn_output,
            'labels': final_labels.detach()
        }

    def distill_with_trajectory_return(self, images, labels, task_classes=None, verbose=True, num_trajectories_to_save=30):
        """
        对任务数据进行蒸馏，并返回用于后续轨迹偏移的trajectories

        Args:
            images: 输入图像 [N, C, H, W]
            labels: 标签 [N]
            task_classes: 当前任务的类别列表
            verbose: 是否打印进度
            num_trajectories_to_save: 保存的trajectory数量

        Returns:
            dict: {
                'images': 蒸馏图像,
                'labels': 对应标签,
                'trajectories': 保存的expert trajectories (最后num_trajectories_to_save个),
                'init_params': 蒸馏时的初始化参数
            }
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        if verbose:
            print(f'Input images range: [{images.min():.3f}, {images.max():.3f}]')

        # 推断当前任务的类别
        if task_classes is None:
            task_classes = torch.unique(labels).tolist()

        num_classes = len(task_classes)

        # 创建类别映射
        class_map = {c: i for i, c in enumerate(task_classes)}
        class_map_inv = {i: c for i, c in enumerate(task_classes)}

        # 按类别组织数据
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels.cpu().tolist()):
            if lab in class_map:
                indices_class[class_map[lab]].append(i)

        def get_images(c, n):
            class_indices = indices_class[c]
            if len(class_indices) == 0:
                return None
            if len(class_indices) >= n:
                idx_shuffle = np.random.permutation(class_indices)[:n]
            else:
                idx_shuffle = np.random.choice(class_indices, size=n, replace=True)
            return images[idx_shuffle]

        # 初始化合成数据
        label_syn = torch.tensor(
            [np.ones(self.ipc, dtype=np.int64) * i for i in range(num_classes)],
            dtype=torch.long, device=self.device
        ).view(-1)

        image_syn = torch.randn(
            size=(num_classes * self.ipc, self.channel, self.im_size[0], self.im_size[1]),
            dtype=torch.float, device=self.device
        )

        if self.pix_init == 'real':
            for c in range(num_classes):
                real_imgs = get_images(c, self.ipc)
                if real_imgs is not None:
                    image_syn.data[c * self.ipc:(c + 1) * self.ipc] = real_imgs.detach().data

        image_syn = image_syn.detach().requires_grad_(True)
        syn_lr = torch.tensor(self.lr_teacher, device=self.device).requires_grad_(True)

        optimizer_img = torch.optim.SGD([image_syn], lr=self.lr_img, momentum=0.5)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=self.lr_lr, momentum=0.5)
        optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(self.device)

        if verbose:
            print(f'Generating expert trajectories...')

        mapped_labels = torch.tensor([class_map[l.item()] for l in labels], device=self.device)
        trajectories = self._generate_expert_trajectory(
            images, mapped_labels,
            num_experts=100,
            train_epochs=30
        )
        buffer = trajectories

        # 保存初始化参数（用于后续轨迹偏移）
        saved_init_params = copy.deepcopy(self.meta_init_params) if self.meta_init_params is not None else None

        if verbose:
            print(f'Starting distillation for {self.iteration} iterations...')

        # 蒸馏循环
        for it in tqdm(range(self.iteration), disable=not verbose):
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]

            max_start = max(1, len(expert_trajectory) - self.expert_epochs - 1)
            start_epoch = np.random.randint(0, min(self.max_start_epoch, max_start))
            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[min(start_epoch + self.expert_epochs, len(expert_trajectory) - 1)]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            student_net = self._get_network()
            student_net = ReparamModule(student_net)
            student_net.train()

            num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

            batch_syn = num_classes * self.ipc
            indices_chunks = []
            for step in range(self.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(image_syn))
                    indices_chunks = list(torch.split(indices, batch_syn))

                these_indices = indices_chunks.pop()
                x = image_syn[these_indices]
                this_y = label_syn[these_indices]

                x = DiffAugment(x, self.dsa_strategy, param=self.dsa_param)

                forward_params = student_params[-1]
                output = student_net(x, flat_param=forward_params)
                ce_loss = criterion(output, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss /= num_params
            param_dist /= num_params

            param_loss = param_loss / (param_dist + 1e-8)

            grand_loss = param_loss

            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()
            grand_loss.backward()
            optimizer_img.step()
            optimizer_lr.step()

            for _ in student_params:
                del _
            del student_net

            if verbose and it % 10 == 0:
                print(f'Iter {it}, Loss: {grand_loss.item():.4f}')

        final_labels = torch.tensor([class_map_inv[l.item()] for l in label_syn], device=self.device)

        if verbose:
            print(f'Before normalization - Range: [{image_syn.min():.3f}, {image_syn.max():.3f}]')

        image_syn_output = self._normalize_distilled_images(image_syn.detach(), clip_std=7.0)

        if verbose:
            print(f'After normalization - Range: [{image_syn_output.min():.3f}, {image_syn_output.max():.3f}]')

        # 保存最后num_trajectories_to_save个trajectories
        trajectories_to_save = buffer[-num_trajectories_to_save:] if len(buffer) >= num_trajectories_to_save else buffer

        return {
            'images': image_syn_output,
            'labels': final_labels.detach(),
            'trajectories': trajectories_to_save,
            'init_params': saved_init_params,
            'class_map': class_map,
            'class_map_inv': class_map_inv
        }

    def update_distilled_data_with_trajectory_shift(self,
                                                     old_images,
                                                     old_labels,
                                                     old_trajectories,
                                                     old_init_params,
                                                     new_init_params,
                                                     class_map,
                                                     class_map_inv,
                                                     meta_lr=0.1,
                                                     update_iterations=500,
                                                     verbose=True):
        """
        使用轨迹偏移更新旧任务的蒸馏数据

        基于 Large-Scale Meta-Learning with Continual Trajectory Shifting 的思路:
        1. 计算新旧初始化参数的偏移量
        2. 对保存的trajectories进行偏移
        3. 使用偏移后的trajectories更新蒸馏数据

        Args:
            old_images: 旧任务的蒸馏图像 [N, C, H, W]
            old_labels: 旧任务的蒸馏标签 [N]
            old_trajectories: 旧任务保存的expert trajectories
            old_init_params: 旧任务蒸馏时的初始化参数
            new_init_params: 新任务的初始化参数（当前meta-update后的参数）
            class_map: 类别映射
            class_map_inv: 逆类别映射
            meta_lr: 轨迹偏移的学习率
            update_iterations: 更新迭代次数
            verbose: 是否打印进度

        Returns:
            dict: {'images': 更新后的蒸馏图像, 'labels': 标签}
        """
        if old_init_params is None or new_init_params is None:
            if verbose:
                print("No init params available, skipping trajectory shift update")
            return {'images': old_images, 'labels': old_labels}

        if verbose:
            print(f"Updating distilled data with trajectory shift...")
            print(f"Number of trajectories: {len(old_trajectories)}")

        # 计算轨迹偏移量 (参考选中的代码逻辑)
        # Δθ = meta_lr * (new_init_params - old_init_params) (对于非fc层)
        trajectory_shift = []
        for old_p, new_p in zip(old_init_params, new_init_params):
            if old_p.shape == new_p.shape:
                # 使用类似meta_train_ours.py中的偏移计算
                # base_param[name].data -= FLAGS.meta_lr * (base_param[name].data - w.data)
                # 这里简化为直接计算偏移
                shift = meta_lr * (new_p.cpu() - old_p.cpu())
                trajectory_shift.append(shift)
            else:
                # 形状不匹配（如分类器层），不进行偏移
                trajectory_shift.append(torch.zeros_like(old_p))

        # 对trajectories进行偏移
        shifted_trajectories = []
        for trajectory in old_trajectories:
            shifted_trajectory = []
            for timestamp in trajectory:
                shifted_timestamp = []
                for i, param in enumerate(timestamp):
                    if i < len(trajectory_shift):
                        # 应用偏移: shifted_param = param + shift
                        shifted_param = param + trajectory_shift[i]
                        shifted_timestamp.append(shifted_param)
                    else:
                        shifted_timestamp.append(param)
                shifted_trajectory.append(shifted_timestamp)
            shifted_trajectories.append(shifted_trajectory)

        if verbose:
            print(f"Trajectories shifted successfully")

        # 准备更新蒸馏数据
        old_images = old_images.to(self.device)
        old_labels = old_labels.to(self.device)

        num_classes = len(class_map)

        # 将标签映射为连续索引
        mapped_labels = torch.tensor([class_map[l.item()] for l in old_labels], device=self.device)

        # 初始化要更新的合成数据（从旧的蒸馏数据开始）
        image_syn = old_images.clone().detach().requires_grad_(True)
        label_syn = mapped_labels.clone()

        syn_lr = torch.tensor(self.lr_teacher, device=self.device).requires_grad_(True)

        optimizer_img = torch.optim.SGD([image_syn], lr=self.lr_img, momentum=0.5)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=self.lr_lr, momentum=0.5)
        optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(self.device)

        buffer = shifted_trajectories

        if verbose:
            print(f'Updating distilled data for {update_iterations} iterations...')

        # 更新循环（使用偏移后的trajectories）
        for it in tqdm(range(update_iterations), disable=not verbose):
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]

            max_start = max(1, len(expert_trajectory) - self.expert_epochs - 1)
            start_epoch = np.random.randint(0, min(self.max_start_epoch, max_start))
            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[min(start_epoch + self.expert_epochs, len(expert_trajectory) - 1)]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params_flat = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            student_net = self._get_network()
            student_net = ReparamModule(student_net)
            student_net.train()

            num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

            batch_syn = num_classes * self.ipc
            indices_chunks = []
            for step in range(self.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(image_syn))
                    indices_chunks = list(torch.split(indices, batch_syn))

                these_indices = indices_chunks.pop()
                x = image_syn[these_indices]
                this_y = label_syn[these_indices]

                x = DiffAugment(x, self.dsa_strategy, param=self.dsa_param)

                forward_params = student_params[-1]
                output = student_net(x, flat_param=forward_params)
                ce_loss = criterion(output, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params_flat, target_params, reduction="sum")

            param_loss /= num_params
            param_dist /= num_params

            param_loss = param_loss / (param_dist + 1e-8)

            grand_loss = param_loss

            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()
            grand_loss.backward()
            optimizer_img.step()
            optimizer_lr.step()

            for _ in student_params:
                del _
            del student_net

            if verbose and it % 50 == 0:
                print(f'Update Iter {it}, Loss: {grand_loss.item():.4f}')

        # 将标签映射回原始类别
        final_labels = torch.tensor([class_map_inv[l.item()] for l in label_syn], device=self.device)

        # 归一化
        image_syn_output = self._normalize_distilled_images(image_syn.detach(), clip_std=7.0)

        if verbose:
            print(f'Update complete - Range: [{image_syn_output.min():.3f}, {image_syn_output.max():.3f}]')

        return {
            'images': image_syn_output,
            'labels': final_labels.detach()
        }

