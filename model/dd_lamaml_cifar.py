import random
from random import shuffle
import numpy as np
import ipdb
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.lamaml_base import *
from distiller.task_distiller import TaskDistiller


class Net(BaseNet):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__(n_inputs,
                                  n_outputs,
                                  n_tasks,
                                  args)
        self.nc_per_task = n_outputs / n_tasks
        self.n_tasks = n_tasks

        # 蒸馏buffer，大小与原始buffer一致
        self.M_distill = []

        # 用于存储每个任务的trajectories和初始化参数（用于轨迹偏移更新）
        # 格式: {task_id: {'trajectories': [...], 'init_params': [...], 'class_map': {...}, 'class_map_inv': {...}}}
        self.task_trajectories = {}

        # 用于收集当前任务的所有数据（用于蒸馏）
        self.current_task_data = {'images': [], 'labels': []}

        # 确定图像尺寸和通道数
        if args.dataset == 'tinyimagenet':
            self.im_size = (64, 64)
            self.channel = 3
        elif args.dataset == 'cifar100':
            self.im_size = (32, 32)
            self.channel = 3
        else:
            self.im_size = (32, 32)
            self.channel = 3

        # 从args获取蒸馏超参数
        self.ipc = getattr(args, 'ipc', 50)
        self.distill_iterations = getattr(args, 'distill_iterations', 1000)
        self.distill_lr_img = getattr(args, 'distill_lr_img', 1000)
        self.distill_lr_lr = getattr(args, 'distill_lr_lr', 1e-5)
        self.distill_lr_teacher = getattr(args, 'distill_lr_teacher', 0.01)
        self.distill_expert_epochs = getattr(args, 'distill_expert_epochs', 3)
        self.distill_syn_steps = getattr(args, 'distill_syn_steps', 20)
        self.distill_max_start_epoch = getattr(args, 'distill_max_start_epoch', 25)
        self.distill_pix_init = getattr(args, 'distill_pix_init', 'real')
        self.dsa_strategy = getattr(args, 'dsa_strategy', 'color_crop_cutout_flip_scale_rotate')

        self.distiller = TaskDistiller(
            channel=self.channel,
            num_classes=int(self.nc_per_task),
            im_size=self.im_size,
            ipc=self.ipc,
            device='cuda' if args.cuda else 'cpu',
            iteration=self.distill_iterations,
            lr_img=self.distill_lr_img,
            lr_lr=self.distill_lr_lr,
            lr_teacher=self.distill_lr_teacher,
            expert_epochs=self.distill_expert_epochs,
            syn_steps=self.distill_syn_steps,
            max_start_epoch=self.distill_max_start_epoch,
            batch_train=128,
            dsa_strategy=self.dsa_strategy,
            pix_init=self.distill_pix_init
        )

    def take_loss(self, logits, y):
        # compute loss on data from a single task
        # offset1, offset2 = self.compute_offsets(t)
        # loss = self.loss(logits[:, offset1:offset2], y - offset1)
        loss = self.loss(logits, y)
        return loss

    def take_multitask_loss(self, logits, y):
        # loss = 0.0
        #
        # for i, ti in enumerate(bt):
        #     offset1, offset2 = self.compute_offsets(ti)
        #     loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0) - offset1)
        # return loss / len(bt)
        loss = self.loss(logits, y)
        return loss

    def reshape_input(self, x):
        """Reshape flattened input to image format for CNN"""
        if self.args.dataset == 'tinyimagenet':
            return x.view(-1, 3, 64, 64)
        elif self.args.dataset == 'cifar100':
            return x.view(-1, 3, 32, 32)
        return x

    def forward(self, x, t):
        x = self.reshape_input(x)
        output = self.net.forward(x)
        return output

    def meta_loss(self, x, fast_weights, y, t):
        """
        differentiate the loss through the network updates wrt alpha
        Class incremental: use all outputs up to current max class
        """
        x = self.reshape_input(x)
        _, offset2 = self.compute_offsets(t)

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss_q = self.take_multitask_loss(logits, y)

        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        Class incremental: use all outputs up to current max class
        """
        x = self.reshape_input(x)
        _, offset2 = self.compute_offsets(t)

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss = self.take_loss(logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))

        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)

        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))

        return fast_weights

    def observe(self, x, y, t):
        self.net.train()

        # 收集当前任务的所有数据用于蒸馏（只在第一个epoch收集，避免重复）
        if self.real_epoch == 0 and self.pass_itr == 0:
            # 将数据reshape为图像格式并存储
            x_img = self.reshape_input(x).detach().cpu()
            y_cpu = y.detach().cpu()
            for i in range(x_img.size(0)):
                self.current_task_data['images'].append(x_img[i])
                self.current_task_data['labels'].append(y_cpu[i].item())

        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.M = self.M_new.copy()
                self.current_task = t

            batch_sz = x.shape[0]
            n_batches = self.args.cifar_batches
            rough_sz = math.ceil(batch_sz / n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            # get a batch by augmented incoming data with old task data, used for
            # computing meta-loss
            bx, by, _ = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)

            for i in range(n_batches):

                batch_x = x[i * rough_sz: (i + 1) * rough_sz]
                batch_y = y[i * rough_sz: (i + 1) * rough_sz]

                # assuming labels for inner update are from the same
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)
                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch
                if (self.real_epoch == 0):
                    self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, t)

                meta_losses[i] += meta_loss

            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            meta_loss = sum(meta_losses) / len(meta_losses)
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            if self.args.learn_lr:
                self.opt_lr.step()

            # if sync-update is being carried out (as in sync-maml) then update the weights using the optimiser
            # otherwise update the weights with sgd using updated LRs as step sizes
            if (self.args.sync_update):
                self.opt_wt.step()
            else:
                for i, p in enumerate(self.net.parameters()):
                    # using relu on updated LRs to avoid negative values
                    p.data = p.data - p.grad * nn.functional.relu(self.net.alpha_lr[i])
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return meta_loss.item()

    def getBatch(self, x, y, t, batch_size=None):
        """
        重写父类的getBatch方法
        组合当前任务数据 + 蒸馏buffer中的历史数据
        用于计算meta-loss
        """
        if x is not None:
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int) * t
        else:
            mxi = np.empty(shape=(0, 0))
            myi = np.empty(shape=(0, 0))
            mti = np.empty(shape=(0, 0))

        bxs = []
        bys = []
        bts = []

        batch_size = self.batchSize if batch_size is None else batch_size

        # 1. 从蒸馏buffer采样历史任务知识（如果存在且不是第一个任务）
        if len(self.M_distill) > 0 and t > 0:
            distill_size = min(batch_size // 2, len(self.M_distill))
            order = list(range(len(self.M_distill)))
            shuffle(order)
            for j in range(distill_size):
                k = order[j]
                x_d, y_d, t_d = self.M_distill[k]
                # 蒸馏数据已经是图像格式 [C, H, W]，需要展平
                xi = x_d.cpu().numpy().flatten()
                yi = y_d.cpu().numpy() if isinstance(y_d, torch.Tensor) else np.array(y_d)
                ti = t_d.cpu().numpy() if isinstance(t_d, torch.Tensor) else np.array(t_d)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # 2. 从当前任务buffer (M_new) 采样
        if self.args.use_old_task_memory and t > 0:
            MEM = self.M
        else:
            MEM = self.M_new

        if len(MEM) > 0:
            # 如果已经从蒸馏buffer采样了，则减少从M_new采样的数量
            current_batch_size = batch_size // 2 if (len(self.M_distill) > 0 and t > 0) else batch_size
            osize = min(current_batch_size, len(MEM))
            order = list(range(len(MEM)))
            shuffle(order)
            for j in range(osize):
                k = order[j]
                x_m, y_m, t_m = MEM[k]
                xi = np.array(x_m)
                yi = np.array(y_m)
                ti = np.array(t_m)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # 3. 添加当前batch的新数据
        for j in range(len(myi)):
            bxs.append(mxi[j])
            bys.append(myi[j])
            bts.append(mti[j])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)

        # handle gpus if specified
        if self.cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs, bys, bts

    def end_task(self, task_id):
        """
        任务结束后进行数据蒸馏，并使用轨迹偏移更新旧任务的蒸馏数据

        流程:
        1. 获取当前meta-update后的参数
        2. 使用轨迹偏移更新所有旧任务的蒸馏数据
        3. 蒸馏当前任务的数据，并保存trajectories
        4. 清空current_task_data准备下一个任务
        """
        print(f"\n===== Task {task_id} ended, starting distillation =====")

        # 1. 检查是否有收集到的数据
        if len(self.current_task_data['images']) == 0:
            print("No data collected for current task, skipping distillation")
            return

        print(f"Found {len(self.current_task_data['images'])} samples for task {task_id}")

        # 2. 转换为tensor
        task_images = torch.stack(self.current_task_data['images']).float()
        task_labels = torch.tensor(self.current_task_data['labels'], dtype=torch.long)

        if self.cuda:
            task_images = task_images.cuda()
            task_labels = task_labels.cuda()

        # 3. 获取当前网络的meta-update后的参数
        print("Getting meta-updated parameters...")
        meta_params = [p.detach().cpu().clone() for p in self.net.parameters()]
        self.distiller.set_meta_init_params(meta_params)

        # 4. 使用轨迹偏移更新旧任务的蒸馏数据
        if len(self.task_trajectories) > 0:
            print(f"\n--- Updating {len(self.task_trajectories)} previous tasks with trajectory shift ---")
            self._update_old_tasks_with_trajectory_shift(meta_params)

        # 5. 蒸馏当前任务的数据
        print(f"\n--- Starting distillation for task {task_id} with IPC={self.ipc} ---")
        try:
            # 使用distill_with_trajectory_return来获取trajectories
            distilled_data = self.distiller.distill_with_trajectory_return(
                task_images,
                task_labels,
                verbose=True,
                num_trajectories_to_save=30
            )

            distilled_images = distilled_data['images']
            distilled_labels = distilled_data['labels']
            trajectories = distilled_data['trajectories']
            init_params = distilled_data['init_params']
            class_map = distilled_data['class_map']
            class_map_inv = distilled_data['class_map_inv']

            print(f"Distillation complete. Generated {len(distilled_images)} synthetic samples")
            print(f"Saved {len(trajectories)} trajectories for future trajectory shift")

            # 6. 保存当前任务的trajectories和初始化参数
            self.task_trajectories[task_id] = {
                'trajectories': trajectories,
                'init_params': init_params,
                'class_map': class_map,
                'class_map_inv': class_map_inv
            }

            # 7. 将蒸馏数据添加到M_distill
            for i in range(len(distilled_images)):
                distill_sample = [
                    distilled_images[i].cpu(),
                    distilled_labels[i].cpu(),
                    torch.tensor(task_id)
                ]
                self._add_to_distill_buffer(distill_sample)

            print(f"M_distill now has {len(self.M_distill)} samples")

        except Exception as e:
            print(f"Distillation failed with error: {e}")
            import traceback
            traceback.print_exc()

        # 8. 清空current_task_data，准备下一个任务
        self.current_task_data = {'images': [], 'labels': []}

        # 保留M_new的数据到M中，然后清空M_new
        self.M = self.M_new.copy()
        self.M_new = []
        self.age = 0

        print(f"===== Distillation for task {task_id} complete =====\n")

    def _update_old_tasks_with_trajectory_shift(self, new_init_params):
        """
        使用轨迹偏移更新所有旧任务的蒸馏数据

        Args:
            new_init_params: 当前meta-update后的参数
        """
        # 按任务ID收集M_distill中的数据
        task_distill_data = {}
        task_distill_indices = {}

        for idx, sample in enumerate(self.M_distill):
            img, label, t_id = sample
            t_id_val = t_id.item() if isinstance(t_id, torch.Tensor) else t_id

            if t_id_val not in task_distill_data:
                task_distill_data[t_id_val] = {'images': [], 'labels': []}
                task_distill_indices[t_id_val] = []

            task_distill_data[t_id_val]['images'].append(img)
            task_distill_data[t_id_val]['labels'].append(label)
            task_distill_indices[t_id_val].append(idx)

        # 对每个旧任务进行轨迹偏移更新
        for old_task_id, traj_info in self.task_trajectories.items():
            if old_task_id not in task_distill_data:
                print(f"Task {old_task_id} has no distilled data in M_distill, skipping")
                continue

            print(f"Updating task {old_task_id} with trajectory shift...")

            # 获取旧任务的蒸馏数据
            old_images = torch.stack(task_distill_data[old_task_id]['images'])
            old_labels = torch.stack(task_distill_data[old_task_id]['labels'])

            # 调用轨迹偏移更新方法
            updated_data = self.distiller.update_distilled_data_with_trajectory_shift(
                old_images=old_images,
                old_labels=old_labels,
                old_trajectories=traj_info['trajectories'],
                old_init_params=traj_info['init_params'],
                new_init_params=new_init_params,
                class_map=traj_info['class_map'],
                class_map_inv=traj_info['class_map_inv'],
                meta_lr=0.1,
                update_iterations=500,
                verbose=True
            )

            # 更新M_distill中对应的数据
            updated_images = updated_data['images']
            updated_labels = updated_data['labels']

            indices = task_distill_indices[old_task_id]
            for i, idx in enumerate(indices):
                if i < len(updated_images):
                    self.M_distill[idx] = [
                        updated_images[i].cpu(),
                        updated_labels[i].cpu(),
                        torch.tensor(old_task_id)
                    ]

            # 更新task_trajectories中的init_params为新的参数
            self.task_trajectories[old_task_id]['init_params'] = [
                p.clone() for p in new_init_params
            ]

            print(f"Task {old_task_id} updated successfully")

    def _add_to_distill_buffer(self, distill_sample):
        """
        将蒸馏样本添加到M_distill
        使用reservoir sampling来管理固定大小的buffer
        """
        if len(self.M_distill) < self.memories:
            # 还有空间，直接添加
            self.M_distill.append(distill_sample)
        else:
            # Buffer已满，使用reservoir sampling替换
            # 计算替换概率
            idx = random.randint(0, len(self.M_distill))
            if idx < self.memories:
                self.M_distill[idx] = distill_sample

