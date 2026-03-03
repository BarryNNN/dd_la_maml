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

        # 存储所有任务的蒸馏数据，用于 Phase 2 后训练
        # 格式: {task_id: {'images': Tensor, 'labels': Tensor}}
        self.all_distilled_data = {}

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

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y - offset1)
        return loss

    def take_multitask_loss(self, bt, logits, y, task_id, current_batch_size):
        # compute loss on data from multiple tasks
        # class incremental setting: different loss computation for current task data and replay buffer data
        # Note: getBatch returns [memory_buffer_data, current_task_data]
        # so current task data starts at index (len(bt) - current_batch_size)
        loss = 0.0
        n_total = len(bt)
        n_buffer = n_total - current_batch_size  # number of samples from memory buffer

        for i, ti in enumerate(bt):
            if i < n_buffer:  # replay buffer data (comes first from getBatch)
                _, offset2 = self.compute_offsets(task_id)
                loss += 0.5 * self.loss(logits[i, 0:offset2].unsqueeze(0), y[i].unsqueeze(0))
            else:  # current task data
                offset1, offset2 = self.compute_offsets(ti)
                loss += self.loss(logits[i, offset1:].unsqueeze(0), y[i].unsqueeze(0) - offset1)
        return loss / n_total

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

    def meta_loss(self, x, fast_weights, y, bt, t, batch_size=32):
        """
        differentiate the loss through the network updates wrt alpha
        class incremental setting
        """
        x = self.reshape_input(x)

        logits = self.net.forward(x, fast_weights)
        loss_q = self.take_multitask_loss(bt, logits, y, t, batch_size)

        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        Class incremental: use all outputs up to current max class
        """
        x = self.reshape_input(x)

        logits = self.net.forward(x, fast_weights)
        loss = self.take_loss(t, logits, y)

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
            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)

            for i in range(n_batches):

                batch_x = x[i * rough_sz: (i + 1) * rough_sz]
                batch_y = y[i * rough_sz: (i + 1) * rough_sz]

                # assuming labels for inner update are from the same
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)
                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch
                if (self.real_epoch == 0):
                    self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, bt, t, batch_size=batch_sz)

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

    def end_task(self, task_id):
        """
        任务结束后进行数据蒸馏，并使用轨迹偏移更新旧任务的蒸馏数据
        蒸馏数据仅存储，不参与 Phase 1 训练，留给 Phase 2 使用

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

            # 7. 将蒸馏数据存储到 all_distilled_data（用于 Phase 2）
            self.all_distilled_data[task_id] = {
                'images': distilled_images.cpu(),
                'labels': distilled_labels.cpu()
            }

            total_distilled = sum(len(d['images']) for d in self.all_distilled_data.values())
            print(f"Total distilled samples across all tasks: {total_distilled}")

        except Exception as e:
            print(f"Distillation failed with error: {e}")
            import traceback
            traceback.print_exc()

        # 8. 清空current_task_data，准备下一个任务
        self.current_task_data = {'images': [], 'labels': []}

        print(f"===== Distillation for task {task_id} complete =====\n")

    def _update_old_tasks_with_trajectory_shift(self, new_init_params):
        """
        使用轨迹偏移更新所有旧任务的蒸馏数据

        Args:
            new_init_params: 当前meta-update后的参数
        """
        for old_task_id, traj_info in self.task_trajectories.items():
            if old_task_id not in self.all_distilled_data:
                print(f"Task {old_task_id} has no distilled data, skipping")
                continue

            print(f"Updating task {old_task_id} with trajectory shift...")

            old_images = self.all_distilled_data[old_task_id]['images']
            old_labels = self.all_distilled_data[old_task_id]['labels']

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

            # 更新 all_distilled_data 中对应任务的数据
            self.all_distilled_data[old_task_id] = {
                'images': updated_data['images'].cpu(),
                'labels': updated_data['labels'].cpu()
            }

            # 更新task_trajectories中的init_params为新的参数
            self.task_trajectories[old_task_id]['init_params'] = [
                p.clone() for p in new_init_params
            ]

            print(f"Task {old_task_id} updated successfully")


    def train_on_distilled_data(self, args, val_tasks=None, test_tasks=None, evaluator=None):
        """
        Phase 2: 使用全部蒸馏数据在 θ_T（Phase 1 最终参数）上继续训练

        对应 Algorithm 1 的第 23-29 行:
        - 收集所有任务的蒸馏数据 DD_R = ∪ D_syn^t
        - 用 θ_T 初始化（即当前 self.net 的参数，Phase 1 最后一次 meta-update 的结果）
        - 用蒸馏数据训练模型

        Args:
            args: 训练参数
            val_tasks: 验证任务（用于训练过程中评估）
            test_tasks: 测试任务
            evaluator: 评估函数
        """
        if len(self.all_distilled_data) == 0:
            print("No distilled data available for Phase 2 training")
            return

        print("\n" + "=" * 60)
        print("Phase 2: Training on distilled data")
        print("=" * 60)

        # 1. 收集所有任务的蒸馏数据 DD_R = ∪ D_syn^t
        all_images = []
        all_labels = []
        for task_id in sorted(self.all_distilled_data.keys()):
            data = self.all_distilled_data[task_id]
            all_images.append(data['images'])
            all_labels.append(data['labels'])
            print(f"Task {task_id}: {len(data['images'])} distilled samples")

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        print(f"Total distilled data: {len(all_images)} samples")

        # 2. θ_dd ← θ_T（当前网络参数即为 Phase 1 最终的 meta-updated 参数）
        # 不需要额外操作，self.net 已经包含 θ_T
        print("Using Phase 1 final parameters (θ_T) as initialization")

        # 3. 创建 DataLoader
        if self.cuda:
            all_images = all_images.cuda()
            all_labels = all_labels.cuda()

        dataset = torch.utils.data.TensorDataset(all_images, all_labels)
        dd_batch_size = getattr(args, 'dd_batch_size', min(64, len(all_images)))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=dd_batch_size, shuffle=True, num_workers=0
        )

        # 4. 设置 Phase 2 训练超参数
        num_dd_epochs = getattr(args, 'num_dd_epochs', 100)
        dd_lr = getattr(args, 'dd_lr', 0.01)
        dd_weight_decay = getattr(args, 'dd_weight_decay', 1e-4)

        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=dd_lr,
            momentum=0.9, weight_decay=dd_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_dd_epochs
        )
        criterion = nn.CrossEntropyLoss()

        # 5. 训练循环
        self.net.train()
        for ep in range(num_dd_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_images, batch_labels in dataloader:
                optimizer.zero_grad()

                # 蒸馏数据已经是 [C, H, W] 格式，直接前向传播
                logits = self.net.forward(batch_images)
                loss = criterion(logits, batch_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # 定期评估
            if evaluator is not None and val_tasks is not None and (ep + 1) % 10 == 0:
                val_acc = evaluator(self, val_tasks, args)
                avg_val = sum(val_acc).item() / len(val_acc)
                print(f"Phase 2 Epoch {ep+1}/{num_dd_epochs} | Loss: {avg_loss:.4f} | Val Acc: {avg_val:.4f}")
            elif (ep + 1) % 10 == 0:
                print(f"Phase 2 Epoch {ep+1}/{num_dd_epochs} | Loss: {avg_loss:.4f}")

        # 6. 最终评估
        print("\n--- Phase 2 Training Complete ---")
        if evaluator is not None:
            if val_tasks is not None:
                val_acc = evaluator(self, val_tasks, args)
                print(f"Phase 2 Final Val Accuracy: Total={sum(val_acc).item()/len(val_acc):.4f} | Per-task={val_acc}")
            if test_tasks is not None:
                test_acc = evaluator(self, test_tasks, args)
                print(f"Phase 2 Final Test Accuracy: Total={sum(test_acc).item()/len(test_acc):.4f} | Per-task={test_acc}")
        print("=" * 60)

