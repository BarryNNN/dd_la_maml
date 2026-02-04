import random
import numpy as np
import ipdb
import math
import torch
import torch.nn as nn
from model.lamaml_base import *

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

        # Dataset distillation buffer
        self.distill_buffer = []  # stores [x, y, t] for distilled data
        self.distill_ratio = 0.9  # ratio of distilled data in getBatch (1:9 = current:distill)
        self.ipc = 50  # images per class for distillation

        # Load pre-computed distilled data
        self.distill_data_path = getattr(args, 'distill_data_path', '/root/La-MAML/distill_data')
        self._load_distill_data()

    def _load_distill_data(self):
        """Load pre-computed distilled dataset"""
        import os
        images_path = os.path.join(self.distill_data_path, 'images_best.pt')
        labels_path = os.path.join(self.distill_data_path, 'labels_best.pt')

        if os.path.exists(images_path) and os.path.exists(labels_path):
            self.distill_images = torch.load(images_path)  # [10000, 3, 32, 32]
            self.distill_labels_soft = torch.load(labels_path)  # [10000, 100] soft labels
            self.distill_labels_hard = self.distill_labels_soft.argmax(dim=1)  # [10000] hard labels
            print(f"Loaded distilled data: {self.distill_images.shape[0]} samples")
        else:
            print(f"Warning: Distilled data not found at {self.distill_data_path}")
            self.distill_images = None
            self.distill_labels_soft = None
            self.distill_labels_hard = None

    def _get_distill_data_for_task(self, task_id):
        """Get distilled data for a specific task (classes in that task)"""
        if self.distill_images is None:
            return [], [], []

        offset1, offset2 = self.compute_offsets(task_id)
        # Select samples belonging to classes in this task
        mask = (self.distill_labels_hard >= offset1) & (self.distill_labels_hard < offset2)
        task_images = self.distill_images[mask]
        task_labels = self.distill_labels_hard[mask]

        # Subsample to ipc per class
        selected_x, selected_y = [], []
        for c in range(int(offset1), int(offset2)):
            class_mask = task_labels == c
            class_images = task_images[class_mask]
            class_labels = task_labels[class_mask]

            n_samples = min(self.ipc, class_images.shape[0])
            if n_samples > 0:
                indices = torch.randperm(class_images.shape[0])[:n_samples]
                selected_x.append(class_images[indices])
                selected_y.append(class_labels[indices])

        if len(selected_x) > 0:
            selected_x = torch.cat(selected_x, dim=0)
            selected_y = torch.cat(selected_y, dim=0)
            return selected_x, selected_y, task_id
        return [], [], []

    def _add_task_to_distill_buffer(self, task_id):
        """Add distilled data for completed task to distill buffer"""
        distill_x, distill_y, t = self._get_distill_data_for_task(task_id)
        if len(distill_x) > 0:
            # Flatten images for storage (consistent with M_new format)
            distill_x_flat = distill_x.view(distill_x.shape[0], -1)
            for i in range(distill_x_flat.shape[0]):
                self.distill_buffer.append([
                    distill_x_flat[i].cpu(),
                    distill_y[i].cpu(),
                    torch.tensor(t)
                ])
            print(f"Added {distill_x.shape[0]} distilled samples for task {task_id} to distill_buffer. Total: {len(self.distill_buffer)}")

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        return loss

    def take_multitask_loss(self, bt, t, logits, y):
        # compute loss on data from a multiple tasks
        # separate from take_loss() since the output positions for each task's
        # logit vector are different and we nly want to compute loss on the relevant positions
        # since this is a task incremental setting

        loss = 0.0

        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


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
        # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def meta_loss(self, x, fast_weights, y, bt, t):
        """
        differentiate the loss through the network updates wrt alpha
        """
        x = self.reshape_input(x)
        offset1, offset2 = self.compute_offsets(t)

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss_q = self.take_multitask_loss(bt, t, logits, y)

        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        x = self.reshape_input(x)
        offset1, offset2 = self.compute_offsets(t)

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss = self.take_loss(t, logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))

        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))

        return fast_weights


    def end_task(self, task_id):
        """Called at the end of each task to add distilled data to buffer and clear M_new"""
        # Add distilled data for the completed task
        self._add_task_to_distill_buffer(task_id)
        # Clear M_new for the next task
        self.M_new = []
        self.age = 0

    def observe(self, x, y, t):
        self.net.train()
        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.epoch += 1
            self.zero_grads()

            # Update current task (without triggering distillation)
            if t != self.current_task:
                self.current_task = t

            batch_sz = x.shape[0]
            n_batches = self.args.cifar_batches
            rough_sz = math.ceil(batch_sz/n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            # get a batch by augmented incming data with old task data, used for 
            # computing meta-loss
            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)             

            for i in range(n_batches):

                batch_x = x[i*rough_sz : (i+1)*rough_sz]
                batch_y = y[i*rough_sz : (i+1)*rough_sz]

                # assuming labels for inner update are from the same 
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)   
                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch     
                if(self.real_epoch == 0):
                    self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, bt, t) 
                
                meta_losses[i] += meta_loss

            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            meta_loss = sum(meta_losses)/len(meta_losses)            
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            if self.args.learn_lr:
                self.opt_lr.step()

            # if sync-update is being carried out (as in sync-maml) then update the weights using the optimiser
            # otherwise update the weights with sgd using updated LRs as step sizes
            if(self.args.sync_update):
                self.opt_wt.step()
            else:            
                for i,p in enumerate(self.net.parameters()):          
                    # using relu on updated LRs to avoid negative values           
                    p.data = p.data - p.grad * nn.functional.relu(self.net.alpha_lr[i])            
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return meta_loss.item()

    def getBatch(self, x, y, t, batch_size=None):
        """
        Given the new data points, create a batch with 1:9 ratio of
        current task data (from M_new) : distilled data (from distill_buffer)
        """

        if(x is not None):
            mxi = np.array(x)
            # Flatten if needed (for CNN input which is (batch, 3, 32, 32))
            if len(mxi.shape) > 2:
                mxi = mxi.reshape(mxi.shape[0], -1)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int)*t
        else:
            mxi = np.empty( shape=(0, 0) )
            myi = np.empty( shape=(0, 0) )
            mti = np.empty( shape=(0, 0) )

        bxs = []
        bys = []
        bts = []

        batch_size = self.batchSize if batch_size is None else batch_size

        # Calculate sample sizes based on 1:9 ratio (current:distill)
        if len(self.distill_buffer) > 0:
            # 1:9 ratio means 10% from current, 90% from distill
            n_current = max(1, int(batch_size * (1 - self.distill_ratio)))
            n_distill = batch_size - n_current
        else:
            # No distilled data yet (first task), use all from current
            n_current = batch_size
            n_distill = 0

        # Sample from M_new (current task buffer)
        MEM = self.M_new
        if len(MEM) > 0:
            order = list(range(len(MEM)))
            osize = min(n_current, len(MEM))
            for j in range(osize):
                shuffle(order)
                k = order[j]
                x_mem, y_mem, t_mem = MEM[k]

                # Ensure consistent flattening
                if isinstance(x_mem, torch.Tensor):
                    xi = x_mem.view(-1).numpy()
                else:
                    xi = np.array(x_mem).flatten()
                yi = np.array(y_mem)
                ti = np.array(t_mem)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # Sample from distill_buffer (distilled data from previous tasks)
        if len(self.distill_buffer) > 0 and n_distill > 0:
            order = list(range(len(self.distill_buffer)))
            osize = min(n_distill, len(self.distill_buffer))
            for j in range(osize):
                shuffle(order)
                k = order[j]
                x_dist, y_dist, t_dist = self.distill_buffer[k]

                # Ensure consistent flattening
                if isinstance(x_dist, torch.Tensor):
                    xi = x_dist.view(-1).numpy()
                else:
                    xi = np.array(x_dist).flatten()
                yi = np.array(y_dist)
                ti = np.array(t_dist)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # Add incoming data (current batch)
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

        return bxs,bys,bts
    
    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if(self.real_epoch > 0 or self.pass_itr>0):
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()              
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0,self.age)  
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]

