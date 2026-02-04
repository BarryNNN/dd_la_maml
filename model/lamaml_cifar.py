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

    def take_loss(self, logits, y):
        loss = self.loss(logits, y)
        return loss

    def take_multitask_loss(self, logits, y):
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
            grads[i] = torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))

        return fast_weights


    def observe(self, x, y, t):
        self.net.train() 
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
            rough_sz = math.ceil(batch_sz/n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            # get a batch by augmented incoming data with old task data, used for
            # computing meta-loss
            bx, by, _ = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)

            for i in range(n_batches):

                batch_x = x[i*rough_sz : (i+1)*rough_sz]
                batch_y = y[i*rough_sz : (i+1)*rough_sz]

                # assuming labels for inner update are from the same 
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)   
                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch     
                if(self.real_epoch == 0):
                    self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, t)
                
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

