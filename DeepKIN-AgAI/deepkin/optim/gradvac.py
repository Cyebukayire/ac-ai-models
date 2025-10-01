import random

import torch
import torch.distributed as dist

class GradVacc:

    def __init__(self, num_tasks, optimizer: torch.optim.Optimizer, DEVICE, scaler: torch.cuda.amp.GradScaler = None, beta = 1e-2, cpu_offload: bool = True):
        self.device = torch.device('cpu') if cpu_offload else DEVICE
        self.num_tasks = num_tasks
        self.cpu_offload = cpu_offload
        self.beta = beta
        self._scaler, self._optim = scaler, optimizer
        # Setup default accumulated gradient
        num_els, self.gradient_shapes = self._init_shapes()
        self.accumulated_gradients_list = [torch.zeros(num_els, dtype=torch.float32, device=self.device) for _ in range(self.num_tasks)]
        self.vaccine_temp1_gradient = torch.zeros(num_els, dtype=torch.float32, device=self.device) if ((dist.get_rank() == 0) and (self.num_tasks > 1)) else None
        self.vaccine_temp2_gradient = torch.zeros(num_els, dtype=torch.float32, device=self.device) if ((dist.get_rank() == 0) and (self.num_tasks > 1))  else None
        self.rho_T = torch.zeros(self.num_tasks, self.num_tasks, device=self.device)
        self.flat_cpu_param = None
        return

    def state_dict(self) -> dict:
        if self._scaler is not None:
            return {'scaler': self._scaler.state_dict(), 'optimizer': self._optim.state_dict(), 'rho_T': self.rho_T}
        else:
            return {'optimizer': self._optim.state_dict(), 'rho_T': self.rho_T}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict['scaler'])
            self._optim.load_state_dict(state_dict['optimizer'])
            self.rho_T.copy_(state_dict['rho_T'])
        else:
            self._optim.load_state_dict(state_dict['optimizer'])
            self.rho_T.copy_(state_dict['rho_T'])

    @property
    def optimizer(self):
        return self._optim

    @property
    def scaler(self):
        return self._scaler

    def zero_grad(self):
        ret = self._optim.zero_grad()
        # Setup zero accumulated gradient
        for i in range(self.num_tasks):
            self.accumulated_gradients_list[i].zero_()
        return ret

    def step(self):
        # 1. Reduce
        for i in range(len(self.accumulated_gradients_list)):
            dist.reduce(self.accumulated_gradients_list[i], 0, async_op=False)
        # 2. [optional] Vaccine
        if (dist.get_rank() == 0) and (self.num_tasks > 1):
            self._apply_grad_vaccine_to_vaccine_grad0(self.accumulated_gradients_list, self.vaccine_temp1_gradient, self.vaccine_temp2_gradient)
            self.accumulated_gradients_list[0].zero_()
            self.accumulated_gradients_list[0].copy_(self.vaccine_temp2_gradient)

        # 3. Broadcast
        dist.broadcast(self.accumulated_gradients_list[0], 0, async_op=False)
        # 4. Use locally
        self._set_grad_from_vaccine_grad0(self.accumulated_gradients_list[0])
        # 5. Optimizer step/Update params
        if self._scaler is not None:
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            self._optim.step()

        return self.zero_grad()

    def backward(self, mt_losses):
        # Gradient accumulation
        for loss_id, loss in enumerate(mt_losses):
            if loss.item() > 0.0:
                self._optim.zero_grad()
                retain_graph = (loss_id < (self.num_tasks - 1))
                if self._scaler is not None:
                    self._scaler.scale(loss).backward(retain_graph = retain_graph)
                else:
                    loss.backward(retain_graph=retain_graph)
                self._accum_gradients(loss_id, self.accumulated_gradients_list)
        self._optim.zero_grad()

    def _apply_grad_vaccine_to_vaccine_grad0(self, accum_grads_list, temp_grad, final_grad):
        num_task = len(accum_grads_list)
        gnorms = [g.norm() for g in accum_grads_list]
        final_grad.zero_()
        for tn_i in range(num_task):
            temp_grad.copy_(accum_grads_list[tn_i])
            task_index = list(range(num_task))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(temp_grad, accum_grads_list[tn_j]) / (temp_grad.norm() * gnorms[tn_j])
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = temp_grad.norm() * (self.rho_T[tn_i, tn_j] * (1 - rho_ij ** 2).sqrt() - rho_ij * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt()) / (gnorms[tn_j] * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt())
                    temp_grad += accum_grads_list[tn_j] * w
                    self.rho_T[tn_i, tn_j] = (1 - self.beta) * self.rho_T[tn_i, tn_j] + self.beta * rho_ij
            final_grad += temp_grad

    def _set_grad_from_vaccine_grad0(self, final_grad):
        count = 0
        start_idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                end_idx = start_idx + p.numel()
                p.grad = final_grad[start_idx:end_idx].view(self.gradient_shapes[count]).clone()
                start_idx = end_idx
                count += 1

    def _export_flat_cpu_params(self, cpu_flat_param):
        count = 0
        start_idx = 0
        with torch.no_grad():
            for group in self._optim.param_groups:
                for param in group['params']:
                    end_idx = start_idx + param.numel()
                    cpu_flat_param[start_idx:end_idx].copy_(param.data.flatten())
                    start_idx = end_idx
                    count += 1

    def _copy_flat_gpu_params(self, gpu_flat_param):
        count = 0
        start_idx = 0
        with torch.no_grad():
            for group in self._optim.param_groups:
                for param in group['params']:
                    end_idx = start_idx + param.numel()
                    param.data.copy_(gpu_flat_param[start_idx:end_idx].view(self.gradient_shapes[count]))
                    start_idx = end_idx
                    count += 1

    def _init_shapes(self):
        shapes = []
        num_els = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                num_els += p.numel()
                shapes.append(p.shape)
        return num_els, shapes

    def _accum_gradients(self, idx, accum_grads_list):
        start_idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                end_idx = start_idx + p.numel()
                if p.grad is not None:
                    accum_grads_list[idx][start_idx:end_idx] += p.grad.clone().flatten()
                start_idx = end_idx
