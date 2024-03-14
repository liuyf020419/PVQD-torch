import ml_collections
import torch
from typing import List, Dict, Any
from ml_collections import ConfigDict
import torch.optim as optim
import math 
from itertools import chain
from typing import Optional

class FP32Adam(optim.Optimizer):
    r"""reimplement Adam with fp32 cast, all states are fp32 though params and grads may be (bfloat16,float16)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(FP32Adam, self).__init__(params, defaults)
    
    def state_dict(self):
        return super().state_dict()
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. change all states back to fp32 here
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain(*(g["params"] for g in saved_groups)),
                chain(*(g["params"] for g in groups)),
            )
        }
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                self.state[param] = v

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


class DynamicLossScaler(object):
    def __init__(
        self,
        init_scale=2.0 ** 15,
        scale_factor=2.0,
        scale_window=2000,
        tolerance=0.0,
        threshold=None,
        min_loss_scale=1e-4,
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0
        self.min_loss_scale = min_loss_scale

    def scale(self, outputs):
        return self.loss_scale * outputs

    def update(self):
        if (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1

    def _decrease_loss_scale(self):
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    def check_overflow(self, grad_norm):
        # detect inf and nan
        if grad_norm == float("inf") or grad_norm != grad_norm:
            # overflow has occured
            prev_scale = self.loss_scale
            iter_since_rescale = self._iter - self._last_rescale_iter

            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0

            if self.loss_scale <= self.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                self.loss_scale = prev_scale
                raise FloatingPointError(
                    (
                        "Minimum loss scale reached ({}). Your loss is probably exploding. "
                        "Try lowering the learning rate, using gradient clipping or "
                        "increasing the batch size."
                    ).format(self.min_loss_scale)
                )

            self._iter += 1
            raise OverflowError("setting loss scale to: " + str(self.loss_scale))

class LRScheduler(object):
    def __init__(self, cfg:ConfigDict):
        self.warmup_steps=cfg.warmup_steps
        self.init_value= cfg.init_value
        self.peak_value=cfg.peak_value
        self.decay_rate= cfg.decay_rate
        self.transition_begin=cfg.transition_begin
        self.transition_steps=cfg.transition_steps
    
    def step(self, num_updates):
        if num_updates < self.warmup_steps:
            lr = self.init_value +(self.peak_value - self.init_value) * num_updates / self.warmup_steps
        elif num_updates <= self.warmup_steps + self.transition_begin:
            lr= self.peak_value
        else:
            lr= self.peak_value*self.decay_rate**((num_updates-self.warmup_steps - self.transition_begin)/self.transition_steps)
        return lr


class GradSaver(object):
    def __init__(self, params:List[torch.Tensor]):
        self.grads={p:torch.zeros_like(p) for p in params}
    
    def accum_grads(self):
        with torch.no_grad():
            for p,g in self.grads.items():
                if p.grad is None:
                    continue
                g.add_(p.grad)
                p.grad.zero_()
    
    def back_grads(self):
        with torch.no_grad():
            for p,g in self.grads.items():
                if p.grad is None:
                    continue
                p.grad.copy_(g)
                g.zero_()


@torch.no_grad()
def calc_grad_norm_(params) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in params if grad_exists(p)]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(g, p=2, dtype=torch.float32) for g in grads]
        )
    )
    return total_norm


class BigOptimizerNoScale(object):
    def __init__(
        self, 
        optimizer:optim.Optimizer,
        lrscheduler: LRScheduler,
        gradsaver:GradSaver
    ):
        self.optimizer= optimizer
        self.lrscheduler= lrscheduler
        self.gradsaver= gradsaver
    
    @classmethod
    def build_optimizer(cls, cfg:ConfigDict, params):
        optimizer= FP32Adam(
            params,
            lr= cfg.init_lr,
            betas= cfg.betas,
            weight_decay= cfg.weight_decay
        )
        lr_scheduler= LRScheduler(cfg.warmup)
        if cfg.fp16:
            raise NotImplementedError('float16 is not supported now')
        gsaver= GradSaver(params)
        return cls(optimizer, lr_scheduler, gsaver)
    
    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def backward(self, loss):
        loss.backward()
    
    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    # def _real_multiply_grads(self, c):
    #     for p in self.params:
    #         if p.grad is not None:
    #             if torch.is_tensor(c):
    #                 c = c.to(p.grad.device)
    #             p.grad.data.mul_(c)
    
    # def _unscale_grads(self):
    #     if (
    #         torch.is_tensor(self._multiply_factor)
    #         or self._multiply_factor != 1.0
    #     ):
    #         self._real_multiply_grads(self._multiply_factor)
    #         self._multiply_factor = 1.0
    
    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = lr
    
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm and updates dynamic loss scaler."""
        max_norm = float(max_norm)
        grad_norm = calc_grad_norm_(self.params)
        clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
        self.multiply_grads(clip_coef)
        return grad_norm
    
    def all_reduce_grads(self, module):
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()
    
    def step(self, num_updates):
        """Performs a single optimization step."""
        lr= self.lrscheduler.step(num_updates)
        self.set_lr(lr)
        self.optimizer.step()


    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad()
    
    def accum_grads(self):
        self.gradsaver.accum_grads()
    def back_grads(self):
        self.gradsaver.back_grads()

        


class BigOptimizer(object):
    """ put optimizer, lrscheduler and amp scaler together
        I cannot find a appropriate name for this, just call it BigOp...
    """
    def __init__(
        self, 
        optimizer:optim.Optimizer,
        lrscheduler: LRScheduler,
        scaler: Optional[DynamicLossScaler] = None
    ):
        self.optimizer= optimizer
        self.lrscheduler= lrscheduler
        self.scaler= scaler
        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)
        else:
            self._multiply_factor = 1.0

    @classmethod
    def build_optimizer(cls, cfg:ConfigDict, params):
        optimizer= FP32Adam(
            params,
            lr= cfg.init_lr,
            betas= cfg.betas,
            weight_decay= cfg.weight_decay
        )
        lr_scheduler= LRScheduler(cfg.warmup)
        if cfg.fp16:
            scaler= DynamicLossScaler(
                init_scale= 2**15,
                scale_factor=2,
                scale_window=2000
            )
        else:
            scaler= None
        return cls(optimizer, lr_scheduler, scaler)
    
    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.optimizer.state_dict()
        if self.scaler is not None:
            state_dict["loss_scale"] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict):
        if "loss_scale" in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict["loss_scale"]
        self.optimizer.load_state_dict(state_dict)
    
    def backward(self, loss):
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()
    
    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def _real_multiply_grads(self, c):
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)
    
    def _unscale_grads(self):
        if (
            torch.is_tensor(self._multiply_factor)
            or self._multiply_factor != 1.0
        ):
            self._real_multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0
    
    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = lr
    
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        self._multiply_factor *= c

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm and updates dynamic loss scaler."""
        max_norm = float(max_norm)
        grad_norm = self._multiply_factor * calc_grad_norm_(self.params)
        if self.scaler is not None:
            grad_norm_cpu = float(grad_norm)
            if grad_norm_cpu > max_norm > 0.0:
                self._multiply_factor *= max_norm / grad_norm_cpu

            # detect overflow and adjust loss scale
            self.scaler.check_overflow(grad_norm_cpu)
        elif max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
            self._multiply_factor *= clip_coef
        return grad_norm
    
    def all_reduce_grads(self, module):
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()
    
    def step(self, num_updates):
        """Performs a single optimization step."""
        lr= self.lrscheduler.step(num_updates)
        self.set_lr(lr)
        self._unscale_grads()
        self.optimizer.step()
        if self.scaler is not None:
            self.scaler.update()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)
        else:
            self._multiply_factor = 1.0
    
