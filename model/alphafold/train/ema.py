from collections import OrderedDict
import copy
import torch
import torch.nn as nn


class ExponentialMovingAverage(object):
    """
        Suppl. 1.11.7 use EMA with decay 0.999
        we implement only float32 version, for possible precision problems
    """
    def __init__(self, model:nn.Module, decay:float):
        state_dict= model.state_dict()
        self.params= {
            k:v.clone().detach().float()
            for k,v in state_dict.items()
        }
        self.decay= decay
    
    def load_state_dict(self, state_dict):
        if 'params' not in state_dict:
            raise KeyError('EMA state key error')
        pstate= state_dict['params']
        for k in self.params.keys():
            if k not in pstate:
                raise RuntimeError(f"EMA model key {k} not exists")
            self.params[k].copy_(pstate[k])
    
    def state_dict(self):
        return {"params":self.params}
    
    def update(self, model:nn.Module):
        with torch.no_grad():
            for k,p in model.state_dict().items():
                p_self = self.params[k]
                if p.dtype in {torch.bfloat16, torch.float16}:
                    p = p.float()
                diff= (1-self.decay)*(p_self -p)
                p_self -= diff


