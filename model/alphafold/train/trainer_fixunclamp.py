import contextlib
import logging
import sys
from collections import OrderedDict
import time
import numpy as np
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, List
from ml_collections import ConfigDict
import torch
import torch.nn as nn
from alphafold.data import NpzDataset, DataIterator,GroupedIterator, AlphafoldDataset, AlphafoldDatasetFullChainMSA
import sys,os
from . import distributed, optimizer, utils
from torch.utils.tensorboard import SummaryWriter
import shutil
from .ema import ExponentialMovingAverage
from torch.profiler import profile, record_function, ProfilerActivity
from alphafold.model2.layers import LayerNormFP32,LinearFp32
from alphafold.data.dataset import numpy_seed

logger = logging.getLogger(__name__)

def tensor_value(tensor):
    if torch.is_tensor(tensor):
        if tensor.ndim>0:
            return tensor.mean().item()
        return tensor.item()
    return tensor

def ret2log(ret):
    logs=dict(
        loss = tensor_value(ret['loss']),
        masked_msa= tensor_value(ret['masked_msa']['loss']),
        distogram=tensor_value(ret['distogram']['loss']),
        experimentally_resolved=tensor_value(ret['experimentally_resolved']['loss']),
        predicted_lddt=tensor_value(ret['predicted_lddt']['loss']),
        # predicted_aligned_error=tensor_value(ret['predicted_aligned_error']['loss']),
        structure_loss =tensor_value(ret['structure_module']['loss']),
        fape_ca=tensor_value(ret['structure_module']['fape']),
        fape_all=tensor_value(ret['structure_module']['sidechain_fape']),
        structure_chi=tensor_value(ret['structure_module']['chi_loss']),
        angle_norm_loss=tensor_value(ret['structure_module']['angle_norm_loss']),
        structure_violations_extreme_ca_ca_distance=tensor_value(ret['structure_module']['metrics']['violations_extreme_ca_ca_distance']),
        structure_violations_between_residue_bond=tensor_value(ret['structure_module']['metrics']['violations_between_residue_bond']),
        structure_violations_between_residue_clash=tensor_value(ret['structure_module']['metrics']['violations_between_residue_clash']),
        structure_violations_within_residue=tensor_value( ret['structure_module']['metrics']['violations_within_residue']),
        structure_violations_per_residue=tensor_value(ret['structure_module']['metrics']['violations_per_residue']),
    )
    return logs

def sum_log(loginfos:List[Dict[str, float]]):
    keys= loginfos[0].keys()
    outinfo=OrderedDict()
    for key in keys:
        outinfo[key] =sum([log[key] for log in loginfos])
    return outinfo


class LogManager(object):
    main_keys = {'loss', 'structure_loss', 'masked_msa', 'fape_ca','fape_all', 'predicted_lddt'} #, 'structure_violations_per_residue'}
    def __init__(self, name='train_0', log_dir= None, log_freq= 1,should_log=True):
        self.name= name
        self.should_log=should_log
        self.log_freq= log_freq
        if log_dir is not None:
            self.writer= SummaryWriter(log_dir)
        else:
            self.writer = None
        self.start_time= time.perf_counter()
        self.pre_time= time.perf_counter()
        self.nsteps= 0
        self.nsamples=0
        self.inc_samples= 0
        self.ma_dict= OrderedDict(
            {k:0 for k in self.main_keys}
        )
    
    def reset(self):
        self.start_time= time.perf_counter()
        self.pre_time= time.perf_counter()
        self.nsamples= 0
        self.inc_samples =0
        self.ma_dict= OrderedDict(
            {k:0 for k in self.main_keys}
        )
    
    def print(self):
        den= max(self.nsamples, 1)
        logs = ['{}={}'.format(k,v/den) for k,v in self.ma_dict.items()]
        logs= ', '.join(logs)
        logger.info(f"{self.name} TOTAL, {logs}")
    
    def logging(self, loginfo, grad_norm, lr, nsamples, nupdates):
        # loginfo= ret2log(ret)
        for key in self.ma_dict:
            if key in loginfo:
                # self.ma_dict[key] += loginfo[key]
                # do not log average, for NAN about 
                self.ma_dict[key]= loginfo[key]
       
        self.nsamples += nsamples
        self.inc_samples +=nsamples
        if not self.should_log:
            return
        den= max(nsamples,1)
        if nupdates % self.log_freq ==0 :
            logs = ['{}={:.4f}'.format(k,v/den) for k,v in self.ma_dict.items()]
            ellapsed = time.perf_counter()-self.pre_time
            speed_curr= float(self.inc_samples)/ellapsed
            self.inc_samples = 0
            self.pre_time = time.perf_counter()
            avg_speed= float(self.nsamples)/(time.perf_counter()-self.start_time)
            logs.extend([
                'avg_speed={:.4f} sample/s'.format(avg_speed),
                'curr_speed={:.4f} sample/s'.format(speed_curr),
                f'lr={lr:.4e}',f'grad_norm={grad_norm:.4f}'
                ])
            logs= ', '.join(logs)
            logger.info(f"{self.name}:num_updates={nupdates},sample_processed= {self.nsamples}, {logs}")
        if self.writer is not None:
            for k,v in loginfo.items():
                if k =="fape_ca" and v/den >1.0:
                    k="fape_ca_noclamp"
                self.writer.add_scalar(f'{self.name}/{k}', v/den,nupdates)
            self.writer.add_scalar(f"{self.name}/lr", lr,nupdates)
            # remove globalsteps,for possible dlp error. we need some proof 
            self.writer.add_scalar(f"{self.name}/grad_norm", grad_norm,nupdates)
            # self.writer.add_scalar(f"{self.name}/grad_norm", grad_norm,nupdates)



class Trainer(object):
    def __init__(self, cfg:ConfigDict, model:nn.Module):
        self.cfg= cfg
        self._model= model
        self.dtype= torch.float32
        if cfg.train.fp16 and cfg.train.bfp16:
            logger.warning("fp16 and bfp16 are all True in config, set to bfloat16 by default")
            cfg.train.fp16=False
        if cfg.train.fp16:
            self._model =self._model.half()
            self.dtype= torch.float16
        elif cfg.train.bfp16:
            self._model= self._model.bfloat16()
            self.dtype=torch.bfloat16
        else:
            self._model = self._model.float()
        
        def conditional_fp32(module):
            fp32_layers=set([LayerNormFP32,LinearFp32])
            if type(module) in fp32_layers:
                module.float()
        self._model.apply(conditional_fp32)


        self._model.cuda()
        self._wrapped_model= None
        self._optimizer= None
        self._num_updates = 0
        
        self.log_dir= f'{cfg.args.root_dir}/log_dir'
        self.checkpoint_dir=f'{cfg.args.root_dir}/checkpoint'
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if cfg.train.no_ema:
            self._ema = None
        else:
            self._ema= ExponentialMovingAverage(self._model, cfg.train.ema_decay)
        
        self.train_logger= LogManager(
            'train', log_dir= self.log_dir, log_freq= cfg.train.log_every,
            should_log= self.data_parallel_rank ==0
            )
        self.valid_logger= LogManager(
            'valid', log_dir= self.log_dir, log_freq= 1,
            should_log= self.data_parallel_rank ==0)
        # self.train_data= self.load_dataset(split='train', epoch=0, mode_sel='CLAMP')
        # self.train_data2= self.load_dataset(split='train', epoch=0, mode_sel='UNCLAMP')
        self.train_data= self.load_dataset(split='train', epoch=0, mode_sel='ALL')
        self.valid_data= self.load_dataset(split='valid', epoch =0, mode_sel="ALL")

        if self.cfg.args.pretrain is not None and self.checkpoint_exists():
            logger.error('checkpoint exists while --pretrain given, try with different working path')
            raise FileExistsError('checkpoint exits with pretrain given')
        
        if self.cfg.args.pretrain is not None:
            self.load_pretrain(self.cfg.args.pretrain)
            logger.info('load from pretrained model')
        elif self.checkpoint_exists():
            logger.info('load from checkpoint')
            self.load_checkpoint()
        else:
            logger.info('random initializing')
        self._dummy_batch= None
        
    
    @property
    def dummy_batch(self):
        if self._dummy_batch is None:
            self._dummy_batch= self._prepare_sample(self.train_data.dummy_batch)[0]
        return self._dummy_batch

    
    def load_dataset(self, split='train', epoch =0, mode_sel="ALL"):
        if split=='train':
            datafile= self.cfg.train.train_list
        else:
            datafile= self.cfg.train.eval_list
        # dataset= NpzDataset(datafile )
        crop_seq_msa = getattr(self.cfg.train, 'crop_seq_msa', True)
        if crop_seq_msa:
            if mode_sel !="ALL":
                raise NotImplementedError('crop_seq_msa mode not support mode_sel')
            dataset = AlphafoldDataset(self.cfg,datafile, train= split=='train')
        else:
            dataset = AlphafoldDatasetFullChainMSA(self.cfg, datafile, train= split=='train',mode_sel=mode_sel)
        logger.info(f'loading data {datafile}')
        num_workers= self.cfg.args.num_workers
        if split !='train':
            num_workers=1
        if mode_sel== 'UNCLAMP':
            num_workers=2
        data_iter=DataIterator(
            dataset,
            num_shards= self.data_parallel_world_size,
            shard_id= self.data_parallel_rank,
            epoch= epoch,
            batch_size= self.cfg.data.common.batch_size,
            shuffle= split=='train',
            num_workers= num_workers,
            seed = self.cfg.args.seed
        )
        return data_iter

    

    @property
    def data_parallel_world_size(self):
        if self.cfg.args.distributed_world_size == 1:
            return 1
        return distributed.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.args.distributed_world_size == 1:
            return 0
        return distributed.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.data_parallel_world_size >1:
                self._wrapped_model= distributed.FairseqDistributedDataParallel(
                    self._model,
                    self.data_parallel_process_group
                )
            else:
                self._wrapped_model= self._model
        return self._wrapped_model
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            params= list(
                filter(
                    lambda p:p.requires_grad,
                    self.model.parameters()
                )
            )
            self._optimizer = optimizer.BigOptimizerNoScale.build_optimizer(
                self.cfg.train, params
            )
            
        return self._optimizer
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates
        
    
    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return self.dummy_batch, True
        sample = utils.move_to_cuda(sample)
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t
        if self.cfg.train.fp16:
            sample= utils.apply_to_sample(apply_half, sample)
        elif self.cfg.train.bfp16:
            sample= utils.apply_to_sample(apply_bfloat16, sample)
        # if self._dummy_batch is None:
        #     self._dummy_batch = sample
        # for k,v in sample.items():
        #     print(f"{k}:{v.shape}")
       
        return sample, False
    
    def fwd_step(self, batch, recycle_num=None):
        ret, losses = self.model(batch,recycle_num=recycle_num)
        batch_size= batch['aatype'].shape[0]
        loss = losses.sum()
        ret['loss']= loss
        loss= loss*batch_size
        loginfo= ret2log(ret)
        loginfo={k:v*batch_size for k,v in loginfo.items()}
        return loss, loginfo, batch_size
    
    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()
    
    def update(self):
        self.optimizer.step(self.get_num_updates())
        self.set_num_updates(self.get_num_updates()+1)
        if self._ema is not None:
            self._ema.update(self._model)

    
    def train_step(self, samples):
        self.model.train()
        self.zero_grad()
        loginfos, nsamples= [], 0
        
        def gen_recycle_num(nupdates, nsample):
            max_num_recycle= self.cfg.model.max_num_recycle
            seed= nupdates* self.cfg.train.update_every + nsample +self.cfg.args.seed
            with numpy_seed(seed):
                # Note: randint in numpy is half open, which is different from random.randint (all close)
                recy_num= np.random.randint(1, max_num_recycle+1)
            return recy_num
        
        use_clamp= True
        if (self.get_num_updates() % 10)==0:
            use_clamp=False
        
        for i, sample in enumerate(samples):
            sample, ignore= self._prepare_sample(sample)
            # ensure same batch on different node do same
            if 'use_clamped_fape' in sample:
                sample['use_clamped_fape'][:]=use_clamp
            def maybe_no_sync():
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack() 

            try:
                with maybe_no_sync():
                    # forward and backward
                    
                    loss, loginfo, sample_size= self.fwd_step(
                        sample, recycle_num= gen_recycle_num(self.get_num_updates(), i)
                    )
                    if ignore:
                        loss*=0
                        sample_size=0
                        loginfo= self.zero_log(loginfo)
                    
                    nres_sqrt = torch.sqrt(sample['seq_mask'][0,0].sum())
                    loss= loss*nres_sqrt
                    
                    self.optimizer.backward(loss)
                    gnorm = self.optimizer.clip_grad_norm(self.cfg.train.grad_clip_thresh)
                    self.optimizer.accum_grads()
                    loginfo['gnorm']=gnorm
                    loginfos.append(loginfo)
                    nsamples += sample_size
                    del loss
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self._log_oom(e)
                    raise e
                else:
                    raise e
        loginfo= sum_log(loginfos)
        if self.data_parallel_world_size >1:
            loginfo, nsamples= self.aggre_loginfo(loginfo, nsamples)
        updated= False
      
        self.optimizer.back_grads()
        # gnorm= optimizer.calc_grad_norm_(self.optimizer.params)
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                if self.data_parallel_world_size >1:
                    self.optimizer.all_reduce_grads(self.model)
            nsamples= max(nsamples,1) # for all node run with dummy
            # replace average with sum. note: grad averaged
            self.optimizer.multiply_grads(float(self.data_parallel_world_size)/nsamples)
            # self.optimizer.multiply_grads(float(self.data_parallel_world_size)/ nsamples)
            # with torch.autograd.profiler.record_function("clip-grads"):
            #     self.optimizer.multiply_grads(float(self.data_parallel_world_size)/ nsamples)
            #     grad_norm = self.optimizer.clip_grad_norm(self.cfg.train.grad_clip_thresh)
            # if not torch.isfinite(grad_norm).all():
            #     raise FloatingPointError("gradients are Nan/Inf")
            self.update()
            updated= True
        except FloatingPointError as e:
            raise e
        except OverflowError as e:
            overflow = True
            logger.info(f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}")
            # loginfo= self.zero_log(loginfo)
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise 
        grad_norm= loginfo['gnorm']/nsamples
        del loginfo['gnorm']
        if self.is_data_parallel_master:
            self.train_logger.logging(loginfo, grad_norm, self.optimizer.get_lr(), nsamples, self.get_num_updates())
        return updated
        
    
    def zero_log(self, loginfo):
        return {k:0 for k in loginfo}
    
    def aggre_loginfo(self, loginfo, sample_size):
        keys= list(loginfo.keys())
        buf = torch.Tensor(len(keys)+1).cuda()
        buf[0]= sample_size
        buf[1:] = torch.Tensor([loginfo[k] for k in keys])
        distributed.all_reduce(buf, group= self.data_parallel_process_group, op='sum')
        sample_size= buf[0].item()
        oinfo={}
        for i,k in enumerate(keys):
            oinfo[k]= buf[i+1].item()
        return oinfo, sample_size
    
    def state_dict(self):
        state_dict={
            "cfg":self.cfg,
            "model":self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "data": self.train_data.state_dict(),
            # "data2": self.train_data2.state_dict(),
            "update_num":self.get_num_updates(),
        }
        if self._ema is not None:
            state_dict['ema'] = self._ema.state_dict()
        return state_dict

    def load_state_dict(self,state_dict):
        if "update_num" in state_dict:
            self.set_num_updates(state_dict['update_num'])
        else:
            logger.info('reset update step to 1')
        if 'model' not in state_dict:
            raise RuntimeError(f'bad checkpoint: no \"model\" params')
        self._model.load_state_dict(state_dict['model'])
        
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if "data" in state_dict:
            self.train_data.load_state_dict(state_dict['data'])
        # if "data2" in state_dict:
        #     self.train_data2.load_state_dict(state_dict['data2'])
        if self._ema is not None:
            if 'ema' in state_dict:
                self._ema.load_state_dict(state_dict['ema'])
            else:
                # we reload model params, so directly reconstruct ema
                self._ema= ExponentialMovingAverage(self._model, self.cfg.train.ema_decay)
    
    def save_checkpoint(self):
        checkpoint_name= os.path.join(self.checkpoint_dir, f"checkpoint_{self.get_num_updates()}.pt")
        logger.info(f'num_update= {self.get_num_updates()}, save checkpoint to {checkpoint_name}')
        with open(checkpoint_name, 'wb') as f:
            torch.save(self.state_dict(), f)
        last_cp= os.path.join(self.checkpoint_dir, f"checkpoint_last.pt")
        if os.path.exists(last_cp):
            os.remove(last_cp)
        shutil.copy(checkpoint_name, last_cp)
        
    
    def checkpoint_exists(self):
        last_cp= os.path.join(self.checkpoint_dir, f"checkpoint_last.pt")
        if os.path.exists(last_cp):
            logger.info(f'{last_cp} exists')
            return True
        return False

    
    def load_checkpoint(self):
        last_cp= os.path.join(self.checkpoint_dir, f"checkpoint_last.pt")
        if not os.path.exists(last_cp):
            logger.warning(f'checkpoint file {last_cp} not exist, ignore load_checkpoint')
            return False
        with open(last_cp,'rb') as f:
            state = torch.load(f, map_location=torch.device("cpu"))
        logger.info(f'load from {last_cp}')
        self.load_state_dict(state)
        return True
    
    def load_pretrain(self, modelfile):
        if not os.path.exists(modelfile):
            logger.warning(f'pretrain file {modelfile} not exist, ignore ')
            return False
        with open(modelfile,'rb') as f:
            state = torch.load(f, map_location=torch.device("cpu"))
        if 'model' in state:
            self.load_state_dict(state)
        else:
            try:
                self._model.load_state_dict(state)
            except Exception as e:
                raise FileNotFoundError(f'bad pretrain {modelfile}')
            if self._ema is not None:
                self._ema = ExponentialMovingAverage(self._model,self.cfg.train.ema_decay)


    
    def profile(self):
        self.cfg.train.num_steps=1
        logger.info(f'profile 1 batch training steps')
        with profile(
            activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA,],
            profile_memory=True, record_shapes=True
        )as prof: 
            with record_function('train-step'):
                self.train()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")
    
    def check_data(self):
        logger.info('start check data')
        dataiter = self.train_data.next_epoch_itr()
        total= dataiter.total
        try:
            for i, sample in enumerate(dataiter):
                logger.info(f'{i}/{total} processed')
        except Exception as e:
            logger.info(f'error in processing data {e}')
       
 
    def train(self):
        max_updates= self.cfg.train.num_steps
        save_freq= self.cfg.train.checkpoint_every
        should_save= self.data_parallel_rank ==0
        logger.info(f'start training, update number is {self.get_num_updates()}')
        train_iter1= self.train_data.next_epoch_itr()
        train_iter1= GroupedIterator(train_iter1, chunk_size= self.cfg.train.update_every)
        while self.get_num_updates() < max_updates:

            
            if not train_iter1.has_next():
                train_iter1 = self.train_data.next_epoch_itr()
                train_iter1= GroupedIterator(train_iter1, chunk_size= self.cfg.train.update_every)
            train_data, train_iter = self.train_data, train_iter1
            samples= next(train_iter)
            logger.info(f"Train {train_iter.n}|{train_iter.total} in epoch {train_data.epoch}")     
            updated = self.train_step(samples)
            if (self.get_num_updates()+1) % save_freq ==0 and updated:
                self.valid()
                if should_save:
                    self.save_checkpoint()
            if self.get_num_updates() >= max_updates:
                logger.info('Training finished')
                self.train_logger.print()
                return
        logger.info('Training finished')
        self.train_logger.print()
        return

    def valid(self, log_step=False):
        self.model.eval()
        logger.info(f'start validation for model step {self.get_num_updates()}')
        valid_iter= self.valid_data.next_epoch_itr()
        loginfos, nsamples=[],0
        self.valid_logger.reset()
        with torch.no_grad():
            for i,sample in enumerate(valid_iter):
                sample, ignore= self._prepare_sample(sample)
                loss, loginfo, sample_size= self.fwd_step(sample)
                if(ignore):
                    sample_size=0
                    loginfo= self.zero_log(loginfo)
                nsamples += sample_size
                
                if log_step:
                    self.valid_logger.logging(loginfo, 0, self.optimizer.get_lr(), sample_size, i)
                # self.valid_logger.logging(loginfo, 0, self.optimizer.get_lr(), nsamples, self.get_num_updates())
                loginfos.append(loginfo)
                # if i>10:
                #     break
            loginfo= sum_log(loginfos)
            if self.data_parallel_world_size >1:
                loginfo, nsamples= self.aggre_loginfo(loginfo, nsamples)
        
        if self.is_data_parallel_master:
            self.valid_logger.logging(loginfo, 0, self.optimizer.get_lr(), nsamples, self.get_num_updates())




        


    


        
