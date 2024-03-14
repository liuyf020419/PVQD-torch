import logging
import os
import subprocess
import torch
import argparse
import torch
import torch.distributed as dist
from collections import OrderedDict
import random
import warnings
import socket
import pickle,struct
import torch.nn as nn
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def is_master(args:argparse.Namespace):
    return args.distributed_rank == 0

def add_distributed_args(parser:argparse.ArgumentParser):
    dgroup = parser.add_argument_group('distributed setup')
    dgroup.add_argument('--distributed-world-size', default= max(1, torch.cuda.device_count()), type=int,
                        help = 'total number of GPUs across all nodes')
    dgroup.add_argument('--distributed-rank', default=0, type=int, help='rank of current worker')
    dgroup.add_argument('--distributed-init-method', default= None, 
                        help="typically tcp://hostname:port for initial connetion")
    dgroup.add_argument('--distributed-port', default= -1, 
                        help="port number")
    dgroup.add_argument('--distributed-no-spawn', default= False, action='store_true', 
                        help="do not spawn multiple processes for distributed env")
    dgroup.add_argument('--device-id', "--local_rank", default= 0, type=int, help='device id for curr process')


def infer_init_method(args:argparse.Namespace):

    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        # support torch.distributed.launch
        _infer_torch_distributed_launch_init(args)
    elif args.distributed_world_size > 1:
        _infer_single_node_init(args)


def _infer_torch_distributed_launch_init(args:argparse.Namespace):
    args.distributed_init_method = "env://"
    args.distributed_world_size = int(os.environ["WORLD_SIZE"])
    args.distributed_rank = int(os.environ["RANK"])
    # processes are created by torch.distributed.launch
    args.distributed_no_spawn = True

def _infer_single_node_init(args:argparse.Namespace):
    assert (
        args.distributed_world_size <= torch.cuda.device_count()
    ), f"world size is {args.distributed_world_size} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    args.distributed_init_method = "tcp://localhost:{port}".format(port=port)



def distributed_init(args: argparse.Namespace):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        warnings.warn(
            "Distributed is already initialized, cannot initialize twice!"
        )
    else:
        logger.info(
            "distributed init (rank {}): {}".format(
                args.distributed_rank,
                args.distributed_init_method,
            )
        )
        dist.init_process_group(
            backend='nccl', #args.distributed_backend,
            init_method=args.distributed_init_method,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank,
        )
        logger.info(
            "initialized host {} as rank {}".format(
                socket.gethostname(),
                args.distributed_rank,
            )
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    args.distributed_rank = torch.distributed.get_rank()

    if is_master(args):
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    return args.distributed_rank


def distributed_main(i, main, args: argparse.Namespace, kwargs):
    args.device_id = i
    if torch.cuda.is_available() :
        torch.cuda.set_device(args.device_id)
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = kwargs.pop("start_rank", 0) + i

    args.distributed_rank = distributed_init(args)

    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        args = after_distributed_init_fn(args)

    main(args, **kwargs)

    if torch.distributed.is_initialized():
        torch.distributed.barrier(get_global_group())


def call_main(args: argparse.Namespace, main, **kwargs):
    if args.distributed_init_method is None:
        infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(main, args, kwargs),
                nprocs=min(
                    torch.cuda.device_count(),
                    args.distributed_world_size,
                ),
                join=True,
            )
        else:
            distributed_main(args.device_id, main, args, kwargs)
    else:
        # single GPU main
        main(args, **kwargs)


def get_global_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def get_rank(group):
    return dist.get_rank(group=group)


def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1



def get_global_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_global_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def get_data_parallel_group():
    return get_global_group()


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())


def all_reduce(tensor, group, op="sum"):
    if op == "sum":
        op = dist.ReduceOp.SUM
    elif op == "max":
        op = dist.ReduceOp.MAX
    else:
        raise NotImplementedError
    dist.all_reduce(tensor, op=op, group=group)
    return tensor


def broadcast(tensor, src, group):
    dist.broadcast(tensor, src=src, group=group)


def all_to_all(tensor, group):
    """Perform an all-to-all operation on a 1D Tensor."""
    assert tensor.dim() == 1
    split_count = get_world_size(group=group)
    assert tensor.numel() % split_count == 0
    output = torch.zeros_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)
    return output


def all_gather(tensor, group, return_tensor=False):
    """Perform an all-gather operation."""
    world_size = get_world_size(group=group)
    rank = get_rank(group=group)
    tensor_list = [
        tensor if i == rank else torch.empty_like(tensor) for i in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    if return_tensor:
        return torch.stack(tensor_list, dim=0)
    else:
        return tensor_list


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    from . import utils

    if group is None:
        group = get_global_group()
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4  # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )

    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )



class FairseqDistributedDataParallel(nn.Module):
    """DDP implementation from fairseq, for multi-step communication
    """

    def __init__(self, module, process_group, buffer_size=2 ** 28):
        super().__init__()

        self.module = module
        self.process_group = process_group
        self.world_size = get_world_size(self.process_group)

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in module.parameters()))
        self.buffer = None

        # We can also forcibly accumulate grads locally and only do the
        # all-reduce at some later time
        self.accumulate_grads = False

        # make per-device lists of parameters
        paramlists = OrderedDict()
        for param in self.module.parameters():
            device = param.device
            if paramlists.get(device) is None:
                paramlists[device] = []
            paramlists[device] += [param]
        self.per_device_params = list(paramlists.values())
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # Sync params and buffers
        self._sync_params_and_buffers(authoritative_rank=0)
    
    def _sync_params_and_buffers(self, authoritative_rank=0):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._distributed_broadcast_coalesced(
                module_states,
                self.broadcast_bucket_size,
                authoritative_rank)
    
    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """

        def all_reduce_params(params):
            buffer = self.buffer
            nonzero_buffer = False
            if len(params) > 1:
                offset = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        buffer[offset : offset + sz].copy_(p.grad.data.view(-1))
                        nonzero_buffer = True
                    else:
                        buffer[offset : offset + sz].zero_()
                    offset += sz
            else:
                # we only have a single grad to all-reduce
                p = params[0]
                if p.grad is not None:
                    buffer = p.grad.data
                    nonzero_buffer = True
                elif p.numel() <= self.buffer.numel():
                    buffer = buffer[: p.numel()]
                    buffer.zero_()
                else:
                    buffer = torch.zeros_like(p)

            if nonzero_buffer:
                buffer.div_(self.world_size)

            all_reduce(buffer, self.process_group)

            # copy all-reduced grads back into their original place
            offset = 0
            for p in params:
                sz = p.numel()
                if p.grad is not None:
                    p.grad.data.copy_(buffer[offset : offset + sz].view_as(p))
                else:
                    p.grad = buffer[offset : offset + sz].view_as(p).clone()
                offset += sz

        def reduction_fn():
            # This function only needs to be called once
            if self.accumulate_grads:
                return

            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)

            for params in self.per_device_params:
                # All-reduce the gradients in buckets
                offset = 0
                buffered_params = []
                for param in params:
                    if not param.requires_grad:
                        continue
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                    if hasattr(param, 'expert'):
                        # Skip gradient sync for unshared parameters
                        continue

                    if param.grad.requires_grad:
                        raise RuntimeError(
                            "DistributedDataParallel only works "
                            "with gradients that don't require "
                            "grad"
                        )
                    sz = param.numel()
                    if sz > self.buffer.numel():
                        # all-reduce big params directly
                        all_reduce_params([param])
                    else:
                        if offset + sz > self.buffer.numel():
                            all_reduce_params(buffered_params)
                            offset = 0
                            buffered_params.clear()
                        buffered_params.append(param)
                        offset += sz

                if len(buffered_params) > 0:
                    all_reduce_params(buffered_params)

        reduction_fn()