# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of JAX utility functions for use in protein folding."""

import torch


def final_init(config):
  if config.zero_init:
    return 'zeros'
  else:
    return 'linear'




def mask_mean(mask, value, dims=None, eps=1e-10):
  if dims is None:
    dims = list(range(len(value.shape)))

  broadcast_factor = 1.
  for axis_ in dims:
    value_size = value.size(axis_)
    mask_size = mask.size(axis_)
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      assert mask_size == value_size
  return torch.sum( mask *value, dim=dims) / (torch.sum(mask, dim=dims) * broadcast_factor + eps)


def moveaxis(data, source, destination):
  n_dims = len(data.shape)
  dims = [i for i in range(n_dims)]
  if source < 0:
    source += n_dims
  if destination < 0:
    destination += n_dims

  if source < destination:
    dims.pop(source)
    dims.insert(destination, source)
  else:
    dims.pop(source)
    dims.insert(destination, source)

  return data.permute(*dims)


  
