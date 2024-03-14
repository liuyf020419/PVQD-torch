import torch
import numpy as np
from .dataset import BaseDataset

class NpzDataset(BaseDataset):
    '''
    load sample from npz dump file, just for code tuning
    '''
    def __init__(self, data_list):
        super().__init__()
        self.data_list= data_list
        self.filelist= [fn.strip() for fn in open(data_list)]

    def reset_data(self):
        self.filelist= [fn.strip() for fn in open(self.data_list)]

    
    def __getitem__(self, index: int) :
        if index >= len(self.filelist):
            raise IndexError(f'bad index {index}')
        with open(self.filelist[index], 'rb') as f:
            sample= np.load(f,allow_pickle=True).item()
        
        def convert_data(x, name):
            dtype = x.dtype
            # small size for debug
            # if name.startswith('extra_'):
            #     x = x[:, :3000]
            # if name.startswith('msa_') or name in ['bert_mask', 'true_msa']:
            #     x = x[:, :512]

            if dtype == np.int32 or dtype == np.int64:
                x = x.astype(np.int64)
            elif dtype == np.float32 or dtype == np.float64:
                x = x.astype(np.float32)   
            return x
        sample= {k:torch.from_numpy(convert_data(v,k)) for k,v in sample.items()}
        return sample
    
    def __len__(self):
        return len(self.filelist)

