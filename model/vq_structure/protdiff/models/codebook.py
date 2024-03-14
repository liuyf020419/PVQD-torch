import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


class Codebook1(nn.Module):
    def __init__(self, args, input_dim, out_dim=None):
        super(Codebook1, self).__init__()
        self.input_dim = input_dim
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.l2norm_h = args.l2norm_h
        if out_dim is None:
            out_dim = self.input_dim
        self.beta = args.beta

        # self.layer_norm = nn.LayerNorm(input_dim)
        self.quant_act = nn.Linear(input_dim, self.latent_dim)

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.post_quant = nn.Linear(self.latent_dim, out_dim)


    def forward(self, z):
        batchsize, res_num = z.shape[:2]
        device = z.device

        # z = self.layer_norm(z)
        z = self.quant_act(z)

        z_flattened = z.view(-1, self.latent_dim) # BxHxW, C
        z_q_emb = self.embedding.weight.detach()
        if self.l2norm_h:
            z_flattened = l2norm(z_flattened)
            z_q_emb = l2norm(z_q_emb)
        # import pdb; pdb.set_trace()
        # d = (z - w)^2
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(z_q_emb**2, dim=1) - 2*( torch.matmul(z_flattened, z_q_emb.t()) )
        # min_idx = argmin(d)
        min_encoding_indices = torch.argmin(d, dim=1) # BxHxW
        z_q = self.embedding(min_encoding_indices).view(z.shape) # B, H, W, C
        # compute loss for embedding
        encoder_qloss = torch.mean((z_q.detach() - z)**2, -1)
        code_qloss = torch.mean((z_q - z.detach())**2, -1)
        loss = encoder_qloss + code_qloss * self.beta
        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = self.post_quant(z_q)

        loss_dict = {}
        min_encoding_indices = [min_encoding_indices]

        return z_q, min_encoding_indices, loss, loss_dict



class Codebook2(nn.Module):
    def __init__(self, args, input_dim, out_dim=None):
        super(Codebook2, self).__init__()
        self.input_dim = input_dim
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        if out_dim is None:
            out_dim = self.input_dim
        self.beta = args.beta

        self.quant_act = nn.Linear(input_dim, self.latent_dim * 2)

        self.embedding_1 = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding_1.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.embedding_2 = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding_2.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.post_quant = nn.Linear(self.latent_dim * 2, out_dim)


    def forward(self, z):
        z = self.quant_act(z)
        batchsize, res_num = z.shape[:2]
        device = z.device
        z_1, z_2 = torch.split(z, self.latent_dim, -1)
        # z = z.permute(0, 2, 3, 1).contiguous() # B, C, H, W -> B, H, W, C
        z_flattened_1 = z_1.view(-1, self.latent_dim) # BxHxW, C
        z_flattened_2 = z_2.view(-1, self.latent_dim) # BxHxW, C

        # d = (z - w)^2
        d_1 = torch.sum(z_flattened_1**2, dim=1, keepdim=True) + torch.sum(self.embedding_1.weight**2, dim=1) - \
            2*( torch.matmul(z_flattened_1, self.embedding_1.weight.t()) )
        # min_idx = argmin(d)
        min_encoding_indices_1 = torch.argmin(d_1, dim=1) # BxHxW
        z_q_1 = self.embedding_1(min_encoding_indices_1).view(z_1.shape) # B, H, W, C
        # compute loss for embedding
        encoder_qloss_1 = torch.mean((z_q_1.detach() - z_1)**2, -1)
        code_qloss_1 = torch.mean((z_q_1 - z_1.detach())**2, -1)
        loss_1 = encoder_qloss_1 + self.beta * code_qloss_1
        # preserve gradients
        z_q_1 = z_1 + (z_q_1 - z_1).detach()

        # d = (z - w)^2
        d_2 = torch.sum(z_flattened_2**2, dim=1, keepdim=True) + torch.sum(self.embedding_2.weight**2, dim=1) - \
            2*( torch.matmul(z_flattened_2, self.embedding_2.weight.t()) )
        # min_idx = argmin(d)
        min_encoding_indices_2 = torch.argmin(d_2, dim=1) # BxHxW
        z_q_2 = self.embedding_2(min_encoding_indices_2).view(z_2.shape) # B, H, W, C
        # compute loss for embedding
        # B, L, C
        encoder_qloss_2 = torch.mean((z_q_2.detach() - z_2)**2, -1) 
        code_qloss_2 = torch.mean((z_q_2 - z_2.detach())**2, -1)
        loss_2 = encoder_qloss_2 + self.beta * code_qloss_2
        # preserve gradients
        z_q_2 = z_2 + (z_q_2 - z_2).detach()


        min_encoding_indices = (min_encoding_indices_1, min_encoding_indices_2)
        loss = 0.5 * (loss_1 + loss_2)
        z_q = torch.cat([z_q_1, z_q_2], -1)
        
        z_q = self.post_quant(z_q)

        loss_dict = {
            'q_loss_1': loss_1.detach(),
            'q_loss_2': loss_2.detach()
        }

        return z_q, min_encoding_indices, loss, loss_dict



class Codebook(nn.Module):
    def __init__(self, args, input_dim, out_dim=None, head_num=1, seperate_codebook_per_head=True):
        super(Codebook, self).__init__()
        self.input_dim = input_dim
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.l2norm_h = args.l2norm_h
        self.head_num= head_num
        self.seperate_codebook_per_head = seperate_codebook_per_head

        if out_dim is None:
            out_dim = self.input_dim
        self.beta = args.beta

        self.quant_act = nn.Linear(input_dim, self.latent_dim * head_num)

        if seperate_codebook_per_head:
            self.codebook_layer = []
            for _ in range(head_num):
                self.codebook_layer.append(
                    nn.Embedding(self.num_codebook_vectors, self.latent_dim)
                )
        else:
            self.codebook_layer = [nn.Embedding(self.num_codebook_vectors, self.latent_dim)]
        self.codebook_layer  = nn.ModuleList(self.codebook_layer)

        self.post_quant = nn.Linear(self.latent_dim * head_num, out_dim)

        self.init_codebooks()

    
    def init_codebooks(self):
        for codebook_l in self.codebook_layer:
            codebook_l.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)


    def compute_each_codebook(self, codebook_l, z):
        z_flattened = z.view(-1, self.latent_dim) # BxHxW, C
        z_q_emb = codebook_l.weight.detach()
        if self.l2norm_h:
            z_flattened = l2norm(z_flattened)
            z_q_emb = l2norm(z_q_emb)

        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(z_q_emb**2, dim=1) - 2*( torch.matmul(z_flattened, z_q_emb.t()) )

        # min_idx = argmin(d)
        min_encoding_indices = torch.argmin(d, dim=1) # BxHxW
        z_q = codebook_l(min_encoding_indices).view(z.shape) # B, H, W, C
        # compute loss for embedding
        encoder_qloss = torch.mean((z_q.detach() - z)**2, -1)
        code_qloss = torch.mean((z_q - z.detach())**2, -1)
        loss = encoder_qloss + code_qloss * self.beta
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, min_encoding_indices, loss


    def forward(self, z, return_all_indices=False):
        batchsize, res_num = z.shape[:2]
        device = z.device
        dtype = z.dtype

        z = self.quant_act(z.float())
        z_list= torch.split(z, self.latent_dim, -1)
        z_q_list = []
        min_encoding_indices_list = []
        loss = 0
        loss_dict = {}
        for h_idx in range(self.head_num):
            if self.seperate_codebook_per_head:
                z_q_l, min_encoding_indices_l, loss_l = self.compute_each_codebook(self.codebook_layer[h_idx], z_list[h_idx])
            else:
                z_q_l, min_encoding_indices_l, loss_l = self.compute_each_codebook(self.codebook_layer[0], z_list[h_idx])
            z_q_list.append(z_q_l)
            min_encoding_indices_list.append(min_encoding_indices_l)
            loss = loss + loss_l
            if (self.head_num == 2):
                loss_dict.update({f'q_loss_{h_idx+1}': loss_l.detach()})

        loss = loss * 1/len(self.codebook_layer)

        z_q = torch.cat(z_q_list, -1)
        z_q = self.post_quant(z_q)

        if return_all_indices:
            return z_q.to(dtype), min_encoding_indices_list, loss.to(dtype), loss_dict
        else:
            if (self.head_num == 2):
                return z_q.to(dtype), min_encoding_indices_list, loss.to(dtype), loss_dict
            else:
                return z_q.to(dtype), [min_encoding_indices_list[0]], loss.to(dtype), loss_dict




class ResidualCodebook(nn.Module):
    def __init__(self, args, input_dim, out_dim=None, codebook_num=1, shared_codebook=True, codebook_dropout=False):
        super(ResidualCodebook, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.l2norm_h = args.l2norm_h
        self.codebook_num = self.head_num = codebook_num
        self.shared_codebook = shared_codebook
        self.codebook_dropout = codebook_dropout

        if out_dim is None:
            out_dim = self.input_dim
        self.beta = args.beta

        self.quant_act = nn.Linear(input_dim, self.latent_dim)

        # import pdb; pdb.set_trace()
        if shared_codebook:
            self.codebook_layer = []
            for _ in range(codebook_num):
                self.codebook_layer.append(
                    nn.Embedding(self.num_codebook_vectors, self.latent_dim)
                )
        else:
            self.codebook_layer = [nn.Embedding(self.num_codebook_vectors, self.latent_dim)]
        self.codebook_layer  = nn.ModuleList(self.codebook_layer)

        self.post_quant = nn.Linear(self.latent_dim, out_dim)

        self.init_codebooks()

    
    def init_codebooks(self):
        for codebook_l in self.codebook_layer:
            codebook_l.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)


    def compute_each_codebook(self, codebook_l, z):
        z_flattened = z.reshape(-1, self.latent_dim) # BxHxW, C
        z_q_emb = codebook_l.weight.detach()
        if self.l2norm_h:
            z_flattened = l2norm(z_flattened)
            z_q_emb = l2norm(z_q_emb)

        z_flattened = z.reshape(-1, self.latent_dim)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(z_q_emb**2, dim=1) - 2*( torch.matmul(z_flattened, z_q_emb.t()) )

        # min_idx = argmin(d)
        min_encoding_indices = torch.argmin(d, dim=1) # BxHxW
        z_q = codebook_l(min_encoding_indices).reshape(z.shape) # B, H, W, C
        # compute loss for embedding
        encoder_qloss = torch.mean((z_q.detach() - z)**2, -1)
        code_qloss = torch.mean((z_q - z.detach())**2, -1)
        loss = encoder_qloss + code_qloss * self.beta
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, min_encoding_indices, loss


    def forward(self, z, return_all_indices=False, use_codebook_num=4):
        batchsize, res_num = z.shape[:2]
        device = z.device
        dtype = z.dtype

        z = self.quant_act(z.float())
        min_encoding_indices_list = []
        loss = 0
        loss_dict = {}
        z_q_out = 0
        z_q_list = []
        residual = z

        if (self.codebook_dropout and self.training):
            if self.args.only_codebook_num is not None:
                curr_codebook_num = self.args.only_codebook_num
            else:
                codebook_dropout_from = self.args.codebook_dropout_from
                curr_codebook_num = np.random.randint(codebook_dropout_from, self.codebook_num+1, 1)[0]
        else:
            curr_codebook_num = min(use_codebook_num, self.codebook_num)

        for h_idx in range(curr_codebook_num):
            if self.shared_codebook:
                z_q_l, min_encoding_indices_l, loss_l = self.compute_each_codebook(
                    self.codebook_layer[h_idx], residual)
            else:
                z_q_l, min_encoding_indices_l, loss_l = self.compute_each_codebook(
                    self.codebook_layer[0], residual)
            
            z_q_out = z_q_out + z_q_l
            residual = residual - z_q_l
            loss = loss + loss_l
            z_q_list.append(z_q_l[:, None])

            min_encoding_indices_list.append(min_encoding_indices_l)

        loss = loss * 1/len(self.codebook_layer)

        z_q = self.post_quant(z_q_out)

        if return_all_indices:
            return z_q.to(dtype), min_encoding_indices_list, loss.to(dtype), loss_dict, torch.cat(z_q_list, 1), z_q_out
        else:
            return z_q.to(dtype), [min_encoding_indices_list[0]], loss.to(dtype), loss_dict, torch.cat(z_q_list, 1), z_q_out


    def get_feature_from_indices(self, indices):
        assert len(indices.shape) == 3
        # assert self.codebook_num == indices.shape[0] # Nc, B, Nr
        codbk_num = indices.shape[0]
        z_q_out = 0

        for h_idx in range(codbk_num):
            if self.shared_codebook:
                z_q_l = self.codebook_layer[h_idx](indices[h_idx])
            else:
                z_q_l = self.codebook_layer[0](indices[h_idx])
            
            z_q_out = z_q_out + z_q_l

        z_q = self.post_quant(z_q_out)
        return z_q