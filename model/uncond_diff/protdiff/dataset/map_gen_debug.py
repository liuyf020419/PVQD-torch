
import torch
import numpy as np

# def gen_fine_grained_map(config, ca_coord):
def gen_fine_grained_map(ca_coord):
    seq_len = ca_coord.shape[0]
    # p_spatial = config.p_spatial
    # min_knn = p_spatial[0] # 0.1
    # max_knn = p_spatial[1] # 0.5
    min_knn = 0.1
    max_knn = 0.5
    knn = int((torch.rand([1]) * (max_knn-min_knn) + min_knn).item() * seq_len)

    central_absidx = torch.randint(0, seq_len, [1])
    ca_map = torch.mean(ca_coord[None] - ca_coord[:, None], -1)
    central_knnid = ca_map[central_absidx.item()]
    knn_idx = torch.argsort(central_knnid)[:knn]
    mask_seq = torch.zeros(seq_len)
    mask_seq = mask_seq.scatter(0, knn_idx, torch.ones_like(knn_idx).float())
    mask_map = mask_seq[:, None] * mask_seq[None]
    masked_ca_map = mask_map * ca_map

    return mask_map, masked_ca_map

if __name__ == "__main__":
    mask_map, masked_ca_map = gen_fine_grained_map(torch.rand(156, 3))
    import pdb; pdb.set_trace()
