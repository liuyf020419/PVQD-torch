import torch
import torch.nn.functional as F
import numpy as np

# from protein_utils import rigid, atom_coords

ss_dict = ['X', 'H', 'L', 'E']
ss_id2letter = {i: v for i, v in enumerate(ss_dict)}
ss_letter2id = {v: i for i, v in enumerate(ss_dict)}


def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths).cumsum(dim=1).t() > lengths).t()
    return mask.float()
    

def sstype_to_ss_segments(sstype):
    dsstype = sstype[1:] - sstype[:-1]
    ss_starts = torch.where(dsstype != 0)[0] + 1
    ss_starts = F.pad(ss_starts, [1, 0])
    ss_ends = torch.zeros_like(ss_starts) + sstype.shape[0]
    ss_ends[:-1] = ss_starts[1:]
    seg_types = sstype[ss_starts]

    ss_segs = []
    for i in range(seg_types.shape[0]):
        ss_segs.append((ss_id2letter[seg_types[i].item()], ss_starts[i].item(), ss_ends[i].item()))
    return ss_segs


def make_adj_parallel_matrix(ss_segs, ca_coords, dropout = 0.0):
    N = ca_coords.shape[0]
    # adj = torch.zeros((N, N)).float()
    ss_assamble_adj = torch.zeros((N, N)).float()

    nonloop_ss_segs = [ss for ss in ss_segs if ss[0] != 'L']
    num_nonloop_segs = len(nonloop_ss_segs)
    for i in range(num_nonloop_segs):
        for j in range(i+1, num_nonloop_segs):
            st_i, ed_i = nonloop_ss_segs[i][1], nonloop_ss_segs[i][2]
            st_j, ed_j = nonloop_ss_segs[j][1], nonloop_ss_segs[j][2]
            len1 = ed_i - st_i
            len2 = ed_j - st_j
            if len1 <= 1 or len2 <=1:
                continue

            ca_i = ca_coords[st_i: ed_i]
            ca_j = ca_coords[st_j: ed_j]
            dist = torch.sqrt(torch.sum((ca_i[:, None] - ca_j[None])**2, -1))
            min_dist = torch.min(dist)

            ss_i, ss_j = nonloop_ss_segs[i][0], nonloop_ss_segs[j][0]
            is_adj = False
            is_ss_assamble = 0.0
            if ss_i == 'E' and ss_j == 'E': # EE
                if min_dist <= 5.0:
                    is_adj = True
                    orient_i = ca_coords[ed_i - 1] - ca_coords[st_i]
                    orient_j = ca_coords[ed_j - 1] - ca_coords[st_j]
                    dot_orient = torch.sum(orient_i * orient_j)
                    is_ss_assamble = 1.0 if dot_orient >= 0.0 else 2.0 # EE_p, EE_ap
            else:
                if min_dist <= 7.0:
                    is_adj = True
                    if (ss_i == 'E' and ss_j == 'H') or (ss_i == 'H' and ss_j == 'E'):
                        is_ss_assamble = 3.0 # HE
                    else:
                        is_ss_assamble = 4.0 # HH
            if is_adj and dropout > 0.0:
                if np.random.uniform() < dropout:
                    is_adj = False

            if is_adj:
                mask_i = torch.zeros((N,))
                mask_j = torch.zeros((N,))
                mask_i[st_i:ed_i] = 1.0
                mask_j[st_j:ed_j] = 1.0
                # adj += mask_i[:, None] * mask_j[None]
                ss_assamble_adj += is_ss_assamble * mask_i[:, None] * mask_j[None]
    # adj = adj + torch.transpose(adj.clone(), 0, 1)
    ss_assamble_adj = ss_assamble_adj + torch.transpose(ss_assamble_adj.clone(), 0, 1)   
    return ss_assamble_adj.long()


def make_SS_condition(sstype, ca_coords, adj_dropout = 0.0):
    ss_segs = sstype_to_ss_segments(sstype)
    return ss_segs, make_adj_parallel_matrix(ss_segs, ca_coords, adj_dropout)


if __name__ == "__main__":
    sstype = torch.LongTensor([0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1])
    ca_coords = torch.rand(sstype.shape[0], 3)
    ss_segs = sstype_to_ss_segments(sstype, ca_coords)