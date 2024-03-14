import torch

from alphafold.common import residue_constants

res_id = {
    "X": -1,
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

id_to_letter = {v: k for k, v in res_id.items()}



def convert_to_af_aatype(aatype):
    new_aatype = aatype.view(-1)
    for i in range(new_aatype.shape[0]):
        new_aatype[i] = residue_constants.restype_order_with_x[id_to_letter[new_aatype[i].item()]]
    new_aatype = torch.reshape(new_aatype, aatype.shape)
    return new_aatype
    
