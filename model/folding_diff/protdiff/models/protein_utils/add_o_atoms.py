import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_nerf(a, b, c, l, theta, chi):
    assert -np.pi <= theta <= np.pi, "theta must be in radians and in [-pi, pi]. theta = " + str(theta)
    assert len(a.shape) == 3
    assert len(b.shape) == 3
    assert len(c.shape) == 3
    assert len(chi.shape) == 2

    batch_size, res_num = a.shape[:2]
    device = a.device
    W_hat = torch.nn.functional.normalize(b - a, dim=-1)
    x_hat = torch.nn.functional.normalize(c-b, dim=-1)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=-1)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=-1) # B, L, 3, 3
    # import pdb; pdb.set_trace()
    # calculate coord pre rotation matrix

    D = torch.stack([
        -l * torch.cos(theta) * torch.ones(batch_size, res_num).to(device), 
        l * torch.sin(theta) * torch.cos(chi), 
        l * torch.sin(theta) * torch.sin(chi)], -1) # B, L, 3
    # d = torch.stack([torch.squeeze(-l * torch.cos(theta)),
    #                  torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
    #                  torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])

    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    # d = d.unsqueeze(1).to(torch.float32) 3, 1
    D = D.unsqueeze(-1) # B, L, 3, 1
    res = c + (M @ D)[..., 0] # B, L, 3
    return res


def res_nerf(a, b, c, l, theta, chi):
    # calculate unit vectors AB and BC
    assert -np.pi <= theta <= np.pi, "theta must be in radians and in [-pi, pi]. theta = " + str(theta)
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert len(c.shape) == 2
    assert len(chi.shape) == 1

    res_num = a.shape[0]
    device = a.device
    W_hat = torch.nn.functional.normalize(b - a, dim=-1)
    x_hat = torch.nn.functional.normalize(c-b, dim=-1)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=-1)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=-1) # L, 3, 3
    # import pdb; pdb.set_trace()
    # calculate coord pre rotation matrix

    D = torch.stack([
        -l * torch.cos(theta) * torch.ones(res_num).to(device), 
        l * torch.sin(theta) * torch.cos(chi), 
        l * torch.sin(theta) * torch.sin(chi)], -1) # L, 3
    # d = torch.stack([torch.squeeze(-l * torch.cos(theta)),
    #                  torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
    #                  torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])

    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    # d = d.unsqueeze(1).to(torch.float32) 3, 1
    D = D.unsqueeze(-1) # L, 3, 1
    res = c + (M @ D)[..., 0] # L, 3
    return res


def nerf(a, b, c, l, theta, chi):
    """F
    Natural extension reference frame method for placing the 4th atom given
    atoms 1-3 and the relevant angle inforamation. This code was originally
    written by Rohit Bhattacharya (rohit.bhattachar@gmail.com,
    https://github.com/rbhatta8/protein-design/blob/master/nerf.py) and I
    have extended it to work with PyTorch. His original documentation is
    below:
    Nerf method of finding 4th coord (d) in cartesian space
        Params:
            a, b, c : coords of 3 points
            l : bond length between c and d
            theta : bond angle between b, c, d (in degrees)
            chi : dihedral using a, b, c, d (in degrees)
        Returns:
            d: tuple of (x, y, z) in cartesian space
    """
    # calculate unit vectors AB and BC
    assert -np.pi <= theta <= np.pi, "theta must be in radians and in [-pi, pi]. theta = " + str(theta)

    W_hat = torch.nn.functional.normalize(b - a, dim=0)
    x_hat = torch.nn.functional.normalize(c-b, dim=0)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=0)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=1)
    # import pdb; pdb.set_trace()
    # calculate coord pre rotation matrix
    D = torch.stack([torch.squeeze(-l * torch.cos(theta)),
                     torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
                     torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])
    # import pdb; pdb.set_trace()
    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    D = D.unsqueeze(1).to(torch.float32)
    res = c + torch.mm(M, D).squeeze()
    return res.squeeze()



def torsion_ch(x1, x2, x3, x4, degrees=False, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1) + 1e-10
    b1 = x3 - x2 + 1e-10
    b2 = x4 - x3 + 1e-10
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / (torch.norm(b1, dim=axis, keepdim=True) + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, dim=axis, keepdim=True) * b1
    w = b2 - torch.sum(b2*b1, dim=axis, keepdim=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*w, dim=axis)
    b1xv = torch.cross(b1, v)
    y = torch.sum(b1xv*w, dim=axis)
    # import pdb; pdb.set_trace()
    if degrees:
        return np.float32(180.0 / np.pi) * torch.atan2(y, x)
    else:
        return torch.atan2(y, x)



def torsion_v0(x1, x2=None, x3=None, x4=None, degrees = False, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    if (x2 is None) or (x3 is None) or (x4 is None):
        x1, x2, x3, x4 = x1
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, axis=axis, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.sum(b0*b1, axis=axis, keepdims=True) * b1
    w = b2 - np.sum(b2*b1, axis=axis, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x

    x = np.sum(v*w, axis=axis)
    b1xv = np.cross(b1, v, axisa=axis, axisb=axis)
    y = np.sum(b1xv*w, axis=axis)
    if degrees:
        return np.float32(180.0 / np.pi) * np.arctan2(y, x)
    else:
        return np.arctan2(y, x)


def add_atom_O(coord3):
    CO_bond = 1.229
    CACO_angle = torch.tensor([2.0944]).float()
    assert isinstance(coord3, np.ndarray)
    assert len(coord3.shape) == 3
    seqlen, _, _ = coord3.shape

    def calc_psi_tors(coord3):
        assert len(coord3.shape) == 3
        N_atoms = coord3[:, 0]
        CA_atoms = coord3[:, 1]
        C_atoms = coord3[:, 2]

        n1_atoms = N_atoms[:-1]
        ca_atoms = CA_atoms[:-1]
        c_atoms = C_atoms[:-1]
        n2_atoms = N_atoms[1:]

        psi_tors = torsion_v0(n1_atoms, ca_atoms, c_atoms, n2_atoms, axis=1)
        return np.concatenate([psi_tors, [0]])

    psi_tors = torch.from_numpy(calc_psi_tors(coord3)).float()
    coord3 = torch.from_numpy(coord3).float()
    # import pdb; pdb.set_trace()
    atomO_coord = [nerf(atom3[-3], atom3[-2], atom3[-1], CO_bond, CACO_angle, psi_tors[resabsD]-np.pi) \
                                    for resabsD, atom3 in enumerate(coord3)]
    new_coord = torch.cat([coord3.reshape(seqlen, -1), torch.stack(atomO_coord)], 1).reshape(seqlen, 4, 3)

    return new_coord.numpy()


def add_atom_O_ch(coord3):
    device = coord3.device
    CO_bond = 1.229
    CACO_angle = torch.tensor([2.0944]).float().to(device)
    assert len(coord3.shape) == 3
    seqlen, _, _ = coord3.shape

    def calc_psi_tors(coord3):
        assert len(coord3.shape) == 3
        N_atoms = coord3[:, 0]
        CA_atoms = coord3[:, 1]
        C_atoms = coord3[:, 2]

        n1_atoms = N_atoms[:-1]
        ca_atoms = CA_atoms[:-1]
        c_atoms = C_atoms[:-1]
        n2_atoms = N_atoms[1:]

        psi_tors = torsion_ch(n1_atoms, ca_atoms, c_atoms, n2_atoms, axis=1)
        return F.pad(psi_tors, (0,1), 'constant', 0)

    psi_tors = calc_psi_tors(coord3)
    import pdb; pdb.set_trace()
    # atomO_coord = [nerf(atom3[-3], atom3[-2], atom3[-1], CO_bond, CACO_angle, psi_tors[resabsD]-np.pi) \
    #                                 for resabsD, atom3 in enumerate(coord3)]
    res_atomO_coord = res_nerf(coord3[:, -3], coord3[:, -2], coord3[:, -1], CO_bond, CACO_angle, psi_tors-np.pi)
    new_coord = torch.cat([coord3.reshape(seqlen, -1), res_atomO_coord], 1).reshape(seqlen, 4, 3).to(device)

    return new_coord


def batch_add_atom_O(batch_coords):
    device = batch_coords.device
    assert len(batch_coords.shape) == 4
    return torch.stack([add_atom_O_ch(coords) for coords in batch_coords]).to(device)


def batch_add_atom_O_new(batch_coords): # B, L, 3, 3
    device = batch_coords.device
    assert len(batch_coords.shape) == 4
    CO_bond = 1.229
    CACO_angle = torch.tensor([2.0944]).float().to(device)

    def calc_psi_tors(coord3):
        assert len(coord3.shape) == 4
        N_atoms = coord3[:, :, 0]
        CA_atoms = coord3[:, :, 1]
        C_atoms = coord3[:, :, 2]

        n1_atoms = N_atoms[:, :-1]
        ca_atoms = CA_atoms[:, :-1]
        c_atoms = C_atoms[:, :-1]
        n2_atoms = N_atoms[:, 1:]

        psi_tors = torsion_ch(n1_atoms, ca_atoms, c_atoms, n2_atoms, axis=2)
        return F.pad(psi_tors, (0,1), 'constant', 0)

    psi_tors = calc_psi_tors(batch_coords)
    batch_atomO_coord = batch_nerf(batch_coords[:, :, -3], batch_coords[:, :, -2], batch_coords[:, :, -1], CO_bond, CACO_angle, psi_tors-np.pi)
    new_coord = torch.cat([batch_coords, batch_atomO_coord[:, :, None]], 2).to(device)
    
    return new_coord



def rebiuld_from_atom_crd(crd_list, chain="A", filename='testloop.pdb', natom=4, natom_dict=None):
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB import PDBIO
    from Bio.PDB.Atom import Atom
    if natom_dict is None:
        natom_dict = {3: {0:'N', 1:'CA', 2: 'C'},
                      4: {0:'N', 1:'CA', 2: 'C', 3:'O'}}
    natom_num = natom_dict[natom]
    sb = StructureBuilder()
    sb.init_structure("pdb")
    sb.init_seg(" ")
    sb.init_model(0)
    chain_id = chain
    sb.init_chain(chain_id)
    for num, line in enumerate(crd_list):
        name = natom_num[num % natom]

        line = np.around(np.array(line, dtype='float'), decimals=3)
        res_num = num // natom
        # print(num//4,line)
        atom = Atom(name=name, coord=line, element=name[0:1], bfactor=1, occupancy=1, fullname=name,
                    serial_number=num,
                    altloc=' ')
        sb.init_residue("GLY", " ", res_num, " ")  # Dummy residue
        sb.structure[0][chain_id].child_list[res_num].add(atom.copy())

    structure = sb.structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)


if __name__ == "__main__":
    import os
    import sys
    from write_pdb import write_multichain_from_atoms

    sys.path.append('/train14/superbrain/yfliu25/structure_refine/monomer_joint_PriorDDPM_ESM1b_Dnet_LE_MPNN_LC_trans_newmask_20221123/pdb_utils/data_parser')
    from protein_coord_parser import PoteinCoordsParser
    pdbparser = PoteinCoordsParser(poteinfile = "/train14/superbrain/lhchen/data/PDB/20220102/mmcif/ub/1ubq.cif")

    crd = pdbparser.chain_main_crd_array.reshape(-1, 5, 3)
    np_coord = add_atom_O(crd[:, :3])
    ch_coord = batch_add_atom_O(torch.from_numpy(crd[:, :3]).float()[None].detach().requires_grad_(True))
    # write_multichain_from_atoms([ch_coord.reshape(-1, 3).numpy()], 'ch_add_o.pdb')

    import pdb; pdb.set_trace()
