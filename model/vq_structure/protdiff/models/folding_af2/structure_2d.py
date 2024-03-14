import torch
import torch.nn.functional as F

import numpy as np

from .common import residue_constants

MAINCHAINATOMS = ["N", "CA", "C", "O", "CB"]
MAPSNAME = ["CB_dist_map", "omega_torsion_map", "theta_torsion_map", "phi_angle_map"]
MASK_DISTS = 20

GLY_ID = residue_constants.restype_order_with_x['G']
ATOM_INDEX = {a: residue_constants.atom_types.index(a) for a in MAINCHAINATOMS}

ANGLE_BINS = [-180, 180, 36]
DIST_BINS = [0, 20, 40]


def descrete_angle(angle, edges=None):
    if edges is None:
        edges = ANGLE_BINS
    min_, max_, nbin_ = edges
    angle = (angle - min_) * nbin_ / (max_ - min_)
    angle = angle.float()
    angle = torch.clamp(angle, min=0, max=nbin_-1)
    return angle


def descrete_dist(dist, edges=None):
    if edges is None:
        edges = DIST_BINS
    min_, max_, nbin_ = edges
    dist = (dist - min_) * nbin_ / (max_ - min_)
    dist = dist.long()
    dist = torch.clamp(dist, min=0, max=nbin_-1)
    return dist

def torsion(x1, x2, x3, x4, degrees = True):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / (torch.linalg.norm(b1, dim=2, keepdims=True) + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, dim=2, keepdims=True) * b1
    w = b2 - torch.sum(b2*b1, dim=2, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*(w + 1e-4), axis=2)
    # b1xv = torch.cross(b1, v, axisa=2, axisb=2)c
    b1xv = torch.cross(b1, v, dim=2)
    y = torch.sum(b1xv*w, dim=2)

    # tan_ = y / x
    # tan_ = torch.sign(tan_).detach() * tan_.abs().clamp(max=30)

    # x = torch.sign(x) * x.clamp(min=1e-4)

    angle_ = torch.atan2(y, x)
    # angle_ = torch.atan(tan_)
    # angle_ = np.nan_to_num(angle_)
    if degrees:
        return (180.0 / torch.pi).float() * angle_
    else:
        return angle_


def angle(x1, x2, x3, degrees=True):
    """
    calc_angle of point(x1), point(x2), point(x3)
    """
    ba = x1 - x2
    ba = ba / (torch.linalg.norm(ba, dim=2, keepdims=True) + 1e-10)
    bc = x3 - x2
    bc = bc / (torch.linalg.norm(bc, dim=2, keepdims=True) + 1e-10)
    cosine_angle = torch.sum(ba*bc, dim=2)
    angle_ = torch.acos(torch.clamp(cosine_angle, min=-1, max=1))
    # angle_ =  np.nan_to_num(angle_)
    if degrees:
        return (180.0 / torch.pi).float() * angle_ #np.degrees(angle_)
    else:
        return angle_



def torsion_batch(x1, x2, x3, x4, degrees = True):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / (torch.linalg.norm(b1, dim=3, keepdims=True) + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, dim=3, keepdims=True) * b1
    w = b2 - torch.sum(b2*b1, dim=3, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*(w + 1e-4), axis=3)
    # b1xv = torch.cross(b1, v, axisa=2, axisb=2)c
    b1xv = torch.cross(b1, v, dim=3)
    y = torch.sum(b1xv*w, dim=3)

    # tan_ = y / x
    # tan_ = torch.sign(tan_).detach() * tan_.abs().clamp(max=30)

    # x = torch.sign(x) * x.clamp(min=1e-4)

    angle_ = torch.atan2(y, x)
    # angle_ = torch.atan(tan_)
    # angle_ = np.nan_to_num(angle_)
    if degrees:
        return (180.0 / torch.pi).float() * angle_
    else:
        return angle_


def angle_batch(x1, x2, x3, degrees=True):
    """
    calc_angle of point(x1), point(x2), point(x3)
    """
    # import pdb; pdb.set_trace()
    ba = x1 - x2
    ba = ba / (torch.linalg.norm(ba, dim=3, keepdims=True) + 1e-10)
    bc = x3 - x2
    bc = bc / (torch.linalg.norm(bc, dim=3, keepdims=True) + 1e-10)
    cosine_angle = torch.sum(ba*bc, dim=3)
    angle_ = torch.acos(torch.clamp(cosine_angle, min=-1, max=1))
    # angle_ =  np.nan_to_num(angle_)
    if degrees:
        return (180.0 / torch.pi).float() * angle_ #np.degrees(angle_)
    else:
        return angle_


def make_psudo_beta(atom_pos, aatype):
    is_gly = (aatype == GLY_ID).float().to(aatype.device) #.astype(np.float32)

    cb_idx = ATOM_INDEX['CB']
    ca_idx = ATOM_INDEX['CA']

    peudo_beta_atoms = atom_pos
    peudo_beta_atoms[:, cb_idx] = atom_pos[:, ca_idx] * is_gly + atom_pos[:, cb_idx] * (1.0 - is_gly)
    return peudo_beta_atoms


def get_atom(atom_pos, aatype, name):
    if name == 'CB':
        is_gly = (aatype == GLY_ID).float().to(aatype.device) #.astype(np.float32)
        cb_idx = ATOM_INDEX['CB']
        ca_idx = ATOM_INDEX['CA']
        vec_ca = atom_pos[:, ca_idx]
        vec_n = atom_pos[:, ATOM_INDEX['N']]
        vec_c = atom_pos[:, ATOM_INDEX['C']]
        b = vec_ca - vec_n
        c = vec_c - vec_ca
        a = torch.cross(b, c)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca
        CB = CB * is_gly[..., None] + atom_pos[:, cb_idx] * (1.0 - is_gly[..., None])

        return CB
    else:
        return atom_pos[:, ATOM_INDEX[name]]


def calc_beta_dist(beta_coords):
    dist = torch.sqrt(torch.sum(torch.square(beta_coords[:, None] - beta_coords[None]), -1) + 1e-4)
    return dist


def calc_beta_dist_batch(beta_coords):
    dist = torch.sqrt(torch.sum(torch.square(beta_coords[:, :, None] - beta_coords[:, None]), -1) + 1e-4)
    return dist


def extract_2d_maps(
        atom_pos, atom_mask, aatype, degrees = False,
    ):
    res_mask = torch.sum(atom_mask[:, :3], axis=-1) == 3
    res_mask = res_mask.float()

    res_len = atom_pos.shape[0]
    # atom_coords = {name: get_atom(atom_pos, aatype, name) for name in MAINCHAINATOMS}
    CA = get_atom(atom_pos, aatype, 'CA')
    CB = get_atom(atom_pos, aatype, 'CB')
    N = get_atom(atom_pos, aatype, 'N')

    mask2d = res_mask[:, None] * res_mask[None, :] * (1.0 - torch.eye(res_len).to(aatype.device))

    # cacluate CB dist map
    CB_dist_map = calc_beta_dist(CB) * mask2d
    
    # import pdb; pdb.set_trace()
    # calculate omega torsion map
    ca1 = torch.tile(CA[:, None], (1, res_len, 1))
    cb1 = torch.tile(CB[:, None], (1, res_len, 1))
    cb2 = torch.tile(CB[None], (res_len, 1, 1))
    ca2 = torch.tile(CA[None], (res_len, 1, 1))
    omega_torsion_map = torsion(ca1, cb1, cb2, ca2, degrees=degrees) * mask2d

    # calculate phi angle map
    # same x1, x2 and x3
    phi_angle_map = angle(ca1, cb1, cb2, degrees=degrees) * mask2d

    # calculate theta torsion map
    n1 = torch.tile(N[:, None], (1, res_len, 1))
    theta_torsion_map = torsion(n1, ca1, cb1, cb2, degrees=degrees) * mask2d

    return {
        "Dist": CB_dist_map.clamp(min=1e-4),
        "Omega": omega_torsion_map,
        "Theta": theta_torsion_map,
        "Phi": phi_angle_map,
    }


def extract_2d_maps_batch(
        atom_pos, atom_mask, aatype, degrees = False,
    ):
    assert len(atom_pos.shape) == 4

    B, N_res, N_a, _ = atom_pos.shape
    atom_pos = atom_pos.reshape(-1, N_a, 3)
    res_mask = torch.sum(atom_mask[:, :, :3], axis=-1) == 3
    res_mask = res_mask.float()
    aatype = aatype.repeat(B)

    # atom_coords = {name: get_atom(atom_pos, aatype, name) for name in MAINCHAINATOMS}
    CA = get_atom(atom_pos, aatype, 'CA').reshape(B, N_res, 3)
    CB = get_atom(atom_pos, aatype, 'CB').reshape(B, N_res, 3)
    N = get_atom(atom_pos, aatype, 'N').reshape(B, N_res, 3)

    mask2d = res_mask[:, :, None] * res_mask[:, None, :] * (1.0 - torch.eye(N_res).to(aatype.device))

    # cacluate CB dist map
    CB_dist_map = calc_beta_dist_batch(CB) * mask2d

    # calculate omega torsion map
    ca1 = torch.tile(CA[:, :, None], (1, 1, N_res, 1))
    cb1 = torch.tile(CB[:, :, None], (1, 1, N_res, 1))
    cb2 = torch.tile(CB[:, None], (1, N_res, 1, 1))
    ca2 = torch.tile(CA[:, None], (1, N_res, 1, 1))
    omega_torsion_map = torsion_batch(ca1, cb1, cb2, ca2, degrees=degrees) * mask2d

    # calculate phi angle map
    # same x1, x2 and x3
    phi_angle_map = angle_batch(ca1, cb1, cb2, degrees=degrees) * mask2d

    # calculate theta torsion map
    n1 = torch.tile(N[:, :, None], (1, 1, N_res, 1))
    theta_torsion_map = torsion_batch(n1, ca1, cb1, cb2, degrees=degrees) * mask2d

    return {
        "Dist": CB_dist_map.clamp(min=1e-4),
        "Omega": omega_torsion_map,
        "Theta": theta_torsion_map,
        "Phi": phi_angle_map
    }


def get_backbone_tors(atom_pos, aatype, degreee=False):
    CA = get_atom(atom_pos, aatype, 'CA')
    N = get_atom(atom_pos, aatype, 'N')
    C = get_atom(atom_pos, aatype, 'C')
    # calculate phi backbone torsion
    phi_c1 = C[:-1][None, :]
    phi_n  = N[1:][None, :]
    phi_ca = CA[1:][None, :]
    phi_c2 = C[1:][None, :]

    phi_tors_torsion_map = torsion(phi_c1, phi_n, phi_ca, phi_c2, degrees=degreee)

    # calculate psi backbone torsion
    psi_n1 = N[:-1][None, :]
    psi_ca = CA[:-1][None, :]
    psi_c  = C[:-1][None, :]
    psi_n2 = N[1:][None, :]
    psi_tors_torsion_map = torsion(psi_n1, psi_ca, psi_c, psi_n2, degrees=degreee)

    return {
        "phi_bb": F.pad(phi_tors_torsion_map, (1, 0), mode="constant", value=np.pi),
        "psi_bb": F.pad(psi_tors_torsion_map, (0, 1), mode="constant", value=np.pi)
    }


def get_backbone_tors_batch(atom_pos, aatype, degreee=False):
    assert len(atom_pos.shape) == 4
    B, N_res, N_a, _ = atom_pos.shape
    atom_pos = atom_pos.reshape(-1, N_a, 3)
    aatype = aatype.repeat(B)
    CA = get_atom(atom_pos, aatype, 'CA').reshape(B, N_res, 3)
    N = get_atom(atom_pos, aatype, 'N').reshape(B, N_res, 3)
    C = get_atom(atom_pos, aatype, 'C').reshape(B, N_res, 3)
    # calculate phi backbone torsion
    phi_c1 = C[:, :-1]
    phi_n  = N[:, 1:]
    phi_ca = CA[:, 1:]
    phi_c2 = C[:, 1:]

    phi_tors_torsion_map = torsion(phi_c1, phi_n, phi_ca, phi_c2, degrees=degreee)
    
    # calculate psi backbone torsion
    psi_n1 = N[:, :-1]
    psi_ca = CA[:, :-1]
    psi_c  = C[:, :-1]
    psi_n2 = N[:, 1:]
    psi_tors_torsion_map = torsion(psi_n1, psi_ca, psi_c, psi_n2, degrees=degreee)

    return {
        "phi_bb": F.pad(phi_tors_torsion_map, (1, 0), mode="constant", value=np.pi),
        "psi_bb": F.pad(psi_tors_torsion_map, (0, 1), mode="constant", value=np.pi)
    }


if __name__ == "__main__":

    import matplotlib.pyplot as plt
        
    def plot_geom_maps(geom_maps, name, sigsize):
        # Distance
        plt.figure(figsize=sigsize)

        plt.subplot(141)
        plt.imshow(geom_maps[0]).set_cmap("hot")
        plt.colorbar()

        plt.subplot(142)
        plt.imshow(geom_maps[1]).set_cmap("hot")
        plt.colorbar()

        plt.subplot(143)
        plt.imshow(geom_maps[2]).set_cmap("hot")
        plt.colorbar()

        plt.subplot(144)
        plt.imshow(geom_maps[3]).set_cmap("hot")
        plt.colorbar()

        plt.savefig(name, bbox_inches='tight', dpi=600, transparent=False)


    atom_pos = torch.rand(16, 10, 5, 3)
    res_mask = torch.ones(16, 10, 5)
    aatype = torch.Tensor(atom_pos.shape[1] * [7])
    bb_tors = get_backbone_tors_batch(atom_pos=atom_pos, aatype=aatype)
    print(bb_tors)
    geom_mps = extract_2d_maps_batch(atom_pos, res_mask, aatype)
    print(geom_mps)

    atom_pos = torch.rand(10, 5, 3)
    res_mask = torch.ones(10, 5)
    aatype = torch.Tensor(atom_pos.shape[0] * [7])
    extract_2d_maps(atom_pos, res_mask, aatype)

    
    # plt_gmps = geom_mps["Dist"][0], geom_mps["Omega"][0], geom_mps["Phi"][0], geom_mps["Theta"][0]
    # plot_geom_maps(plt_gmps, "batch_2dtest.png", [20, 5])






