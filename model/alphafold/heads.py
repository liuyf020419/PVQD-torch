import torch
from torch import nn
from torch.nn import functional as F

from .layers import *
from . import quat_affine
from . import lddt

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return loss #torch.asarray(loss)

def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = F.logsigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
    log_not_p = F.logsigmoid(-logits)
    loss = -labels * log_p - (1. - labels) * log_not_p
    return loss


def _distogram_log_loss(logits, bin_edges, batch, num_bins):
    """Log loss of a distogram."""

    assert len(logits.shape) == 3
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']

    assert positions.shape[-1] == 3

    sq_breaks = torch.square(bin_edges)

    dist2 = torch.sum(
        torch.square(
            positions.unsqueeze(-2) - positions.unsqueeze(-3)
        ),
        dim=-1,
        keepdims=True)

    true_bins = torch.sum(dist2 > sq_breaks, dim=-1)

    errors = softmax_cross_entropy(
        labels=F.one_hot(true_bins, num_bins), logits=logits)

    square_mask = mask.unsqueeze(-2) *mask.unsqueeze(-1)

    avg_error = (
        torch.sum(errors * square_mask, dim=(-2, -1)) /
        (1e-6 + torch.sum(square_mask, dim=(-2, -1))))
    dist2 = dist2[..., 0]
    return dict(loss=avg_error, true_dist=torch.sqrt(1e-6 + dist2))


class MaskedMsaHead(nn.Module):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """
    def __init__(self, config, global_config, input_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.num_output = config.num_output
        self.logits = Linear(input_channel, config.num_output, initializer=final_init(global_config))

    def forward(self, representations, batch):
        """Builds MaskedMsaHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [N_seq, N_res, c_m].
        batch: Batch, unused.
        is_training: Whether the module is in training mode.

        Returns:
        Dictionary containing:
            * 'logits': logits of shape [N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        del batch
        logits = self.logits(representations['msa'])
        return dict(logits=logits)

    def loss(self, value, batch):
        errors = softmax_cross_entropy(
            labels = F.one_hot(batch['true_msa'], num_classes=self.num_output),
            logits = value['logits']
        )
        loss = (torch.sum(errors * batch['bert_mask'], dim=(-2, -1)) /
            (1e-8 + torch.sum(batch['bert_mask'], dim=(-2, -1))))
        return {'loss': loss}


class DistogramHead(nn.Module):
    """Head to predict a distogram.

    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """
    def __init__(self, config, global_config, pair_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.half_logits = Linear(pair_channel, config.num_bins, initializer=final_init(global_config))

    def forward(self, representations, batch):
        """Builds DistogramHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'pair': pair representation, shape [N_res, N_res, c_z].
        batch: Batch, unused.
        is_training: Whether the module is in training mode.

        Returns:
        Dictionary containing:
            * logits: logits for distogram, shape [N_res, N_res, N_bins].
            * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
        """
        half_logits = self.half_logits(representations['pair'])
        logits = half_logits + torch.swapaxes(half_logits, -2, -3)
        breaks = torch.linspace(self.config.first_break, self.config.last_break,
                                self.config.num_bins - 1).to(logits.device)
        return dict(logits=logits, bin_edges=breaks)

    def loss(self, value, batch):
        return _distogram_log_loss(value['logits'], value['bin_edges'],
                                   batch, self.config.num_bins)


class PredictedLDDTHead(nn.Module):
    """Head to predict the per-residue LDDT to be used as a confidence measure.

    Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
    Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
    """
    def __init__(self, config, global_config, msa_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.input_layer_norm = nn.LayerNorm(msa_channel)
        self.act_0 = Linear(msa_channel, config.num_channels, initializer='relu')
        self.act_1 = Linear(config.num_channels, config.num_channels, initializer='relu')
        self.logits = Linear(config.num_channels, config.num_bins, initializer=final_init(global_config))


    def forward(self, representations, batch):
        act = representations['structure_module']

        act = self.input_layer_norm(act)
        act = self.act_0(act)
        act = F.relu(act)
        act = self.act_1(act)
        act = F.relu(act)

        logits = self.logits(act)
        return dict(logits=logits)

    def loss(self, value, batch):
        # Shape (num_res, 37, 3)
        pred_all_atom_pos = value['structure_module']['final_atom_positions']
        # Shape (num_res, 37, 3)
        true_all_atom_pos = batch['all_atom_positions']
        # Shape (num_res, 37)
        all_atom_mask = batch['all_atom_mask']
        # Shape (num_res,)
        lddt_ca = lddt.lddt(
            # Shape (batch_size, num_res, 3)
            predicted_points=pred_all_atom_pos[None, :, 1, :],
            # Shape (batch_size, num_res, 3)
            true_points=true_all_atom_pos[None, :, 1, :],
            # Shape (batch_size, num_res, 1)
            true_points_mask=all_atom_mask[None, :, 1:2].float(),
            cutoff=15.,
            per_residue=True)
        lddt_ca = lddt_ca.detach()

        num_bins = self.config.num_bins
        bin_index = torch.floor(lddt_ca * num_bins).long()
        # protect against out of range for lddt_ca == 1
        bin_index = bin_index.clamp(max=num_bins-1) #torch.min(bin_index, num_bins - 1)
        lddt_ca_one_hot = F.one_hot(bin_index, num_classes=num_bins)

        # Shape (num_res, num_channel)
        logits = value['predicted_lddt']['logits']
        errors = softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

        # Shape (num_res,)
        mask_ca = all_atom_mask[:, residue_constants.atom_order['CA']]
        mask_ca = mask_ca.float()
        loss = torch.sum(errors * mask_ca) / (torch.sum(mask_ca) + 1e-8)
        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            loss = loss * ((batch['resolution'] >= self.config.min_resolution)
                    & (batch['resolution'] <= self.config.max_resolution)).float().sum()

        output = {'loss': loss}
        return output


class ExperimentallyResolvedHead(nn.Module):
    """Head to predict the per-residue LDDT to be used as a confidence measure.

    Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
    Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
    """
    def __init__(self, config, global_config, input_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.logits = Linear(input_channel, 37, initializer=final_init(global_config))


    def forward(self, representations, batch):
        logits = self.logits(representations['single'])
        return dict(logits=logits)

    def loss(self, value, batch):
        logits = value['logits']
        assert len(logits.shape) == 2

        # Does the atom appear in the amino acid?
        atom_exists = batch['atom37_atom_exists']
        # Is the atom resolved in the experiment? Subset of atom_exists,
        # *except for OXT*
        all_atom_mask = batch['all_atom_mask'].float()

        xent = sigmoid_cross_entropy(labels=all_atom_mask, logits=logits)
        loss = torch.sum(xent * atom_exists) / (1e-8 + torch.sum(atom_exists))

        if self.config.filter_by_resolution:
            # NMR & distillation examples have resolution = 0.
            loss = loss * ((batch['resolution'] >= self.config.min_resolution)
                    & (batch['resolution'] <= self.config.max_resolution)).float().sum()

        output = {'loss': loss}
        return output


class PredictedAlignedErrorHead(nn.Module):
    """Head to predict the distance errors in the backbone alignment frames.

    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """
    def __init__(self, config, global_config, input_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.logits = Linear(input_channel, config.num_bins, final_init(global_config))

    def forward(self, representations, batch):
        act = representations['pair']
        logits = self.logits(act)
        breaks = torch.linspace(
            0., self.config.max_error_bin, self.config.num_bins - 1).to(act.device)
        return dict(logits=logits, breaks=breaks)

    def loss(self, value, batch):
        # Shape (num_res, 7)
        predicted_affine = quat_affine.QuatAffine.from_tensor(
            value['structure_module']['final_affines'])
        # Shape (num_res, 7)
        true_affine = quat_affine.QuatAffine.from_tensor(
            batch['backbone_affine_tensor'])
        # Shape (num_res)
        mask = batch['backbone_affine_mask']
        # Shape (num_res, num_res)
        square_mask = mask[:, None] * mask[None, :]
        num_bins = self.config.num_bins
        # (1, num_bins - 1)
        breaks = value['predicted_aligned_error']['breaks']
        # (1, num_bins)
        logits = value['predicted_aligned_error']['logits']

         # Compute the squared error for each alignment.
        def _local_frame_points(affine):
            points = [x.unsqueeze(-2) for x in affine.translation]
            return affine.invert_point(points, extra_dims=1)

        error_dist2_xyz = [
            torch.square(a - b)
            for a, b in zip(_local_frame_points(predicted_affine),
                            _local_frame_points(true_affine))]
        error_dist2 = sum(error_dist2_xyz)
        # Shape (num_res, num_res)
        # First num_res are alignment frames, second num_res are the residues.
        error_dist2 = error_dist2.detach()

        sq_breaks = torch.square(breaks)
        true_bins = torch.sum((
            error_dist2[..., None] > sq_breaks).long(), axis=-1)

        errors = softmax_cross_entropy(
            labels=F.one_hot(true_bins, num_bins), logits=logits)

        loss = (torch.sum(errors * square_mask, dim=(-2, -1)) /
                (1e-8 + torch.sum(square_mask, dim=(-2, -1))))

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            loss = loss * ((batch['resolution'] >= self.config.min_resolution)
                    & (batch['resolution'] <= self.config.max_resolution)).float().sum()

        output = {'loss': loss}
        return output
