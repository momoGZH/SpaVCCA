#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import dgl


def kl_div(mu, var):
    return kl_divergence(Normal(mu, var.sqrt()),
                         Normal(torch.zeros_like(mu), torch.ones_like(var))).sum(dim=1).mean()


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def _median_heuristic_sigma(z, subsample=1000):
    """Estimate RBF sigma using median heuristic.
    Uses torch.pdist to get condensed pairwise distances (upper triangle),
    which is memory-efficient compared to forming full (N,N) distance matrix.
    Returns scalar float sigma (bandwidth).
    """
    N = z.size(0)
    if N == 0:
        return 1.0
    if N > subsample:
        idx = torch.randperm(N, device=z.device)[:subsample]
        z_s = z[idx]
    else:
        z_s = z

    # pdist returns pairwise distances (condensed upper triangle)
    if z_s.size(0) < 2:
        # not enough points to compute distances
        return 1.0

    d = torch.pdist(z_s, p=2)  # distances, not squared
    if d.numel() == 0:
        return 1.0

    median_dist = float(d.median().item())
    # avoid degenerate tiny sigma
    if median_dist < 1e-12:
        return 1.0
    return median_dist


def rbf_kernel(x, y=None, sigma=None):
    """Compute RBF kernel matrix between x and y: K_ij = exp(-||x_i-y_j||^2 / (2*sigma^2))
    - If y is None, computes between x and itself.
    - sigma can be python float or a tensor; it will be converted to tensor matching x's dtype/device.
    """
    if y is None:
        y = x
    # ensure contiguous for efficient matmul
    x = x.contiguous()
    y = y.contiguous()

    if sigma is None:
        # estimate from concatenation (subsample inside)
        xy = torch.cat([x, y], dim=0)
        sigma = _median_heuristic_sigma(xy)

    # convert sigma to tensor on same device/dtype for safe math
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype)

    # squared distances via efficient matmul formula
    x2 = (x ** 2).sum(dim=1, keepdim=True)   # (n_x, 1)
    y2 = (y ** 2).sum(dim=1, keepdim=True)   # (n_y, 1)
    dist2 = x2 + y2.t() - 2.0 * (x @ y.t())  # (n_x, n_y)
    dist2 = torch.clamp(dist2, min=0.0)

    denom = 2.0 * (sigma ** 2) + 1e-8  # small eps for safety
    K = torch.exp(-dist2 / denom)
    return K


def mmd_between_sets(x, y, sigma=None):
    """Compute biased MMD^2 between x and y using RBF kernel (biased estimator = means).
    Returns a scalar tensor (non-negative, clamped).
    """
    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)
    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return torch.clamp(mmd2, min=0.0)


def mmd_loss_to_reference(z, batch_idx, reference_batch=None, sigma=None,
                          subsample_per_batch=500, mode="to_reference"):
    """
    Compute average MMD^2 between a chosen reference batch and every other batch in the minibatch.
    - z: (N, D) tensor
    - batch_idx: (N,) integer tensor labeling batches
    - reference_batch: None or int (if None, pick first unique)
    - subsample_per_batch: cap per-batch samples to reduce cost
    Returns: scalar tensor average MMD^2
    """
    device = z.device
    dtype = z.dtype

    unique_batches = torch.unique(batch_idx)
    B = unique_batches.numel()
    if B <= 1:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # choose reference
    if reference_batch is None:
        ref = int(unique_batches[0].item())
    else:
        ref = int(reference_batch)

    # prepare per-batch sampling function
    def subsample_tensor(x, cap):
        n = x.size(0)
        if n > cap:
            idx = torch.randperm(n, device=x.device)[:cap]
            return x[idx]
        return x

    # get reference samples (and fallback if empty)
    ref_mask = (batch_idx == ref)
    z_ref_all = z[ref_mask]
    if z_ref_all.size(0) == 0:
        # fallback to first unique if specified reference empty
        ref = int(unique_batches[0].item())
        ref_mask = (batch_idx == ref)
        z_ref_all = z[ref_mask]

    z_ref = subsample_tensor(z_ref_all, subsample_per_batch)

    # compute sigma from subsampled points if not provided (cheaper than full)
    if sigma is None:
        # combine small subsample of each batch for robust estimate
        # here we sample up to 2*subsample_per_batch points total (ref + a random other set)
        sample_pool = [z_ref]
        # try adding one other batch (if available) to avoid single-batch estimation bias
        for bi in unique_batches:
            bi = int(bi.item())
            if bi == ref:
                continue
            xi = z[batch_idx == bi]
            if xi.numel() > 0:
                sample_pool.append(subsample_tensor(xi, subsample_per_batch))
                break
        sigma = _median_heuristic_sigma(torch.cat(sample_pool, dim=0))

    loss_sum = z_ref.new_tensor(0.0)
    count = 0

    # iterate other batches
    for bi in unique_batches:
        bi = int(bi.item())
        if bi == ref:
            continue
        xi = z[batch_idx == bi]
        if xi.size(0) == 0:
            continue
        xi = subsample_tensor(xi, subsample_per_batch)
        mmd2 = mmd_between_sets(z_ref, xi, sigma=sigma)
        loss_sum = loss_sum + mmd2  # keep as tensor for grad
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return (loss_sum / float(count)).to(device=device, dtype=dtype)


def graph_rl(block, z, device):
    # 1. positive
    # ======================
    pos_src, pos_dst = block.edges()
    pos_src = pos_src.to(device)
    pos_dst = pos_dst.to(device)
    # remove self-loop
    mask = pos_src != pos_dst
    pos_src = pos_src[mask]
    pos_dst = pos_dst[mask]
    if pos_src.numel() == 0:
        struct_loss = torch.tensor(0.0, device=device)
        return struct_loss
    else:
        k = len(pos_src)
        neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(block, k)
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)
        # 3. loss
        pos_logits = torch.sum(z[pos_src] * z[pos_dst], dim=1)
        neg_logits = torch.sum(z[neg_src] * z[neg_dst], dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        struct_loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return struct_loss


def cov_offdiag_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    z: [B, D]
    return: scalar
    """
    B, D = z.shape
    zc = z - z.mean(dim=0, keepdim=True)                 # center
    cov = (zc.T @ zc) / (B - 1 + eps)                    # [D, D]
    offdiag = cov - torch.diag(torch.diag(cov))
    return (offdiag ** 2).sum() / D                      # /D 让尺度更稳定


def var_floor_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Encourage per-dimension std >= gamma
    """
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)  # [D]
    return torch.relu(gamma - std).mean()


def barlow_twins_loss_single_view(z: torch.Tensor, lambd_offdiag: float = 0.005, eps: float = 1e-4) -> torch.Tensor:
    """
    Single-view variant: encourage correlation matrix ~ I
    """
    B, D = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)
    zn = zc / std                                        # z-score

    corr = (zn.T @ zn) / (B - 1 + eps)                   # [D, D], approx correlation
    on_diag = (torch.diag(corr) - 1).pow(2).sum()
    off_diag = (corr - torch.diag(torch.diag(corr))).pow(2).sum()
    return (on_diag + lambd_offdiag * off_diag) / D


import torch


def _subsample_tensor(x, cap):
    """Randomly subsample rows if needed."""
    n = x.size(0)
    if n > cap:
        idx = torch.randperm(n, device=x.device)[:cap]
        return x[idx]
    return x


def _match_sample_count(x, y):
    """
    Make two tensors have the same number of samples by random subsampling
    the larger one. Returns x_matched, y_matched.
    """
    nx = x.size(0)
    ny = y.size(0)

    if nx == 0 or ny == 0:
        return x[:0], y[:0]

    n = min(nx, ny)

    if nx > n:
        idx = torch.randperm(nx, device=x.device)[:n]
        x = x[idx]
    if ny > n:
        idx = torch.randperm(ny, device=y.device)[:n]
        y = y[idx]

    return x, y


def _center(x):
    """Center features column-wise."""
    return x - x.mean(dim=0, keepdim=True)


def _cca_correlation_loss(
    x,
    y,
    r1=1e-4,
    r2=1e-4,
    eps=1e-8,
    use_all_singular_values=True,
    outdim_size=None,
):
    """
    Deep CCA-style correlation loss between two views x and y.

    Args:
        x: (n, d1)
        y: (n, d2)
        r1, r2: regularization for covariance matrices
        eps: numerical stability
        use_all_singular_values: if True, maximize sum of all canonical correlations
        outdim_size: if not using all singular values, sum top-k singular values

    Returns:
        scalar tensor loss = negative correlation objective
    """
    n = x.size(0)
    if n < 2:
        return x.new_tensor(0.0)

    x = _center(x)
    y = _center(y)

    d1 = x.size(1)
    d2 = y.size(1)

    # covariance matrices
    scale = 1.0 / max(n - 1, 1)

    sigma_xx = scale * (x.t() @ x) + r1 * torch.eye(d1, device=x.device, dtype=x.dtype)
    sigma_yy = scale * (y.t() @ y) + r2 * torch.eye(d2, device=y.device, dtype=y.dtype)
    sigma_xy = scale * (x.t() @ y)

    # eigen-decomposition for inverse sqrt
    eigvals_x, eigvecs_x = torch.linalg.eigh(sigma_xx)
    eigvals_y, eigvecs_y = torch.linalg.eigh(sigma_yy)

    eigvals_x = torch.clamp(eigvals_x, min=eps)
    eigvals_y = torch.clamp(eigvals_y, min=eps)

    sigma_xx_inv_sqrt = eigvecs_x @ torch.diag(eigvals_x.pow(-0.5)) @ eigvecs_x.t()
    sigma_yy_inv_sqrt = eigvecs_y @ torch.diag(eigvals_y.pow(-0.5)) @ eigvecs_y.t()

    # T = Cxx^{-1/2} Cxy Cyy^{-1/2}
    T = sigma_xx_inv_sqrt @ sigma_xy @ sigma_yy_inv_sqrt

    # singular values are canonical correlations
    singular_values = torch.linalg.svdvals(T)
    singular_values = torch.clamp(singular_values, min=0.0)

    if use_all_singular_values:
        corr = singular_values.sum()
    else:
        if outdim_size is None:
            outdim_size = min(d1, d2)
        k = min(outdim_size, singular_values.numel())
        corr = singular_values[:k].sum()

    # maximize correlation => minimize negative correlation
    loss = -corr
    return loss


def cca_between_sets(
    x,
    y,
    subsample_to_min=True,
    r1=1e-4,
    r2=1e-4,
    eps=1e-8,
    use_all_singular_values=True,
    outdim_size=None,
):
    """
    Compute Deep CCA-style loss between two sets.

    Args:
        x: (n_x, d)
        y: (n_y, d)
        subsample_to_min: if True, subsample the larger set to match the smaller one

    Returns:
        scalar tensor loss (negative correlation; lower is better)
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return x.new_tensor(0.0)

    if subsample_to_min:
        x, y = _match_sample_count(x, y)

    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_tensor(0.0)

    return _cca_correlation_loss(
        x=x,
        y=y,
        r1=r1,
        r2=r2,
        eps=eps,
        use_all_singular_values=use_all_singular_values,
        outdim_size=outdim_size,
    )


def cca_loss_to_reference(
    z,
    batch_idx,
    reference_batch=None,
    sigma=None,
    subsample_per_batch=500,
    mode="to_reference",
    # ---- CCA-specific args ----
    subsample_to_min=True,
    r1=1e-4,
    r2=1e-4,
    eps=1e-8,
    use_all_singular_values=True,
    outdim_size=None,
    # ---- optional stats ----
    return_stats=False,
):
    """
    Minimal-change drop-in replacement:
    keep the old MMD-like interface, but internally compute Deep CCA-style loss.

    IMPORTANT:
    - `sigma` is kept only for API compatibility and is unused.
    - Returned loss is a NEGATIVE correlation objective.
      This means:
          * lower (more negative) is stronger correlation
          * if added directly to total loss, it encourages alignment
    - If you want a non-negative loss instead, see note below.

    Args:
        z: (N, D) latent tensor
        batch_idx: (N,) integer tensor labeling batches
        reference_batch: same meaning as before; used if mode="to_reference"
        sigma: unused, only kept for API compatibility
        subsample_per_batch: cap samples per batch for efficiency
        mode: "to_reference" or "pairwise"

        subsample_to_min: whether to match batch sizes by subsampling to min size
        r1, r2: covariance regularization
        eps: numerical stability
        use_all_singular_values: sum all canonical correlations
        outdim_size: top-k singular values to sum if not using all

    Returns:
        scalar tensor average CCA-style loss
        or (loss, stats) if return_stats=True
    """
    device = z.device
    dtype = z.dtype

    unique_batches = torch.unique(batch_idx)
    B = unique_batches.numel()

    if B <= 1:
        zero = z.new_tensor(0.0)
        if return_stats:
            return zero, {"num_pairs": 0, "mode": mode}
        return zero

    def collect_batch_tensors():
        batch_to_z = {}
        for bi in unique_batches:
            bi_int = int(bi.item())
            xi = z[batch_idx == bi_int]
            if xi.size(0) == 0:
                continue
            xi = _subsample_tensor(xi, subsample_per_batch)
            batch_to_z[bi_int] = xi
        return batch_to_z

    batch_to_z = collect_batch_tensors()

    if len(batch_to_z) <= 1:
        zero = z.new_tensor(0.0)
        if return_stats:
            return zero, {"num_pairs": 0, "mode": mode}
        return zero

    loss_sum = z.new_tensor(0.0)
    count = 0

    if mode == "to_reference":
        if reference_batch is None:
            ref = int(unique_batches[0].item())
        else:
            ref = int(reference_batch)

        if ref not in batch_to_z:
            ref = int(unique_batches[0].item())

        z_ref = batch_to_z[ref]
        if z_ref.size(0) < 2:
            zero = z.new_tensor(0.0)
            if return_stats:
                return zero, {"num_pairs": 0, "mode": mode}
            return zero

        for bi in unique_batches:
            bi = int(bi.item())
            if bi == ref:
                continue
            if bi not in batch_to_z:
                continue

            xi = batch_to_z[bi]
            if xi.size(0) < 2:
                continue

            cca_loss = cca_between_sets(
                z_ref,
                xi,
                subsample_to_min=subsample_to_min,
                r1=r1,
                r2=r2,
                eps=eps,
                use_all_singular_values=use_all_singular_values,
                outdim_size=outdim_size,
            )
            loss_sum = loss_sum + cca_loss
            count += 1

    elif mode == "pairwise":
        batch_keys = sorted(batch_to_z.keys())

        for i in range(len(batch_keys)):
            zi = batch_to_z[batch_keys[i]]
            if zi.size(0) < 2:
                continue

            for j in range(i + 1, len(batch_keys)):
                zj = batch_to_z[batch_keys[j]]
                if zj.size(0) < 2:
                    continue

                cca_loss = cca_between_sets(
                    zi,
                    zj,
                    subsample_to_min=subsample_to_min,
                    r1=r1,
                    r2=r2,
                    eps=eps,
                    use_all_singular_values=use_all_singular_values,
                    outdim_size=outdim_size,
                )
                loss_sum = loss_sum + cca_loss
                count += 1
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'to_reference' or 'pairwise'.")

    if count == 0:
        zero = z.new_tensor(0.0)
        if return_stats:
            return zero, {"num_pairs": 0, "mode": mode}
        return zero

    loss = (loss_sum / float(count)).to(device=device, dtype=dtype)

    if return_stats:
        return loss, {"num_pairs": count, "mode": mode}
    return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class FastGroupedBlockContrastiveLoss(nn.Module):
    def __init__(
        self,
        # temperature: float = 0.5,
        temperature: float = 1,
        # num_negatives: int = 1,
        num_negatives: int = 1,
        use_cosine: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.use_cosine = use_cosine
        self.eps = eps

    def _similarity(self, x, y):
        if self.use_cosine:
            x = F.normalize(x, dim=-1, eps=self.eps)
            y = F.normalize(y, dim=-1, eps=self.eps)
            if y.dim() == 2:
                return (x * y).sum(dim=-1)
            else:
                return (x.unsqueeze(1) * y).sum(dim=-1)
        else:
            if y.dim() == 2:
                return -(x - y).pow(2).sum(dim=-1)
            else:
                return -(x.unsqueeze(1) - y).pow(2).sum(dim=-1)

    def _sample_negatives_grouped(self, src_idx, group_ids, num_src, K, device):
        """
        Vectorized negative sampling within the same group.
        Excludes anchor itself, but does NOT exclude all true neighbors.
        """
        # sort source nodes by group id
        perm = torch.argsort(group_ids)                      # [N]
        sorted_groups = group_ids[perm]                      # [N]

        # unique groups and counts
        unique_groups, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts[:-1]]), dim=0
        )                                                   # [G]
        ends = starts + counts                              # [G]

        # map each src node to its rank position in perm
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(num_src, device=device)

        # for each edge anchor, get its group and group interval
        anchor_groups = group_ids[src_idx]                  # [E]

        # search group interval for each anchor
        group_pos = torch.searchsorted(unique_groups, anchor_groups)
        group_starts = starts[group_pos]                    # [E]
        group_ends = ends[group_pos]                        # [E]
        group_sizes = group_ends - group_starts             # [E]

        # sample positions within each anchor's group
        E = src_idx.shape[0]
        rand_offsets = torch.floor(
            torch.rand(E, K, device=device) * group_sizes.unsqueeze(1).float()
        ).long()                                            # [E, K]
        sampled_pos = group_starts.unsqueeze(1) + rand_offsets

        # map sorted positions -> original src node ids
        neg_idx = perm[sampled_pos]                         # [E, K]

        # exclude anchor itself with a few vectorized retries
        anchor_expand = src_idx.unsqueeze(1)
        same_mask = neg_idx.eq(anchor_expand)

        # Retry a few times; usually enough unless group size is tiny
        for _ in range(3):
            if not same_mask.any():
                break
            resample_offsets = torch.floor(
                torch.rand(same_mask.sum(), device=device) *
                group_sizes.unsqueeze(1).expand(E, K)[same_mask].float()
            ).long()
            resample_pos = group_starts.unsqueeze(1).expand(E, K)[same_mask] + resample_offsets
            neg_idx[same_mask] = perm[resample_pos]
            same_mask = neg_idx.eq(anchor_expand)

        # final fallback for groups of size 1 or persistent self-hit
        if same_mask.any():
            global_rand = torch.randint(0, num_src, (same_mask.sum(),), device=device)
            neg_idx[same_mask] = global_rand

            same_mask2 = neg_idx.eq(anchor_expand)
            if same_mask2.any():
                neg_idx[same_mask2] = (neg_idx[same_mask2] + 1) % num_src

        return neg_idx

    def forward(self, block, mu: torch.Tensor, group_ids: torch.Tensor = None):
        src_idx, dst_idx = block.edges(order="eid")
        if src_idx.numel() == 0:
            return mu.new_tensor(0.0)

        device = mu.device
        num_src = block.num_src_nodes()
        E = src_idx.shape[0]
        K = self.num_negatives

        z_anchor = mu[src_idx]   # [E, D]
        z_pos = mu[dst_idx]      # [E, D]

        if group_ids is None:
            neg_idx = torch.randint(0, num_src, (E, K), device=device)

            same_mask = neg_idx.eq(src_idx.unsqueeze(1))
            for _ in range(3):
                if not same_mask.any():
                    break
                neg_idx[same_mask] = torch.randint(0, num_src, (same_mask.sum(),), device=device)
                same_mask = neg_idx.eq(src_idx.unsqueeze(1))

            if same_mask.any():
                neg_idx[same_mask] = (neg_idx[same_mask] + 1) % num_src
        else:
            group_ids = group_ids.to(device)
            neg_idx = self._sample_negatives_grouped(
                src_idx=src_idx,
                group_ids=group_ids,
                num_src=num_src,
                K=K,
                device=device,
            )

        z_neg = mu[neg_idx]  # [E, K, D]

        pos_sim = self._similarity(z_anchor, z_pos) / self.temperature
        neg_sim = self._similarity(z_anchor, z_neg) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(E, dtype=torch.long, device=device)

        return F.cross_entropy(logits, labels)
