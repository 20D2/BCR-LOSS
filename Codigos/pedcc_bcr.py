"""
Author: Nicolas Larue (nicolas.larue@ensea.fr)
PEDCC points generation + BCR loss
"""

import random
import math

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def classical_gs(A):
    dim = A.shape
    # print(dim, type(dim))
    Q = np.zeros(dim)  # initialize Q
    R = np.zeros((dim[1], dim[1]))  # initialize R
    for j in range(dim[1]):
        y = np.copy(A[:, j])
        for i in range(j):
            R[i, j] = np.matmul(np.transpose(Q[:, i]), A[:, j])
            y -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]
    return Q

def pedcc_frame_2(n: int, k: int = None) -> np.ndarray:
    assert 0 < k <= n + 1
    zero = [0] * (n - k + 1)
    u0 = [-1][:0] + zero + [-1][0:]
    u1 = [1][:0] + zero + [1][0:]
    u = np.stack((u0, u1)).tolist()
    for i in range(k - 2):
        c = np.insert(u[len(u) - 1], 0, 0)
        for j in range(len(u)):
            p = np.append(u[j], 0).tolist()
            s = len(u) + 1
            u[j] = math.sqrt(s * (s - 2)) / (s - 1) * np.array(p) - 1 / (
                s - 1
            ) * np.array(c)
        u.append(c)
    return np.array(u)


def pedcc_pod(
    n: int, k: int = None, qr: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    U = pedcc_frame_2(n=n, k=k)
    r = np.random.RandomState(seed)
    V = r.rand(n, n)  # [0, 1)
    if qr:
        Q, R = np.linalg.qr(V)
        V = Q
    else:
        V = classical_gs(V)
    points = np.dot(U, V)
    return points


def verif(points: np.ndarray, debug=False, verbose=True, rtol=1e-05, atol=1e-08) -> bool:
    """ verify that the given points are indeed in even-distributed n-hypersphere"""
    k, n = points.shape
    sim_pedcc = -1 / (k - 1)
    p_sum = np.sum(points, axis=0)
    p_norms = np.linalg.norm(points, axis=1)
    # p_sim    = cosine_similarity(points)

    # compute tensors
    points_normalized = points / p_norms[:, np.newaxis]
    p_sim = points_normalized @ points_normalized.T
    p_sim[np.tril_indices(k, k=0)] = 0
    if debug:
        print("p_sum  ", p_sum)
        print("p_norms", p_norms)
        print("p_sim  ", p_sim)

    # compute condition on previous tensors
    centered: bool = np.allclose(p_sum, np.zeros(n), rtol=rtol, atol=atol)
    normalized: bool = np.allclose(p_norms, np.ones(k), rtol=rtol, atol=atol)
    evenly_distributed: bool = np.allclose(
        p_sim[np.triu_indices(k, k=1)], sim_pedcc * np.ones(k * (k - 1) // 2), 
        rtol=rtol, atol=atol
    )
    is_pedcc = all((normalized, centered, evenly_distributed))

    if verbose:
        print(f"PEDCC(k={k}, n={n}) ? is_pedcc={is_pedcc}")
        print(
            f"normalized : {normalized}",
            f"centered : {centered}",
            f"evenly-distributed {evenly_distributed}",
            sep=" | ",
        )

    return is_pedcc


def generate(
    n: int,
    k: int,
    method: str = "pod",
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate k evenly distributed R^n points in a unit (n-1)-hypersphere 
    Args:
        n (int): dimension of the Euclidean space
        k (int): number of points to generate
        method (str): method to generate the points. Defaults to "simplex".
        filename (str, optional): filename to save the points. Defaults to None.
        seed (int, optional): seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: k evenly distributed points in a unit (n-1)-hypersphere

    >>> generate(2, 3, method="simplex")
    array([[ 0.        ,  0.        ],
           [ 0.70710678,  0.70710678],
    """
    if seed is None or not (isinstance(seed, int) and 0 <= seed < 2 ** 32):
        print("[pedcc.generate] seed must be an integer between 0 and 2**32-1")
        seed = random.randrange(2 ** 32)
        print("[pedcc.generate] seed set to", seed)

    if method == "simplex":
        raise Exception("simplex method not implemented")
    elif method == "frame":
        points = pedcc_frame_2(n, k)
    elif method == "pod":
        points = pedcc_pod(n, k, seed=seed)
    else:
        raise NotImplementedError(f"method not supported: {method}")

    if not verif(points, debug=False, verbose=True):
        raise ValueError(f"constructed points are not PEDCC")

    if filename:
        suffix = (f"_seed{seed}" if method == "pod" else "") + ".npy"
        path = f"{filename}_n{n}_k{k}_{method}{suffix}"
        np.save(path, points)

    return points




class BCR(nn.Module):
    """BCR loss proposed in SeeABLE (https://arxiv.org/abs/2211.11296)
    It also supports the unsupervised contrastive loss in SimCLR
    
    Usage:
        loss_fn = BCR(
            num_times=1, # number of (contrastive) views
            temperature=0.07, # temperature for the contrastive loss
            base_temperature=0.07, # usualy same as temperature
            contrast_mode="all", # contrastive anchors: all, one
            n=100, # latent dimension of vectors (eg. resnet50 -> 2048)
            k=2, # number of classes (eg. CIFAR10 -> 10)
            method="pod", # method to generate the pedcc points
            seed=42, # seed for the random number generator
        )

        ...

        for x, y in loader:
            z = model(x) # shape (batch_size, n)

            loss = loss_fn(z, labels=y)
    """

    def __init__(
        self,
        num_times: int = 1,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",
        # pedcc
        n: int = None,
        k: int = None,
        method: str = "pod",
        seed: Optional[int] = None,
    ):
        super(BCR, self).__init__()
        self.num_times = num_times
        # assert num_times == 2
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

        # generate pedcc points
        self.n = n
        self.k = k
        points = generate(
            n, k, method=method, filename="BCR", seed=seed
        )
        self.points = torch.from_numpy(points)
        print(self)

    def __str__(self):
        lines = []
        lines.append(f"pedcc({self.k}, {self.n})")
        lines.append(f"num_times: {self.num_times}")
        lines.append(f"temperature: {self.temperature}")
        lines.append(f"base_temperature: {self.base_temperature}")
        lines.append(f"contrast_mode: {self.contrast_mode}")
        return "[BCR]" + "\n" + "\n".join(["\t" + l for l in lines])

    def unstack(self, features, labels):
        """
        in:
            features torch.Size([bsz, d])
            labels torch.Size([bsz])

        out:
            features torch.Size([bsz//num_times, num_times, d])
            labels torch.Size([bsz//num_times])
        """
        batch_size = features.shape[0] // self.num_times
        features = torch.cat(
            [
                fi.unsqueeze(1)
                for fi in torch.split(features, [batch_size] * self.num_times, dim=0)
            ],
            dim=1,
        )
        if labels is not None:
            labels = labels[:batch_size]
        return features, labels

    def get_mask(self, labels, mask, batch_size: int):
        # labels|  mask | mask_output
        #   0   |   0   | torch.eye(batch_size)
        #   0   |   1   | mask
        #   1   |   0   | torch.eq(labels, labels.T)
        #   1   |   1   | "Cannot define both `labels` and `mask`"
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            return torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            return torch.eq(labels, labels.T).float()
        else:
            return mask.float()

    def forward(self, features, labels=None, mask=None, simclr=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        features, labels = self.unstack(features, labels)
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, contrast_count = features.shape[:2]

        if not simclr:
            mask = self.get_mask(labels, mask, batch_size)
        else:
            mask = self.get_mask(None, None, batch_size)
        mask = mask.to(features)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_count = 1
            anchor_feature = features[:, 0]
        elif self.contrast_mode == "all":
            anchor_count = contrast_count
            anchor_feature = contrast_feature
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # ----------------------------------------------------------------------
        # replace self-contrast cases with pedcc sim
        # ----------------------------------------------------------------------

        self.points = self.points.to(features)
        centroids_feature = self.points[labels.repeat(anchor_count)]
        points_dot_contrast = torch.div(
            torch.mul(anchor_feature, centroids_feature).sum(1), self.temperature
        )
        I = torch.arange(centroids_feature.size(0))
        anchor_dot_contrast[I, I] = points_dot_contrast

        # ----------------------------------------------------------------------

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # fix : https://github.com/HobbitLong/SupContrast/pull/86
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # loss
        # the gradient scales inversely with choice of temperature τ;
        # therefore we rescale the loss by τ during training for stability
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

