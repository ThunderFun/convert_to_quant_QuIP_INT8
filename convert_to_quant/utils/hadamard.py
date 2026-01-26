import torch
from torch import Tensor
import math
from typing import Optional


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return (n > 0) and (n & (n - 1) == 0)


def next_power_of_two(n: int) -> int:
    """Return the next power of two greater than or equal to n."""
    if n <= 0:
        return 1
    return 2**(n - 1).bit_length()


def hadamard_matrix(n: int) -> Tensor:
    """
    Generate normalized Hadamard matrix of size n x n.
    n must be a power of 2.
    
    H_1 = [1]
    H_2 = 1/sqrt(2) * [[1,  1],
                       [1, -1]]
    H_2k = 1/sqrt(2) * [[H_k,  H_k],
                        [H_k, -H_k]]
    """
    if not is_power_of_two(n):
        raise ValueError(f"n must be a power of 2, got {n}")

    if n == 1:
        return torch.ones((1, 1))

    h_prev = hadamard_matrix(n // 2)
    h_n = torch.cat([
        torch.cat([h_prev, h_prev], dim=1),
        torch.cat([h_prev, -h_prev], dim=1)
    ], dim=0)
    
    return h_n / math.sqrt(2)


def fast_hadamard_transform(x: Tensor, normalize: bool = True, inplace: bool = False) -> Tensor:
    """
    Apply Fast Hadamard Transform (FHT) in O(n log n) time.
    x can be a vector or a matrix (transform applied to last dimension).
    n must be a power of 2.
    
    Vectorized implementation for PyTorch with Triton acceleration.
    """
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    # Try Triton acceleration if on CUDA
    if x.is_cuda:
        try:
            from ..comfy.hadamard_kernels import HAS_TRITON_HADAMARD, triton_hadamard_transform
            if HAS_TRITON_HADAMARD:
                # Triton implementation is currently not in-place and always normalizes
                # TODO: Support inplace and normalize=False in Triton
                return triton_hadamard_transform(x)
        except ImportError:
            pass

    original_shape = x.shape
    if not inplace:
        x = x.clone()
    
    x = x.reshape(-1, n)
    
    h = 1
    while h < n:
        x = x.view(-1, n // (h * 2), 2, h)
        # Butterfly operation: [a, b] -> [a+b, a-b]
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        # We can't easily do this in-place with torch.stack,
        # but we can avoid the initial clone if inplace=True
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    
    x = x.reshape(original_shape)
    if normalize:
        x = x / math.sqrt(n)
        
    return x


def random_orthogonal_matrix(n: int, seed: Optional[int] = None, device: str = "cpu") -> Tensor:
    """
    Generate random orthogonal matrix via QR decomposition of random Gaussian.
    
    Algorithm:
    1. G = randn(n, n)
    2. Q, R = QR(G)
    3. Q = Q @ diag(sign(diag(R)))  # Ensure uniform distribution on O(n)
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    g = torch.randn(n, n, device=device)
    q, r = torch.linalg.qr(g)
    
    # Ensure uniform distribution
    d = torch.diag(r).sign()
    # Handle zeros in sign (unlikely but possible)
    d[d == 0] = 1
    
    q = q * d.unsqueeze(0)
    return q
