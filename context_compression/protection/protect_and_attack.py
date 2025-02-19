# ideas behind protect-and-attack algorithm:

# we are simulating something like a turn-by-turn game between an attacker and defender.
# in this case, we are simulating it across a sequence of tokens in a context window.
# we will be computing the "health" of a certain token K (that lives in the KV cache) at all turns of the game (i.e. all Q positions in the context window).

# our transformer is using causal attention. We let future tokens "mask" past tokens (if they're no longer useful) or possibly "protect" them from being masked.
# so for any given K, it will not be protected or attacked by the tokens before it. It can, however, protect itself.

# to simplify the problem, we can just say that every Q that appears before K has protect=0 and attack=0 towards K.

# so the core algorithm we need to implement is:
# given an array of protections of length N, and attacks of length N, output an array of healths of length N.
# Health starts at zero. It can only go negative. It can never increase.
# The array of healths will tell us the health of K at all turns of the game.
# The K token starts every turn with a running health stat and a running protection stat, based on past turns.
# The turn starts with the attack phase: the attack will first subtract from protection points, and if protection points hit zero, it will subtract from health.
# Then the protection phase starts. The protection points will be added to the token's running protection stat.

# We also want this algorithm to be differentiable w.r.t. the protection and attack arrays (which are continuous!). This means going from a gradient w.r.t. the array of turn-wise healths -> a gradient w.r.t. the protection and attack arrays.
# This is kinda complicated actually.
  # So you should start by constructing a record of in which turns running_protection <= attack at the start of the turn. This mask will tell you in which turns the health was vulnerable.
  # This should be computed during the forward pass.
  # Then, during the backward pass, we will start from the end of the dhealth, attack, and protect arrays, and scan our way backwards to the start of the array.
  # We will keep two running sums of dhealth: one of all the dhealths I have encountered so far (all_dhealths), and one for all the dhealths I had encountered at the earliest vulnerable turn I've scanned to so far (dhealths_after_earliest_vulnerable_turn).
  # Gradient of health and protection at my current turn is just dhealths_after_earliest_vulnerable_turn.

# Now let's impl it as a python function.
# Some simple test cases:

# A = [1]
# P = any of [0], [1], or [2]
# Then
# H = [-1]
# Assuming dH = [1], then
# dA = [-1]
# dP = [0]

# A = [0]
# P = [0]
# Then
# H = [0]
# Assuming dH = [1], then
# dA = [-1]
# dP = [0]

# A = [0, 1]
# P = [2, 0]
# Then
# H = [0, 0]
# Assuming dH = [1, 1], then
# dA = [-2, 0]
# dP = [0, 0]

# A = [0, 1]
# P = [1, 0]
# Then
# H = [0, 0]
# Assuming dH = [1, 1], then
# dA = [-2, -1]
# dP = [1, 0]

# A = [1, 1, 10]
# P = [5, 0, 0]
# Then
# H = [-1, -1, -6]
# Assuming dH = [1, 1, 1], then
# dA = [-3, -2, -1]
# dP = [1, 1, 0]

from typing import List, Callable, Tuple
import torch  # <--- added torch import for the pytorch implementation

def protect_and_attack_raw(A: List[float], P: List[float]) -> Tuple[List[float], Callable[[List[float]], Tuple[List[float], List[float]]]]:

    assert all(a >= 0 for a in A)
    assert all(p >= 0 for p in P)

    H_running = 0
    P_running = 0

    took_damage = []
    H = []

    for a, p in zip(A, P):
        if (P_running - a) <= 0:
            H_running -= (a - P_running)
            P_running = 0
            took_damage.append(True)
        else:
            # in this case we are guaranteed that the protection stat has not run out yet. Since P_running > a, we know that P_running - a > 0.
            P_running -= a
            took_damage.append(False)
        H.append(H_running)

        P_running += p
    
    def backward_pass(dH: List[float]) -> Tuple[List[float], List[float]]:
        dA = [0] * len(A)
        dP = [0] * len(P)

        all_dhealths = 0
        all_dhealths_after_earliest_seen_vulnerable_turn = 0

        print("took_damage", took_damage)
        print("dH", dH)

        for i in range(len(H) - 1, -1, -1):
            dP[i] = all_dhealths_after_earliest_seen_vulnerable_turn
            all_dhealths += dH[i]
            if took_damage[i]:
                all_dhealths_after_earliest_seen_vulnerable_turn = all_dhealths
            dA[i] = -all_dhealths_after_earliest_seen_vulnerable_turn

        return dA, dP
    
    return H, backward_pass

import pytest
@pytest.mark.parametrize("A, P, dH, expected_H, expected_dA, expected_dP", [
    ([1], [0], [1], [-1], [-1], [0]),
    ([0], [0], [1], [0], [-1], [0]),
    ([0, 1], [2, 0], [1, 1], [0, 0], [-2, 0], [0, 0]),
    ([0, 1], [1, 0], [1, 1], [0, 0], [-2, -1], [1, 0]),
    ([1, 1, 10], [5, 0, 0], [1, 1, 1], [-1, -1, -7], [-3, -1, -1], [1, 1, 0]),
])
def test_protect_and_attack(A, P, dH, expected_H, expected_dA, expected_dP):
    H, backward_pass = protect_and_attack_raw(A, P)
    assert H == expected_H, f"H: {H}, expected_H: {expected_H}"
    print("H", H)
    assert backward_pass(dH) == (expected_dA, expected_dP), f"backward_pass(dH): {backward_pass(dH)}, expected_dA: {expected_dA}, expected_dP: {expected_dP}"

# -------------------------------------------------------------------
# New PyTorch implementation of the protect-and-attack function

class ProtectAndAttackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, P, dim):
        # Ensure A and P have the same shape
        if A.shape != P.shape:
            raise ValueError("A and P must have the same shape")
        # Bring the processing dimension to the end.
        dims = list(range(A.dim()))
        if dim < 0:
            dim = A.dim() + dim
        new_order = dims[:dim] + dims[dim+1:] + [dim]
        A_perm = A.permute(*new_order).contiguous()
        P_perm = P.permute(*new_order).contiguous()
        orig_shape = A_perm.shape  # shape: (batch_dims..., N)
        batch_size = A_perm.view(-1, A_perm.size(-1)).size(0)
        N = A_perm.size(-1)
        A_flat = A_perm.view(batch_size, N)
        P_flat = P_perm.view(batch_size, N)
        
        # Allocate output and record which turns took damage.
        H_flat = torch.empty_like(A_flat)
        took_damage = torch.zeros_like(A_flat, dtype=torch.bool)
        for b in range(batch_size):
            H_running = 0.0
            P_running = 0.0
            for i in range(N):
                a_val = A_flat[b, i].item()
                p_val = P_flat[b, i].item()
                if (P_running - a_val) <= 0:
                    damage = a_val - P_running
                    H_running = H_running - damage
                    P_running = 0.0
                    took_damage[b, i] = True
                else:
                    P_running = P_running - a_val
                H_flat[b, i] = H_running
                P_running += p_val

        # Reshape the computed health back to the permuted shape.
        H_perm = H_flat.view(orig_shape)
        # Compute inverse permutation to restore original order.
        inverse_order = [0] * len(new_order)
        for i, p in enumerate(new_order):
            inverse_order[p] = i
        H = H_perm.permute(*inverse_order)

        # Save variables for backward.
        ctx.save_for_backward(A, P, took_damage)
        ctx.dim = dim
        ctx.new_order = new_order
        ctx.inverse_order = inverse_order
        ctx.batch_size = batch_size
        ctx.N = N
        ctx.orig_shape = orig_shape
        return H

    @staticmethod
    def backward(ctx, grad_H):
        A, P, took_damage = ctx.saved_tensors
        new_order = ctx.new_order
        batch_size = ctx.batch_size
        N = ctx.N
        # Permute grad_H to match the shape of the forward pass.
        grad_H_perm = grad_H.permute(*new_order).contiguous()
        grad_H_flat = grad_H_perm.view(batch_size, N)

        dA_flat = torch.zeros_like(grad_H_flat)
        dP_flat = torch.zeros_like(grad_H_flat)
        for b in range(batch_size):
            all_dhealths = 0.0
            all_dhealths_after = 0.0
            for i in range(N-1, -1, -1):
                dP_flat[b, i] = all_dhealths_after
                all_dhealths += grad_H_flat[b, i].item()
                if took_damage[b, i]:
                    all_dhealths_after = all_dhealths
                dA_flat[b, i] = -all_dhealths_after

        dA_perm = dA_flat.view(ctx.orig_shape)
        dP_perm = dP_flat.view(ctx.orig_shape)
        inverse_order = ctx.inverse_order
        dA = dA_perm.permute(*inverse_order)
        dP = dP_perm.permute(*inverse_order)
        return dA, dP, None

def protect_and_attack_pytorch(A, P, dim=-1):
    """
    A PyTorch wrapper for the protect-and-attack function.
    A and P are tensors with the same shape.
    The processing is done along the dimension `dim`.
    """
    return ProtectAndAttackFunction.apply(A, P, dim)

# -------------------------------------------------------------------
# New tests for the PyTorch version

@pytest.mark.parametrize("A_list, P_list, dH_list, expected_H, expected_dA, expected_dP", [
    # These are the same as your original test cases.
    ([1], [0], [1], [-1], [-1], [0]),
    ([0], [0], [1], [0], [-1], [0]),
    ([0, 1], [2, 0], [1, 1], [0, 0], [-2, 0], [0, 0]),
    ([0, 1], [1, 0], [1, 1], [0, 0], [-2, -1], [1, 0]),
    ([1, 1, 10], [5, 0, 0], [1, 1, 1], [-1, -1, -7], [-3, -1, -1], [1, 1, 0]),
])
def test_protect_and_attack_pytorch_single_dim(A_list, P_list, dH_list, expected_H, expected_dA, expected_dP):
    A = torch.tensor(A_list, dtype=torch.float32, requires_grad=True)
    P = torch.tensor(P_list, dtype=torch.float32, requires_grad=True)
    dH = torch.tensor(dH_list, dtype=torch.float32)
    H = protect_and_attack_pytorch(A, P, dim=0)
    # Check forward pass.
    assert torch.allclose(H, torch.tensor(expected_H, dtype=torch.float32)), f"H: {H}, expected: {expected_H}"
    # Compute gradients.
    H.backward(dH)
    assert torch.allclose(A.grad, torch.tensor(expected_dA, dtype=torch.float32)), f"dA: {A.grad}, expected: {expected_dA}"
    assert torch.allclose(P.grad, torch.tensor(expected_dP, dtype=torch.float32)), f"dP: {P.grad}, expected: {expected_dP}"

def test_protect_and_attack_pytorch_multi_dim():
    # Test on a 2D tensor processing along dim=1.
    A = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32, requires_grad=True)
    P = torch.tensor([[2, 0], [1, 0]], dtype=torch.float32, requires_grad=True)
    # For row 0:
    #   i=0: a=0 -> condition (0-0)<=0 so damage occurs → H[0]=0, took_damage=True.
    #         then add p=2 → P_running becomes 2.
    #   i=1: a=1 -> condition (2-1)>0 so no damage, H remains 0.
    # For row 1:
    #   i=0: a=1 -> condition (0-1)<=0 → H becomes -1, took_damage=True, then add p=1 → P_running becomes 1.
    #   i=1: a=0 -> no damage → H remains -1.
    expected_H = torch.tensor([[0, 0], [-1, -1]], dtype=torch.float32)
    A_clone = A.clone().detach().requires_grad_(True)
    P_clone = P.clone().detach().requires_grad_(True)
    H = protect_and_attack_pytorch(A_clone, P_clone, dim=1)
    assert torch.allclose(H, expected_H)
    dH = torch.ones_like(H)
    H.backward(dH)
    # Backward calculations based on the iterative loop (see inline comments).
    # For both rows, the backward loop yields:
    #   For index i=1: dP = 0, dA = 0.
    #   For index i=0: dP = 0, dA = - (sum of dH from this turn onward) = -2.
    expected_dA = torch.tensor([[-2, 0], [-2, 0]], dtype=torch.float32)
    expected_dP = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    assert torch.allclose(A_clone.grad, expected_dA), f"multi-dim dA: {A_clone.grad}"
    assert torch.allclose(P_clone.grad, expected_dP), f"multi-dim dP: {P_clone.grad}"

# -------------------------------------------------------------------
# Triton kernel implementation of the protect-and-attack function

import triton
import triton.language as tl

@triton.jit
def kernel_protect_and_attack_forward(
    A_ptr, P_ptr, H_ptr, T_ptr,
    L: tl.constexpr, stride: tl.constexpr
):
    """
    Each program instance handles one row (sequence).
    A_ptr, P_ptr, H_ptr, and T_ptr point to 1D arrays that represent a row,
    and stride is the number of elements in each row (i.e. L).
    """
    b = tl.program_id(0)
    base = b * stride
    H_running = 0.0
    P_running = 0.0
    for i in range(L):
        a = tl.load(A_ptr + base + i)
        p = tl.load(P_ptr + base + i)
        # if current protection minus attack is <= 0 then we suffer damage.
        if (P_running - a) <= 0:
            damage = a - P_running
            H_running = H_running - damage
            P_running = 0.0
            tl.store(T_ptr + base + i, 1)  # mark "took damage"
        else:
            P_running = P_running - a
            tl.store(T_ptr + base + i, 0)
        tl.store(H_ptr + base + i, H_running)
        P_running = P_running + p


@triton.jit
def kernel_protect_and_attack_backward(
    grad_H_ptr, T_ptr, dA_ptr, dP_ptr,
    L: tl.constexpr, stride: tl.constexpr
):
    """
    Backward pass computed in reverse order.
    """
    b = tl.program_id(0)
    base = b * stride
    all_dhealths = 0.0
    all_dhealths_after = 0.0
    # Loop backwards: from i=L-1 down to 0.
    for i in range(L, 0, -1):
        idx = i - 1
        grad_val = tl.load(grad_H_ptr + base + idx)
        tl.store(dP_ptr + base + idx, all_dhealths_after)
        all_dhealths += grad_val
        flag = tl.load(T_ptr + base + idx)
        if flag != 0:
            all_dhealths_after = all_dhealths
        tl.store(dA_ptr + base + idx, -all_dhealths_after)


class ProtectAndAttackTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, P, dim):
        # Ensure tensors are on CUDA.
        if not A.is_cuda:
            A = A.cuda()
        if not P.is_cuda:
            P = P.cuda()
        # Permute inputs so that processing dim becomes the last dimension.
        dims = list(range(A.dim()))
        if dim < 0:
            dim = A.dim() + dim
        new_order = dims[:dim] + dims[dim+1:] + [dim]
        A_perm = A.permute(*new_order).contiguous()
        P_perm = P.permute(*new_order).contiguous()
        orig_shape = A_perm.shape  # shape: (..., L)
        if A_perm.ndim == 1:
            B = 1
            L = A_perm.shape[0]
            A_flat = A_perm.view(1, L)
            P_flat = P_perm.view(1, L)
            batch_shape = (1,)
        else:
            batch_shape = A_perm.shape[:-1]  # all dimensions except the last
            L = A_perm.shape[-1]
            B = 1
            for s in batch_shape:
                B *= s
            A_flat = A_perm.reshape(B, L)
            P_flat = P_perm.reshape(B, L)
        
        # Allocate output and a tensor for "took_damage" flags.
        H_flat = torch.empty_like(A_flat)
        T_flat = torch.empty_like(A_flat, dtype=torch.int32)
        grid = (B,)

        assert A_flat.dtype == torch.float32
        assert P_flat.dtype == torch.float32
        assert H_flat.dtype == torch.float32
        # Launch the forward Triton kernel.
        kernel_protect_and_attack_forward[grid](A_flat, P_flat, H_flat, T_flat, L, L)
        H_perm = H_flat.view(*orig_shape)
        # Compute inverse permutation to restore original order.
        inverse_order = [0] * len(new_order)
        for i, p in enumerate(new_order):
            inverse_order[p] = i
        H_out = H_perm.permute(*inverse_order)
        # Save variables for backward.
        ctx.save_for_backward(A, P, T_flat)
        ctx.dim = dim
        ctx.new_order = new_order
        ctx.inverse_order = inverse_order
        ctx.orig_shape = orig_shape
        ctx.batch_shape = batch_shape
        ctx.B = B
        ctx.L = L

        # Added runtime assert when protection is all zeros.
        if torch.all(P == 0):
            # The expected forward result when protection is all zeros is -cumsum(A)
            expected_H = torch.cumsum(-A, dim=dim)
            assert torch.allclose(
                H_out, expected_H, atol=1e-5
            ), f"Forward pass assert failed: H_out = {H_out}, expected = {expected_H}"

        return H_out

    @staticmethod
    def backward(ctx, grad_H):
        A, P, T_flat = ctx.saved_tensors
        new_order = ctx.new_order
        inverse_order = ctx.inverse_order
        orig_shape = ctx.orig_shape  # shape after permuting (e.g., (L,) for 1D, or (..., L) for higher dims)
        B = ctx.B
        L = ctx.L
        batch_shape = ctx.batch_shape

        # Permute grad_H to match the shape of the forward pass and flatten
        grad_H_perm = grad_H.permute(*new_order).contiguous().reshape(B, L)
        dA_flat = torch.empty((B, L), dtype=grad_H.dtype, device=grad_H.device)
        dP_flat = torch.empty((B, L), dtype=grad_H.dtype, device=grad_H.device)
        grid = (B,)
        assert grad_H_perm.dtype == torch.float32
        assert T_flat.dtype == torch.int32
        assert dA_flat.dtype == grad_H.dtype
        assert dP_flat.dtype == grad_H.dtype
        kernel_protect_and_attack_backward[grid](grad_H_perm, T_flat, dA_flat, dP_flat, L, L)
        
        dA_perm = dA_flat.reshape(*batch_shape, L)
        dP_perm = dP_flat.reshape(*batch_shape, L)
        
        # If the original input was 1D then our "batch" dimension is artificial.
        # In that case, we simply squeeze out the batch dim instead of calling permute.
        if len(orig_shape) == 1:
            dA = dA_perm.squeeze(0)
            dP = dP_perm.squeeze(0)
        else:
            dA = dA_perm.permute(*inverse_order)
            dP = dP_perm.permute(*inverse_order)

        # Added runtime assert in the backward pass for the case P is all zeros.
        if torch.all(P == 0):
            with torch.no_grad():
                # For the equivalent forward pass H = -cumsum(A, dim=dim), 
                # the gradient with respect to A is given by:
                #    dA[i] = - sum_{j=i}^{N-1} grad_H[j]
                # To compute this, we flip grad_H along the processing dimension,
                # compute cumsum, then flip back.
                grad_H_perm_for_calc = grad_H.permute(*new_order).contiguous()
                flipped = torch.flip(grad_H_perm_for_calc, dims=[-1])
                expected_dA_perm = -torch.flip(
                    torch.cumsum(flipped, dim=-1),
                    dims=[-1]
                )
                expected_dA_full = expected_dA_perm.view(ctx.orig_shape).permute(*inverse_order)
                # Assert that dA from our Triton backward is close to the expected value.
                assert torch.allclose(
                    dA, expected_dA_full, atol=1e-5
                ), f"Backward pass assert failed for dA: computed {dA}, expected {expected_dA_full}"
                # # For P, when protection is all zeros no contribution should propagate.
                # assert torch.allclose(
                #     dP, torch.zeros_like(dP), atol=1e-5
                # ), f"Backward pass assert failed for dP: computed {dP}, expected zeros"

        return dA, dP, None

def protect_and_attack_triton(A, P, dim=-1):
    """
    A Triton-based PyTorch wrapper for the protect-and-attack function.
    A and P are tensors with the same shape.
    Processing is done along the specified dimension `dim`.
    """
    return ProtectAndAttackTritonFunction.apply(A, P, dim)

# -------------------------------------------------------------------
# New tests for the Triton version

# -------------------------------------------------------------------
# Triton kernel implementation of cumulative sum

@triton.jit
def kernel_cumsum_forward(
    input_ptr, output_ptr,
    L: tl.constexpr, stride: tl.constexpr
):
    """
    Computes the cumulative sum along a 1D row of length L.
    """
    b = tl.program_id(0)
    base = b * stride
    acc = tl.load(input_ptr + base)
    for i in range(L):
        tl.store(output_ptr + base + i, acc)
        if i < L - 1:
            x = tl.load(input_ptr + base + i + 1)
            acc = acc + x

@triton.jit
def kernel_cumsum_backward(
    grad_ptr, grad_input_ptr,
    L: tl.constexpr, stride: tl.constexpr
):
    """
    Backward pass for cumulative sum.
    For y = cumsum(x), the derivative dL/dx[i] = sum_{j=i}^{L-1} dL/dy[j].
    """
    b = tl.program_id(0)
    base = b * stride
    acc = tl.load(grad_ptr + base)
    for i in range(L, 0, -1):
        idx = i - 1
        tl.store(grad_input_ptr + base + idx, acc)
        if i > 1:
            g = tl.load(grad_ptr + base + idx)
            acc = acc + g

# -------------------------------------------------------------------
# Triton-based cumulative sum autograd function

class CumsumTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, dim):
        # Permute X so that processing dim becomes the last dimension.
        dims = list(range(X.dim()))
        if dim < 0:
            dim = X.dim() + dim
        new_order = dims[:dim] + dims[dim+1:] + [dim]
        X_perm = X.permute(*new_order).contiguous()
        orig_shape = X_perm.shape  # shape: (..., L)
        if X_perm.ndim == 1:
            B = 1
            L = X_perm.shape[0]
            X_flat = X_perm.view(1, L)
        else:
            batch_shape = X_perm.shape[:-1]
            L = X_perm.shape[-1]
            B = 1
            for s in batch_shape:
                B *= s
            X_flat = X_perm.reshape(B, L)
        
        output_flat = torch.empty_like(X_flat)
        grid = (B,)
        kernel_cumsum_forward[grid](X_flat, output_flat, L, L)
        output_perm = output_flat.view(orig_shape)
        # Compute inverse permutation.
        inverse_order = [0] * len(new_order)
        for i, p in enumerate(new_order):
            inverse_order[p] = i
        out = output_perm.permute(*inverse_order)
        ctx.save_for_backward(X)
        ctx.dim = dim
        ctx.new_order = new_order
        ctx.inverse_order = inverse_order
        ctx.orig_shape = orig_shape
        ctx.B = B
        ctx.L = L
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        new_order = ctx.new_order
        inverse_order = ctx.inverse_order
        orig_shape = ctx.orig_shape
        B = ctx.B
        L = ctx.L

        grad_output_perm = grad_output.permute(*new_order).contiguous().view(B, L)
        grad_input_flat = torch.empty((B, L), dtype=grad_output.dtype, device=grad_output.device)
        grid = (B,)
        # Reuse the cumulative-sum backward kernel (which computes reverse cumsum).
        kernel_cumsum_backward[grid](grad_output_perm, grad_input_flat, L, L)
        
        grad_input = grad_input_flat.view(orig_shape).permute(*inverse_order)
        return grad_input, None

def cumsum_triton(X, dim=-1, parallel_scan=False):
    """
    Triton-based cumulative sum implementation.
    
    Args:
        X (torch.Tensor): Input float tensor on CUDA.
        dim (int): The dimension along which to compute the cumulative sum.
        parallel_scan (bool): If True, uses the one-phase scan (assumes row length is a power of 2).
    
    Returns:
        torch.Tensor: A tensor of the same shape as X containing the cumulative sum.
    """
    if parallel_scan:
        return CumsumTritonEfficientParallelScanFunction.apply(X, dim)
    else:
        return CumsumTritonFunction.apply(X, dim)

# -------------------------------------------------------------------
# New tests for Triton-based cumulative sum

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")
@pytest.mark.parametrize("parallel_scan", [True, False])
def test_triton_cumsum_vector(parallel_scan):
    # Test on a 1D tensor (vector).
    X = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda', requires_grad=True)
    expected = torch.tensor([1.0, 3.0, 6.0], device='cuda')
    Y = cumsum_triton(X, dim=0, parallel_scan=parallel_scan)
    assert torch.allclose(Y, expected, atol=1e-5), f"Forward pass: expected {expected}, got {Y}"
    # Test backward: for cumsum, with grad outputs ones, dX = [3, 2, 1].
    Y.backward(torch.ones_like(Y))
    expected_grad = torch.tensor([3.0, 2.0, 1.0], device='cuda')
    assert torch.allclose(X.grad, expected_grad, atol=1e-5), f"Backward pass: expected gradient {expected_grad}, got {X.grad}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")
@pytest.mark.parametrize("parallel_scan", [True, False])
def test_triton_cumsum_multi_dim(parallel_scan):
    # Test on a 2D tensor, processing along dim=1.
    X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda', requires_grad=True)
    if parallel_scan:
        # two-phase approach
        Y = CumsumTritonEfficientParallelScanFunction.apply(X, 1)
    else:
        # for example, fallback to the older single-phase CumsumTritonFunction
        Y = CumsumTritonFunction.apply(X, 1)
    expected = torch.cumsum(X, dim=1)
    assert torch.allclose(Y, expected, atol=1e-5), f"Forward pass: expected {expected}, got {Y}"

    # For each row, if grad outputs ones, expected gradient is [3, 2, 1].
    Y.backward(torch.ones_like(Y))

# Make sure you have at least Triton 2.2.0 for numerical stability.
assert triton.__version__ != '2.1.0', (
    'Triton 2.1.0 is missing enable_fp_fusion. '
    'Triton 2.2.0 is required for numerical stability of this implementation.'
)

# --- New Kernel Helpers Using tl.associative_scan ---

# Manual tuple packing by @jackd from https://github.com/openai/triton/issues/2359
@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return a, b

@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    return a | b

@triton.jit
def first_order_op(l, r):
    """
    See https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf Section 1.4.1.
    This implements the combine function used in the associative scan.
    """
    xl, fl = unpack64(l)
    xr, fr = unpack64(r)
    x = xl + xr
    f = fl
    return pack64(x, f)

@triton.jit
def forward_scan(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
    POW2_SEQUENCE_LENGTH: tl.constexpr,
):
    """
    Our new forward kernel for cumulative sum using an associative scan.
    For cumulative sum we set the "gates" to ones.
    Each program instance handles one entire sequence.
    """
    # Here we assume a 2D grid so that grid = (B, 1)
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    strides_in_row = tl.arange(0,POW2_SEQUENCE_LENGTH)
    strides = strides_in_row + sequence_id * SEQUENCE_LENGTH

    tokens_ = tl.load(tokens + strides, mask=strides_in_row < SEQUENCE_LENGTH)
    gates_ = tl.load(gates + strides, mask=strides_in_row < SEQUENCE_LENGTH)

    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + strides, output_tokens_, mask=strides_in_row < SEQUENCE_LENGTH)
    

# --- New Cumsum Triton Function using the Associative Scan Kernel ---

class CumsumTritonEfficientParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, dim):
        # Permute X so that the processing dimension becomes the last dimension.
        dims = list(range(X.dim()))
        if dim < 0:
            dim = X.dim() + dim
        new_order = dims[:dim] + dims[dim+1:] + [dim]
        X_perm = X.permute(*new_order).contiguous()
        orig_shape = X_perm.shape  # shape: (..., L)
        if X_perm.ndim == 1:
            B = 1
            L = X_perm.shape[0]
            X_flat = X_perm.view(1, L)
        else:
            batch_shape = X_perm.shape[:-1]
            L = X_perm.shape[-1]
            B = 1
            for s in batch_shape:
                B *= s
            X_flat = X_perm.reshape(B, L)
        
        # For cumulative sum we treat the original input as the "tokens" and use ones for the "gates."
        gates = torch.ones_like(X_flat)
        # Allocate output tensor.
        output = torch.empty_like(X_flat)
        
        # Launch the new forward_scan kernel.
        grid = (B, 1)  # 2D grid: one workgroup per sequence.
        closest_power_of_2 = 2 ** (L - 1).bit_length()
        forward_scan[grid](gates, X_flat, output, L, closest_power_of_2,enable_fp_fusion=False)
        
        output_view = output.view(orig_shape)
        # Compute inverse permutation to restore original dimensions.
        inverse_order = [0] * len(new_order)
        for i, p in enumerate(new_order):
            inverse_order[p] = i
        out = output_view.permute(*inverse_order)
        
        # Save context for the backward pass.
        ctx.save_for_backward(X)
        ctx.dim = dim
        ctx.new_order = new_order
        ctx.inverse_order = inverse_order
        ctx.orig_shape = orig_shape
        ctx.B = B
        ctx.L = L
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        new_order = ctx.new_order
        inverse_order = ctx.inverse_order
        orig_shape = ctx.orig_shape
        B = ctx.B
        L = ctx.L
        
        # Permute grad_output to the same shape used in the forward pass.
        grad_output_perm = grad_output.permute(*new_order).contiguous().view(B, L)
        grad_input_flat = torch.empty((B, L), dtype=grad_output.dtype, device=grad_output.device)
        
        grid = (B,)
        # Here we reuse the existing kernel for reverse cumulative sum.
        kernel_cumsum_backward[grid](grad_output_perm, grad_input_flat, L, L,enable_fp_fusion=False)
        
        grad_input = grad_input_flat.view(orig_shape).permute(*inverse_order)
        return grad_input, None


import torch
from typing import Callable

def bliasson_associative_scan(x: torch.Tensor, merge_fn: Callable, dim: int=-1):
    x = x.transpose(dim, x.ndim-1) if dim != -1 else x
    *rest, n = x.shape
    bit_length = n.bit_length()
    closest_power_of_2 = 2**bit_length
    x_big = torch.zeros((*rest, closest_power_of_2), dtype=x.dtype, device=x.device)
    x_big[...,:n] = x

    # ok so now we're going to do several rounds of summing.
    # each round will have half as many active nodes as the previous one. the final round will have 1 active node. after that round, we'll just return the sum.
    # ok so we can compute this by just using a diff set of indices each time.

    indices = torch.arange(0, closest_power_of_2)

    while len(indices) > 1:
        assert len(indices) % 2 == 0
        next_indices = indices[1::2]
        x_big[...,next_indices] = merge_fn(x_big[...,indices[::2]], x_big[...,indices[1::2]])
        indices = next_indices
    
    # ok now we're going to propagate the info back down the tree, from top-down.

    for i in range(bit_length,1,-1):
        end_of_first_chunk = torch.arange(2 ** (i-1),closest_power_of_2,2 ** (i-1)) - 1
        end_of_first_half_of_second_chunk = end_of_first_chunk + 2 ** (i - 2)

        x_big[...,end_of_first_half_of_second_chunk] = merge_fn(x_big[...,end_of_first_chunk], x_big[...,end_of_first_half_of_second_chunk])
    
    raw_out = x_big[...,:n]

    return raw_out.transpose(dim, x.ndim-1) if dim != -1 else raw_out

def bliasson_cumsum(x: torch.Tensor, dim: int=-1):
    return bliasson_associative_scan(x, lambda l, r: l + r, dim)

class CumsumBliassonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        return bliasson_cumsum(x, dim)
    
    @staticmethod
    def backward(ctx, grad_output):
        return bliasson_cumsum(grad_output.flip(ctx.dim), ctx.dim).flip(ctx.dim), None


def cumsum_bliasson(x, dim=-1):
    return CumsumBliassonFunction.apply(x, dim)

from .packing import pack_tensors, unpack_tensor

def merge_fn(C1,C2):
    # unpack l into U and V
    U1,V1 = unpack_tensor(C1)
    # unpack r into W and X
    U2,V2 = unpack_tensor(C2)

    V = V1 + V2 - torch.minimum(U1,V2)
    U = U2 + torch.relu(U1-V2)

    return pack_tensors(U,V)

def bliasson_protect_and_attack(A,P,dim=-1):
    A_hat = (A - P).float()
    U = torch.relu(-A_hat)
    V = torch.relu(A_hat)

    C = pack_tensors(U,V)

    C_out = bliasson_associative_scan(C,merge_fn,dim)

    U_out,V_out = unpack_tensor(C_out)

    T = U_out == 0.0 # whether this token took damage in this turn. taking zero damage IS included - as long as the token was not protected!

    return -1*V_out,T

# attack and protect means the attack phase in every turn happens before the protection for that turn.
def bliasson_attack_and_protect(A: torch.Tensor, P: torch.Tensor, dim: int=-1):
    rolled_P = P.roll(shifts=1, dims=dim)
    # then set the first element of P to 0
    index = [slice(None)] * P.ndim
    index[dim] = 0
    rolled_P[tuple(index)] = 0

    return bliasson_protect_and_attack(A,rolled_P,dim)


# NOTE: this assumes that the dim in question is the last dimension.
# So we need to transpose the tensor accordingly to make this true.
@triton.jit
def kernel_bliasson_protect_and_attack_backward(
    reverse_cumsum_grad_H_ptr, T_ptr, dA_ptr, dP_ptr,
    L: tl.constexpr, stride: tl.constexpr
):
    """
    Backward pass computed in reverse order.
    """
    b = tl.program_id(0)
    base = b * stride
    all_dhealths = 0.0
    all_dhealths_after = 0.0
    # Loop backwards: from i=L-1 down to 0.
    for i in range(L, 0, -1):
        idx = i - 1
        grad_val = tl.load(reverse_cumsum_grad_H_ptr + base + idx)
        tl.store(dP_ptr + base + idx, all_dhealths_after)
        all_dhealths = grad_val
        flag = tl.load(T_ptr + base + idx)
        if flag != 0:
            all_dhealths_after = all_dhealths
        tl.store(dA_ptr + base + idx, -all_dhealths_after)


class AttackAndProtectBliassonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, P, dim):
        ctx.dim = dim
        ctx.orig_shape = A.shape
        H,T = bliasson_attack_and_protect(A,P,dim)
        ctx.save_for_backward(T)
        return H
    
    @staticmethod
    def backward(ctx, grad_H):
        T, = ctx.saved_tensors

        flipped_cumsum_grad_H = bliasson_cumsum(grad_H.flip(ctx.dim),ctx.dim).flip(ctx.dim)

        dA = torch.zeros(*ctx.orig_shape, dtype=grad_H.dtype, device=grad_H.device)
        dP = torch.zeros(*ctx.orig_shape, dtype=grad_H.dtype, device=grad_H.device)

        final_dim = ctx.orig_shape[ctx.dim]

        # now let's invoke the triton function here.
        flipped_cumsum_grad_H_transposed = flipped_cumsum_grad_H.transpose(ctx.dim, len(ctx.orig_shape)-1)
        flipped_cumsum_grad_H_flat = flipped_cumsum_grad_H_transposed.reshape(-1,final_dim)
        dA_transposed = dA.transpose(ctx.dim, len(ctx.orig_shape)-1)
        dA_flat = dA_transposed.reshape(-1,final_dim)
        dP_transposed = dP.transpose(ctx.dim, len(ctx.orig_shape)-1)
        dP_flat = dP_transposed.reshape(-1,final_dim)

        T_transposed = T.transpose(ctx.dim, len(ctx.orig_shape)-1)
        T_flat = T_transposed.reshape(-1,final_dim)

        combined_batch_dim = flipped_cumsum_grad_H_flat.shape[0]

        grid = (combined_batch_dim,)
        kernel_bliasson_protect_and_attack_backward[grid](flipped_cumsum_grad_H_flat, T_flat, dA_flat, dP_flat, final_dim, final_dim)

        dA_transposed = dA_flat.reshape(dA_transposed.shape)
        dA = dA_transposed.transpose(ctx.dim, len(ctx.orig_shape)-1)

        dP_transposed = dP_flat.reshape(dP_transposed.shape)
        dP = dP_transposed.transpose(ctx.dim, len(ctx.orig_shape)-1)

        return dA,dP,None
    
def attack_and_protect_bliasson(A,P,dim=-1):
    return AttackAndProtectBliassonFunction.apply(A,P,dim)

protect_and_attack_fn = attack_and_protect_bliasson

@pytest.mark.parametrize("A_list, P_list, dH_list, expected_H, expected_dA, expected_dP", [
    # These are the same as your original test cases.
    ([1], [0], [1], [-1], [-1], [0]),
    # ([0], [0], [1], [0], [-1], [0]),
    # ([0, 1], [2, 0], [1, 1], [0, 0], [-2, 0], [0, 0]),
    # ([0, 1], [1, 0], [1, 1], [0, 0], [-2, -1], [1, 0]),
    # ([1, 1, 10], [5, 0, 0], [1, 1, 1], [-1, -1, -7], [-3, -1, -1], [1, 1, 0]),
])
def test_protect_and_attack_triton_single_dim(A_list, P_list, dH_list, expected_H, expected_dA, expected_dP):
    # Create tensors on CUDA.
    A = torch.tensor(A_list, dtype=torch.float32, requires_grad=True, device="cuda")
    P = torch.tensor(P_list, dtype=torch.float32, requires_grad=True, device="cuda")
    dH = torch.tensor(dH_list, dtype=torch.float32, device="cuda")
    H = protect_and_attack_fn(A, P, dim=0)
    # Check forward pass (move result back to CPU for comparison).
    assert torch.allclose(H.cpu(), torch.tensor(expected_H, dtype=torch.float32)), f"H: {H.cpu()}, expected: {expected_H}"
    # Compute gradients.
    H.backward(dH)
    assert torch.allclose(A.grad.cpu(), torch.tensor(expected_dA, dtype=torch.float32)), f"dA: {A.grad.cpu()}, expected: {expected_dA}"
    assert torch.allclose(P.grad.cpu(), torch.tensor(expected_dP, dtype=torch.float32)), f"dP: {P.grad.cpu()}, expected: {expected_dP}"

def test_protect_and_attack_triton_multi_dim():
    # Test on a 2D tensor processing along dim=1.
    A = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32, requires_grad=True, device="cuda")
    P = torch.tensor([[2, 0], [1, 0]], dtype=torch.float32, requires_grad=True, device="cuda")
    # For row 0:
    #   i=0: a=0 -> condition (0-0)<=0 so damage occurs → H[0]=0, took_damage=True.
    #         then add p=2 → P_running becomes 2.
    #   i=1: a=1 -> condition (2-1)>0 so no damage, H remains 0.
    # For row 1:
    #   i=0: a=1 -> condition (0-1)<=0 → H becomes -1, took_damage=True, then add p=1 → P_running becomes 1.
    #   i=1: a=0 -> no damage → H remains -1.
    expected_H = torch.tensor([[0, 0], [-1, -1]], dtype=torch.float32)
    A_clone = A.clone().detach().requires_grad_(True)
    P_clone = P.clone().detach().requires_grad_(True)
    H = protect_and_attack_fn(A_clone, P_clone, dim=1)
    assert torch.allclose(H.cpu(), expected_H)
    dH = torch.ones_like(H, device="cuda")
    H.backward(dH)
    # Backward calculations based on the iterative loop (see inline comments).
    # For both rows, the backward loop yields:
    #   For index i=1: dP = 0, dA = 0.
    #   For index i=0: dP = 0, dA = - (sum of dH from this turn onward) = -2.
    expected_dA = torch.tensor([[-2, 0], [-2, 0]], dtype=torch.float32)
    expected_dP = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    assert torch.allclose(A_clone.grad.cpu(), expected_dA), f"multi-dim dA: {A_clone.grad.cpu()}"
    assert torch.allclose(P_clone.grad.cpu(), expected_dP), f"multi-dim dP: {P_clone.grad.cpu()}"

# -------------------------------------------------------------------
# New tests for 3D tensors (non-degenerate) for both the PyTorch-based and Triton-based implementations.

import pytest

def test_protect_and_attack_pytorch_3d():
    # We create a 3D tensor of shape (2, 2, 2) where processing is done along the last dimension.
    # For each slice the simulation is as follows:
    #
    # Batch 0, row 0: A = [0, 1], P = [2, 0]
    #   token0: (0-0)<=0 -> damage flag True, H becomes 0, then add p=2 → P_running becomes 2.
    #   token1: (2-1)>0 -> no damage, H remains 0.
    #   => H = [0, 0]; backward yields: dA = [-2, 0], dP = [0, 0].
    #
    # Batch 0, row 1: A = [1, 0], P = [0, 1]
    #   token0: (0-1)<=0 -> damage, H becomes -1, then add protection.
    #   token1: damage flag True -> H remains -1.
    #   => H = [-1, -1]; backward yields: dA = [-2, -1], dP = [1, 0].
    #
    # Batch 1, row 0: A = [0, 1], P = [1, 0]
    #   token0: damage flag True -> H = 0, then add protection.
    #   token1: (1-1)<=0 -> damage, H remains 0.
    #   => H = [0, 0]; backward yields: dA = [-2, -1], dP = [1, 0].
    #
    # Batch 1, row 1: A = [1, 0], P = [2, 0]
    #   token0: (0-1)<=0 -> damage (damage=1), H becomes -1, then add protection (P_running becomes 2).
    #   token1: (2-0)>0 -> no damage, H remains -1.
    #   => H = [-1, -1]; backward yields: dA = [-2, 0], dP = [0, 0].

    A = torch.tensor([
            [[0, 1],
             [1, 0]],
            [[0, 1],
             [1, 0]]
        ], dtype=torch.float32, requires_grad=True,device="cuda")
    
    P = torch.tensor([
            [[2, 0],
             [0, 1]],
            [[1, 0],
             [2, 0]]
        ], dtype=torch.float32, requires_grad=True,device="cuda")

    expected_H = torch.tensor([
            [[0,  0],
             [-1, -1]],
            [[0,  0],
             [-1, -1]]
        ], dtype=torch.float32,device="cuda")

    H = protect_and_attack_fn(A, P, dim=-1)
    assert torch.allclose(H, expected_H), f"Forward pass error: H = {H}, expected {expected_H}"

    dH = torch.ones_like(H)
    H.backward(dH)

    expected_dA = torch.tensor([
            [[-2,  0],
             [-2, -1]],
            [[-2, -1],
             [-2,  0]]
        ], dtype=torch.float32,device="cuda")
    expected_dP = torch.tensor([
            [[0, 0],
             [1, 0]],
            [[1, 0],
             [0, 0]]
        ], dtype=torch.float32,device="cuda")

    assert torch.allclose(A.grad, expected_dA), f"PyTorch backward dA error:\n{A.grad}, expected:\n{expected_dA}"
    assert torch.allclose(P.grad, expected_dP), f"PyTorch backward dP error: {P.grad}, expected: {expected_dP}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")
def test_protect_and_attack_triton_3d():
    # Similar 3D test for the Triton implementation.
    A = torch.tensor([
            [[0, 1],
             [1, 0]],
            [[0, 1],
             [1, 0]]
        ], dtype=torch.float32, requires_grad=True, device="cuda")
    
    P = torch.tensor([
            [[2, 0],
             [0, 1]],
            [[1, 0],
             [2, 0]]
        ], dtype=torch.float32, requires_grad=True, device="cuda")

    expected_H = torch.tensor([
            [[0,  0],
             [-1, -1]],
            [[0,  0],
             [-1, -1]]
        ], dtype=torch.float32)
    
    H = protect_and_attack_fn(A, P, dim=-1)
    assert torch.allclose(H.cpu(), expected_H), f"Triton forward error: H = {H.cpu()}, expected {expected_H}"

    dH = torch.ones_like(H, device="cuda")
    H.backward(dH)

    expected_dA = torch.tensor([
            [[-2,  0],
             [-2, -1]],
            [[-2, -1],
             [-2,  0]]
        ], dtype=torch.float32)
    expected_dP = torch.tensor([
            [[0, 0],
             [1, 0]],
            [[1, 0],
             [0, 0]]
        ], dtype=torch.float32)

    assert torch.allclose(A.grad.cpu(), expected_dA), f"Triton backward dA error:\n{A.grad.cpu()}, expected:\n{expected_dA}"
    assert torch.allclose(P.grad.cpu(), expected_dP), f"Triton backward dP error:\n{P.grad.cpu()}, expected:\n{expected_dP}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")
def test_protect_and_attack_triton_random_A_P0():
    """
    Test that when P is zero and A is random the health output equals -cumsum(A)
    and that with dH=ones the gradients are:
        dA = -[N, N-1, ..., 1]
        dP = [N-1, N-2, ..., 0]
    """
    N = 10
    A = torch.rand(N, dtype=torch.float32, device="cuda", requires_grad=True)
    P = torch.zeros_like(A, requires_grad=True)
    H = protect_and_attack_fn(A, P, dim=0)
    expected_H = -torch.cumsum(A, dim=0)
    assert torch.allclose(H.cpu(), expected_H.cpu(), atol=1e-5), (
        f"Output health {H.cpu()} does not match expected {expected_H.cpu()}"
    )
    dH = torch.ones_like(H)
    H.backward(dH)
    # Expected gradients:
    # For a sequence of length N, since every token is "vulnerable", the backward pass computes:
    #   dA[0] = -N, dA[1] = -(N-1), ..., dA[N-1] = -1.
    #   dP[0] = N-1, dP[1] = N-2, ..., dP[N-1] = 0.
    expected_dA = -torch.arange(N, 0, -1, dtype=torch.float32, device="cuda")
    expected_dP = torch.arange(N-1, -1, -1, dtype=torch.float32, device="cuda")
    assert torch.allclose(A.grad, expected_dA, atol=1e-5), (
        f"dA {A.grad} does not match expected {expected_dA}"
    )
    assert torch.allclose(P.grad, expected_dP, atol=1e-5), (
        f"dP {P.grad} does not match expected {expected_dP}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")
def test_protect_and_attack_triton_A0_Ppositive():
    """
    Test that when A is zero and P is positive, the simulation produces
    zero health at every turn and the backward pass produces gradients where
    only the first token gets nonzero gradient:
        dA[0] = -N  and all other entries zero,
        dP = all zeros.
    """
    N = 10
    A = torch.zeros(N, dtype=torch.float32, device="cuda", requires_grad=True)
    # Choose a positive protection value; here we use ones.
    P = torch.ones(N, dtype=torch.float32, device="cuda", requires_grad=True)
    H = protect_and_attack_fn(A, P, dim=0)
    expected_H = torch.zeros_like(A)
    assert torch.allclose(H.cpu(), expected_H.cpu(), atol=1e-5), (
        f"Output health {H.cpu()} does not match expected {expected_H.cpu()}"
    )
    dH = torch.ones_like(H)
    H.backward(dH)
    # With A==0 and P>0, only token 0 is vulnerable:
    expected_dA = torch.zeros_like(A)
    expected_dA[0] = -float(N)  # gradient accumulates all the dH values = -N at token 0
    expected_dP = torch.zeros_like(P)
    assert torch.allclose(A.grad, expected_dA, atol=1e-5), (
        f"dA {A.grad} does not match expected {expected_dA}"
    )
    assert torch.allclose(P.grad, expected_dP, atol=1e-5), (
        f"dP {P.grad} does not match expected {expected_dP}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")
def test_protect_and_attack_triton_A0_P0():
    """
    Test that when both A and P are zero the health output equals -cumsum(A)=0,
    and with dH=ones the backward gradients are:
        dA = -[N, N-1, ..., 1]
        dP = [N-1, N-2, ..., 0]
    (which is the same as when P is zero, since the condition always triggers).
    """
    N = 10
    A = torch.zeros(N, dtype=torch.float32, device="cuda", requires_grad=True)
    P = torch.zeros(N, dtype=torch.float32, device="cuda", requires_grad=True)
    H = protect_and_attack_fn(A, P, dim=0)
    expected_H = -torch.cumsum(A, dim=0)  # which is all zeros
    assert torch.allclose(H.cpu(), expected_H.cpu(), atol=1e-5), (
        f"Output health {H.cpu()} does not match expected {expected_H.cpu()}"
    )
    dH = torch.ones_like(H)
    H.backward(dH)
    expected_dA = -torch.arange(N, 0, -1, dtype=torch.float32, device="cuda")
    expected_dP = torch.arange(N-1, -1, -1, dtype=torch.float32, device="cuda")
    assert torch.allclose(A.grad, expected_dA, atol=1e-5), (
        f"dA {A.grad} does not match expected {expected_dA}"
    )
    assert torch.allclose(P.grad, expected_dP, atol=1e-5), (
        f"dP {P.grad} does not match expected {expected_dP}"
    )
