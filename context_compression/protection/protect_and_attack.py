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
        all_dhealths = all_dhealths + grad_val
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

@pytest.mark.parametrize("A_list, P_list, dH_list, expected_H, expected_dA, expected_dP", [
    # These are the same as your original test cases.
    ([1], [0], [1], [-1], [-1], [0]),
    ([0], [0], [1], [0], [-1], [0]),
    ([0, 1], [2, 0], [1, 1], [0, 0], [-2, 0], [0, 0]),
    ([0, 1], [1, 0], [1, 1], [0, 0], [-2, -1], [1, 0]),
    ([1, 1, 10], [5, 0, 0], [1, 1, 1], [-1, -1, -7], [-3, -1, -1], [1, 1, 0]),
])
def test_protect_and_attack_triton_single_dim(A_list, P_list, dH_list, expected_H, expected_dA, expected_dP):
    # Create tensors on CUDA.
    A = torch.tensor(A_list, dtype=torch.float32, requires_grad=True, device="cuda")
    P = torch.tensor(P_list, dtype=torch.float32, requires_grad=True, device="cuda")
    dH = torch.tensor(dH_list, dtype=torch.float32, device="cuda")
    H = protect_and_attack_triton(A, P, dim=0)
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
    H = protect_and_attack_triton(A_clone, P_clone, dim=1)
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
    #   token0: (0-0)<=0 -> damage flag True, H becomes 0, then add protection (P_running becomes 2).
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
        ], dtype=torch.float32, requires_grad=True)
    
    P = torch.tensor([
            [[2, 0],
             [0, 1]],
            [[1, 0],
             [2, 0]]
        ], dtype=torch.float32, requires_grad=True)

    expected_H = torch.tensor([
            [[0,  0],
             [-1, -1]],
            [[0,  0],
             [-1, -1]]
        ], dtype=torch.float32)

    H = protect_and_attack_pytorch(A, P, dim=-1)
    assert torch.allclose(H, expected_H), f"Forward pass error: H = {H}, expected {expected_H}"

    dH = torch.ones_like(H)
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

    assert torch.allclose(A.grad, expected_dA), f"PyTorch backward dA error: {A.grad}, expected: {expected_dA}"
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
    
    H = protect_and_attack_triton(A, P, dim=-1)
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

    assert torch.allclose(A.grad.cpu(), expected_dA), f"Triton backward dA error: {A.grad.cpu()}, expected {expected_dA}"
    assert torch.allclose(P.grad.cpu(), expected_dP), f"Triton backward dP error: {P.grad.cpu()}, expected {expected_dP}"