# ok so we have a slightly diff merge function.
# acc, for now, let's use unpack and pack.
# then we can just drop in a fully custom merge function into our bliasson algorithm.

import math
import torch
import triton
import triton.language as tl
from typing import Tuple

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

# -------------------------------------------------------------------
# These are the new Triton kernels that wrap the util functions.
# They operate elementwise over flattened arrays.
# -------------------------------------------------------------------

@triton.jit
def pack_kernel(a_ptr, b_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for packing two FP32 arrays into one FP64 output.
    
    a_ptr : pointer to the first FP32 input tensor.
    b_ptr : pointer to the second FP32 input tensor.
    out_ptr : pointer to the output FP64 tensor. (will hold the bit-packed result)
    numel : number of elements to process
    BLOCK_SIZE : compile-time constant block size.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load FP32 values from the two inputs.
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Pack the two FP32 values into a single uint64.
    merged = pack64(a, b)
    # Bitcast our 64-bit unsigned integer into a FP64.
    merged_fp64 = merged.to(tl.float64, bitcast=True)
    tl.store(out_ptr + offsets, merged_fp64, mask=mask)

@triton.jit
def unpack_kernel(in_ptr, a_ptr, b_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for unpacking an FP64 array into two FP32 outputs.
    
    in_ptr : pointer to the input FP64 tensor (packed data)
    a_ptr : pointer to the first FP32 output tensor.
    b_ptr : pointer to the second FP32 output tensor.
    numel : number of elements to process
    BLOCK_SIZE : compile-time constant block size.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load the packed FP64 data then bitcast it to uint64.
    merged_fp64 = tl.load(in_ptr + offsets, mask=mask)
    merged = merged_fp64.to(tl.uint64, bitcast=True)
    a, b = unpack64(merged)
    tl.store(a_ptr + offsets, a, mask=mask)
    tl.store(b_ptr + offsets, b, mask=mask)

# -------------------------------------------------------------------
# PyTorch wrappers for the new Triton kernels.
# These functions ensure the inputs are properly formatted, then launch the kernels.
# -------------------------------------------------------------------

def pack_tensors(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Packs two FP32 tensors elementwise into one FP64 tensor using a Triton kernel.
    
    Args:
        a (torch.Tensor): Input tensor of type torch.float32.
        b (torch.Tensor): Input tensor of type torch.float32 (must have same shape as a).
        out (torch.Tensor, optional): Output tensor to store the result.
            If None, a new tensor is allocated.
    
    Returns:
        torch.Tensor: Packed tensor of type torch.float64.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors 'a' and 'b' must have the same shape.")

    numel = a.numel()
    device = a.device

    if out is None:
        out = torch.empty_like(a, dtype=torch.float64, device=device)
    else:
        if out.shape != a.shape:
            raise ValueError("Output tensor must have the same shape as input tensors.")
        if out.dtype != torch.float64:
            raise ValueError("Output tensor must be of type torch.float64.")

    # Ensure the tensors are contiguous.
    a = a.contiguous()
    b = b.contiguous()
    out = out.contiguous()

    # Define the grid for launching the kernel.
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    pack_kernel[grid](a, b, out, numel, BLOCK_SIZE=1024)
    return out

def unpack_tensor(merged: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks an FP64 (packed) tensor into two FP32 tensors using a Triton kernel.
    
    Args:
        merged (torch.Tensor): Packed input tensor of type torch.float64.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two unpacked tensors of type torch.float32.
    """
    if merged.dtype != torch.float64:
        raise ValueError("Input tensor must be of type torch.float64.")

    numel = merged.numel()
    device = merged.device

    a = torch.empty_like(merged, dtype=torch.float32, device=device)
    b = torch.empty_like(merged, dtype=torch.float32, device=device)

    # Ensure the tensors are contiguous.
    merged = merged.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    # Define the grid for launching the kernel.
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    unpack_kernel[grid](merged, a, b, numel, BLOCK_SIZE=1024)
    return a, b


# tests

a,b = torch.randn(1000000,device="cuda"), torch.randn(1000000,device="cuda")

c = pack_tensors(a,b)

d,e = unpack_tensor(c)

assert (a == d).all()