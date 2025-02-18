# cumsum_triton

import torch
import triton
import triton.language as tl

@triton.jit
def cumsum_triton(x_ptr, y_ptr, n_elements):
    """
    A simple sequential cumulative sum kernel in fp64.
    
    This kernel reads the input array from x_ptr, initializes the accumulator
    with the first element (to ensure the accumulator is fp64) and writes the
    cumulative sum into y_ptr. It then loops over the remaining elements,
    adds each value to the accumulator, and stores the cumulative sum.
    
    Parameters:
      x_ptr: pointer to the beginning of the input array.
      y_ptr: pointer to the beginning of the output array.
      n_elements: total number of elements to process (expected to be a Python int).
    """
    # Initialize the accumulator with the first element, ensuring the type is fp64.
    acc = tl.load(x_ptr)
    tl.store(y_ptr, acc)
    
    # Process the rest of the elements.
    i = 1
    while i < n_elements:
        val = tl.load(x_ptr + i)
        acc = acc + val
        tl.store(y_ptr + i, acc)
        i = i + 1

def triton_cumsum(x):
    """
    Computes the cumulative sum of the input tensor x using a Triton kernel.
    
    The function allocates an output tensor of the same shape as x and then launches
    the Triton kernel `cumsum_triton` with a grid size of 1 (i.e. a single program instance)
    to perform the sequential accumulation.
    
    Parameters:
      x: a 1-dimensional torch tensor of dtype torch.float64 on the CUDA device.
    
    Returns:
      A torch tensor with the cumulative sum computed elementwise.
    """
    n_elements = x.numel()       # total number of elements in x
    y = torch.empty_like(x)      # allocate output tensor
    # Launch a single kernel instance; note: this kernel assumes n_elements >= 1.
    grid = (1,)
    cumsum_triton[grid](x, y, n_elements)
    return y
