from typing import Any

import torch
from torch import nn, Tensor
from torch_scatter.scatter import scatter
from tree_segmentation.extension._C import get_C_function, get_python_function, try_use_C_extension


class _get_mask_func(torch.autograd.Function):
    _forward_func = get_C_function('tree_seg_get_masks')
    _backward_func = get_C_function('tree_seg_get_masks_backward')

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        P, masks_2d, indices, masks_view, packed, eps = args
        ctx.packed = packed
        ctx.eps = eps
        masks_3d, weights = _get_mask_func._forward_func(P, masks_2d, indices, masks_view, packed, eps)
        ctx.save_for_backward(P, masks_2d, indices, masks_view, masks_3d, weights)
        return masks_3d

    @staticmethod
    def backward(ctx, *grad_outputs):
        P, masks_2d, indices, masks_view, masks_3d, weights = ctx.saved_tensors
        grad_P = _get_mask_func._backward_func(
            P,
            masks_2d,
            indices,
            masks_view,
            masks_3d,
            weights,
            grad_outputs[0],
            ctx.packed,
            ctx.eps
        )
        return grad_P, None, None, None, None, None


@try_use_C_extension(_get_mask_func.apply, "tree_seg_get_masks", 'tree_seg_get_masks_backward')
def get_mask(P: Tensor, masks_2d: Tensor, indices: Tensor, masks_view: Tensor, packed: bool, eps=1e-7):
    """
    Args:
        P (Tensor): shape [K, M], the probability of m-th 2D mask belong k-th 3D mask
        masks_2d (Tensor): shape [M_, F], the 2D masks
        indices (Tensor): shape [M], the view indices of 2D mask
        masks_view (Tensor): shape [V, F], whethere f-th faces can be viewed by v-th view
        packed (bool): whether masks_2d is packed
        eps (float):
    Returns:
        mask (Tensor): shape [K, F]
    """
    V = masks_view.shape[0]  # number of views
    # assert 0 <= indices.min() and indices.max() < V
    # torch.cuda.synchronize()
    weights = scatter(P, indices.long(), dim=1, dim_size=V, reduce='sum')  # shape: [K, V]
    # torch.cuda.synchronize()
    weights = (weights @ masks_view.float()).clamp_min(eps)  # shape: [K, F]
    # if self._masks_2d_packed:
    #     masks = (P @ self._masks_2d_sp) / weights
    # else:
    masks = (P @ masks_2d.float()) / weights
    return masks  # shape: [K, F]


def test_get_mask():
    from tree_segmentation.extension.utils import show_shape
    print()
    K, M = 10, 20
    F = 100
    V = 5
    device = torch.device('cuda')
    masks_2d = torch.randint(0, 2, (M, F), dtype=torch.bool, device=device)
    P = torch.randn(K, M, device=device).softmax(dim=-1)
    indices_view = torch.randint(0, V, (M,), dtype=torch.int32, device=device)
    # masks_view = torch.randint(0, 2, (V, F), dtype=torch.bool, device=device)
    masks_view = scatter(masks_2d.float(), indices_view.long(), dim=0, dim_size=V, reduce='sum') > 0
    print(f'P, masks_2d, indices_view, masks_view: {show_shape(P, masks_2d, indices_view, masks_view)}')
    eps = 1e-7
    py_func = get_python_function('get_mask')
    cu_func = _get_mask_func.apply

    P1 = P.clone().requires_grad_(True)
    P2 = P.clone().requires_grad_(True)
    out_py = py_func(P1, masks_2d, indices_view, masks_view, False, eps)
    out_cu = cu_func(P2, masks_2d, indices_view, masks_view, False, eps)
    error = (out_py - out_cu).abs()  # / out_py.abs().clamp(min=1e-4)
    print('max_error:', error.max())
    idx = error.argmax()
    print('python out:', out_py[idx // F:idx // F + 5, idx % F:idx % F + 5])
    print('cuda output', out_cu[idx // F:idx // F + 5, idx % F:idx % F + 5])

    g = torch.randn_like(out_py)
    torch.autograd.backward(out_py, g)
    torch.autograd.backward(out_cu, g)
    g_error = (P1.grad - P2.grad).abs()
    print('grad error:', g_error.max())
    idx = g_error.argmax()
    print('python out:', P1.grad[idx // M:idx // M + 5, idx % M:idx % M + 5])
    print('cuda output', P2.grad[idx // M:idx // M + 5, idx % M:idx % M + 5])

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=0)
    # ) as prof:
    #     for _ in range(100):
    #         with torch.profiler.record_function('py_fuc'):
    #             out_py = py_func(P, masks_2d, indices_view, masks_view, False, eps)
    #         with torch.profiler.record_function('cu_fuc'):
    #             out_cu = cu_func(P, masks_2d, indices_view, masks_view, False, eps)
    #         prof.step()
    # print(prof.key_averages().table(sort_by='cuda_time_total'))
    # prof.export_chrome_trace("trace.json")


def test_get_mask_packed():
    from tree_segmentation.extension.utils import show_shape
    print()
    K, M = 10, 20
    N = 15
    F = 100
    V = 5
    device = torch.device('cuda')
    masks_2d = torch.zeros((N, F), dtype=torch.int32, device=device)
    for i in range(N):
        s = M * i // N
        e = M * (i + 1) // N
        masks_2d[i] = torch.randint(s, e, (F,), dtype=torch.int32, device=device)
    masks_2d = (masks_2d + 1) * torch.rand((N, F), device=device).ge(0.3).int()
    print('masks_2d:', show_shape(masks_2d), masks_2d.max().item())
    indices = torch.nonzero(masks_2d)
    masks_2d_sp = torch.sparse.FloatTensor(
        torch.stack([masks_2d[indices[:, 0], indices[:, 1]] - 1, indices[:, 1]]),
        torch.ones(indices.shape[0], device=indices.device),
        [M, F],
    )
    print(masks_2d_sp)
    masks_2d_ = masks_2d_sp.to_dense()
    print(masks_2d_)
    P = torch.randn(K, M, device=device).softmax(dim=-1)
    indices_view = torch.randint(0, V, (M,), dtype=torch.int32, device=device)
    # masks_view = torch.randint(0, 2, (V, F), dtype=torch.bool, device=device)
    masks_view = scatter(masks_2d_.float(), indices_view.long(), dim=0, dim_size=V, reduce='sum') > 0
    print(f'P, masks_2d, indices_view, masks_view: {show_shape(P, masks_2d, indices_view, masks_view)}')
    eps = 1e-7
    py_func = get_python_function('get_mask')
    cu_func = _get_mask_func.apply

    P1 = P.clone().requires_grad_(True)
    P2 = P.clone().requires_grad_(True)
    out_py = py_func(P1, masks_2d_sp, indices_view, masks_view, True, eps)
    out_cu = cu_func(P2, masks_2d, indices_view, masks_view, True, eps)
    out_cu2 = cu_func(P2, masks_2d_.bool(), indices_view, masks_view, False, eps)
    error = (out_py - out_cu).abs()  # / out_py.abs().clamp(min=1e-4)
    print('max_error:', error.max().item(), (out_py - out_cu2).abs().max().item())
    idx = error.argmax()
    print('python out:', out_py[idx // F:idx // F + 5, idx % F:idx % F + 5])
    print('cuda output', out_cu[idx // F:idx // F + 5, idx % F:idx % F + 5])

    g = torch.randn_like(out_py)
    torch.autograd.backward(out_py, g)
    torch.autograd.backward(out_cu, g)
    g_error = (P1.grad - P2.grad).abs()
    print('grad error:', g_error.max())
    idx = g_error.argmax()
    print('python out:', P1.grad[idx // M:idx // M + 5, idx % M:idx % M + 5])
    print('cuda output', P2.grad[idx // M:idx // M + 5, idx % M:idx % M + 5])

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=0)
    # ) as prof:
    #     for _ in range(100):
    #         with torch.profiler.record_function('py_fuc'):
    #             out_py = py_func(P, masks_2d, indices_view, masks_view, False, eps)
    #         with torch.profiler.record_function('cu_fuc'):
    #             out_cu = cu_func(P, masks_2d, indices_view, masks_view, False, eps)
    #         prof.step()
    # print(prof.key_averages().table(sort_by='cuda_time_total'))
    # prof.export_chrome_trace("trace.json")
