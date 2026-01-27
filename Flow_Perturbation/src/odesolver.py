from .scheme import Euler, RK2, RK4, Heun, Euler_dSt, RK2_dSt, RK4_dSt, Heun_dSt
import torch

def odesolver(func, z, t, t_next, method = 'RK4'):
    if (method == 'Euler'):
        z_next = Euler().step(func, t, t_next - t, z)
    elif (method == 'RK2'):
        z_next = RK2().step(func, t, t_next - t, z)
    elif (method == 'RK4'):
        z_next = RK4().step(func, t, t_next - t, z)
    elif (method == 'Heun'):
        z_next = Heun().step(func, t, t_next - t, z)
    else:
        print('error unsupported method passed')
        return
    return z_next

from torch.func import vjp, vmap, jacrev, jvp

from functools import partial

def get_jacobian_score(func, t, xt, eps=None):
    with torch.no_grad():
        score = func(xt, t)
    score = score.reshape(xt.shape[0], -1)
    jacobian_score = jacrev(func)
    v_jacobian_score = vmap(jacobian_score, in_dims=(0, None))
    jj_score_xt_new = v_jacobian_score(xt, t)
    div_xt_new = torch.einsum("...ii", jj_score_xt_new)
    return score, div_xt_new

def get_jacobian_score_batch(func, t, xt, eps=None, batch_size=100):
    # Use torch.no_grad to prevent gradient calculation during evaluation
    with torch.no_grad():
        score = func(xt, t)
    
    # Reshape the score to be a 2D tensor (batch_size, flattened_dim)
    score = score.reshape(xt.shape[0], -1)
    
    # Define the Jacobian function and vectorize it using vmap
    jacobian_score = jacrev(func)
    v_jacobian_score = vmap(jacobian_score, in_dims=(0, None))
    
    # Get the number of samples in xt (the batch size)
    n = xt.shape[0]
    jacobian_scores = []
    
    # Iterate over the data in batches
    for i in range(0, n, batch_size):
        # Select the batch from xt, the last batch may have fewer elements
        batch_xt = xt[i:i + batch_size]
        
        # Compute the Jacobian for the current batch
        jj_score_xt_new = v_jacobian_score(batch_xt, t)
        
        # Append the Jacobian result for the batch
        jacobian_scores.append(jj_score_xt_new)
    
    # Concatenate the Jacobian results from all batches
    jj_score_xt_new = torch.cat(jacobian_scores, dim=0)
    
    # Compute the trace by summing the diagonal elements of the Jacobian matrices
    div_xt_new = torch.einsum("...ii", jj_score_xt_new)
    
    return score, div_xt_new

def get_vjp_score(func, t, xt, eps):
    score, vjp_score = vjp(partial(func, t), xt)
    x_grad, = vjp_score(eps)
    return score, torch.sum(eps * x_grad, dim=-1)
def get_jvp_score(func, t, xt, eps):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    func_t = partial(func, t)

    # 计算 score = f(t, xt)
    score = func_t(xt)

    # JVP: (f(xt), J·eps)
    _, jvp_out = jvp(func_t, (xt,), (eps,))

    # Hutchinson inner product
    trace_est = torch.sum(eps * jvp_out, dim=-1)

    return score, trace_est

def get_vjp_score_mnoise(func, t, xt, eps):
    score, vjp_score = vjp(partial(func, t), xt)
    x_grad, = vmap(vjp_score)(eps)
    del xt, eps, x_grad, vjp_score
    torch.cuda.empty_cache()
    return score, torch.mean(torch.sum(eps * x_grad, dim=-1),dim=0)

def get_vjp_score_mnoise(func, t, xt, eps, batch_size=500):
    N = xt.shape[0]
    D = xt.shape[1]
    estimator = []
    scores_list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xt_chunk = xt[start:end]          # [B, D]
        eps_chunk = eps[:, start:end]        
        #print(eps_chunk.shape)

        # vjp
        score_chunk, vjp_fun = vjp(partial(func, t), xt_chunk)
        x_grad_chunk, = vmap(vjp_fun)(eps_chunk)

        # 累加 Hutchinson 估计
        estimator.append(torch.sum(eps_chunk * x_grad_chunk, dim=-1))

        # 如果你需要返回 scores，可以先存一个小列表，或者只返回最后均值
        scores_list.append(score_chunk.detach())

        # 清理显存
        del xt_chunk, eps_chunk, x_grad_chunk, vjp_fun
        torch.cuda.empty_cache()

    estimator = torch.cat(estimator, dim=1).mean(dim=0)
    print("estimator shape:", estimator.shape)
    scores = torch.cat(scores_list, dim=0)
    return scores, estimator
from torch.func import jvp, vmap
from functools import partial

def get_jvp_score_mnoise(func, t, xt, eps):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    # func_t(x) = func(t, x)
    func_t = partial(func, t)

    # 计算主输出：score(x)
    score = func_t(xt)

    # batched jvp: 对 eps 做 vmap
    def single_jvp(e):
        _, jvp_out = jvp(func_t, (xt,), (e,))
        return jvp_out

    jvp_out = vmap(single_jvp)(eps)  # [N, D]

    # Hutchinson trace estimator
    trace_est = torch.mean(torch.sum(eps * jvp_out, dim=-1), dim=0)

    return score, trace_est

def odesolver_Huch_dSt(score_func, xt, t, t_next, method = 'RK4',eps=None,nnoise = 1,eps_type = 'Rademacher',hutch_type = "vjp"):
    if nnoise == 1:
        if hutch_type == "jvp":
            func = lambda t, xt, eps: get_jvp_score(score_func, t, xt, eps)
        elif hutch_type == "vjp":
            func = lambda t, xt, eps: get_vjp_score(score_func, t, xt, eps)
        else:
            ValueError('unsupported hutch type')
        if eps is None:
            if eps_type == 'Rademacher':
                eps = torch.randint(0, 2, xt.shape, device=xt.device, dtype=xt.dtype) * 2 - 1
            elif eps_type == 'Gaussian':
                eps = torch.randn_like(xt, device=xt.device, dtype=xt.dtype)
            else:
                ValueError('unsupported eps type')
    elif nnoise > 1:
        if hutch_type == "jvp":
            func = lambda t, xt, eps: get_jvp_score_mnoise(score_func, t, xt, eps)
        elif hutch_type == "vjp":
            func = lambda t, xt, eps: get_vjp_score_mnoise(score_func, t, xt, eps)
        else:
            ValueError('unsupported hutch type')
        if eps is None:
            if eps_type == 'Rademacher':
                eps = (torch.randint(0, 2, (nnoise, xt.shape[0], xt.shape[1]), device=xt.device, dtype=xt.dtype) * 2 - 1)
            elif eps_type == 'Gaussian':
                eps = torch.randn((nnoise, xt.shape[0], xt.shape[1]), device=xt.device, dtype=xt.dtype)
            else:
                ValueError('unsupported eps type')
    else:
        ndim = xt.shape[-1]
        if ndim >=100:
            func = lambda t, xt, eps: get_jacobian_score_batch(score_func, t, xt, eps)
        else:
            func = lambda t, xt, eps: get_jacobian_score(score_func, t, xt, eps)
        eps = None
    if (method == 'Euler'):
        z_next , div_z_next = Euler_dSt().step(func, t, t_next - t, xt, eps)
    elif (method == 'RK2'):
        z_next , div_z_next = RK2_dSt().step(func, t, t_next - t, xt, eps)
    elif (method == 'RK4'):
        z_next , div_z_next = RK4_dSt().step(func, t, t_next - t, xt, eps)
    elif (method == 'Heun'):
        z_next , div_z_next = Heun_dSt().step(func, t, t_next - t, xt, eps)
    else:
        print('error unsupported method passed')
        return
    return z_next, div_z_next

def odesolver_FP_dSt(score_func, xt, t, t_next, method = 'RK4',eps=None,back_coeff=0.001):
    sampN = xt.shape[0]
    ndim = xt.shape[-1]
    dx0 = back_coeff * eps
    x0_next = odesolver(score_func, xt, t, t_next, method)
    x0p = x0_next + dx0
    x0n = x0_next - dx0
    x0_all = torch.cat([x0p, x0n], dim=0)
    xt_all = torch.cat([xt, xt], dim=0)
    dxT_all = xt_all - odesolver(score_func, x0_all, t_next, t, method)
    dxT = (dxT_all[:sampN] - dxT_all[sampN:])/2
    dx0 = dx0.reshape(sampN, -1)
    dxT = dxT.reshape(sampN, -1)

    dx0_normsquare = torch.sum(dx0**2, dim=-1)
    dxT_normsquare = torch.sum(dxT**2, dim=-1)

    alpha_epsz_square = dxT_normsquare / dx0_normsquare # the streching factor
    deltaSt =  - ndim/2 * torch.log(alpha_epsz_square)

    return x0_next, deltaSt

def odesolver_FP_dSt_jvp(score_func, xt, t, t_next, method='RK4', eps=None):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    """
    Compute deltaSt using a JVP-based estimate of ||(∂f^{-1}/∂x) ε|| / ||ε||,
    where f maps x_t at time t to x_{t_next}, and f^{-1} maps x_{t_next} back to x_t.

    Args:
        score_func: Callable taking (t, x) -> score, as used by `odesolver`.
        xt: Current state at time t, shape (batch, ndim).
        t: Current time scalar.
        t_next: Next time scalar.
        method: ODE integration scheme name.
        eps: Optional perturbation tensor same shape as xt. If None, random normal.

    Returns:
        x0_next: The mapped state at time t_next (same as in odesolver_FP_dSt).
        deltaSt: Tensor of shape (batch,) with the FP divergence contribution.
    """
    sampN = xt.shape[0]
    ndim = xt.shape[-1]
    if eps is None:
        eps = torch.randn_like(xt, device=xt.device, dtype=xt.dtype)

    # Map forward (backward in diffusion time): xt @ t -> x0_next @ t_next
    x0_next = odesolver(score_func, xt, t, t_next, method)

    # Define inverse map f^{-1}: x0 @ t_next -> xt @ t
    def inv_map(x0):
        return odesolver(score_func, x0, t_next, t, method)

    # Forward-mode JVP to get (∂f^{-1}/∂x) ε at x0_next
    _, dxT = jvp(inv_map, (x0_next,), (eps,))

    dx0 = eps.reshape(sampN, -1)
    dxT = dxT.reshape(sampN, -1)

    dx0_normsquare = torch.sum(dx0**2, dim=-1)
    dxT_normsquare = torch.sum(dxT**2, dim=-1)

    alpha_epsz_square = dxT_normsquare / dx0_normsquare
    deltaSt = - ndim/2 * torch.log(alpha_epsz_square)

    return x0_next, deltaSt

def odesolver_FP_dSt_vjp(score_func, xt, t, t_next, method='RK4', eps=None):
    """
    Compute deltaSt using a VJP-based estimate of ε (∂f^{-1}/∂x),
    where f maps x_t at time t to x_{t_next}, and f^{-1} maps x_{t_next} back to x_t.

    This computes the norm ratio ||ε J|| / ||ε|| with J = ∂f^{-1}/∂x evaluated at x0_next,
    by using a vector-Jacobian product (VJP) with respect to the output of f^{-1}.

    Args:
        score_func: Callable taking (t, x) -> score, as used by `odesolver`.
        xt: State at time t, shape (batch, ndim).
        t: Current time scalar.
        t_next: Next time scalar.
        method: ODE integration scheme name.
        eps: Optional left-multiplier for VJP (same shape as xt); if None, standard normal.

    Returns:
        x0_next: The mapped state at time t_next (same as in odesolver_FP_dSt).
        deltaSt: Tensor of shape (batch,) with the FP divergence contribution.
    """
    sampN = xt.shape[0]
    ndim = xt.shape[-1]
    if eps is None:
        eps = torch.randn_like(xt, device=xt.device, dtype=xt.dtype)

    # Forward map and define inverse map
    x0_next = odesolver(score_func, xt, t, t_next, method)

    def inv_map(x0):
        return odesolver(score_func, x0, t_next, t, method)

    # Vector-Jacobian product: given output cotangent eps, compute eps J
    _, vjp_fn = vjp(inv_map, x0_next)
    vjp_out = vjp_fn(eps)[0]

    eps_flat = eps.reshape(sampN, -1)
    vjp_flat = vjp_out.reshape(sampN, -1)

    denom = torch.sum(eps_flat**2, dim=-1)
    numer = torch.sum(vjp_flat**2, dim=-1)
    alpha_sq = numer / denom
    deltaSt = - ndim/2 * torch.log(alpha_sq)
    return x0_next, deltaSt
