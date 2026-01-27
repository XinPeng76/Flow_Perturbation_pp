import torch
import numpy as np
from torch.func import jvp, vjp
def finite_difference(fun, xT, eps,time_backward, back_coeff=0.001):
    sampN,ndim = xT.shape
    dx0 = back_coeff * eps
    time_forward = time_backward[::-1]
    x0 = fun(xT, time_backward)  # backward dynamics
    x0p = x0 + dx0
    x0n = x0 - dx0
    x0_all = torch.cat([x0p, x0n], dim=0)
    xT_all = torch.cat([xT, xT], dim=0)
    dxT_all = xT_all - fun(x0_all, time_forward)
    dxT = (dxT_all[:sampN] - dxT_all[sampN:])/2
    dx0 = dx0.reshape(sampN, -1)
    dxT = dxT.reshape(sampN, -1)
    return dx0, dxT, x0



def get_log_omega_FP(xT,eps,exact_dynamics,time_backward,get_energy,tmax=1.0,back_coeff = 0.001,fp_impl='finite'):
    sampN,ndim = xT.shape
    if fp_impl == 'finite':
        dx0, dxT, x0 = finite_difference(exact_dynamics, xT, eps,time_backward, back_coeff)
    else:
        raise NotImplementedError('Only finite difference implementation is available currently.')
    dx0_normsquare = torch.sum(dx0**2, dim=-1)
    dxT_normsquare = torch.sum(dxT**2, dim=-1)
    alpha_epsz_square = dxT_normsquare / dx0_normsquare # the streching factor
    #deltaSt =  - ndim/2 * torch.log(alpha_epsz_square) + np.log(gamma(ndim/2) / (np.pi**(ndim/2)*2))
    deltaSt =  - ndim/2 * torch.log(alpha_epsz_square)
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy(x0).reshape(-1)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

def get_log_omega_FP_batch(xT,eps,exact_dynamics,time_backward,get_energy,tmax=1.0,back_coeff = 0.001):
    """
    xT:           Tensor of shape (sampN, ndim)
    eps:          Tensor of shape (M, sampN, ndim)
    back_coeff:   float

    Returns:
      log_omega_mean: Tensor of shape (sampN,) — the mean over M log_omega values
      x0_mean:        Tensor of shape (sampN, ndim)
      ux_mean:        Tensor of shape (sampN,)
    """
    M, sampN, ndim = eps.shape
    time_forward = time_backward[::-1]
    # 1) Expand xT to shape (M, sampN, ndim), then flatten to (M*sampN, ndim)
    xT_exp = xT.unsqueeze(0).expand(M, sampN, ndim)       # (M, sampN, ndim)
    xT_flat = xT_exp.reshape(M * sampN, ndim)             # (M*sampN, ndim)
    eps_flat = eps.reshape(M * sampN, ndim)               # (M*sampN, ndim)
    
    # 2) Compute dx0 and prepare x0_flat
    dx0 = back_coeff * eps_flat                           # (M*sampN, ndim)
    x0 = exact_dynamics(xT, time_backward)                # (sampN, ndim)
    x0_flat = x0.unsqueeze(0).expand(M, sampN, ndim)      # (M, sampN, ndim)
    x0_flat = x0_flat.reshape(M * sampN, ndim)            # (M*sampN, ndim)
    
    # 3) Form positive and negative perturbations and concatenate
    x0p = x0_flat + dx0
    x0n = x0_flat - dx0
    x0_all = torch.cat([x0p, x0n], dim=0)                 # (2*M*sampN, ndim)
    xT_all = torch.cat([xT_flat, xT_flat], dim=0)         # (2*M*sampN, ndim)
    
    # 4) Push forward to get dxT_all, then recover dxT
    dxT_all = xT_all - exact_dynamics(x0_all, time_forward)  # (2*M*sampN, ndim)
    S = M * sampN
    dxT = (dxT_all[:S] - dxT_all[S:]) / 2                    # (M*sampN, ndim)
    
    # 5) Compute squared norms
    dx0_norm2 = torch.sum(dx0**2, dim=-1)                   # (M*sampN,)
    dxT_norm2 = torch.sum(dxT**2, dim=-1)                   # (M*sampN,)
    
    # 6) 计算 deltaSt_flat
    alpha2 = dxT_norm2 / dx0_norm2                           # (M*sampN,)
    alpha2 = alpha2.reshape(M, sampN)                      # (M, sampN)
    log_a = -ndim / 2 * torch.log(alpha2)
    m, _ = log_a.max(dim=0)

    deltaSt =  m + torch.log(torch.exp(log_a - m).mean(dim=0))  # (sampN,)
    
    # 7) log_omega
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi)
    ux = get_energy(x0).reshape(-1)
    log_omega = -ux + deltaSt + uz
    
    return log_omega, x0, ux

def get_log_omega_J(xT, eps, exact_dynamics_dSt, time_backward, get_energy, tmax=1.0, nnoise=1,method = 'RK4', if_fp = False, if_K_eps=True, eps_type = 'Gaussian', impl='finite'):
    ndim = xT.shape[-1]
    if if_K_eps:
        eps0 = eps.clone()
    else:
        eps0 = None
    #print('if_K_eps',if_K_eps)
    x0, deltaSt = exact_dynamics_dSt(xT, time_backward, method = method, nnoise = nnoise, if_fp = if_fp, eps=eps0, eps_type=eps_type, impl=impl)
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy(x0).reshape(-1)
    log_omega = -ux + deltaSt + uz
    
    return log_omega, x0, ux

def get_log_omega_SNF(xT, eps, exact_dynamics_dSt, time_backward, get_energy, langevin_layer, tmax=1.0, method = 'RK4'):
    ndim = xT.shape[-1]
    x0, deltaSt = exact_dynamics_dSt(xT, time_backward, method = method,nnoise = -1)
    x0_next, log_prob_ratio = langevin_layer(x0, None)
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy(x0_next).reshape(-1)
    log_omega = -ux + deltaSt + uz + log_prob_ratio
    
    return log_omega, x0, ux
