import math
import torch


def _normalize(v, eps=1e-12):
    return v / torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)


def _as_batch_vector(x, B, name, device, dtype):
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError(f"{name} must have shape [B] or [B,1], got {tuple(x.shape)}")
    if x.shape[0] != B:
        raise ValueError(f"{name} must have leading dimension B={B}, got {tuple(x.shape)}")
    return x


def _stable_log_sinh(x):
    out = torch.empty_like(x)
    small = x < 20.0
    out[small] = torch.log(torch.sinh(x[small]))
    xs = x[~small]
    out[~small] = xs - math.log(2.0) + torch.log1p(-torch.exp(-2.0 * xs))
    return out


def _stable_one_minus_kappa_coth(kappa):
    out = torch.empty_like(kappa)
    small = kappa < 1e-4
    ks = kappa[small]
    out[small] = -(ks**2) / 3.0 + (ks**4) / 45.0 - 2.0 * (ks**6) / 945.0
    large = ~small
    kl = kappa[large]
    out[large] = 1.0 - kl / torch.tanh(kl)
    return out


def _resultant_to_kappa_approx_3d(r):
    r = torch.clamp(r, 1e-8, 1.0 - 1e-8)
    return (3.0 * r - r**3) / (1.0 - r**2)


def tangent_basis(gamma1):
    gamma1 = _normalize(gamma1)
    eye = torch.eye(3, device=gamma1.device, dtype=gamma1.dtype)

    idx = torch.argmin(torch.abs(gamma1), dim=1)
    a = eye[idx]

    u = a - (a * gamma1).sum(dim=1, keepdim=True) * gamma1
    u = _normalize(u)
    v = _normalize(torch.cross(gamma1, u, dim=1))
    return torch.stack([u, v], dim=-1)  # [B,3,2]


def rotmat_from_quat_raw(qraw):
    """
    qraw: [B,4], interpreted as (w,x,y,z), unnormalized
    returns R: [B,3,3]
    """
    q = _normalize(qraw)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)], dim=-1),
    ], dim=-2)
    return R


def quat_raw_from_rotmat(R):
    B = R.shape[0]
    q = torch.empty((B, 4), dtype=R.dtype, device=R.device)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    m0 = trace > 0
    if m0.any():
        t = torch.sqrt(1.0 + trace[m0]) * 2.0
        q[m0, 0] = 0.25 * t
        q[m0, 1] = (R[m0, 2, 1] - R[m0, 1, 2]) / t
        q[m0, 2] = (R[m0, 0, 2] - R[m0, 2, 0]) / t
        q[m0, 3] = (R[m0, 1, 0] - R[m0, 0, 1]) / t

    m1 = (~m0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m1.any():
        t = torch.sqrt(1.0 + R[m1, 0, 0] - R[m1, 1, 1] - R[m1, 2, 2]) * 2.0
        q[m1, 0] = (R[m1, 2, 1] - R[m1, 1, 2]) / t
        q[m1, 1] = 0.25 * t
        q[m1, 2] = (R[m1, 0, 1] + R[m1, 1, 0]) / t
        q[m1, 3] = (R[m1, 0, 2] + R[m1, 2, 0]) / t

    m2 = (~m0) & (~m1) & (R[:, 1, 1] > R[:, 2, 2])
    if m2.any():
        t = torch.sqrt(1.0 + R[m2, 1, 1] - R[m2, 0, 0] - R[m2, 2, 2]) * 2.0
        q[m2, 0] = (R[m2, 0, 2] - R[m2, 2, 0]) / t
        q[m2, 1] = (R[m2, 0, 1] + R[m2, 1, 0]) / t
        q[m2, 2] = 0.25 * t
        q[m2, 3] = (R[m2, 1, 2] + R[m2, 2, 1]) / t

    m3 = (~m0) & (~m1) & (~m2)
    if m3.any():
        t = torch.sqrt(1.0 + R[m3, 2, 2] - R[m3, 0, 0] - R[m3, 1, 1]) * 2.0
        q[m3, 0] = (R[m3, 1, 0] - R[m3, 0, 1]) / t
        q[m3, 1] = (R[m3, 0, 2] + R[m3, 2, 0]) / t
        q[m3, 2] = (R[m3, 1, 2] + R[m3, 2, 1]) / t
        q[m3, 3] = 0.25 * t

    return q


def _kentlike_u_from_raw(raw_u, kappa, embedding_dim=3.0):
    D = torch.as_tensor(embedding_dim, dtype=kappa.dtype, device=kappa.device)
    upper_ln = 0.5 * torch.log1p(kappa / D)

    safe_log_u = torch.where(
        upper_ln > 1e-14,
        raw_u / torch.sqrt(1.0 + (raw_u / upper_ln) ** 2),
        torch.zeros_like(raw_u),
    )
    u = torch.exp(safe_log_u)
    return u, safe_log_u, upper_ln


def _raw_from_target_u(u_target, kappa, embedding_dim=3.0):
    D = torch.as_tensor(embedding_dim, dtype=kappa.dtype, device=kappa.device)
    upper_ln = 0.5 * torch.log1p(kappa / D)

    y = torch.log(torch.clamp(u_target, min=1e-12))
    y = torch.where(
        upper_ln > 1e-14,
        torch.clamp(y, min=-0.999 * upper_ln, max=0.999 * upper_ln),
        torch.zeros_like(y),
    )

    denom = torch.sqrt(torch.clamp(
        1.0 - (y / torch.clamp(upper_ln, min=1e-14)) ** 2,
        min=1e-12,
    ))
    raw = torch.where(
        upper_ln > 1e-14,
        y / denom,
        torch.zeros_like(y),
    )
    return raw


def _quat_raw_grad_from_rotgrad(qraw, gradR):
    """
    gradR: [B,3,3] = dL/dR
    returns grad wrt raw quaternion [B,4]
    """
    s = qraw.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q = qraw / s
    w, x, y, z = q.unbind(dim=-1)

    zeros = torch.zeros_like(w)

    dRdw = torch.stack([
        torch.stack([zeros, -2*z,  2*y], dim=-1),
        torch.stack([ 2*z, zeros, -2*x], dim=-1),
        torch.stack([-2*y,  2*x, zeros], dim=-1),
    ], dim=-2)

    dRdx = torch.stack([
        torch.stack([zeros,  2*y,  2*z], dim=-1),
        torch.stack([ 2*y, -4*x, -2*w], dim=-1),
        torch.stack([ 2*z,  2*w, -4*x], dim=-1),
    ], dim=-2)

    dRdy = torch.stack([
        torch.stack([-4*y,  2*x,  2*w], dim=-1),
        torch.stack([ 2*x, zeros,  2*z], dim=-1),
        torch.stack([-2*w,  2*z, -4*y], dim=-1),
    ], dim=-2)

    dRdz = torch.stack([
        torch.stack([-4*z, -2*w,  2*x], dim=-1),
        torch.stack([ 2*w, -4*z,  2*y], dim=-1),
        torch.stack([ 2*x,  2*y, zeros], dim=-1),
    ], dim=-2)

    gw = (gradR * dRdw).sum(dim=(1, 2))
    gx = (gradR * dRdx).sum(dim=(1, 2))
    gy = (gradR * dRdy).sum(dim=(1, 2))
    gz = (gradR * dRdz).sum(dim=(1, 2))

    gq = torch.stack([gw, gx, gy, gz], dim=1)
    proj = (gq * q).sum(dim=1, keepdim=True)
    grad_raw = (gq - q * proj) / s
    return grad_raw


def zlpkent_value_grads_per_item(X, log_kappa, raw_u, qraw, need_q_grad=True):
    """
    Exact value and exact gradients for the Kent-like ZLP fit (https://arxiv.org/abs/2510.04762).

    X: [B,N,3]
    log_kappa: [B]
    raw_u: [B]
    qraw: [B,4]

    returns
    -------
    ll: [B]
    kappa: [B]
    u: [B]
    g_log_kappa: [B]
    g_raw_u: [B]
    g_qraw: [B,4] or None
    """
    B, N, _ = X.shape
    log_kappa = log_kappa.reshape(B)
    raw_u = raw_u.reshape(B)

    kappa = torch.exp(log_kappa).clamp_min(1e-10)

    D = torch.as_tensor(3.0, dtype=X.dtype, device=X.device)
    L = 0.5 * torch.log1p(kappa / D)
    denom = torch.sqrt(L * L + raw_u * raw_u).clamp_min(1e-15)

    safe_log_u = torch.where(
        L > 1e-14,
        raw_u * L / denom,
        torch.zeros_like(raw_u),
    )
    u = torch.exp(safe_log_u)
    inv_u = 1.0 / u

    R = rotmat_from_quat_raw(qraw)
    Y = torch.einsum("bij,bnj->bni", R.transpose(1, 2), X)

    y1 = Y[:, :, 0]
    y2 = Y[:, :, 1]
    y3 = Y[:, :, 2]

    A = y1 * y1 * (inv_u[:, None] ** 2)
    Bv = y2 * y2 * (u[:, None] ** 2)
    C = y3 * y3
    r2 = (A + Bv + C).clamp_min(1e-15)
    r = torch.sqrt(r2)
    z3 = y3 / r

    log_norm = torch.log(kappa) - math.log(4.0 * math.pi) - _stable_log_sinh(kappa)
    ll = N * log_norm + kappa * z3.sum(dim=1) - 1.5 * torch.log(r2).sum(dim=1)

    # dℓ / d(safe_log_u)
    diff = Bv - A
    g_safe_log_u = -torch.sum((kappa[:, None] * z3 + 3.0) * diff / r2, dim=1)

    # chain from safe_log_u(raw_u, kappa)
    dl_dg = torch.where(
        L > 1e-14,
        (L ** 3) / (denom ** 3),
        torch.zeros_like(L),
    )
    dl_dL = torch.where(
        L > 1e-14,
        (raw_u ** 3) / (denom ** 3),
        torch.zeros_like(L),
    )
    dL_dloga = kappa / (2.0 * (D + kappa))

    g_log_kappa_direct = N * _stable_one_minus_kappa_coth(kappa) + kappa * z3.sum(dim=1)
    g_log_kappa = g_log_kappa_direct + g_safe_log_u * dl_dL * dL_dloga
    g_raw_u = g_safe_log_u * dl_dg

    g_qraw = None
    if need_q_grad:
        common = (kappa[:, None] * z3 + 3.0) / r2
        gy1 = -(inv_u[:, None] ** 2) * y1 * common
        gy2 = -(u[:, None] ** 2) * y2 * common
        gy3 = kappa[:, None] * (A + Bv) / (r2 * r) - 3.0 * y3 / r2

        GY = torch.stack([gy1, gy2, gy3], dim=-1)
        gradR = torch.einsum("bnj,bni->bji", X, GY)
        g_qraw = _quat_raw_grad_from_rotgrad(qraw, gradR)

    return ll, kappa, u, g_log_kappa, g_raw_u, g_qraw


def _masked_adam_step_(param, grad, m, v, step_row, active, lr, beta1, beta2, eps):
    if grad is None:
        return

    if param.ndim == 1:
        m[active] = beta1 * m[active] + (1.0 - beta1) * grad[active]
        v[active] = beta2 * v[active] + (1.0 - beta2) * (grad[active] * grad[active])

        t = step_row[active].to(param.dtype)
        mhat = m[active] / (1.0 - beta1 ** t)
        vhat = v[active] / (1.0 - beta2 ** t)

        # ascent
        param[active] += lr * mhat / (torch.sqrt(vhat) + eps)

    elif param.ndim == 2:
        m[active] = beta1 * m[active] + (1.0 - beta1) * grad[active]
        v[active] = beta2 * v[active] + (1.0 - beta2) * (grad[active] * grad[active])

        t = step_row[active].to(param.dtype).unsqueeze(1)
        mhat = m[active] / (1.0 - beta1 ** t)
        vhat = v[active] / (1.0 - beta2 ** t)

        # ascent
        param[active] += lr * mhat / (torch.sqrt(vhat) + eps)

    else:
        raise ValueError(f"Unsupported param ndim={param.ndim}")


def _fastpath_hessian_fd(X, a, g, qraw, h_scale=1e-4):
    """
    Finite-difference Hessian of the exact gradients for fast_path.

    Returns
    -------
    ll: [M]
    grad: [M,2]
    H: [M,2,2]
    """
    M = a.shape[0]

    ll, _, _, ga, gg, _ = zlpkent_value_grads_per_item(X, a, g, qraw, need_q_grad=False)
    grad = torch.stack([ga, gg], dim=1)

    ha = h_scale * (1.0 + a.abs())
    hg = h_scale * (1.0 + g.abs())

    _, _, _, ga_p, gg_p, _ = zlpkent_value_grads_per_item(X, a + ha, g, qraw, need_q_grad=False)
    _, _, _, ga_m, gg_m, _ = zlpkent_value_grads_per_item(X, a - ha, g, qraw, need_q_grad=False)

    _, _, _, ga_gp, gg_gp, _ = zlpkent_value_grads_per_item(X, a, g + hg, qraw, need_q_grad=False)
    _, _, _, ga_gm, gg_gm, _ = zlpkent_value_grads_per_item(X, a, g - hg, qraw, need_q_grad=False)

    H11 = (ga_p - ga_m) / (2.0 * ha)
    H21 = (gg_p - gg_m) / (2.0 * ha)

    H12 = (ga_gp - ga_gm) / (2.0 * hg)
    H22 = (gg_gp - gg_gm) / (2.0 * hg)

    H12s = 0.5 * (H12 + H21)

    H = torch.zeros((M, 2, 2), dtype=a.dtype, device=a.device)
    H[:, 0, 0] = H11
    H[:, 0, 1] = H12s
    H[:, 1, 0] = H12s
    H[:, 1, 1] = H22

    return ll, grad, H


def fit_zlpkent_batch_quat(
    X,
    mu_vmf=None,
    kappa_vmf=None,
    max_iter=300,
    lr=0.03,
    beta1=0.9,
    beta2=0.999,
    adam_eps=1e-8,
    grad_tol=1e-5,
    rel_obj_tol=1e-8,
    fast_path=False,
    newton_fd_eps=1e-4,
    newton_armijo_c1=1e-4,
    return_numpy=True,
):
    """
    Batched Kent-like ZLP ML fit on S^2.
    https://arxiv.org/abs/2510.04762

    fast_path:
      uses per-row damped Newton on (log_kappa, raw_u), keeping the frame fixed.

    full path:
      uses exact gradients + custom masked Adam on (log_kappa, raw_u, qraw).
    """
    
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must have shape [B,N,3], got {tuple(X.shape)}")
    

    device=X.device
    dtype=X.dtype
    B, N, _ = X.shape

    xbar = X.mean(dim=1)
    S = torch.einsum("bni,bnj->bij", X, X) / N

    if mu_vmf is None:
        gamma1_0 = _normalize(xbar)
    else:
        mu_vmf = torch.as_tensor(mu_vmf, dtype=dtype, device=device)
        if mu_vmf.shape != (B, 3):
            raise ValueError(f"mu_vmf must have shape [B,3], got {tuple(mu_vmf.shape)}")
        gamma1_0 = _normalize(mu_vmf)

    U = tangent_basis(gamma1_0)
    T = torch.matmul(U.transpose(1, 2), torch.matmul(S, U))

    evals, evecs = torch.linalg.eigh(T)
    lam_min = evals[:, 0].clamp_min(1e-10)
    lam_max = evals[:, 1].clamp_min(1e-10)

    gamma2_0 = torch.einsum("bij,bj->bi", U, evecs[:, :, 1])
    gamma3_0 = torch.einsum("bij,bj->bi", U, evecs[:, :, 0])

    handed = torch.einsum("bi,bi->b", torch.cross(gamma2_0, gamma3_0, dim=1), gamma1_0)
    flip = handed < 0
    gamma3_0[flip] = -gamma3_0[flip]

    R0 = torch.stack([gamma2_0, gamma3_0, gamma1_0], dim=-1)
    q0 = quat_raw_from_rotmat(R0)

    if kappa_vmf is None:
        r = xbar.norm(dim=1).clamp(1e-8, 1.0 - 1e-8)
        kappa_iso0 = _resultant_to_kappa_approx_3d(r).clamp_min(1e-4)
    else:
        kappa_iso0 = _as_batch_vector(kappa_vmf, B, "kappa_vmf", device, dtype).clamp_min(1e-4)

    u0 = (lam_max / lam_min).pow(0.25).clamp_min(1.0)
    u0_sq = u0 * u0
    kappa0 = 0.5 * (u0_sq + 1.0 / u0_sq) * kappa_iso0
    kappa0 = kappa0.clamp_min(1e-4)
    raw_u0 = _raw_from_target_u(u0, kappa0, embedding_dim=3.0)

    log_kappa = torch.log(kappa0).clone()
    raw_u = raw_u0.clone()
    qraw = q0.clone()

    converged = torch.zeros(B, dtype=torch.bool, device=device)
    prev_ll = torch.full((B,), float("-inf"), dtype=dtype, device=device)

    num_iter=0
    if fast_path:
        for _ in range(max_iter):
            active = ~converged
            if not torch.any(active):
                break

            idx = torch.where(active)[0]

            Xa = X[idx]
            aa = log_kappa[idx]
            gg = raw_u[idx]
            qa = qraw[idx]  # fixed

            ll, grad, H = _fastpath_hessian_fd(Xa, aa, gg, qa, h_scale=newton_fd_eps)

            gmax = grad.abs().max(dim=1).values

            H11 = H[:, 0, 0]
            H12 = H[:, 0, 1]
            H22 = H[:, 1, 1]
            det = H11 * H22 - H12 * H12

            # valid Newton step for local maximum: negative definite Hessian
            negdef = (H11 < 0.0) & (H22 < 0.0) & (det > 1e-12)

            g1 = grad[:, 0]
            g2 = grad[:, 1]

            p_newton_1 = (-H22 * g1 + H12 * g2) / torch.clamp(det, min=1e-12)
            p_newton_2 = ( H12 * g1 - H11 * g2) / torch.clamp(det, min=1e-12)
            p_newton = torch.stack([p_newton_1, p_newton_2], dim=1)

            p_grad = grad / torch.clamp(grad.norm(dim=1, keepdim=True), min=1e-12)

            ascent_newton = (grad * p_newton).sum(dim=1) > 0.0
            use_newton = negdef & ascent_newton

            p = torch.where(use_newton[:, None], p_newton, p_grad)

            slope = (grad * p).sum(dim=1)

            alpha = torch.ones_like(slope)
            a_try = aa.clone()
            g_try = gg.clone()
            ll_try = ll.clone()

            for _ls in range(20):
                a_try = aa + alpha * p[:, 0]
                g_try = gg + alpha * p[:, 1]

                ll_try, _, _, _, _, _ = zlpkent_value_grads_per_item(
                    Xa, a_try, g_try, qa, need_q_grad=False
                )

                rhs = ll + newton_armijo_c1 * alpha * slope
                ok = ll_try >= rhs
                if torch.all(ok):
                    break
                alpha = torch.where(ok, alpha, 0.5 * alpha)

            log_kappa[idx] = a_try
            raw_u[idx] = g_try
            log_kappa.clamp_(min=-12.0, max=40.0)

            rel_gain = (ll_try - ll) / (1.0 + ll.abs().clamp_min(1.0))

            just_done = ((gmax < grad_tol) | ((rel_gain >= 0.0) & (rel_gain < rel_obj_tol))) 
            #just_done = ((gmax < grad_tol) | ( (rel_gain < rel_obj_tol))) 
            #print("ONV .. ",rel_gain, gmax)
            converged[idx[just_done]] = True
            prev_ll[idx] = ll_try

            num_iter+=1

    else:
        step_row = torch.zeros(B, dtype=torch.long, device=device)

        m_log_kappa = torch.zeros_like(log_kappa)
        v_log_kappa = torch.zeros_like(log_kappa)

        m_raw_u = torch.zeros_like(raw_u)
        v_raw_u = torch.zeros_like(raw_u)

        m_qraw = torch.zeros_like(qraw)
        v_qraw = torch.zeros_like(qraw)


        for _ in range(max_iter):
            active = ~converged
            if not torch.any(active):
                break

            ll, _, _, g_log_kappa, g_raw_u, g_qraw = zlpkent_value_grads_per_item(
                X, log_kappa, raw_u, qraw, need_q_grad=True
            )

            grad_blocks = [g_log_kappa.abs(), g_raw_u.abs(), g_qraw.abs().max(dim=1).values]
            gmax = torch.stack(grad_blocks, dim=1).max(dim=1).values

            step_row[active] += 1

            _masked_adam_step_(
                log_kappa, g_log_kappa, m_log_kappa, v_log_kappa, step_row,
                active, lr, beta1, beta2, adam_eps
            )
            _masked_adam_step_(
                raw_u, g_raw_u, m_raw_u, v_raw_u, step_row,
                active, lr, beta1, beta2, adam_eps
            )
            _masked_adam_step_(
                qraw, g_qraw, m_qraw, v_qraw, step_row,
                active, lr, beta1, beta2, adam_eps
            )

            log_kappa.clamp_(min=-12.0, max=15.0)

            ll_new, _, _, _, _, _ = zlpkent_value_grads_per_item(
                X, log_kappa, raw_u, qraw, need_q_grad=False
            )
            rel_gain = (ll_new - prev_ll) / (1.0 + prev_ll.abs().clamp_min(1.0))

            just_done = (gmax < grad_tol) | ((rel_gain >= 0.0) & (rel_gain < rel_obj_tol))
            converged |= just_done
            prev_ll = ll_new.clone()

    if fast_path:
        gamma1 = gamma1_0
        gamma2 = gamma2_0
        gamma3 = gamma3_0
    else:
        R = rotmat_from_quat_raw(qraw)
        gamma2 = R[:, :, 0]
        gamma3 = R[:, :, 1]
        gamma1 = R[:, :, 2]

    ll, kappa, u, _, _, _ = zlpkent_value_grads_per_item(
        X, log_kappa, raw_u, qraw, need_q_grad=False
    )
    _, safe_log_u, upper_ln = _kentlike_u_from_raw(raw_u, kappa, embedding_dim=3.0)

    out = {
        "kappa": kappa.detach(),
        "u": u.detach(),
        "safe_log_u": safe_log_u.detach(),
        "u_log_upper_bound": upper_ln.detach(),
        "quat_raw": qraw.detach(),
        "gamma1": gamma1.detach(),
        "gamma2": gamma2.detach(),
        "gamma3": gamma3.detach(),
        "loglik": ll.detach(),
        "converged": converged.detach(),
        "fast_path": fast_path,
        "xbar": xbar.detach(),
        "S": S.detach(),
        "lam_tangent_major_init": lam_max.detach(),
        "lam_tangent_minor_init": lam_min.detach(),
        "init_gamma1": gamma1_0.detach(),
        "init_gamma2": gamma2_0.detach(),
        "init_gamma3": gamma3_0.detach(),
        "init_kappa_iso": kappa_iso0.detach(),
        "init_kappa": kappa0.detach(),
        "init_u": u0.detach(),
        "num_iter": num_iter
    }

    if not fast_path:
        out["adam_steps_per_row"] = step_row.detach()

    if return_numpy:
        return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in out.items()}
    return out