import numpy as np


def _normalize(v, eps=1e-15):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _frame_from_mu(mu):
    """
    Return rotation R with columns (e1, e2, mu),
    so canonical [x,y,z] maps to target via rowvec @ R.T.
    """
    mu = _normalize(mu)
    eye = np.eye(3)
    a = eye[np.argmin(np.abs(mu))]
    e1 = a - np.dot(a, mu) * mu
    e1 = _normalize(e1)
    e2 = np.cross(mu, e1)
    e2 = _normalize(e2)
    return np.stack([e1, e2, mu], axis=-1)


def sample_vmf_s2(mu, kappa, n, rng=None):
    """
    Exact, numerically stable vMF sampler on S^2.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float)
    if mu.shape != (3,):
        raise ValueError("mu must have shape (3,)")
    if kappa < 0:
        raise ValueError("kappa must be >= 0")

    if kappa == 0:
        x = rng.normal(size=(n, 3))
        return _normalize(x)

    # u ~ Uniform(0,1), phi ~ Uniform(0,2pi)
    u = rng.random(n)
    phi = 2.0 * np.pi * rng.random(n)

    # Stable inverse CDF for z in [-1,1]:
    # z = 1 + ( log(u + (1-u)e^{-2k}) ) / k
    log_term = np.logaddexp(np.log(u), np.log1p(-u) - 2.0 * kappa)
    z = 1.0 + log_term / kappa
    z = np.clip(z, -1.0, 1.0)

    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    pts_can = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)

    R = _frame_from_mu(mu)
    return pts_can @ R.T

def _rotation_from_gammas(gamma1, gamma2, gamma3):
    """
    Return rotation R with columns (gamma2, gamma3, gamma1),
    so canonical [x,y,z] maps to target via rowvec @ R.T.
    """
    g1 = _normalize(gamma1)
    g2 = gamma2 - np.dot(gamma2, g1) * g1
    g2 = _normalize(g2)
    g3 = np.cross(g1, g2)
    g3 = _normalize(g3)
    if np.dot(g3, gamma3) < 0:
        g2 = -g2
        g3 = -g3
    return np.stack([g2, g3, g1], axis=-1)


def sample_zlpkent_s2(gamma1, gamma2, gamma3, kappa, u, n, rng=None):
    """
    Exact, numerically stable sampler for the Kent-like ZLP flow on S^2.
    https://arxiv.org/abs/2510.04762
    """
    if rng is None:
        rng = np.random.default_rng()

    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if u <= 0:
        raise ValueError("u must be > 0")

    # 1) uniform base sample on S^2
    base = rng.normal(size=(n, 3))
    base = _normalize(base)

    x0 = base[:, 0]
    y0 = base[:, 1]
    z0 = np.clip(base[:, 2], -1.0, 1.0)

    # 2) stable forward Fisher zoom in canonical coordinates
    # z1 = 1 + ( log((1+z0) + (1-z0)e^{-2k}) - log 2 ) / k
    log_term = np.logaddexp(np.log1p(z0), np.log1p(-z0) - 2.0 * kappa)
    z1 = 1.0 + (log_term - np.log(2.0)) / kappa
    z1 = np.clip(z1, -1.0, 1.0)

    phi = np.arctan2(y0, x0)
    rho1 = np.sqrt(np.clip(1.0 - z1 * z1, 0.0, None))
    zoom = np.stack([rho1 * np.cos(phi), rho1 * np.sin(phi), z1], axis=-1)

    # 3) LP/project with A = diag(u, 1/u, 1)
    y1 = u * zoom[:, 0]
    y2 = (1.0 / u) * zoom[:, 1]
    y3 = zoom[:, 2]

    norm = np.sqrt(y1 * y1 + y2 * y2 + y3 * y3)
    can = np.stack([y1 / norm, y2 / norm, y3 / norm], axis=-1)

    # 4) rotate to fitted target frame
    R = _rotation_from_gammas(gamma1, gamma2, gamma3)
    return can @ R.T