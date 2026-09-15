"""Self-contained rigid-body dynamics for the 7-DOF Franka Emika Panda.

Provides the mass matrix M(q), gravity torque g(q), the Coriolis/centrifugal
matrix C(q, qd) (Christoffel factorization) and the transpose product
C(q, qd)^T qd. These are the quantities needed by the generalized-momentum
collision observer of

    A. De Luca, A. Albu-Schaffer, S. Haddadin, G. Hirzinger,
    "Collision Detection and Safe Reaction with the DLR-III Lightweight
    Manipulator Arm," IEEE/RSJ IROS 2006, pp. 1623-1630.

whose residual is

    r = K_I [ p - integral( tau + C^T qd - g + r ) dt - p(0) ],   p = M(q) qd

Everything here is plain NumPy so it runs inside the CoppeliaSim/PyRep `aha`
environment with no extra dependency. The kinematic (standard Denavit-Hartenberg)
and inertial parameters below are taken verbatim from roboticstoolbox's
``rtb.models.DH.Panda`` so that M, C and g match that (validated) model
element-for-element; see the companion validation in the collision detector's
tests. Standard-DH recursive Newton-Euler (Craig / Featherstone conventions for
the standard-DH parameterisation) is used throughout.
"""

from aha_publish import paths

import numpy as np

# Standard-DH links, exactly as rtb.models.DH.Panda. Each link:
#   a, d, alpha, offset : standard-DH parameters (theta_i = q_i + offset)
#   m                   : link mass
#   r                   : centre of mass in the link frame
#   I                   : 3x3 inertia tensor about the centre of mass
_LINKS = [
    dict(a=0.0, d=0.333, alpha=0.0, offset=0.0,
         m=4.970684, r=[0.0, 0.0, 0.0],
         I=[[0.70337, -0.000139, 0.006772], [-0.000139, 0.70661, 0.019169], [0.006772, 0.019169, 0.009117]]),
    dict(a=0.0, d=0.0, alpha=-1.5707963267948966, offset=0.0,
         m=0.646926, r=[0.0, 0.0, 0.0],
         I=[[0.007962, -0.003925, 0.010254], [-0.003925, 0.02811, 0.000704], [0.010254, 0.000704, 0.025995]]),
    dict(a=0.0, d=0.316, alpha=1.5707963267948966, offset=0.0,
         m=3.228604, r=[0.0, 0.0, 0.0],
         I=[[0.037242, -0.004761, -0.011396], [-0.004761, 0.036155, -0.012805], [-0.011396, -0.012805, 0.01083]]),
    dict(a=0.0825, d=0.0, alpha=1.5707963267948966, offset=0.0,
         m=3.587895, r=[0.0, 0.0, 0.0],
         I=[[0.025853, 0.007796, -0.001332], [0.007796, 0.019552, 0.008641], [-0.001332, 0.008641, 0.028323]]),
    dict(a=-0.0825, d=0.384, alpha=-1.5707963267948966, offset=0.0,
         m=1.225946, r=[0.0, 0.0, 0.0],
         I=[[0.035549, -0.002117, -0.004037], [-0.002117, 0.029474, 0.000229], [-0.004037, 0.000229, 0.008627]]),
    dict(a=0.0, d=0.0, alpha=1.5707963267948966, offset=0.0,
         m=1.666555, r=[0.0, 0.0, 0.0],
         I=[[0.001964, 0.000109, -0.001158], [0.000109, 0.004354, 0.000341], [-0.001158, 0.000341, 0.005433]]),
    dict(a=0.088, d=0.107, alpha=1.5707963267948966, offset=0.0,
         m=0.735522, r=[0.0, 0.0, 0.0],
         I=[[0.012516, -0.000428, -0.001196], [-0.000428, 0.010027, -0.000741], [-0.001196, -0.000741, 0.004815]]),
]

N = len(_LINKS)
_GRAVITY = np.array([0.0, 0.0, -9.81])

# Pre-parsed per-link arrays.
_A = np.array([l['a'] for l in _LINKS])
_D = np.array([l['d'] for l in _LINKS])
_ALPHA = np.array([l['alpha'] for l in _LINKS])
_OFFSET = np.array([l['offset'] for l in _LINKS])
_M = np.array([l['m'] for l in _LINKS])
_R = np.array([l['r'] for l in _LINKS], dtype=float)      # (N,3)
_I = np.array([l['I'] for l in _LINKS], dtype=float)      # (N,3,3)


def _link_rotations_positions(q):
    """Per-joint modified-DH rotation R_i (frame i expressed in i-1) and origin p_i.

    Modified DH (Craig): T_i = Rx(alpha_{i-1}) Tx(a_{i-1}) Rz(theta_i) Tz(d_i),
    where the stored per-link (a, alpha) are (a_{i-1}, alpha_{i-1}) and
    theta_i = q_i + offset_i. rtb.models.DH.Panda is a modified-DH robot.
    Returns R_i (3x3, maps a frame-i vector into frame i-1) and p_i (origin of
    frame i expressed in frame i-1).
    """
    Rs, ps = [], []
    for i in range(N):
        th = q[i] + _OFFSET[i]
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(_ALPHA[i]), np.sin(_ALPHA[i])
        R = np.array([
            [ct,        -st,      0.0],
            [st * ca,  ct * ca,   -sa],
            [st * sa,  ct * sa,    ca],
        ])
        p = np.array([_A[i], -sa * _D[i], ca * _D[i]])
        Rs.append(R)
        ps.append(p)
    return Rs, ps


def rne(q, qd, qdd, gravity=None):
    """Recursive Newton-Euler inverse dynamics for the modified-DH Panda.

    Returns the joint torque vector tau such that
        tau = M(q) qdd + C(q, qd) qd + g(q)
    with g(q) produced by the base gravitational acceleration `gravity`
    (default: Panda's [0, 0, -9.81]). All joints revolute (sigma=0). Craig's
    modified-DH outward/inward recursion.
    """
    q = np.asarray(q, float).reshape(-1)
    qd = np.asarray(qd, float).reshape(-1)
    qdd = np.asarray(qdd, float).reshape(-1)
    grav = _GRAVITY if gravity is None else np.asarray(gravity, float).reshape(-1)

    Rs, ps = _link_rotations_positions(q)
    z = np.array([0.0, 0.0, 1.0])

    # Outward recursion: angular vel/acc and linear acc of each frame + COM,
    # every quantity expressed in its own link frame i. R.T maps an (i-1)-frame
    # vector into frame i (R maps frame i -> i-1).
    w = np.zeros(3)                 # angular velocity of frame i-1 (in i-1)
    wd = np.zeros(3)                # angular acceleration
    vd = -grav                      # linear accel of base origin (gravity trick)
    Fs, Ns = [], []                 # net force / moment at each COM (frame i)
    for i in range(N):
        RT = Rs[i].T               # (i-1) -> i
        p = ps[i]                  # origin of frame i in frame i-1
        w_i = RT @ w + qd[i] * z
        wd_i = RT @ wd + np.cross(RT @ w, qd[i] * z) + qdd[i] * z
        vd_i = RT @ (vd + np.cross(wd, p) + np.cross(w, np.cross(w, p)))
        # COM linear acceleration (COM offset r_i expressed in frame i).
        r_i = _R[i]
        vcd = vd_i + np.cross(wd_i, r_i) + np.cross(w_i, np.cross(w_i, r_i))
        Fs.append(_M[i] * vcd)
        Ii = _I[i]
        Ns.append(Ii @ wd_i + np.cross(w_i, Ii @ w_i))
        w, wd, vd = w_i, wd_i, vd_i

    # Inward recursion: propagate forces/moments to joints -> torques. R_next
    # (rotation of frame i+1 in frame i) maps an (i+1)-frame vector into frame i.
    f = np.zeros(3)                 # force exerted on link i by link i+1 (frame i)
    n = np.zeros(3)                 # moment likewise
    tau = np.zeros(N)
    for i in range(N - 1, -1, -1):
        if i + 1 < N:
            R_next = Rs[i + 1]     # frame i+1 -> i
            p_next = ps[i + 1]     # origin of frame i+1 in frame i
        else:
            R_next = np.eye(3)
            p_next = np.zeros(3)
        r_i = _R[i]
        f_next = R_next @ f
        n = (Ns[i]
             + R_next @ n
             + np.cross(r_i, Fs[i])
             + np.cross(p_next, f_next))
        f = f_next + Fs[i]
        tau[i] = n @ z             # revolute: projection on joint axis z
    return tau


def gravity_torque(q, gravity=None):
    """g(q): joint torques required to hold against gravity (qd = qdd = 0)."""
    return rne(q, np.zeros(N), np.zeros(N), gravity=gravity)


def coriolis_vector(q, qd):
    """C(q, qd) qd: the Coriolis/centrifugal joint-torque vector (no gravity)."""
    return rne(q, qd, np.zeros(N), gravity=np.zeros(3))


def mass_matrix(q):
    """M(q) via unit-acceleration columns of RNEA (gravity off, qd = 0)."""
    q = np.asarray(q, float).reshape(-1)
    M = np.zeros((N, N))
    zero = np.zeros(N)
    for i in range(N):
        e = np.zeros(N)
        e[i] = 1.0
        M[:, i] = rne(q, zero, e, gravity=np.zeros(3))
    return 0.5 * (M + M.T)          # symmetrise tiny numerical asymmetry


def _dM_dq(q, k, eps=1e-6):
    """Central finite difference dM/dq_k."""
    qp = q.copy(); qp[k] += eps
    qm = q.copy(); qm[k] -= eps
    return (mass_matrix(qp) - mass_matrix(qm)) / (2.0 * eps)


def coriolis_matrix(q, qd):
    """Coriolis matrix C(q, qd) via Christoffel symbols (matches rtb).

        C_ij = sum_k 0.5 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) qd_k

    Uses finite-difference partials of M. C(q,qd) qd equals coriolis_vector,
    and C(q,qd)^T qd is what the momentum observer integrates.
    """
    q = np.asarray(q, float).reshape(-1)
    qd = np.asarray(qd, float).reshape(-1)
    dM = np.stack([_dM_dq(q, k) for k in range(N)], axis=0)   # (k,i,j)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += 0.5 * (dM[k, i, j] + dM[j, i, k] - dM[i, j, k]) * qd[k]
            C[i, j] = s
    return C


def coriolis_transpose_qd(q, qd):
    """C(q, qd)^T qd, the term integrated by the momentum observer.

    Computed model-consistently as  C^T qd = Mdot qd - C qd, where
    Mdot qd = (sum_k dM/dq_k * qd_k) qd and C qd = coriolis_vector(q, qd).
    This avoids forming the full C matrix (7x faster) yet is exact because
    Mdot = C + C^T for the Christoffel factorization.
    """
    q = np.asarray(q, float).reshape(-1)
    qd = np.asarray(qd, float).reshape(-1)
    Mdot_qd = np.zeros(N)
    for k in range(N):
        Mdot_qd += (_dM_dq(q, k) @ qd) * qd[k]
    return Mdot_qd - coriolis_vector(q, qd)
