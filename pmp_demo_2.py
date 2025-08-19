import math
import numpy as np
import pygame

# ===========================
# Config: set DoF here (4..8)
# ===========================
N_DOF = 5
assert 4 <= N_DOF <= 8

# Link lengths (m) - provide N_DOF entries
# You can tweak these; keep total reach reasonable for the screen.
LINKS = np.array([0.30, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12][:N_DOF], dtype=float)

# Joint weights/damping W (diagonal >0)
W_diag = np.ones(N_DOF, dtype=float)

# Posture preference (radians)
q_star = np.linspace(0.0, 0.5, N_DOF) * np.array([1, 1, -1, 1, -1, 1, -1, 1][:N_DOF])

# Posture stiffness (diagonal >=0)
Kp_diag = np.linspace(1.5, 0.3, N_DOF)  # stronger at base, softer at tip

# Task-space spring gain (1/s)
K_task = 2.8

# Damped least-squares regularization for (J W^{-1} J^T) inverse
#MU = 2e-3
MU = 0

# Integration / numerics
DT = 1.0 / 120.0
VEL_LIMIT = 5.0  # rad/s clamp
CLAMP_VEL = True

# ===========================
# Drawing / UI
# ===========================
SCREEN_W, SCREEN_H = 1100, 700
ORIGIN = np.array([SCREEN_W // 2, SCREEN_H // 2 + 120], dtype=float)  # base pixel position
PIXELS_PER_M = 220.0  # scale (m -> px)

BG = (18, 18, 22)
ARM = (240, 240, 240)
JOINT = (120, 180, 255)
EE = (255, 220, 120)
TARGET = (255, 90, 90)
TEXT = (210, 210, 210)
AUX = (120, 255, 160)

# ===========================
# Helpers
# ===========================
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def to_px(p_m):
    """Meters (world) -> screen pixels (flip y)."""
    return ORIGIN + np.array([p_m[0] * PIXELS_PER_M, -p_m[1] * PIXELS_PER_M])

def from_px(p_px):
    """Screen pixels -> meters (world)."""
    v = (np.array(p_px, dtype=float) - ORIGIN) / PIXELS_PER_M
    v[1] = -v[1]
    return v

def fk_points(q):
    """
    Forward kinematics for planar serial chain:
    returns joint positions (meters), joint positions (pixels), end-effector (meters).
    """
    th_cum = np.cumsum(q)  # shape (N,)
    cs = np.cos(th_cum)
    sn = np.sin(th_cum)

    # segment vectors and cumulative joint coordinates in meters
    segs = (LINKS[:, None] * np.stack([cs, sn], axis=1))  # (N,2)
    joints = np.zeros((N_DOF + 1, 2), dtype=float)
    for i in range(N_DOF):
        joints[i + 1] = joints[i] + segs[i]

    ee = joints[-1]
    joints_px = np.array([to_px(p) for p in joints])
    return joints, joints_px, ee

def jacobian(q):
    """
    Planar 2xN Jacobian for end-effector (x,y).
    J[:, j] = partial(EE) / partial(q_j) = sum_{k=j..N} R(theta_k) * d/dq_j (link_k)
            = [- sum L_k sin(theta_k), sum L_k cos(theta_k)] for k>=j (planar)
    """
    th_cum = np.cumsum(q)               # (N,)
    cs = np.cos(th_cum)                 # (N,)
    sn = np.sin(th_cum)                 # (N,)
    # Build cumulative sums from the end to efficiently form columns
    # S_jx = - sum_{k=j..N} L_k sin(theta_k)
    # S_jy =   sum_{k=j..N} L_k cos(theta_k)
    Lsn = LINKS * sn
    Lcs = LINKS * cs
    # reverse cumulative sums
    Sx = -np.cumsum(Lsn[::-1])[::-1]    # (N,)
    Sy =  np.cumsum(Lcs[::-1])[::-1]    # (N,)
    J = np.vstack([Sx, Sy])             # (2,N)
    return J

# ===========================
# PMP step (closed-form)
# dot q = G xdot + (I - GJ) W^{-1} grad(h)
# ===========================
def pmp_step(q, x_des, W_diag, K_task, Kp_diag, q_star, mu=1e-3, dt=DT):
    # FK and J
    joints_m, joints_px, x = fk_points(q)
    J = jacobian(q)                       # (2,N)
    N = q.size

    W = np.diag(W_diag)                   # (N,N)
    Winv = np.diag(1.0 / W_diag)          # (N,N)

    # Task-space desired velocity (spring to target)
    dx = x_des - x                        # (2,)
    xdot = K_task * dx

    # Posture gradient (attractive torque): grad(h) = -Kp (q - q_star)
    grad_h = -np.diag(Kp_diag) @ (q - q_star)  # (N,)

    # Weighted pseudoinverse with damped least squares:
    # G = W^{-1} J^T (J W^{-1} J^T + mu I)^-1
    JWJt = J @ Winv @ J.T                 # (2,2)
    A = JWJt + mu * np.eye(2)
    Ainv = np.linalg.inv(A)
    G = Winv @ J.T @ Ainv                 # (N,2)

    # Null-space projector
    Nproj = np.eye(N) - G @ J

    # Joint velocity
    qdot = G @ xdot + Nproj @ (Winv @ grad_h)

    if CLAMP_VEL:
        qdot = np.clip(qdot, -VEL_LIMIT, VEL_LIMIT)

    q_next = q + qdot * dt
    q_next = np.array([wrap_angle(a) for a in q_next])
    return q_next, joints_px, x, xdot, grad_h, Nproj

# ===========================
# Drawing
# ===========================
def draw_text(screen, txt, xy, size=18, color=TEXT):
    font = pygame.font.SysFont("consolas", size)
    surf = font.render(txt, True, color)
    screen.blit(surf, xy)

def draw_arm(screen, joints_px):
    for i in range(len(joints_px) - 1):
        a = joints_px[i]
        b = joints_px[i + 1]
        pygame.draw.line(screen, ARM, a, b, 5)
        pygame.draw.circle(screen, JOINT, a.astype(int), 6)
    pygame.draw.circle(screen, EE, joints_px[-1].astype(int), 7)

def draw_target(screen, x_des):
    p = to_px(x_des)
    pygame.draw.circle(screen, TARGET, p.astype(int), 8, 2)
    pygame.draw.line(screen, TARGET, p + np.array([-8, 0]), p + np.array([8, 0]), 2)
    pygame.draw.line(screen, TARGET, p + np.array([0, -8]), p + np.array([0, 8]), 2)

# ===========================
# Main
# ===========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"PMP demo — {N_DOF}-DoF planar manipulator (Left-click to set target)")
    clock = pygame.time.Clock()

    # Initial joint configuration
    q = np.linspace(0.15, -0.15, N_DOF)

    # Initial target: current EE
    joints_m, joints_px, ee = fk_points(q)
    x_des = ee.copy()

    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x_des = from_px(event.pos)

        # PMP step
        q, joints_px, x, xdot, grad_h, Nproj = pmp_step(
            q, x_des, W_diag, K_task, Kp_diag, q_star, mu=MU, dt=DT
        )

        # Draw
        screen.fill(BG)
        draw_target(screen, x_des)
        draw_arm(screen, joints_px)

        # UI text (ASCII-safe; no special symbols)
        draw_text(screen, f"DoF: {N_DOF}", (10, 10))
        draw_text(screen, f"Target (m): [{x_des[0]:+.3f}, {x_des[1]:+.3f}]", (10, 34))
        draw_text(screen, f"EE (m):     [{x[0]:+.3f}, {x[1]:+.3f}]", (10, 56))
        draw_text(screen, "Law: qdot = G*xdot + (I - GJ) * W^-1 * grad(h)", (10, 78))
        draw_text(screen, f"K_task={K_task:.2f}  mu={MU:.0e}  dt={DT:.4f}", (10, 100))
        draw_text(screen, "Tip: Increase mu if it jitters near singular postures.", (10, 122))

        pygame.display.flip()
        clock.tick(int(1 / DT))

    pygame.quit()

if __name__ == "__main__":
    main()
