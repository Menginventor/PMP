import math
import numpy as np
import pygame

# ===========================
# Parameters
# ===========================
SCREEN_W, SCREEN_H = 900, 600
ORIGIN = np.array([SCREEN_W//2, SCREEN_H//2 + 100], dtype=float)  # base in screen coords
PIXELS_PER_M = 180.0  # scale for drawing (m -> px)

# Link lengths (meters)
LINKS = np.array([0.35, 0.30, 0.25, 0.20], dtype=float)

# Damping / weight matrix W (diagonal)
W_diag = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

# Posture preference q* (radians)
q_star = np.array([0.0, 0.4, -0.6, 0.4], dtype=float)

# Posture stiffness (diagonal)
Kp_diag = np.array([1.5, 1.2, 1.0, 0.8], dtype=float)

# Task-space spring gain (1/s)
K_task = 2.5

# DLS regularization for (J W^{-1} J^T) inverse
MU = 0
# Integration
DT = 1/120.0  # seconds
VEL_LIMIT = 4.0  # rad/s clamp
FRICTION_CLAMP = True  # clamp dot q

# Trace
TRACE_LIFETIME = 2.0   # seconds before trace fades
TRACE_INTERVAL = 3     # frames between samples

# Colors
BG = (18, 18, 22)
ARM = (240, 240, 240)
JOINT = (120, 180, 255)
TARGET = (255, 90, 90)
TEXT = (200, 200, 200)
TRACE_COLOR = (255, 200, 80)

# ===========================
# Helper math
# ===========================
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def fk_points(q):
    th1, th2, th3, th4 = np.cumsum(q)
    p1 = LINKS[0]*np.array([math.cos(th1), math.sin(th1)])
    p2 = LINKS[1]*np.array([math.cos(th2), math.sin(th2)])
    p3 = LINKS[2]*np.array([math.cos(th3), math.sin(th3)])
    p4 = LINKS[3]*np.array([math.cos(th4), math.sin(th4)])
    j0 = np.array([0.0, 0.0])
    j1 = j0 + p1
    j2 = j1 + p2
    j3 = j2 + p3
    ee = j3 + p4
    joints_m = [j0, j1, j2, j3, ee]
    joints_px = [to_px(p) for p in joints_m]
    return joints_m, joints_px, ee

def jacobian(q):
    th = np.cumsum(q)
    s = np.sin(th); c = np.cos(th)
    L = LINKS
    J = np.zeros((2, 4))
    J[:,0] = [-L[0]*s[0] - L[1]*s[1] - L[2]*s[2] - L[3]*s[3],
               L[0]*c[0] + L[1]*c[1] + L[2]*c[2] + L[3]*c[3]]
    J[:,1] = [-L[1]*s[1] - L[2]*s[2] - L[3]*s[3],
               L[1]*c[1] + L[2]*c[2] + L[3]*c[3]]
    J[:,2] = [-L[2]*s[2] - L[3]*s[3],
               L[2]*c[2] + L[3]*c[3]]
    J[:,3] = [-L[3]*s[3], L[3]*c[3]]
    return J

def to_px(p_m):
    return ORIGIN + np.array([p_m[0]*PIXELS_PER_M, -p_m[1]*PIXELS_PER_M])

def from_px(p_px):
    v = (np.array(p_px, dtype=float) - ORIGIN)/PIXELS_PER_M
    v[1] = -v[1]
    return v

# ===========================
# PMP step
# ===========================
def pmp_step(q, x_des, W_diag, K_task, Kp_diag, q_star, mu=1e-3, dt=DT):
    (joints_m, joints_px, x) = fk_points(q)
    J = jacobian(q)
    W = np.diag(W_diag)
    Winv = np.diag(1.0/W_diag)

    dx = x_des - x
    xdot = K_task * dx

    grad_h = - np.diag(Kp_diag) @ (q - q_star)

    JWJt = J @ Winv @ J.T
    A = JWJt + mu * np.eye(2)
    Ainv = np.linalg.inv(A)
    G = Winv @ J.T @ Ainv

    N = np.eye(4) - G @ J

    qdot = G @ xdot + N @ (Winv @ grad_h)

    if FRICTION_CLAMP:
        qdot = np.clip(qdot, -VEL_LIMIT, VEL_LIMIT)

    q_next = q + qdot * dt
    q_next = np.array([wrap_angle(a) for a in q_next])
    return q_next, joints_px, x, xdot, grad_h, N

# ===========================
# Pygame drawing
# ===========================
def draw_text(screen, txt, xy, size=18, color=TEXT):
    font = pygame.font.SysFont("consolas", size)
    surf = font.render(txt, True, color)
    screen.blit(surf, xy)

def draw_arm(screen, joints_px):
    for i in range(len(joints_px)-1):
        a = joints_px[i]
        b = joints_px[i+1]
        pygame.draw.line(screen, ARM, a, b, 5)
        pygame.draw.circle(screen, JOINT, a.astype(int), 7)
    pygame.draw.circle(screen, (255,220,120), joints_px[-1].astype(int), 7)

def draw_target(screen, x_des):
    p = to_px(x_des)
    pygame.draw.circle(screen, TARGET, p.astype(int), 8, width=2)
    pygame.draw.line(screen, TARGET, p + np.array([-8,0]), p + np.array([8,0]), 2)
    pygame.draw.line(screen, TARGET, p + np.array([0,-8]), p + np.array([0,8]), 2)

def draw_trace(screen, trace, now):
    for (px, t0) in trace:
        age = now - t0
        if age < TRACE_LIFETIME:
            alpha = max(0, 255 * (1 - age/TRACE_LIFETIME))
            col = (TRACE_COLOR[0], TRACE_COLOR[1], TRACE_COLOR[2], int(alpha))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, col, (2,2), 2)
            screen.blit(s, px - np.array([2,2]))

# ===========================
# Main
# ===========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("PMP demo — 4-DoF planar manipulator")
    clock = pygame.time.Clock()

    q = np.array([0.2, 0.6, -0.5, 0.2], dtype=float)
    joints_m, joints_px, ee = fk_points(q)
    x_des = ee.copy()

    trace = []
    frame_count = 0
    start_time = pygame.time.get_ticks()/1000.0

    running = True
    while running:
        now = pygame.time.get_ticks()/1000.0 - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x_des = from_px(event.pos)

        q, joints_px, x, xdot, grad_h, N = pmp_step(
            q, x_des, W_diag, K_task, Kp_diag, q_star, mu=MU, dt=DT
        )

        # Add trace point every few frames
        frame_count += 1
        if frame_count % TRACE_INTERVAL == 0:
            trace.append((joints_px[-1], now))
        # Remove expired points
        trace = [(p,t) for (p,t) in trace if now - t < TRACE_LIFETIME]

        screen.fill(BG)
        draw_target(screen, x_des)
        draw_arm(screen, joints_px)
        draw_trace(screen, trace, now)

        draw_text(screen, f"Target (m): [{x_des[0]:+.3f}, {x_des[1]:+.3f}]", (10, 10))
        draw_text(screen, f"EE (m):     [{x[0]:+.3f}, {x[1]:+.3f}]", (10, 32))
        draw_text(screen, "Controls: Left-click to set target", (10, 56))
        draw_text(screen, "PMP: qdot = G*xdot + (I-GJ)*W^{-1}*grad(h)", (10, 80))

        pygame.display.flip()
        clock.tick(int(1/DT))

    pygame.quit()

if __name__ == "__main__":
    main()
