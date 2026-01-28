import argparse
import glob
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import cv2
import pygame
import mediapipe as mp

try:
    import nibabel as nib
except Exception:
    nib = None


# ----------------------------
# Dataset loading helpers
# ----------------------------
def _find_first(root: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
        if hits:
            return hits[0]
    return None


def load_fmri_mean_timeseries(bids_root: str) -> tuple[np.ndarray, str]:
    if nib is None:
        raise RuntimeError("nibabel not available; cannot read NIfTI files.")

    bold_path = _find_first(
        bids_root,
        [
            "**/sub-*/func/*_bold.nii.gz",
            "**/sub-*/func/*_bold.nii",
            "**/func/*_bold.nii.gz",
            "**/func/*_bold.nii",
        ],
    )
    if not bold_path:
        raise FileNotFoundError("No *_bold.nii(.gz) found under the BIDS root.")

    img = nib.load(bold_path)
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D fMRI, got shape {img.shape} for {bold_path}")

    t_len = img.shape[3]
    ts = np.zeros(t_len, dtype=np.float32)

    for t in range(t_len):
        vol = np.asanyarray(img.dataobj[..., t], dtype=np.float32)
        ts[t] = float(vol.mean())

    ts = ts - ts.mean()
    ts = ts / (ts.std() + 1e-6)
    return ts, bold_path


def load_t1_slice(bids_root: str) -> tuple[np.ndarray, str] | tuple[None, None]:
    if nib is None:
        return None, None

    t1_path = _find_first(
        bids_root,
        [
            "**/sub-*/anat/*_T1w.nii.gz",
            "**/sub-*/anat/*_T1w.nii",
            "**/anat/*_T1w.nii.gz",
            "**/anat/*_T1w.nii",
        ],
    )
    if not t1_path:
        return None, None

    img = nib.load(t1_path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)

    z = data.shape[2] // 2
    sl = data[:, :, z]

    lo, hi = np.percentile(sl, [2, 98])
    sl = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)
    sl_u8 = (sl * 255).astype(np.uint8)
    return sl_u8, t1_path


# ----------------------------
# Visual primitives
# ----------------------------
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: float
    hue: float


@dataclass
class StrokeDot:
    x: float
    y: float
    life: float
    size: float
    hue: float


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def flow_noise(x: float, y: float, t: float) -> float:
    return math.sin(2.3 * x + 1.7 * y + 0.9 * t) + math.sin(1.1 * x - 2.1 * y + 1.3 * t)


def frame_to_surface_rgb(frame_bgr: np.ndarray) -> pygame.Surface:
    """Convert cv2 BGR frame -> pygame Surface (RGB)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # pygame.surfarray expects (w, h, 3)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surf


# ----------------------------
# Hand tracking
# ----------------------------
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.connections = list(mp.solutions.hands.HAND_CONNECTIONS)

        self.last_x = 0.5
        self.last_y = 0.5
        self.last_pinch = 1.0
        self.last_spread = 0.0
        self.last_landmarks = None  # list[(x,y)] normalized

    def process(self, frame_bgr: np.ndarray) -> tuple[float, float, float, float, list[tuple[float, float]] | None, bool]:
        """
        Returns (x, y, pinch, spread, landmarks_norm, found)
        x,y from index tip in [0,1]
        landmarks_norm = list of (x,y) for 21 points if found else None
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return self.last_x, self.last_y, self.last_pinch, self.last_spread, self.last_landmarks, False

        lm = res.multi_hand_landmarks[0].landmark
        landmarks_norm = [(float(p.x), float(p.y)) for p in lm]

        idx = lm[8]    # index tip
        thb = lm[4]    # thumb tip
        pky = lm[20]   # pinky tip

        x, y = float(idx.x), float(idx.y)
        pinch = math.hypot(idx.x - thb.x, idx.y - thb.y)
        spread = math.hypot(idx.x - pky.x, idx.y - pky.y)

        self.last_x, self.last_y, self.last_pinch, self.last_spread = x, y, pinch, spread
        self.last_landmarks = landmarks_norm
        return x, y, pinch, spread, landmarks_norm, True


def draw_hand_skeleton(screen: pygame.Surface, landmarks_norm, connections, W, H, alpha=200):
    """Draw hand landmarks + connections on main canvas."""
    if not landmarks_norm:
        return

    overlay = pygame.Surface((W, H), pygame.SRCALPHA)

    pts = [(int(x * W), int(y * H)) for (x, y) in landmarks_norm]

    # connections
    for a, b in connections:
        ax, ay = pts[a]
        bx, by = pts[b]
        pygame.draw.line(overlay, (255, 255, 255, alpha), (ax, ay), (bx, by), 2)

    # points
    for i, (px, py) in enumerate(pts):
        r = 6 if i in (4, 8, 12, 16, 20) else 4
        pygame.draw.circle(overlay, (255, 255, 255, alpha), (px, py), r)

    screen.blit(overlay, (0, 0))


def draw_hud(screen: pygame.Surface, lines: list[str], font: pygame.font.Font):
    W, H = screen.get_size()
    padding = 12
    line_h = font.get_linesize()
    box_w = min(W - 24, 900)
    box_h = padding * 2 + line_h * len(lines)

    panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, 160), (0, 0, box_w, box_h), border_radius=12)
    pygame.draw.rect(panel, (255, 255, 255, 40), (0, 0, box_w, box_h), width=1, border_radius=12)

    y = padding
    for line in lines:
        txt = font.render(line, True, (235, 235, 235))
        panel.blit(txt, (padding, y))
        y += line_h

    screen.blit(panel, (12, 12))


# ----------------------------
# Main app
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", type=str, default="ds007185", help="Path to the ds007185 dataset folder")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--w", type=int, default=1280, help="Window width")
    ap.add_argument("--h", type=int, default=720, help="Window height")
    ap.add_argument("--no-mirror", action="store_true", help="Disable left/right flip")
    args = ap.parse_args()

    # Load dataset-driven signals
    ts = None
    bold_path = None
    t1_slice = None
    t1_path = None

    if os.path.isdir(args.bids_root):
        try:
            ts, bold_path = load_fmri_mean_timeseries(args.bids_root)
        except Exception as e:
            print(f"[warn] Could not load fMRI BOLD: {e}")

        try:
            t1_slice, t1_path = load_t1_slice(args.bids_root)
        except Exception as e:
            print(f"[warn] Could not load T1w anatomy: {e}")
    else:
        print(f"[warn] BIDS root not found: {args.bids_root}")

    if ts is None:
        print("[warn] Using synthetic signal (dataset not loaded).")
        t = np.linspace(0, 40 * math.pi, 800).astype(np.float32)
        ts = (np.sin(t) + 0.35 * np.sin(2.7 * t + 1.2) + 0.15 * np.sin(7.3 * t)).astype(np.float32)
        ts = (ts - ts.mean()) / (ts.std() + 1e-6)
        bold_path = "(synthetic)"

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    args.w, args.h = screen.get_size()
    pygame.display.set_caption("Curiosity Hand Art v2 — Mirror Hand + Interactions")
    clock = pygame.time.Clock()

    # nicer fonts
    font = pygame.font.SysFont("consolas", 18)
    font_big = pygame.font.SysFont("consolas", 22, bold=True)

    # Anatomy background
    anat_surface = None
    if t1_slice is not None:
        rgb = np.stack([t1_slice] * 3, axis=-1)
        rgb = np.rot90(rgb)  # orientation tweak
        anat_surface = pygame.surfarray.make_surface(rgb)

    # Camera + hand tracker
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()

    # States
    particles: list[Particle] = []
    strokes: list[StrokeDot] = []

    show_anat = True if anat_surface is not None else False
    show_cam = True
    show_hand_skeleton = True

    freeze_scrub = False
    mirror = not args.no_mirror

    t_index = 0
    gain = 1.0  # controlled by vertical hand position
    last_toggle = 0.0
    last_freeze_toggle = 0.0
    last_cam_toggle = 0.0
    last_hand_toggle = 0.0

    # trails
    fade = pygame.Surface((args.w, args.h), pygame.SRCALPHA)

    # motion history (for “brush velocity”)
    prev_cx, prev_cy = args.w * 0.5, args.h * 0.5

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = time.time()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_a and anat_surface is not None:
                    show_anat = not show_anat
                elif ev.key == pygame.K_c:
                    show_cam = not show_cam
                elif ev.key == pygame.K_h:
                    show_hand_skeleton = not show_hand_skeleton
                elif ev.key == pygame.K_m:
                    mirror = not mirror

        ok, frame = cap.read()
        if not ok:
            frame = None

        hx, hy, pinch, spread, landmarks_norm, found = (0.5, 0.5, 1.0, 0.0, None, False)
        if frame is not None:
            if mirror:
                frame = cv2.flip(frame, 1)  # <-- MIRROR LEFT/RIGHT
            hx, hy, pinch, spread, landmarks_norm, found = tracker.process(frame)

        # Gesture thresholds
        pinch_trigger = pinch < 0.055
        spread_trigger = spread > 0.33

        # Toggle anatomy with wide spread (optional)
        if spread_trigger and (now - last_toggle) > 0.8 and anat_surface is not None:
            show_anat = not show_anat
            last_toggle = now

        # Pinch toggles freeze (but ALSO acts as "grab" mode)
        if pinch_trigger and (now - last_freeze_toggle) > 0.6:
            freeze_scrub = not freeze_scrub
            last_freeze_toggle = now

        # Hand position
        cx = float(np.clip(hx, 0, 1) * args.w)
        cy = float(np.clip(hy, 0, 1) * args.h)

        # Vertical controls gain (top = stronger influence)
        gain = float(np.interp(np.clip(hy, 0, 1), [1.0, 0.0], [0.7, 2.0]))

        # Time scrub only when not frozen AND not pinching hard (feel like "grab stops time")
        if not freeze_scrub and not pinch_trigger:
            t_index = int(np.clip(hx, 0.0, 1.0) * (len(ts) - 1))

        # Signal & derivative
        s = float(ts[t_index]) * gain
        s_prev = float(ts[max(0, t_index - 1)]) * gain
        s_next = float(ts[min(len(ts) - 1, t_index + 1)]) * gain
        ds = 0.5 * (s_next - s_prev)

        # Background
        if show_anat and anat_surface is not None:
            bg = pygame.transform.smoothscale(anat_surface, (args.w, args.h))
            screen.blit(bg, (0, 0))
            fade.fill((0, 0, 0, 70))
            screen.blit(fade, (0, 0))
        else:
            fade.fill((0, 0, 0, 26))
            screen.blit(fade, (0, 0))

        # Camera PIP (so user sees their hand)
        if show_cam and frame is not None:
            cam_surf = frame_to_surface_rgb(frame)
            pip_w = 360
            pip_h = int(pip_w * (cam_surf.get_height() / cam_surf.get_width()))
            cam_surf = pygame.transform.smoothscale(cam_surf, (pip_w, pip_h))
            cam_surf.set_alpha(200)

            # background behind pip for readability
            pip_bg = pygame.Surface((pip_w + 12, pip_h + 12), pygame.SRCALPHA)
            pygame.draw.rect(pip_bg, (0, 0, 0, 130), (0, 0, pip_w + 12, pip_h + 12), border_radius=14)
            pygame.draw.rect(pip_bg, (255, 255, 255, 40), (0, 0, pip_w + 12, pip_h + 12), width=1, border_radius=14)

            screen.blit(pip_bg, (args.w - (pip_w + 24), 12))
            screen.blit(cam_surf, (args.w - (pip_w + 18), 18))

        # Hand skeleton on main canvas
        if show_hand_skeleton and found and landmarks_norm is not None:
            draw_hand_skeleton(screen, landmarks_norm, tracker.connections, args.w, args.h, alpha=180)

        # Brush dynamics: based on hand motion
        vxh = (cx - prev_cx) / max(dt, 1e-6)
        vyh = (cy - prev_cy) / max(dt, 1e-6)
        speed = math.hypot(vxh, vyh)
        prev_cx, prev_cy = cx, cy

        # Color: signal -> hue
        base_hue = (0.60 + 0.10 * s + 0.10 * ds) % 1.0

        # Interaction modes:
        # - Pinch = grab mode: strong attraction to fingertip + fewer spawns (feels like pulling matter)
        # - Open hand (high spread) = spray mode: more spawns + larger brush
        grab = pinch_trigger
        spray = spread > 0.28  # softer than trigger

        # Emission rate
        emit = 30 + int(70 * abs(s) + 120 * abs(ds) + 0.03 * speed)
        if spray:
            emit = int(emit * 1.9)
        if grab:
            emit = int(emit * 0.45)
        emit = max(8, min(emit, 260))

        # Brush radius from spread
        brush_r = 10 + 80 * np.clip((spread - 0.15) / 0.35, 0.0, 1.0)

        # Add stroke dots (a soft “ink” trail that fades)
        if found:
            dot_size = 2.0 + 0.08 * brush_r
            if spray:
                dot_size *= 1.25
            if grab:
                dot_size *= 0.85
            strokes.append(StrokeDot(cx, cy, life=0.45 + 0.35 * random.random(), size=dot_size, hue=base_hue))

        # Spawn particles around fingertip (spray) or near center (normal)
        for _ in range(emit):
            ang = random.random() * 2 * math.pi
            rad = random.random() ** 0.6 * brush_r
            px = cx + math.cos(ang) * rad
            py = cy + math.sin(ang) * rad

            # initial velocity: follow hand direction a bit
            hand_push = 0.0025 * speed
            ivx = (vxh * hand_push) + math.cos(ang) * (0.5 + 1.5 * abs(s))
            ivy = (vyh * hand_push) + math.sin(ang) * (0.5 + 1.5 * abs(s))

            life = 0.5 + 1.4 * random.random()
            size = 1.2 + 2.4 * random.random() + 1.0 * abs(s)
            hue = (base_hue + 0.10 * (random.random() - 0.5)) % 1.0
            particles.append(Particle(px, py, ivx, ivy, life, size, hue))

        # Update strokes (soft ink)
        new_strokes = []
        for d in strokes:
            d.life -= dt
            if d.life > 0:
                v = 0.25 + 0.70 * np.clip(abs(s) + 0.8 * abs(ds), 0.0, 1.6)
                col = hsv_to_rgb(d.hue, 0.85, min(1.0, v))
                pygame.draw.circle(screen, col, (int(d.x), int(d.y)), int(max(1, d.size)))
                new_strokes.append(d)
        strokes = new_strokes[-2400:]

        # Particle field dynamics
        swirl = 0.5 + 2.2 * abs(ds) + 0.9 * abs(s)
        grab_pull = 3.4 if grab else 1.2
        noise_amt = 0.25 + 0.55 * np.clip(abs(s), 0.0, 2.0)

        new_particles = []
        for p in particles:
            nx = p.x / args.w
            ny = p.y / args.h
            n = flow_noise(nx * 8.0, ny * 6.0, now * 0.7)

            dxh = (p.x - cx) / (args.w + 1e-6)
            dyh = (p.y - cy) / (args.h + 1e-6)

            # swirl around fingertip
            p.vx += (-dyh) * swirl * 0.9 + noise_amt * 0.25 * n
            p.vy += ( dxh) * swirl * 0.9 + noise_amt * 0.18 * n

            # attraction/repulsion depending on mode:
            # grab pulls strongly toward fingertip; spray slightly repels (like blowing paint outward)
            if grab:
                p.vx += (-dxh) * grab_pull * 1.6
                p.vy += (-dyh) * grab_pull * 1.6
            else:
                if spray:
                    p.vx += (dxh) * 0.55
                    p.vy += (dyh) * 0.55
                else:
                    p.vx += (-dxh) * 0.65
                    p.vy += (-dyh) * 0.65

            # damping
            damp = 0.92 - 0.05 * np.clip(abs(s), 0.0, 1.8)
            p.vx *= damp
            p.vy *= damp

            p.x += p.vx
            p.y += p.vy
            p.life -= dt

            # wrap edges
            if p.x < 0: p.x += args.w
            if p.x > args.w: p.x -= args.w
            if p.y < 0: p.y += args.h
            if p.y > args.h: p.y -= args.h

            if p.life > 0:
                v = 0.30 + 0.60 * np.clip(abs(s) + 0.6 * abs(ds), 0.0, 1.6)
                col = hsv_to_rgb(p.hue, 0.85, min(1.0, v))
                pygame.draw.circle(screen, col, (int(p.x), int(p.y)), int(max(1, p.size)))
                new_particles.append(p)

        particles = new_particles[-14000:]

        # Cursor ring at fingertip (makes interaction feel tighter)
        ring = pygame.Surface((args.w, args.h), pygame.SRCALPHA)
        ring_alpha = 120 if found else 50
        pygame.draw.circle(ring, (255, 255, 255, ring_alpha), (int(cx), int(cy)), int(brush_r), 2)
        pygame.draw.circle(ring, (255, 255, 255, ring_alpha), (int(cx), int(cy)), 8, 2)
        screen.blit(ring, (0, 0))

        # HUD
        hud = [
            "Hand controls (mirror view):",
            "  Move = paint / emit (index tip)",
            "  Pinch = GRAB field + freezes scrub",
            "  Open hand = SPRAY mode",
            "  X = scrub time (when not pinching),  Y = gain (signal influence)",
            "",
            f"signal z(mean BOLD)*gain: {s:+.3f}   d/dt: {ds:+.3f}   gain: {gain:.2f}",
            f"t = {t_index}/{len(ts)-1}   freeze={freeze_scrub}   grab={grab}   spray={spray}",
            f"mirror={mirror}   cam(PIP)={show_cam}   hand-skel={show_hand_skeleton}",
            f"BOLD: {os.path.basename(bold_path) if bold_path else 'n/a'}",
            f"T1:   {os.path.basename(t1_path) if t1_path else 'n/a'}",
            "Keys: [A] anatomy  [C] camera  [H] hand lines  [M] mirror  [ESC] quit",
        ]
        draw_hud(screen, hud, font)

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
