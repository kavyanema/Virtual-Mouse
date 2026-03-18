"""
Virtual Mouse - Pinch to Click
Compatible with mediapipe >= 0.10

Install:
    pip install opencv-python mediapipe pyautogui numpy

GESTURES:
  Index finger up              -> Move cursor
  Pinch index + thumb          -> Click
  All fingers up               -> Scroll (move hand up/down)
  Fist                         -> Drag
  Press Q                      -> Quit
"""

import cv2
import pyautogui
import numpy as np
import time
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_ID        = 0
FRAME_W, FRAME_H = 640, 480
SMOOTH_FACTOR    = 5
PINCH_DIST       = 40        # pixels — how close thumb+index must be to click
CLICK_COOLDOWN   = 0.5
MARGIN_X         = 80
MARGIN_Y         = 60
SCROLL_SPEED     = 15
MODEL_PATH       = "hand_landmarker.task"
MODEL_URL        = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

# ── Download model if needed ──────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~8 MB), please wait...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.\n")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

SCREEN_W, SCREEN_H = pyautogui.size()

# ── Helpers ───────────────────────────────────────────────────────────────────
def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist(p1, p2):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

def map_range(v, i0, i1, o0, o1):
    v = max(i0, min(i1, v))
    return int((v - i0) / (i1 - i0) * (o1 - o0) + o0)

def get_finger_states(lms, w, h):
    tips = [4, 8, 12, 16, 20]
    dips = [3, 6, 10, 14, 18]
    up = []
    up.append(lm_px(lms[4], w, h)[0] < lm_px(lms[3], w, h)[0])  # thumb
    for t, d in zip(tips[1:], dips[1:]):
        up.append(lm_px(lms[t], w, h)[1] < lm_px(lms[d], w, h)[1])
    return up

def draw_hand(frame, lms, w, h, pinching=False):
    pts = [lm_px(lm, w, h) for lm in lms]
    bone_color = (60, 160, 255) if pinching else (80, 200, 120)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], bone_color, 2)
    for i, p in enumerate(pts):
        r = 6 if i in [4, 8, 12, 16, 20] else 3
        cv2.circle(frame, p, r, (255, 255, 255), -1)
        cv2.circle(frame, p, r, bone_color, 1)

# ── Smoothing ─────────────────────────────────────────────────────────────────
sx, sy = SCREEN_W // 2, SCREEN_H // 2

def smooth(rx, ry):
    global sx, sy
    sx += (rx - sx) / SMOOTH_FACTOR
    sy += (ry - sy) / SMOOTH_FACTOR
    return int(sx), int(sy)

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, label, fps, pinch_dist):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (250, 120), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    clr = {"Move": (0,230,120), "Click": (60,160,255),
           "Scroll": (255,215,0), "Drag": (200,80,255), "—": (160,160,160)}
    c = clr.get(label, (200,200,200))

    cv2.putText(frame, f"Gesture: {label}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Pinch distance bar
    bar_max  = 120
    bar_fill = max(0, min(bar_max, int(bar_max - (pinch_dist / PINCH_DIST) * bar_max)))
    bar_col  = (60,160,255) if pinch_dist < PINCH_DIST else (80,80,80)
    cv2.rectangle(frame, (12, 72), (12 + bar_max, 82), (50,50,50), -1)
    cv2.rectangle(frame, (12, 72), (12 + bar_fill, 82), bar_col, -1)
    cv2.putText(frame, "pinch", (140, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140,140,140), 1)

    cv2.putText(frame, "Q to quit", (12, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)
    cv2.rectangle(frame, (MARGIN_X, MARGIN_Y),
                  (w - MARGIN_X, h - MARGIN_Y), (40,40,80), 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    last_click = 0
    dragging   = False
    scroll_ref = None
    label      = "—"
    pinch_dist = 999

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    prev_t = time.time()
    ts_ms  = 0

    print("\n Virtual Mouse started — press Q to quit\n")
    print("  Index finger up        -> Move cursor")
    print("  Pinch index + thumb    -> Click")
    print("  All fingers up         -> Scroll (move hand up/down)")
    print("  Fist                   -> Drag\n")

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error. Try CAMERA_ID = 1")
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            cur_t  = time.time()
            fps    = 1.0 / max(cur_t - prev_t, 1e-6)
            prev_t = cur_t

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms += 33
            result = detector.detect_for_video(mp_img, ts_ms)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]

                idx_tip = lm_px(lms[8], w, h)
                thm_tip = lm_px(lms[4], w, h)
                pinch_dist = dist(idx_tip, thm_tip)
                pinching   = pinch_dist < PINCH_DIST

                draw_hand(frame, lms, w, h, pinching)

                f = get_finger_states(lms, w, h)
                thumb, index, middle, ring, pinky = f

                # Cursor follows index fingertip
                mx = map_range(idx_tip[0], MARGIN_X, w-MARGIN_X, 0, SCREEN_W)
                my = map_range(idx_tip[1], MARGIN_Y, h-MARGIN_Y, 0, SCREEN_H)
                cx, cy = smooth(mx, my)

                # ── Scroll: all fingers up ──────────────────────────────────
                if all(f):
                    label = "Scroll"
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    if scroll_ref is None:
                        scroll_ref = idx_tip[1]
                    delta = scroll_ref - idx_tip[1]
                    if abs(delta) > 10:
                        pyautogui.scroll(int(delta / SCROLL_SPEED))
                        scroll_ref = idx_tip[1]

                # ── Drag: fist ──────────────────────────────────────────────
                elif not index and not middle and not ring and not pinky:
                    label = "Drag"
                    scroll_ref = None
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                    pyautogui.moveTo(cx, cy)
                    cv2.circle(frame, idx_tip, 14, (200,80,255), -1)

                # ── Move + Pinch to click ───────────────────────────────────
                else:
                    scroll_ref = None
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                    pyautogui.moveTo(cx, cy)

                    if pinching:
                        # Draw pinch line between thumb and index
                        cv2.line(frame, thm_tip, idx_tip, (60,160,255), 3)
                        cv2.circle(frame, idx_tip, 14, (60,160,255), 3)
                        cv2.circle(frame, thm_tip, 14, (60,160,255), 3)

                        if (cur_t - last_click) > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click = cur_t
                        label = "Click"
                    else:
                        label = "Move"
                        cv2.circle(frame, idx_tip, 10, (0,230,120), -1)
                        cv2.circle(frame, idx_tip, 10, (255,255,255), 1)

            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                scroll_ref = None
                label      = "—"
                pinch_dist = 999

            draw_hud(frame, label, fps, pinch_dist)
            cv2.imshow("Virtual Mouse  |  Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("Virtual Mouse closed.")

if __name__ == "__main__":
    main()