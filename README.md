# Virtual-Mouse using Hand Gestures
Control your computer mouse using just your hand via webcam. This project uses MediaPipe + OpenCV + PyAutoGUI to detect hand gestures and convert them into real-time mouse actions.

## Features

* Move cursor using index finger
* Click using pinch (thumb and index finger)
* Scroll with all fingers up
* Drag using a fist
* Smooth cursor movement
* Real-time gesture and FPS display

---

## How it Works

* Detects hand landmarks using MediaPipe
* Tracks finger positions
* Identifies gestures based on finger states and distances
* Maps gestures to mouse actions using PyAutoGUI

---

## Controls

| Gesture               | Action      |
| --------------------- | ----------- |
| Index finger up       | Move cursor |
| Pinch (thumb + index) | Click       |
| All fingers up        | Scroll      |
| Fist                  | Drag        |
| Q key                 | Quit        |

---
