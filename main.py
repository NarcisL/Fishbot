import pyautogui
import pydirectinput
import tkinter as tk
import time
import os
from ultralytics import YOLO
import mss
import numpy as np
import cv2

start_x = start_y = end_x = end_y = 0 

def on_mouse_down(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y

def on_mouse_up(event):
    global end_x, end_y
    end_x, end_y = event.x, event.y
    root.destroy()


print("Current working directory:", os.getcwd())

root = tk.Tk()
root.attributes("-fullscreen", True)
root.attributes("-alpha", 0.3)
root.configure(bg='gray')
root.bind("<ButtonPress-1>", on_mouse_down)
root.bind("<ButtonRelease-1>", on_mouse_up)
tk.Label(root, text="Drag to select the zone (left mouse button)", font=("Arial", 24), bg='gray').pack(pady=20)
root.mainloop()

zone_left = min(start_x, end_x)
zone_top = min(start_y, end_y)
zone_right = max(start_x, end_x)
zone_bottom = max(start_y, end_y)

zone_width = zone_right - zone_left
zone_height = zone_bottom - zone_top

print("Selected zone: ({}, {}) to ({}, {})".format(zone_left, zone_top, zone_right, zone_bottom))

gradient_filenames = ['g1.png', 'g2.png', 'g3.png', 'g4.png']

pydirectinput.moveTo((zone_left+ zone_right) // 2, (zone_top + zone_bottom) // 2)

time.sleep(3)

pydirectinput.click()

MODEL_PATH = "C:/Users/narci/Desktop/fishbot_elve/dataset/runs/detect/train/weights/best.pt"  
print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

mon = {"top": zone_top, "left": zone_left, "width": zone_width, "height": zone_height}
print(f"[INFO] Monitoring region: {mon}")

def boxes_overlap(boxA, boxB):
    # boxA and boxB: (x1, y1, x2, y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    # Compute overlap
    overlap_x1 = max(ax1, bx1)
    overlap_y1 = max(ay1, by1)
    overlap_x2 = min(ax2, bx2)
    overlap_y2 = min(ay2, by2)
    return overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2

def safe_locate_on_screen(image, **kwargs):
    try:
        return pyautogui.locateOnScreen(image, **kwargs)
    except Exception as e:
        print(f"[WARN] Failed to locate {image}: {e}")
        return None

with mss.mss() as sct:
    while True:
        # Check condition before proceeding (optional)
        cond_ok = safe_locate_on_screen("cond_init.png", confidence=0.8)
        if not cond_ok:
            print("Condition not met, pressing space.")
            pydirectinput.press('space')
            time.sleep(2)
            continue

        # Screenshot of the selected region
        frame = np.array(sct.grab(mon))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # YOLO inference
        results = model(frame_rgb)
        fish_boxes = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            fish_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        print(f"[INFO] Fish boxes: {fish_boxes}")

        # Gradient detection (in screen coordinates)
        gradient_boxes = []
        for gradients in gradient_filenames:
            try:
                gradient = safe_locate_on_screen(gradients, confidence=0.9, region=(zone_left, zone_top, zone_width, zone_height))
                if gradient:
                    # Adjust coords relative to zone
                    g_box = (
                        gradient.left - zone_left,
                        gradient.top - zone_top,
                        gradient.left + gradient.width - zone_left,
                        gradient.top + gradient.height - zone_top
                    )
                    gradient_boxes.append(g_box)
                    print("[INFO] Gradient found:", gradients, g_box)
            except Exception as e:
                print(f"Error locating gradient {gradients}: {e}")
                continue

        # Check for any overlap
        overlap_found = False
        for fbox in fish_boxes:
            for gbox in gradient_boxes:
                if boxes_overlap(fbox, gbox):
                    print("[ACTION] Overlap detected! Pressing space.")
                    pydirectinput.keyDown('space')
                    time.sleep(0.1)
                    pydirectinput.keyUp('space')
                    overlap_found = True
                    break
            if overlap_found:
                break

        time.sleep(0.1)