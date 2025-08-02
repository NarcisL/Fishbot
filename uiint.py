import sys
import numpy as np
import mss
import cv2
from ultralytics import YOLO
import time
import os
import pydirectinput
import random
from datetime import datetime
import tkinter as tk
import threading

# Model path for YOLO
MODEL_PATH = "C:/Users/narci/Desktop/fishbot_elve/dataset/runs/detect/train3/weights/best.pt"

# Condition image path
COND_INIT_PATH = "C:/Users/narci/Desktop/fishbot_elve/cond_init.png"

# Specific region for HSV gradient detection
HSV_MONITOR = {"top": 230, "left": 745, "width": 427, "height": 25}

# Entire screen for YOLO fish detection - adjust as needed for your screen
FULL_SCREEN = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust to your screen resolution

# HSV ranges for gradients
GRADIENTS = [
    {"name": "green",  "lower": (40, 40, 80),  "upper": (70, 160, 255), "color": (0, 255, 0)},
    {"name": "purple", "lower": (130, 40, 80), "upper": (170, 255, 255), "color": (128, 0, 128)},
    {"name": "gray",   "lower": (0, 0, 60),    "upper": (180, 70, 180), "color": (128, 128, 128)},
    {"name": "yellow", "lower": (20, 60, 90),  "upper": (35, 255, 255), "color": (0, 255, 255)},
]

# Detection mode: "hsv", "yolo", or "both"
MODE = "both"  # Default to both

# Bot configuration
BOT_ENABLED = False  # Start with bot disabled
LAST_ACTION_TIME = 0
ACTION_COOLDOWN = 1.0  # Seconds between actions
OVERLAP_DELAY = 0.14  # Wait time after overlap before pressing space

# Global variables
INIT_FOUND_LAST_FRAME = False  # Track if init was found in the last frame
RUNNING = True  # Flag to control main loop
TOGGLE_BUTTON = None  # Reference to the toggle button

def detect_hsv(frame):
    """Detect gradients using HSV color ranges"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # We'll keep track of the best box (largest area)
    best_box = None
    max_area = 0
    
    for grad in GRADIENTS:
        mask = cv2.inRange(hsv, np.array(grad["lower"]), np.array(grad["upper"]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_area:  # Keep only the largest area
                max_area = area
                x, y, w, h = cv2.boundingRect(cnt)
                best_box = {
                    "name": grad["name"],
                    "color": grad["color"],
                    "x": x, 
                    "y": y, 
                    "w": w, 
                    "h": h,
                    "area": area,
                    "type": "gradient"
                }
    
    return best_box

def detect_yolo(frame, model):
    """Detect objects using YOLO model"""
    # Ensure the frame is in the right format for YOLO
    if frame.shape[2] == 4:  # Check if we have an alpha channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Run inference with lower confidence threshold for more detections
    results = model(frame, conf=0.4, device='0')  # Lower to 0.2 to catch more fish
    
    # Debug print full YOLO results
    print(f"YOLO detected {len(results[0].boxes) if len(results) > 0 and hasattr(results[0], 'boxes') else 0} objects")
    
    # Find the best detection
    best_box = None
    max_conf = 0
    
    if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for i in range(len(results[0].boxes)):
            box = results[0].boxes.xyxy[i].cpu().numpy()
            conf = results[0].boxes.conf[i].item()
            
            # Get class name if available
            class_id = 0
            if hasattr(results[0].boxes, 'cls') and len(results[0].boxes.cls) > i:
                class_id = int(results[0].boxes.cls[i].item())
            class_name = model.names[class_id] if hasattr(model, 'names') else "Fish"
            
            # Debug each detection
            print(f"  Detection {i}: class={class_name}, conf={conf:.2f}")
            
            # Accept all classes for now - we're debugging
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box)
                best_box = {
                    "name": f"{class_name} {conf:.2f}",
                    "color": (0, 0, 255),  # Red for YOLO detections
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": conf,
                    "type": "fish",
                    "class": class_name
                }
    
    return best_box

def find_template(screen, template, threshold=0.8):
    """Find a template image on the screen using template matching"""
    # Convert images to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # If the best match exceeds our threshold, return the location
    if max_val >= threshold:
        w, h = template_gray.shape[::-1]
        return {
            "x": max_loc[0],
            "y": max_loc[1],
            "w": w,
            "h": h,
            "confidence": max_val
        }
    
    return None

def check_overlap(box1, box2):
    """Check if two boxes overlap with more lenient matching"""
    if not box1 or not box2:
        print("Missing box - can't check overlap")
        return False
    
    # For box1, adjust to screen coordinates if it's a gradient box
    x1 = box1.get("x_screen", box1["x"])
    y1 = box1.get("y_screen", box1["y"])
    w1 = box1["w"]
    h1 = box1["h"]
    
    # For box2, adjust to screen coordinates if it's a gradient box
    x2 = box2.get("x_screen", box2["x"])
    y2 = box2.get("y_screen", box2["y"])
    w2 = box2["w"]
    h2 = box2["h"]
    
    # Print box coordinates for debugging
    print(f"GRADIENT: x={x1}, y={y1}, w={w1}, h={h1}, type={box1.get('name', 'unknown')}")
    print(f"FISH: x={x2}, y={y2}, w={w2}, h={h2}, conf={box2.get('conf', 0)}")
    
    # Calculate vertical distance from fish to gradient (since gradient is usually above fish)
    vertical_distance = abs(y2 - (y1 + h1))
    print(f"Vertical distance from gradient to fish: {vertical_distance} pixels")
    
    # Make detection more lenient by expanding gradient box and allowing vertical alignment
    x1 -= 10
    y1 -= 10
    w1 += 20
    h1 += vertical_distance  # Extend gradient box down to the fish
    
    # Debug expanded box
    print(f"EXPANDED GRADIENT: x={x1}, y={y1}, w={w1}, h={h1}")
    
    # Check for overlap
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        print("✓✓✓ OVERLAP DETECTED! ✓✓✓")
        return True
    
    # If no overlap, print the distances
    x_overlap = not (x1 + w1 < x2 or x1 > x2 + w2)
    y_overlap = not (y1 + h1 < y2 or y1 > y2 + h2)
    print(f"Overlap status: X={x_overlap}, Y={y_overlap}")
    
    # Alternative check: if they align horizontally and are close vertically, consider it a match
    if x_overlap and vertical_distance < 50:  # If within 50 pixels vertically
        print("✓✓✓ VERTICAL ALIGNMENT MATCH! ✓✓✓")
        return True
    
    return False

def perform_action():
    """Press spacebar when fish and gradient overlap"""
    global LAST_ACTION_TIME
    
    # Check cooldown
    current_time = time.time()
    if current_time - LAST_ACTION_TIME < ACTION_COOLDOWN:
        print(f"Cooldown active ({ACTION_COOLDOWN - (current_time - LAST_ACTION_TIME):.2f}s left)")
        return
    
    print("⚡ PRESSING SPACE NOW ⚡")
    
    # Use multiple methods to ensure the keypress is registered
    try:
        # Method 1: Direct press
        pydirectinput.press('space')
        
        # Method 2: Press and release with small delay
        time.sleep(0.02)
        pydirectinput.keyDown('space')
        time.sleep(0.02)
        pydirectinput.keyUp('space')
    except Exception as e:
        print(f"Error pressing space: {e}")
    
    LAST_ACTION_TIME = current_time
    print("✅ Spacebar pressed successfully!")

def init_fishing():
    """Initialize fishing by pressing space and waiting"""
    global LAST_ACTION_TIME
    
    # Check cooldown
    current_time = time.time()
    if current_time - LAST_ACTION_TIME < ACTION_COOLDOWN:
        return
    
    print("🎮 Initializing fishing...")
    pydirectinput.press('space')
    LAST_ACTION_TIME = current_time
    time.sleep(0.5)  # Wait 0.5 seconds as specified
    print("✅ Fishing initialized")

def toggle_bot():
    """Toggle the bot on/off"""
    global BOT_ENABLED, TOGGLE_BUTTON, INIT_FOUND_LAST_FRAME
    
    BOT_ENABLED = not BOT_ENABLED
    
    if BOT_ENABLED:
        TOGGLE_BUTTON.config(text="DISABLE BOT", bg="#ff6b6b")
        print("Bot ENABLED - waiting 1 second before starting")
        time.sleep(1.0)  # Wait 1 second before starting
        INIT_FOUND_LAST_FRAME = False
        print("Starting fishing workflow")
    else:
        TOGGLE_BUTTON.config(text="ENABLE BOT", bg="#4cd137")
        print("Bot DISABLED")

def on_closing():
    """Handle window closing"""
    global RUNNING
    print("Closing fishing bot...")
    RUNNING = False

def create_button_window():
    """Create a minimal window with just an enable/disable button"""
    global TOGGLE_BUTTON
    
    # Create window
    root = tk.Tk()
    root.title("Fishing Bot")
    root.geometry("200x100")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Create and configure button
    TOGGLE_BUTTON = tk.Button(
        root, 
        text="ENABLE BOT", 
        command=toggle_bot,
        font=("Arial", 12, "bold"),
        bg="#4cd137",  # Green
        fg="white",
        height=3
    )
    TOGGLE_BUTTON.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    return root

def bot_loop(model, cond_init_template):
    """The main bot loop that runs continuously"""
    global BOT_ENABLED, INIT_FOUND_LAST_FRAME, RUNNING
    
    print("Bot loop started - running permanently")
    
    # Create a new MSS instance specifically for this thread
    # This is important as MSS is not thread-safe
    with mss.mss() as sct:
        # Loop until the application is closed
        while RUNNING:
            try:
                # Skip processing if bot is disabled
                if not BOT_ENABLED:
                    # Sleep to reduce CPU usage when idle
                    time.sleep(0.1)
                    continue
                
                # STEP 1: Capture screens for processing
                yolo_img = np.array(sct.grab(FULL_SCREEN))
                yolo_frame = cv2.cvtColor(yolo_img, cv2.COLOR_BGRA2BGR)
                
                hsv_img = np.array(sct.grab(HSV_MONITOR))
                hsv_frame = cv2.cvtColor(hsv_img, cv2.COLOR_BGRA2BGR)
                
                # STEP 2: Always check for the initialization condition first
                init_condition_found = False
                
                if cond_init_template is not None:
                    init_match = find_template(yolo_frame, cond_init_template, 0.8)
                    
                    if init_match:
                        # STEP 3A: Condition image found, look for fish and gradient
                        init_condition_found = True
                        
                        # Only announce when we first find it
                        if not INIT_FOUND_LAST_FRAME:
                            print("✅ Found initialization condition on screen")
                        
                        # Look for fish and gradient box
                        gradient_box = None
                        fish_box = None
                        
                        # Detect gradient first
                        if MODE == "hsv" or MODE == "both":
                            gradient_box = detect_hsv(hsv_frame)
                            if gradient_box:
                                # Adjust coordinates to full screen reference
                                gradient_box["x_screen"] = gradient_box["x"] + HSV_MONITOR["left"]
                                gradient_box["y_screen"] = gradient_box["y"] + HSV_MONITOR["top"]
                                print(f"Found gradient: {gradient_box['name']} (area: {gradient_box['area']})")
                            else:
                                print("No gradient detected in this frame")
                        
                        # Then detect fish
                        if (MODE == "yolo" or MODE == "both") and model:
                            print("Running YOLO detection...")
                            fish_box = detect_yolo(yolo_frame, model)
                            if fish_box:
                                print(f"Found fish: {fish_box['name']} at ({fish_box['x']},{fish_box['y']})")
                            else:
                                print("No fish detected in this frame")
                        
                        # Check for overlap and perform action
                        if gradient_box and fish_box:
                            print("Found both gradient and fish boxes, checking for overlap...")
                            if check_overlap(gradient_box, fish_box):
                                print(f"⚠️ OVERLAP DETECTED: Fish {fish_box.get('name', 'unknown')} with {gradient_box.get('name', 'unknown')}")
                                perform_action()
                    else:
                        # STEP 3B: Condition image not found, initialize fishing
                        if INIT_FOUND_LAST_FRAME:
                            print("❌ Lost initialization condition")
                        
                        print("❌ Initialization condition not met, pressing space to start fishing")
                        init_fishing()
                else:
                    # No template available
                    print("⚠️ No init template available, can't check initialization condition")
                
                # Update our tracking flag
                INIT_FOUND_LAST_FRAME = init_condition_found
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in bot loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Sleep on error to avoid tight error loops
    
    print("Bot loop ended")

def main():
    global BOT_ENABLED, RUNNING
    
    print("===== FISHING BOT STARTED =====")
    print("Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-08-02 15:52:21")
    print("Current User's Login: NarcisL")
    print("Starting fishing bot with permanent loop...")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: YOLO model file not found at {MODEL_PATH}")
    
    # Check if condition image exists
    cond_init_template = None
    if not os.path.exists(COND_INIT_PATH):
        print(f"WARNING: Condition image not found at {COND_INIT_PATH}")
    else:
        # Load the condition image template
        cond_init_template = cv2.imread(COND_INIT_PATH)
        if cond_init_template is None:
            print(f"ERROR: Could not load condition image from {COND_INIT_PATH}")
        else:
            print(f"Successfully loaded initialization condition image: {COND_INIT_PATH}")
    
    # Initialize YOLO model
    model = None
    try:
        print(f"Loading YOLO model from {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully")
        
        # Print available classes
        if hasattr(model, 'names'):
            print("Available classes in YOLO model:")
            for class_id, class_name in model.names.items():
                print(f"  {class_id}: {class_name}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        if MODE == "yolo":
            print("Falling back to HSV mode since YOLO model failed to load")
            MODE = "hsv"
    
    # Create the button window
    root = create_button_window()
    
    # Start time for logging
    start_time = time.time()
    
    try:
        # Start bot loop in a separate thread so it doesn't block the UI
        # Don't pass the mss instance - create a new one in the thread
        bot_thread = threading.Thread(target=bot_loop, args=(model, cond_init_template))
        bot_thread.daemon = True  # Thread will close when main program exits
        bot_thread.start()
        
        # Main UI loop
        while RUNNING:
            try:
                root.update()
                time.sleep(0.01)  # Short sleep to reduce CPU usage
            except tk.TclError:
                # Window was closed
                RUNNING = False
                break
    
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Calculate runtime
        runtime = time.time() - start_time
        hours, remainder = divmod(int(runtime), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Bot closed. Total runtime: {hours}h {minutes}m {seconds}s")
        
        # Ensure the window is closed
        try:
            root.destroy()
        except:
            pass

if __name__ == '__main__':
    main()