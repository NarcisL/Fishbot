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
OVERLAP_DELAY = 0.  # Wait time after overlap before pressing space

# Statistics
FISH_CAUGHT = 0
START_TIME = time.time()
INIT_FOUND_LAST_FRAME = False  # Track if init was found in the last frame

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
    
    # Run inference with lower confidence threshold
    results = model(frame, conf=0.3, device='0')  # Lower threshold to catch more fish
    
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
                    "type": "fish"
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
    """Check if two boxes overlap"""
    if not box1 or not box2:
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
    
    # Check for overlap
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def perform_action():
    """Press spacebar when fish and gradient overlap"""
    global LAST_ACTION_TIME, FISH_CAUGHT
    
    # Check cooldown
    current_time = time.time()
    if current_time - LAST_ACTION_TIME < ACTION_COOLDOWN:
        return
    
    # Wait specified delay before pressing space
    print(f"🎣 FISH DETECTED IN BOX! Waiting {OVERLAP_DELAY}s before pressing spacebar...")
    time.sleep(OVERLAP_DELAY)  # Wait exactly as specified
    
    # Press spacebar
    pydirectinput.press('space')
    LAST_ACTION_TIME = current_time
    FISH_CAUGHT += 1
    print("✅ Spacebar pressed! Going back to check initial condition...")

def init_fishing():
    """Initialize fishing by pressing space and waiting"""
    global LAST_ACTION_TIME
    
    # Check cooldown
    current_time = time.time()
    if current_time - LAST_ACTION_TIME < ACTION_COOLDOWN:
        return
    
    print("🎮 Initializing fishing sequence...")
    pydirectinput.press('space')
    LAST_ACTION_TIME = current_time
    time.sleep(0.5)  # Wait 0.5 seconds as specified
    print("✅ Pressed space and waited 0.5s")

def main():
    # Declare global variables
    global MODE, BOT_ENABLED, FISH_CAUGHT, START_TIME, INIT_FOUND_LAST_FRAME
    
    print("Starting advanced fishing bot with continuous init checking...")
    print("Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-08-02 14:15:37")
    print("Current User's Login: NarcisL")
    print(f"HSV Monitor region: {HSV_MONITOR}")
    print(f"YOLO detection region: Full screen")
    print(f"Overlap delay: {OVERLAP_DELAY}s")
    print("Bot starts DISABLED. Press 'E' to enable/disable the bot.")
    
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
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        if MODE == "yolo":
            print("Falling back to HSV mode since YOLO model failed to load")
            MODE = "hsv"
    
    # Initialize screen capture
    sct = mss.mss()
    
    # Create display window
    cv2.namedWindow("Fishing Bot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fishing Bot", 1200, 800)
    
    # Create window to show HSV region
    cv2.namedWindow("HSV Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Region", 427, 25)
    
    # Start the detection loop
    try:
        # Flag to track if we need to wait 1 second after enabling
        wait_after_enable = False
        
        while True:
            # Capture the full screen for init check and YOLO
            yolo_img = np.array(sct.grab(FULL_SCREEN))
            yolo_frame = cv2.cvtColor(yolo_img, cv2.COLOR_BGRA2BGR)
            
            # Capture HSV frame for gradient detection
            hsv_img = np.array(sct.grab(HSV_MONITOR))
            hsv_frame = cv2.cvtColor(hsv_img, cv2.COLOR_BGRA2BGR)
            
            # Show the HSV region
            cv2.imshow("HSV Region", hsv_frame)
            
            # Create a copy of the full screen for display
            display_frame = yolo_frame.copy()
            
            # Status to display
            status_message = ""
            
            # If we just enabled the bot, wait 1 second first
            if wait_after_enable:
                print("Bot enabled, waiting 1 second before starting...")
                time.sleep(1.0)
                wait_after_enable = False
                print("Starting fishing workflow")
            
            # Process if bot is enabled
            if BOT_ENABLED:
                # Always check for the initialization condition first
                init_condition_found = False
                
                if cond_init_template is not None:
                    init_match = find_template(yolo_frame, cond_init_template, 0.8)
                    
                    if init_match:
                        # Condition image found, look for fish and gradient
                        init_condition_found = True
                        
                        # Only announce when we first find it
                        if not INIT_FOUND_LAST_FRAME:
                            print("✅ Found initialization condition on screen")
                        
                        status_message = "Init condition found, looking for fish overlap"
                        
                        # Draw where we found the template
                        cv2.rectangle(
                            display_frame,
                            (init_match["x"], init_match["y"]),
                            (init_match["x"] + init_match["w"], init_match["y"] + init_match["h"]),
                            (0, 255, 0),  # Green
                            2
                        )
                        
                        # Look for fish and gradient box
                        gradient_box = None
                        fish_box = None
                        
                        # Perform detection based on selected mode
                        if MODE == "hsv" or MODE == "both":
                            gradient_box = detect_hsv(hsv_frame)
                            if gradient_box:
                                # Adjust coordinates to full screen reference
                                gradient_box["x_screen"] = gradient_box["x"] + HSV_MONITOR["left"]
                                gradient_box["y_screen"] = gradient_box["y"] + HSV_MONITOR["top"]
                            
                        if (MODE == "yolo" or MODE == "both") and model:
                            fish_box = detect_yolo(yolo_frame, model)
                        
                        # Check for overlap and perform action
                        if gradient_box and fish_box:
                            if check_overlap(gradient_box, fish_box):
                                perform_action()  # This will wait OVERLAP_DELAY before pressing space
                                
                                # Draw a special indicator for overlap
                                cv2.putText(
                                    display_frame,
                                    "OVERLAP DETECTED!",
                                    (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (0, 255, 255),  # Yellow
                                    2
                                )
                        
                        # Draw detection boxes
                        if gradient_box:
                            # Draw the gradient box at its actual screen position
                            cv2.rectangle(
                                display_frame,
                                (gradient_box["x_screen"], gradient_box["y_screen"]),
                                (gradient_box["x_screen"] + gradient_box["w"], gradient_box["y_screen"] + gradient_box["h"]),
                                gradient_box["color"],
                                2
                            )
                            cv2.putText(
                                display_frame,
                                gradient_box["name"],
                                (gradient_box["x_screen"], gradient_box["y_screen"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                gradient_box["color"],
                                2
                            )
                        
                        if fish_box:
                            # Draw the fish box
                            cv2.rectangle(
                                display_frame,
                                (fish_box["x"], fish_box["y"]),
                                (fish_box["x"] + fish_box["w"], fish_box["y"] + fish_box["h"]),
                                fish_box["color"],
                                2
                            )
                            cv2.putText(
                                display_frame,
                                fish_box["name"],
                                (fish_box["x"], fish_box["y"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                fish_box["color"],
                                2
                            )
                    else:
                        # Condition image not found, initialize fishing
                        if INIT_FOUND_LAST_FRAME:
                            print("❌ Lost initialization condition")
                        
                        status_message = "Init condition not found, initializing fishing"
                        print("❌ Initialization condition not met, pressing space to start fishing")
                        init_fishing()
                else:
                    # No template available
                    status_message = "No init template available"
                    print("⚠️ No init template available, can't check initialization condition")
                
                # Update our tracking flag
                INIT_FOUND_LAST_FRAME = init_condition_found
            else:
                # Bot is disabled, just show current screen
                status_message = "Bot is disabled. Press 'E' to enable."
            
            # Draw rectangle around HSV monitor region
            cv2.rectangle(
                display_frame,
                (HSV_MONITOR["left"], HSV_MONITOR["top"]),
                (HSV_MONITOR["left"] + HSV_MONITOR["width"], HSV_MONITOR["top"] + HSV_MONITOR["height"]),
                (255, 255, 255),
                2
            )
            
            # Calculate runtime statistics
            runtime = time.time() - START_TIME
            hours, remainder = divmod(int(runtime), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            
            # Display bot status and mode info
            bot_status = "ENABLED" if BOT_ENABLED else "DISABLED"
            init_status = "FOUND" if INIT_FOUND_LAST_FRAME else "NOT FOUND"
            
            cv2.putText(
                display_frame,
                f"Bot: {bot_status} | Init: {init_status} | Mode: {MODE} | Fish Caught: {FISH_CAUGHT} | Runtime: {runtime_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display status message if any
            if status_message:
                cv2.putText(
                    display_frame,
                    status_message,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),  # Yellow
                    2
                )
            
            # Display help
            cv2.putText(
                display_frame,
                "Controls: [E] Toggle Bot | [H] HSV | [Y] YOLO | [B] Both | [R] Reset Stats | [Q] Quit",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Show the main display frame
            cv2.imshow("Fishing Bot", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):  # Toggle bot
                BOT_ENABLED = not BOT_ENABLED
                status = "ENABLED" if BOT_ENABLED else "DISABLED"
                print(f"Bot {status}")
                
                # If we're enabling, set the flag to wait 1 second
                if BOT_ENABLED:
                    wait_after_enable = True
                    INIT_FOUND_LAST_FRAME = False
            elif key == ord('h'):
                MODE = "hsv"
                print("Switched to HSV detection mode")
            elif key == ord('y'):
                if model:
                    MODE = "yolo"
                    print("Switched to YOLO detection mode")
                else:
                    try:
                        model = YOLO(MODEL_PATH)
                        MODE = "yolo"
                        print("Loaded YOLO model and switched to YOLO detection mode")
                    except Exception as e:
                        print(f"Failed to load YOLO model: {e}")
            elif key == ord('b'):
                if not model:
                    try:
                        model = YOLO(MODEL_PATH)
                        print("Loaded YOLO model")
                    except Exception as e:
                        print(f"Failed to load YOLO model: {e}")
                MODE = "both"
                print("Switched to combined detection mode")
            elif key == ord('r'):  # Reset stats
                FISH_CAUGHT = 0
                START_TIME = time.time()
                print("Statistics reset")
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Fishing bot stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Fishing bot closed")
        print(f"Total fish caught: {FISH_CAUGHT}")
        runtime = time.time() - START_TIME
        hours, remainder = divmod(int(runtime), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total runtime: {hours}h {minutes}m {seconds}s")

if __name__ == '__main__':
    # Print start info
    print("===== FISHING BOT STARTED =====")
    print("Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-08-02 14:15:37")
    print("Current User's Login: NarcisL")
    
    # Start the main loop
    main()