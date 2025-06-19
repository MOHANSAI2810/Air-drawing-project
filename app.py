import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque # For smoothing history
import datetime # For timestamping saved files

# --- 1. Initialize MediaPipe ---
mp_hands = mp.solutions.hands
# Increased min_detection_confidence for more stable tracking
# Added min_tracking_confidence
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Set window size
window_width = 1200
window_height = 600

# Canvas setup
canvas_width = window_width // 2
canvas_height = window_height
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
canvas[:] = (40, 40, 40) # Dark gray background

# Drawing variables
prev_x, prev_y = 0, 0
drawing_active = False
current_color = (0, 200, 255) # Default orange (BGR)
brush_size = 5 # Constant brush size
mode = 'draw' # Default mode is drawing

# Add a state for button click detection to prevent rapid multiple clicks
button_pressed_cooldown = False
button_cooldown_frames = 10 # Number of frames to wait after a button press
cooldown_counter = 0

# --- Smoothing History ---
finger_tip_history = deque(maxlen=7) # Store last N positions for averaging (tune this value)

# --- Save Message Variables ---
show_save_message = False
save_message_display_frames = 60 # Show message for 60 frames (approx 2 seconds at 30 FPS)
save_message_counter = 0
last_saved_filename = ""

# --- 2. Buttons (x1, y1, x2, y2, label, color, highlight_color) ---
# Buttons are relative to the camera_display panel (left side: window_width // 2)
# Added a highlight color for visual feedback
buttons = [
    (10, 10, 110, 60, 'Draw', (100, 100, 100), (0, 200, 200)), # New Draw button
    (120, 10, 220, 60, 'Eraser', (50, 50, 50), (100, 100, 100)), # Eraser button
    (230, 10, 330, 60, 'Clear', (100, 100, 100), (150, 150, 150)),
    (340, 10, 440, 60, 'Save', (100, 100, 100), (150, 150, 150)),

    # Color buttons
    (10, 70, 80, 120, 'Red', (0, 0, 200), (0, 0, 255)),
    (90, 70, 160, 120, 'Green', (0, 200, 0), (0, 255, 0)),
    (170, 70, 240, 120, 'Blue', (200, 0, 0), (255, 0, 0)),
    (250, 70, 320, 120, 'Orange', (0, 140, 255), (0, 165, 255)), # Added Orange button
    (330, 70, 400, 120, 'Black', (0, 0, 0), (50, 50, 50)), # Added Black button
]

# Track button highlight states
button_highlights = {btn[4]: False for btn in buttons}

# Initialize current active button highlights
button_highlights['Draw'] = True # Draw is default mode
button_highlights['Orange'] = True # Default color is orange

def draw_buttons(img, current_color_val, current_mode):
    global button_highlights
    for x1, y1, x2, y2, label, btn_color, highlight_color in buttons:
        display_color = btn_color

        # Persistent highlight for active mode/color
        if (label == 'Draw' and current_mode == 'draw') or \
           (label == 'Eraser' and current_mode == 'erase') or \
           (label == 'Red' and current_color_val == (0, 0, 255)) or \
           (label == 'Green' and current_color_val == (0, 255, 0)) or \
           (label == 'Blue' and current_color_val == (255, 0, 0)) or \
           (label == 'Orange' and current_color_val == (0, 140, 255)) or \
           (label == 'Black' and current_color_val == (0, 0, 0)) or \
           button_highlights[label]: # Or if it's currently highlighted by hover/recent press
            display_color = highlight_color

        cv2.rectangle(img, (x1, y1), (x2, y2), display_color, -1) # Filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2) # White border
        cv2.putText(img, label, (x1 + 10, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def check_button_press(cursor_x, cursor_y, original_frame_width, original_frame_height):
    global current_color, canvas, drawing_active, brush_size, button_pressed_cooldown, button_highlights, mode
    global show_save_message, save_message_counter, last_saved_filename

    if button_pressed_cooldown:
        return # Skip if cooldown is active

    # Map cursor_x, cursor_y (from original frame resolution) to the resized cam_display resolution
    scale_x = (window_width // 2) / original_frame_width
    scale_y = window_height / original_frame_height

    scaled_cursor_x = int(cursor_x * scale_x)
    scaled_cursor_y = int(cursor_y * scale_y)

    for i, (x1, y1, x2, y2, label, _, _) in enumerate(buttons):
        if x1 < scaled_cursor_x < x2 and y1 < scaled_cursor_y < y2:
            # Activate button press
            button_pressed_cooldown = True

            # Clear all non-sticky highlights before setting new ones
            for key in button_highlights:
                # Keep active mode/color highlights
                if not (key == 'Draw' and mode == 'draw') and \
                   not (key == 'Eraser' and mode == 'erase') and \
                   not ((key == 'Red' and current_color == (0, 0, 255)) or \
                        (key == 'Green' and current_color == (0, 255, 0)) or \
                        (key == 'Blue' and current_color == (255, 0, 0)) or \
                        (key == 'Orange' and current_color == (0, 140, 255)) or \
                        (key == 'Black' and current_color == (0, 0, 0))):
                    button_highlights[key] = False

            # Set highlight for the pressed button
            button_highlights[label] = True

            if label == 'Draw':
                mode = 'draw'
            elif label == 'Eraser':
                mode = 'erase'
            elif label == 'Clear':
                canvas[:] = (40, 40, 40)
            elif label == 'Save':
                current_time = datetime.datetime.now()
                filename = f"air_drawing_snapshot_{current_time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, canvas)
                last_saved_filename = filename
                show_save_message = True # Activate save message
                save_message_counter = save_message_display_frames # Reset message timer
                print(f"Drawing saved as {filename}")
            elif label == 'Red':
                current_color = (0, 0, 255)
                mode = 'draw' # Switch to draw mode when a color is selected
            elif label == 'Green':
                current_color = (0, 255, 0)
                mode = 'draw'
            elif label == 'Blue':
                current_color = (255, 0, 0)
                mode = 'draw'
            elif label == 'Orange':
                current_color = (0, 140, 255)
                mode = 'draw'
            elif label == 'Black':
                current_color = (0, 0, 0)
                mode = 'draw'
            return # Only process one button press per interaction

# --- 3. Robust Gesture Recognition ---
def is_pinch_gesture(hand_landmarks, pinch_threshold=0.04): # Normalized distance threshold
    # Thumb tip (4) and Index finger tip (8)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)
    return distance < pinch_threshold

def is_index_finger_extended(hand_landmarks):
    # Check if index finger tip is significantly above its PIP and MCP joints
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

    # Check that tip is above PIP, and PIP is above MCP, and MCP is generally below wrist
    return index_tip_y < index_pip_y and index_pip_y < index_mcp_y and index_mcp_y < wrist_y

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape # Get original frame dimensions
    frame = cv2.flip(frame, 1) # Flip horizontally for natural interaction

    display = np.zeros((window_height, window_width, 3), dtype=np.uint8)

    # Left side - camera feed with controls
    cam_display = cv2.resize(frame, (canvas_width, window_height))

    # Draw buttons on the resized camera display
    draw_buttons(cam_display, current_color, mode)

    display[0:window_height, 0:canvas_width] = cam_display

    # Right side - drawing canvas
    display[0:window_height, canvas_width:window_width] = canvas

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_tip_x_cam, finger_tip_y_cam = 0, 0 # Finger tip on camera_display (scaled)
    finger_tip_x_canvas, finger_tip_y_canvas = 0, 0 # Finger tip on drawing canvas
    hand_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw MediaPipe landmarks and connections (optional, for debugging/visual)
            mp_draw.draw_landmarks(cam_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates (normalized 0-1)
            idx_tip_norm_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            idx_tip_norm_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Convert to pixel coordinates on the *original frame* for button detection
            cursor_x_original = int(idx_tip_norm_x * frame_width)
            cursor_y_original = int(idx_tip_norm_y * frame_height)

            # Convert to pixel coordinates for drawing on the *canvas*
            current_canvas_x = int(idx_tip_norm_x * canvas_width)
            current_canvas_y = int(idx_tip_norm_y * canvas_height)

            # Add current tip to history for smoothing
            finger_tip_history.append((current_canvas_x, current_canvas_y))

            # Calculate smoothed coordinates
            if len(finger_tip_history) == finger_tip_history.maxlen:
                smoothed_x = int(np.mean([p[0] for p in finger_tip_history]))
                smoothed_y = int(np.mean([p[1] for p in finger_tip_history]))
            else:
                # If not enough history, use raw coordinates for now
                smoothed_x, smoothed_y = current_canvas_x, current_canvas_y

            finger_tip_x_canvas, finger_tip_y_canvas = smoothed_x, smoothed_y


            # Convert to pixel coordinates for displaying cursor on *cam_display*
            # Use smoothed coordinates for cursor too for consistency
            finger_tip_x_cam = int(smoothed_x * (canvas_width / canvas_width)) # Rescale from canvas_width to cam_display width
            finger_tip_y_cam = int(smoothed_y * (window_height / canvas_height)) # Rescale from canvas_height to cam_display height

            hand_detected = True

            # --- Draw Circular Pointer on Cam Display (Left Panel) ---
            # Draw a bright circle on the index finger tip in the camera view
            cv2.circle(cam_display, (finger_tip_x_cam, finger_tip_y_cam), 8, (0, 255, 255), -1) # Yellow filled circle
            cv2.circle(cam_display, (finger_tip_x_cam, finger_tip_y_cam), 10, (255, 255, 255), 2) # White border

            # --- Gesture Logic ---
            # If pinching, check for button press
            if is_pinch_gesture(hand_landmarks):
                # Draw a green circle to indicate pinch
                cv2.circle(cam_display, (finger_tip_x_cam, finger_tip_y_cam), 15, (0, 255, 0), -1)
                check_button_press(cursor_x_original, cursor_y_original, frame_width, frame_height)
                # When pinching, stop drawing
                drawing_active = False
                prev_x, prev_y = 0, 0 # Reset prev_x, prev_y when not drawing
                finger_tip_history.clear() # Clear history when not drawing

            # If index finger is extended and NOT pinching, activate drawing
            elif is_index_finger_extended(hand_landmarks):
                drawing_active = True
                # Draw cursor on the combined display (on the canvas side)
                cv2.circle(display, (finger_tip_x_canvas + canvas_width, finger_tip_y_canvas), brush_size + 3, current_color, -1)
            else:
                # If neither pinch nor extended index, drawing is inactive
                drawing_active = False
                prev_x, prev_y = 0, 0 # Reset prev_x, prev_y when not drawing
                finger_tip_history.clear() # Clear history when not drawing

            # --- Drawing Logic ---
            if drawing_active:
                if prev_x != 0 and prev_y != 0:
                    if mode == 'draw':
                        cv2.line(canvas, (prev_x, prev_y), (finger_tip_x_canvas, finger_tip_y_canvas), current_color, brush_size)
                    elif mode == 'erase':
                        cv2.line(canvas, (prev_x, prev_y), (finger_tip_x_canvas, finger_tip_y_canvas), (40, 40, 40), brush_size * 2) # Eraser is thicker and matches canvas background
                prev_x, prev_y = finger_tip_x_canvas, finger_tip_y_canvas
            else:
                prev_x, prev_y = 0, 0 # Reset prev_x, prev_y when drawing is inactive

    else:
        # If no hand is detected, reset drawing state and clear history
        drawing_active = False
        prev_x, prev_y = 0, 0
        finger_tip_history.clear()

    # Button cooldown management
    if button_pressed_cooldown:
        cooldown_counter += 1
        if cooldown_counter > button_cooldown_frames:
            button_pressed_cooldown = False
            cooldown_counter = 0
            # Remove highlight from action-only buttons after cooldown
            for label in button_highlights:
                if label not in ['Draw', 'Eraser', 'Red', 'Green', 'Blue', 'Orange', 'Black']: # These are "sticky" highlights
                    button_highlights[label] = False

    # --- Save Message Display Logic ---
    if show_save_message:
        message = f"Drawing saved as {last_saved_filename}!"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (window_width - text_size[0]) // 2 # Center horizontally
        text_y = window_height - 100 # Position near bottom

        # Draw a translucent background for the message
        overlay = display.copy()
        cv2.rectangle(overlay, (text_x - 20, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 20, text_y + 20), (50, 50, 50), -1) # Dark background
        alpha = 0.6
        display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)

        cv2.putText(display, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        save_message_counter -= 1
        if save_message_counter <= 0:
            show_save_message = False


    # Add instructions
    cv2.putText(display, "Pinch to select buttons", (10, window_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Extend index finger to draw", (10, window_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Add current color and mode display
    cv2.putText(display, f"Color: {current_color}", (canvas_width + 10, window_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 1)
    cv2.putText(display, f"Mode: {mode.capitalize()}", (canvas_width + 10, window_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


    cv2.imshow("Air Drawing", display)

    key = cv2.waitKey(1)
    if key == 27: # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()