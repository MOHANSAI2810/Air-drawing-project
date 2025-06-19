import cv2
import mediapipe as mp
import numpy as np
import math # Import the math module for math.sqrt

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
# The canvas occupies the right half of the display window
canvas_width = window_width // 2
canvas_height = window_height
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
canvas[:] = (40, 40, 40) # Dark gray background

# Drawing variables
prev_x, prev_y = 0, 0
drawing_active = False # Renamed 'drawing' to 'drawing_active' for clarity
current_color = (0, 200, 255) # Default orange (BGR)
brush_size = 5 # Constant brush size

# Add a state for button click detection to prevent rapid multiple clicks
button_pressed_cooldown = False
button_cooldown_frames = 10 # Number of frames to wait after a button press

# --- 2. Buttons (x1, y1, x2, y2, label, color, highlight_color) ---
# Buttons are relative to the camera_display panel (left side: window_width // 2)
# Added a highlight color for visual feedback
buttons = [
    (10, 10, 110, 60, 'Red', (0, 0, 200), (0, 0, 255)),
    (120, 10, 220, 60, 'Green', (0, 200, 0), (0, 255, 0)),
    (230, 10, 330, 60, 'Blue', (200, 0, 0), (255, 0, 0)),
    (340, 10, 440, 60, 'Clear', (100, 100, 100), (150, 150, 150)),
    (450, 10, 550, 60, 'Save', (100, 100, 100), (150, 150, 150)),
    # New: Eraser Button
    (10, 70, 110, 120, 'Eraser', (50, 50, 50), (100, 100, 100))
]

# Track button highlight states
button_highlights = {btn[4]: False for btn in buttons}
# Initialize current color button highlight
if current_color == (0, 200, 255): # Orange default
    button_highlights['Orange'] = True # You might want to add an explicit Orange button
else:
    # Set initial highlight based on default color. Assuming default is Red (0,0,255) in BGR
    # If your default is orange (0,200,255), you might need to add a button for it or adjust this.
    button_highlights['Red'] = True


def draw_buttons(img, current_color_val, current_mode):
    global button_highlights
    for x1, y1, x2, y2, label, btn_color, highlight_color in buttons:
        display_color = btn_color

        # Highlight current color/mode button
        if (label in ['Red', 'Green', 'Blue'] and current_color_val == {
                'Red': (0, 0, 255),
                'Green': (0, 255, 0),
                'Blue': (255, 0, 0)
            }.get(label, (0,0,0))) or \
           (label == 'Eraser' and current_mode == 'erase') or \
           button_highlights[label]: # Or if it's currently highlighted by hover
            display_color = highlight_color

        cv2.rectangle(img, (x1, y1), (x2, y2), display_color, -1) # Filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2) # White border
        cv2.putText(img, label, (x1 + 10, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def check_button_press(cursor_x, cursor_y, original_frame_width, original_frame_height):
    global current_color, canvas, drawing_active, brush_size, button_pressed_cooldown, button_highlights, mode

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
            button_pressed_cooldown = True # Start cooldown
            
            # Reset all highlights first
            for key in button_highlights:
                button_highlights[key] = False
            # Set highlight for the pressed button
            button_highlights[label] = True

            if label in ['Red', 'Green', 'Blue']:
                current_color = {
                    'Red': (0, 0, 255),
                    'Green': (0, 255, 0),
                    'Blue': (255, 0, 0)
                }[label]
                mode = 'draw' # Switch to draw mode when a color is selected
            elif label == 'Clear':
                canvas[:] = (40, 40, 40)
            elif label == 'Save':
                cv2.imwrite("air_drawing_snapshot.png", canvas)
                print(f"Drawing saved as air_drawing_snapshot.png at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            elif label == 'Eraser':
                mode = 'erase'
            return # Only process one button press per interaction

# --- 3. Robust Gesture Recognition ---
# Using distance between specific landmarks for more reliable pinch detection
def is_pinch_gesture(hand_landmarks, pinch_threshold=0.04): # Normalized distance threshold
    # Thumb tip (4) and Index finger tip (8)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)
    return distance < pinch_threshold

# Using angle/Y-coordinates for "pen down" (index finger extended)
def is_index_finger_extended(hand_landmarks, threshold_y_ratio=0.05):
    # Check if index finger tip is significantly above its PIP and MCP joints
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

    # Check if other fingers are NOT extended (heuristic for "pen up")
    # This part needs careful tuning. For simplicity, we'll focus on index for draw, pinch for stop.
    # A better approach for "pen up" is a specific gesture (like a closed fist or open palm without drawing)

    # Simple check for extended index finger
    return index_tip_y < index_pip_y and index_pip_y < index_mcp_y

# A simpler "pen up" gesture could be a very open palm where no fingers are close to each other.
# For now, pinch will act as "pen down" TOGGLE. Or a specific "pen up" gesture.
# Let's redefine the drawing activation for more robustness:
# Drawing is active when index finger is extended AND NOT pinching.
# Drawing stops when pinching or if hand pose is not "index extended".

# --- Main Loop ---
mode = 'draw' # Default mode is drawing

cooldown_counter = 0 # For button press cooldown

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape # Get original frame dimensions
    frame = cv2.flip(frame, 1) # Flip horizontally for natural interaction

    display = np.zeros((window_height, window_width, 3), dtype=np.uint8)

    # Left side - camera feed with controls
    cam_display = cv2.resize(frame, (canvas_width, window_height)) # Resize camera feed
    
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
            mp_draw.draw_landmarks(cam_display, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw on cam_display

            # Get index finger tip coordinates (normalized 0-1)
            idx_tip_norm_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            idx_tip_norm_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Convert to pixel coordinates on the *original frame* for button detection
            cursor_x_original = int(idx_tip_norm_x * frame_width)
            cursor_y_original = int(idx_tip_norm_y * frame_height)

            # Convert to pixel coordinates for drawing on the *canvas*
            finger_tip_x_canvas = int(idx_tip_norm_x * canvas_width)
            finger_tip_y_canvas = int(idx_tip_norm_y * canvas_height)

            # Convert to pixel coordinates for displaying cursor on *cam_display*
            finger_tip_x_cam = int(idx_tip_norm_x * canvas_width)
            finger_tip_y_cam = int(idx_tip_norm_y * window_height) # cam_display is window_height

            hand_detected = True

            # --- Gesture Logic ---
            # If pinching, check for button press
            if is_pinch_gesture(hand_landmarks):
                # Draw a green circle to indicate pinch
                cv2.circle(cam_display, (finger_tip_x_cam, finger_tip_y_cam), 10, (0, 255, 0), -1)
                check_button_press(cursor_x_original, cursor_y_original, frame_width, frame_height)
                # When pinching, stop drawing
                drawing_active = False
                prev_x, prev_y = 0, 0 # Reset prev_x, prev_y when not drawing

            # If index finger is extended and NOT pinching, activate drawing
            elif is_index_finger_extended(hand_landmarks):
                drawing_active = True
                # Draw cursor on the combined display (on the canvas side)
                cv2.circle(display, (finger_tip_x_canvas + canvas_width, finger_tip_y_canvas), brush_size + 3, current_color, -1)
            else:
                # If neither pinch nor extended index, drawing is inactive
                drawing_active = False
                prev_x, prev_y = 0, 0 # Reset prev_x, prev_y when not drawing

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
        # If no hand is detected, reset drawing state
        drawing_active = False
        prev_x, prev_y = 0, 0
        
    # Button cooldown management
    if button_pressed_cooldown:
        cooldown_counter += 1
        if cooldown_counter > button_cooldown_frames:
            button_pressed_cooldown = False
            cooldown_counter = 0
            # Remove highlight from non-sticky buttons after cooldown
            for label in button_highlights:
                if label not in ['Red', 'Green', 'Blue', 'Eraser']: # These are "sticky" highlights
                    button_highlights[label] = False


    # Add instructions
    cv2.putText(display, "Pinch to interact with buttons", (10, window_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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