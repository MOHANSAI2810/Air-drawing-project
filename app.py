import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe and drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Create a white canvas (same size as webcam)
ret, frame = cap.read()
height, width = frame.shape[:2]
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index fingertip
            x = int(hand_landmarks.landmark[8].x * width)
            y = int(hand_landmarks.landmark[8].y * height)

            # Draw only if moving (avoid first point jumping)
            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 4)

            prev_x, prev_y = x, y

    else:
        prev_x, prev_y = 0, 0  # Reset when hand not detected

    # Show combined live preview
    preview = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
    cv2.imshow("Air Drawing", preview)

    key = cv2.waitKey(1)
    if key == ord('c'):
        canvas[:] = 255  # Clear canvas (reset to white)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Show only the clean drawing
cv2.imshow("Your Drawing", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
