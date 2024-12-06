import cv2
import mediapipe as mp
import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s: %(message)s'
)

class GunGestureDetector:
    """
    Detect gun pose gestures and determine when a shot is fired.
    Also handles aiming direction and calculates the position to show the heart effect.
    """
    def __init__(self, forward_distance=50, shot_frames=30, move_threshold=10):
        """
        Args:
            forward_distance (int): Distance from fingertip in aim direction where the heart effect appears.
            shot_frames (int): Number of frames the heart remains visible after a shot.
            move_threshold (int): Minimum vertical fingertip movement (pixels) to detect a shot.
        """
        self.forward_distance = forward_distance
        self.shot_frames = shot_frames
        self.move_threshold = move_threshold

        self.prev_index_finger_y = None
        self.shot_countdown = 0
        self.shot_position = (0,0)
        self.aim_position = None

    def is_finger_extended(self, tip_idx, pip_idx, landmarks):
        """Check if a finger is extended by comparing tip and pip y coordinates."""
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]

    def is_finger_folded(self, tip_idx, pip_idx, landmarks):
        """Check if a finger is folded by comparing tip and pip y coordinates."""
        return landmarks[tip_idx][1] > landmarks[pip_idx][1]

    def detect_gun_pose(self, landmarks):
        """Determine if the hand is in a 'gun pose' gesture."""
        thumb_extended = self.is_finger_extended(4, 3, landmarks)
        index_extended = self.is_finger_extended(8, 6, landmarks)
        middle_folded = self.is_finger_folded(12, 10, landmarks)
        ring_folded = self.is_finger_folded(16, 14, landmarks)
        pinky_folded = self.is_finger_folded(20, 18, landmarks)
        return thumb_extended and index_extended and middle_folded and ring_folded and pinky_folded

    def update_aim_position(self, index_mcp_pos, index_tip_pos):
        """
        Update the aim position if the index finger is held horizontally.
        Aim position is forward_distance pixels away from the fingertip in finger direction.
        """
        dx = index_tip_pos[0] - index_mcp_pos[0]
        dy = index_tip_pos[1] - index_mcp_pos[1]

        # Check if horizontal (abs(dx) > abs(dy)) means more horizontal than vertical.
        if abs(dx) > abs(dy):
            length = (dx**2 + dy**2)**0.5
            length = length if length != 0 else 1
            ux, uy = dx / length, dy / length

            aim_x = int(index_tip_pos[0] + ux * self.forward_distance)
            aim_y = int(index_tip_pos[1] + uy * self.forward_distance)
            self.aim_position = (aim_x, aim_y)
            return True  # Indicates that we are indeed aiming horizontally
        return False

    def detect_shot(self, index_tip_y):
        """
        Check if the vertical movement of the fingertip indicates a shot.
        We compare current index_tip_y to previous frame to see if it moved up sufficiently.
        """
        if self.prev_index_finger_y is not None:
            diff = self.prev_index_finger_y - index_tip_y
            logging.debug(f"Fingertip vertical movement: {diff}px")
            if diff > self.move_threshold and self.aim_position is not None:
                # Shot detected
                self.shot_countdown = self.shot_frames
                self.shot_position = self.aim_position
                logging.info("Shot detected!")
                return True
        return False

    def update_finger_position(self, index_tip_y):
        """Update the stored fingertip position for shot detection."""
        self.prev_index_finger_y = index_tip_y

    def reset_finger_position(self):
        """Reset the previous fingertip position when not in gun pose."""
        self.prev_index_finger_y = None

    def is_shot_active(self):
        """Check if the shot effect is currently active."""
        return self.shot_countdown > 0

    def decrement_shot_countdown(self):
        """Decrement the shot countdown frames."""
        if self.shot_countdown > 0:
            self.shot_countdown -= 1

def draw_heart(frame, center, scale):
    """
    Draw an expanding heart shape on the frame.
    The heart is composed of two circles and a triangular bottom part.
    """
    cx, cy = center
    def s(val): return int(val * scale)

    # Calculate heart shape coordinates
    left_circle_center = (cx - s(10), cy - s(10))
    right_circle_center = (cx + s(10), cy - s(10))
    pts = [
        (cx - s(20), cy - s(2)),
        (cx,          cy + s(20)),
        (cx + s(20),  cy - s(2))
    ]
    pts = cv2.convexHull(np.array(pts).reshape(-1,1,2))

    # Draw heart
    cv2.circle(frame, left_circle_center, s(10), (0,0,255), -1)
    cv2.circle(frame, right_circle_center, s(10), (0,0,255), -1)
    cv2.fillConvexPoly(frame, pts, (0,0,255))
    cv2.putText(frame, "Shot Fired!", (cx - s(50), cy + s(50)), cv2.FONT_HERSHEY_SIMPLEX, scale*0.8, (0,0,255), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera.")
        return

    logging.info("Camera opened successfully.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    detector = GunGestureDetector()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            gun_pose = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    h, w, c = frame.shape
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                    # Check gun pose
                    gun_pose = detector.detect_gun_pose(landmarks)

                    if gun_pose:
                        index_tip_pos = landmarks[8]
                        index_mcp_pos = landmarks[5]

                        # Update aim if horizontal
                        is_horizontal = detector.update_aim_position(index_mcp_pos, index_tip_pos)
                        if is_horizontal:
                            cv2.putText(frame, "Aiming...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        else:
                            cv2.putText(frame, "Not horizontal", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                        # Detect shot
                        shot_fired = detector.detect_shot(index_tip_pos[1])
                        detector.update_finger_position(index_tip_pos[1])
                    else:
                        # Not in gun pose, reset finger position
                        detector.reset_finger_position()

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                # No hands detected
                detector.reset_finger_position()

            # Show gun pose status
            if gun_pose:
                cv2.putText(frame, "Gun Pose", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Handle heart drawing if shot is active
            if detector.is_shot_active():
                # Expand the heart over time
                scale = 0.5 + (1 - detector.shot_countdown/detector.shot_frames)*1.5
                draw_heart(frame, detector.shot_position, scale)
                detector.decrement_shot_countdown()

            cv2.imshow("Gun Gesture Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()