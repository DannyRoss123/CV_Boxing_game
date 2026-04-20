import os

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmark,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")


def lerp_vec(prev: np.ndarray, cur: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if prev is None:
        return cur
    return prev * (1.0 - alpha) + cur * alpha


def landmark_to_px(landmark, w: int, h: int) -> np.ndarray:
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)


def draw_fighter(frame, shoulder, elbow, wrist):
    h, w, _ = frame.shape

    head = (int(w * 0.78), int(h * 0.27))
    torso_top = (int(w * 0.78), int(h * 0.40))
    torso_bottom = (int(w * 0.78), int(h * 0.72))

    cv2.circle(frame, head, 35, (220, 220, 220), 3)
    cv2.line(frame, torso_top, torso_bottom, (200, 200, 200), 6)

    s = tuple(np.int32(shoulder))
    e = tuple(np.int32(elbow))
    wr = tuple(np.int32(wrist))

    cv2.line(frame, s, e, (0, 200, 255), 10)
    cv2.line(frame, e, wr, (0, 120, 255), 10)
    cv2.circle(frame, s, 9, (255, 255, 255), -1)
    cv2.circle(frame, e, 8, (255, 255, 255), -1)
    cv2.circle(frame, wr, 8, (255, 255, 255), -1)


def map_user_arm_to_fighter(
    user_shoulder: np.ndarray,
    user_elbow: np.ndarray,
    user_wrist: np.ndarray,
    fighter_shoulder: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper_vec = user_elbow - user_shoulder
    fore_vec = user_wrist - user_elbow

    upper_len = np.linalg.norm(upper_vec) + 1e-6
    fore_len = np.linalg.norm(fore_vec) + 1e-6

    upper_dir = upper_vec / upper_len
    fore_dir = fore_vec / fore_len

    mapped_upper_len = float(np.clip(upper_len * 0.55, 55, 115))
    mapped_fore_len = float(np.clip(fore_len * 0.55, 50, 120))

    mapped_elbow = fighter_shoulder + upper_dir * mapped_upper_len
    mapped_wrist = mapped_elbow + fore_dir * mapped_fore_len
    return mapped_elbow, mapped_wrist


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    landmarker = PoseLandmarker.create_from_options(options)

    prev_shoulder = None
    prev_elbow = None
    prev_wrist = None
    frame_ts = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33  # ~30fps timestamp increment in ms
        result = landmarker.detect_for_video(mp_image, frame_ts)

        fighter_shoulder = np.array([w * 0.73, h * 0.43], dtype=np.float32)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]

            r_shoulder = landmark_to_px(lm[PoseLandmark.RIGHT_SHOULDER], w, h)
            r_elbow = landmark_to_px(lm[PoseLandmark.RIGHT_ELBOW], w, h)
            r_wrist = landmark_to_px(lm[PoseLandmark.RIGHT_WRIST], w, h)

            r_shoulder = lerp_vec(prev_shoulder, r_shoulder)
            r_elbow = lerp_vec(prev_elbow, r_elbow)
            r_wrist = lerp_vec(prev_wrist, r_wrist)

            prev_shoulder, prev_elbow, prev_wrist = r_shoulder, r_elbow, r_wrist

            fighter_elbow, fighter_wrist = map_user_arm_to_fighter(
                r_shoulder, r_elbow, r_wrist, fighter_shoulder
            )

            draw_fighter(frame, fighter_shoulder, fighter_elbow, fighter_wrist)

            cv2.line(frame, tuple(np.int32(r_shoulder)), tuple(np.int32(r_elbow)), (80, 255, 80), 3)
            cv2.line(frame, tuple(np.int32(r_elbow)), tuple(np.int32(r_wrist)), (80, 255, 80), 3)
            cv2.putText(
                frame,
                "Move your right arm to drive the fighter arm",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
        else:
            draw_fighter(
                frame,
                fighter_shoulder,
                fighter_shoulder + np.array([40, 25], dtype=np.float32),
                fighter_shoulder + np.array([85, 18], dtype=np.float32),
            )
            cv2.putText(
                frame,
                "No pose detected - step back so camera sees your upper body",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Basic CV Boxing Prototype", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
