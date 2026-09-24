import os

import cv2


def extract_frames(video_path: str, frames_path: str, step_frames: int, total_frames: int) -> int:
    os.makedirs(frames_path, exist_ok=True)
    saved_count = 0
    current_frame = 0
    capture = cv2.VideoCapture(video_path)

    while True:
        if step_frames > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = capture.read()
        if not ok:
            break
        output_path = os.path.join(frames_path, f"frame_{saved_count:05d}.jpg")
        if not os.path.exists(output_path):
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved_count += 1
        current_frame += step_frames
        if step_frames == 1:
            continue
        if current_frame >= total_frames:
            break

    if step_frames == 1:
        capture.release()
        capture = cv2.VideoCapture(video_path)
        saved_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output_path = os.path.join(frames_path, f"frame_{saved_count:05d}.jpg")
            if not os.path.exists(output_path):
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

    capture.release()
    return saved_count
