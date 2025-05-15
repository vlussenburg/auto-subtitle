import mediapipe as mp
import numpy as np
from tqdm import tqdm
from slugify import slugify
from dataclasses import dataclass
from typing import List, Optional
import json
import os
from scipy.ndimage import gaussian_filter1d

@dataclass
class FacePoint:
    frame: int
    x: float
    y: float

    def to_dict(self):
        return {"frame": self.frame, "x": self.x, "y": self.y}

def read_from_cache(infile) -> Optional[List[FacePoint]]:
    if not os.path.exists(infile):
        return None
    with open(infile) as f:
        data = json.load(f)
    return [FacePoint(d["frame"], d["x"], d["y"]) for d in data]

def write_to_cache(data: list[FacePoint], out_file) -> None:
    with open(out_file, "w") as f:
        json.dump([pt.to_dict() for pt in data], f, indent=2)

def track_face_centers(video_path, smoothing_window=60, work_dir="work") -> list[FacePoint]:
    slug = slugify(os.path.splitext(os.path.basename(video_path))[0])
    out_file = os.path.join(work_dir, f"{slug}.face_track.json")
    
    cached_results = read_from_cache(out_file)
    if cached_results:
        print(f"Using cached face tracking results from {out_file}")
        return cached_results
    
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_centers = []
    
    with tqdm(total=frame_count, desc="Tracking faces", unit="frames") as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                x_center = (bbox.xmin + bbox.width / 2) * width
                y_center = (bbox.ymin + bbox.height / 2) * height
                face_centers.append((x_center, y_center))
            else:
                face_centers.append((None, None))
            
            pbar.update(1)

    cap.release()

    # Interpolate missing values
    x_vals = np.array([x if x is not None else 0 for x, _ in face_centers])
    y_vals = np.array([y if y is not None else 0 for _, y in face_centers])
    valid = np.array([x is not None for x, _ in face_centers])

    if valid.any():
        x_vals[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), x_vals[valid])
        y_vals[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), y_vals[valid])

        def apply_kalman_1d(data: np.ndarray) -> np.ndarray:
            kf = cv2.KalmanFilter(4, 1)
            kf.measurementMatrix = np.array([[1], [0], [0], [0]], dtype=np.float32)
            kf.transitionMatrix = np.array([[1,1,0.5,0],
                                            [0,1,1,0],
                                            [0,0,1,0],
                                            [0,0,0,1]], dtype=np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            kf.measurementNoiseCov = np.array([[1]], dtype=np.float32)

            kf.statePre = np.array([[data[0]], [0], [0], [0]], dtype=np.float32)
            kf.statePost = kf.statePre.copy()

            smoothed = []
            for val in data:
                pred = kf.predict()
                est = kf.correct(np.array([[np.float32(val)]]))
                smoothed.append(est[0][0])
            return np.array(smoothed)

        x_smooth = apply_kalman_1d(x_vals)
        y_smooth = apply_kalman_1d(y_vals)
    else:
        x_smooth = x_vals
        y_smooth = y_vals

    result = [FacePoint(i, float(x), float(y)) for i, (x, y) in enumerate(zip(x_smooth, y_smooth))]
    write_to_cache(result, out_file)
    return result
    
