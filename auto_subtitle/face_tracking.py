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

def track_face_centers(video_path, frame_sample_interval=60, work_dir="work") -> list[FacePoint]:
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

    frame_idx = 0
    sampled_centers = {}
    with tqdm(total=frame_count, desc="Tracking faces", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_sample_interval == 0:
                results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    x_center = (bbox.xmin + bbox.width / 2) * width
                    y_center = (bbox.ymin + bbox.height / 2) * height
                    sampled_centers[frame_idx] = (x_center, y_center)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Interpolation and smoothing
    x_raw = np.full(frame_count, np.nan)
    y_raw = np.full(frame_count, np.nan)
    for i, (x, y) in sampled_centers.items():
        x_raw[i] = x
        y_raw[i] = y

    if np.any(~np.isnan(x_raw)):
        # Interpolate missing values
        x_interp = np.interp(np.arange(frame_count), np.flatnonzero(~np.isnan(x_raw)), x_raw[~np.isnan(x_raw)])
        y_interp = np.interp(np.arange(frame_count), np.flatnonzero(~np.isnan(y_raw)), y_raw[~np.isnan(y_raw)])
    else:
        # No detections at all: center
        x_interp = np.full(frame_count, width / 2)
        y_interp = np.full(frame_count, height / 2)

    # Apply smoothing
    x_smooth = gaussian_filter1d(x_interp, sigma=2)
    y_smooth = gaussian_filter1d(y_interp, sigma=2)

    result = [FacePoint(i, float(x), float(y)) for i, (x, y) in enumerate(zip(x_smooth, y_smooth))]
    write_to_cache(result, out_file)
    return result
    
