from typing import Optional
import numpy as np
import torch
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceMeshExtractor:
    def __init__(self, model_path: str = "model/face_landmarker.task") -> None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def __del__(self) -> None:
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def extract_landmarks(self, image: Image.Image) -> Optional[np.ndarray]:
        image_np = np.array(image)

        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]

        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        return landmark_array


def extract_face_landmarks(
    image: Image.Image, model_path: str = "model/face_landmarker.task"
) -> Optional[np.ndarray]:
    extractor = FaceMeshExtractor(model_path)
    landmarks = extractor.extract_landmarks(image)
    return landmarks


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    normalized = landmarks.copy()

    x_min = float(normalized[:, 0].min())
    x_max = float(normalized[:, 0].max())
    normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min + 1e-8)

    y_min = float(normalized[:, 1].min())
    y_max = float(normalized[:, 1].max())
    normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min + 1e-8)

    z_mean = float(normalized[:, 2].mean())
    z_std = float(normalized[:, 2].std())
    normalized[:, 2] = (normalized[:, 2] - z_mean) / (z_std + 1e-8)

    return normalized


def create_gaussian_heatmap(
    center_x: float, center_y: float, height: int, width: int, sigma: float = 10.0
) -> np.ndarray:
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    y = y[:, np.newaxis]

    x0 = center_x * width
    y0 = center_y * height

    heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    return heatmap


def create_attention_bias(
    landmarks: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
    sigma: float = 10.0,
) -> torch.Tensor:
    num_patches = image_size // patch_size

    heatmap = np.zeros((image_size, image_size), dtype=np.float32)

    for landmark in landmarks:
        x, y = float(landmark[0]), float(landmark[1])
        landmark_heatmap = create_gaussian_heatmap(x, y, image_size, image_size, sigma)
        heatmap = np.maximum(heatmap, landmark_heatmap)

    patch_heatmap = np.zeros((num_patches, num_patches), dtype=np.float32)
    for i in range(num_patches):
        for j in range(num_patches):
            patch = heatmap[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            patch_heatmap[i, j] = float(patch.mean())

    bias = torch.from_numpy(patch_heatmap.flatten())

    bias_min = float(bias.min())
    bias_max = float(bias.max())
    bias = (bias - bias_min) / (bias_max - bias_min + 1e-8)

    return bias


def create_zero_landmarks(num_landmarks: int = 468) -> np.ndarray:
    return np.zeros((num_landmarks, 3), dtype=np.float32)


def create_zero_attention_bias(
    image_size: int = 224, patch_size: int = 16
) -> torch.Tensor:
    num_patches = image_size // patch_size
    return torch.zeros(num_patches * num_patches)
