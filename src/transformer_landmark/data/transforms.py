from typing import Dict, Any
import torch
import random
import math

from .face_mesh import build_symmetry_map


class LandmarkAugmentation:
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        jitter_std: float = 0.02,
        rotation_degrees: float = 15.0,
        scale_range: tuple[float, float] = (0.95, 1.05),
        dropout_prob: float = 0.1,
        dropout_landmark_prob: float = 0.05,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.jitter_std = jitter_std
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.dropout_prob = dropout_prob
        self.dropout_landmark_prob = dropout_landmark_prob
        self.symmetry_map = build_symmetry_map()

    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = landmarks.clone()

        if random.random() < self.horizontal_flip_prob:
            landmarks = self._horizontal_flip(landmarks)

        landmarks = self._add_jitter(landmarks)
        landmarks = self._rotate(landmarks)
        landmarks = self._scale(landmarks)

        if random.random() < self.dropout_prob:
            landmarks = self._dropout_landmarks(landmarks)

        landmarks = torch.clamp(landmarks, 0.0, 1.0)
        return landmarks

    def _horizontal_flip(self, landmarks: torch.Tensor) -> torch.Tensor:
        flipped = landmarks.clone()
        flipped[:, 0] = 1.0 - flipped[:, 0]

        swapped = flipped.clone()
        for left_idx, right_idx in self.symmetry_map.items():
            if left_idx < right_idx:
                swapped[left_idx] = flipped[right_idx]
                swapped[right_idx] = flipped[left_idx]

        return swapped

    def _add_jitter(self, landmarks: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(landmarks) * self.jitter_std
        return landmarks + noise

    def _rotate(self, landmarks: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        angle_rad = math.radians(angle)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        center_x = landmarks[:, 0].mean()
        center_y = landmarks[:, 1].mean()

        x = landmarks[:, 0] - center_x
        y = landmarks[:, 1] - center_y

        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        landmarks[:, 0] = x_rot + center_x
        landmarks[:, 1] = y_rot + center_y

        return landmarks

    def _scale(self, landmarks: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        center_x = landmarks[:, 0].mean()
        center_y = landmarks[:, 1].mean()

        landmarks[:, 0] = (landmarks[:, 0] - center_x) * scale + center_x
        landmarks[:, 1] = (landmarks[:, 1] - center_y) * scale + center_y

        return landmarks

    def _dropout_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        num_landmarks = landmarks.shape[0]
        dropout_mask = torch.rand(num_landmarks) > self.dropout_landmark_prob
        dropout_mask = dropout_mask.unsqueeze(1)
        return landmarks * dropout_mask.float()


class NoAugmentation:
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        return landmarks


def get_train_transforms(
    config: Dict[str, Any],
) -> LandmarkAugmentation | NoAugmentation:
    aug_config = config.get("augmentation", {}).get("train", {})

    if not aug_config.get("enabled", True):
        return NoAugmentation()

    return LandmarkAugmentation(
        horizontal_flip_prob=aug_config.get("horizontal_flip_prob", 0.5),
        jitter_std=aug_config.get("jitter_std", 0.02),
        rotation_degrees=aug_config.get("rotation_degrees", 15.0),
        scale_range=tuple(aug_config.get("scale_range", [0.95, 1.05])),
        dropout_prob=aug_config.get("dropout_prob", 0.1),
        dropout_landmark_prob=aug_config.get("dropout_landmark_prob", 0.05),
    )


def get_val_transforms(config: Dict[str, Any]) -> NoAugmentation:
    return NoAugmentation()


def get_test_transforms(config: Dict[str, Any]) -> NoAugmentation:
    return NoAugmentation()

