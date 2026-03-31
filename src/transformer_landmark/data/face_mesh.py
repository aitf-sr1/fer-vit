import numpy as np


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
