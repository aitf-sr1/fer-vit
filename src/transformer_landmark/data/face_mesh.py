import numpy as np
import torch
from typing import Tuple, Dict

NUM_LANDMARKS = 478

def get_face_mesh_edges():
    edges = []
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    for i in range(len(face_oval) - 1):
        edges.append((face_oval[i], face_oval[i + 1]))
    edges.append((face_oval[-1], face_oval[0]))
    
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    for i in range(len(left_eye) - 1):
        edges.append((left_eye[i], left_eye[i + 1]))
    edges.append((left_eye[-1], left_eye[0]))
    
    right_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    for i in range(len(right_eye) - 1):
        edges.append((right_eye[i], right_eye[i + 1]))
    edges.append((right_eye[-1], right_eye[0]))
    
    return edges

def get_iris_edges():
    edges = []
    right_iris = list(range(468, 473))
    for i in range(len(right_iris) - 1):
        edges.append((right_iris[i], right_iris[i + 1]))
    edges.append((right_iris[-1], right_iris[0]))
    
    left_iris = list(range(473, 478))
    for i in range(len(left_iris) - 1):
        edges.append((left_iris[i], left_iris[i + 1]))
    edges.append((left_iris[-1], left_iris[0]))
    
    edges.extend([(468, 33), (473, 263)])
    return edges

def get_all_edges():
    all_edges = get_face_mesh_edges() + get_iris_edges()
    edge_list = []
    for src, dst in all_edges:
        edge_list.append([src, dst])
        edge_list.append([dst, src])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return edge_index, edge_index.shape[1]

def build_symmetry_map():
    pairs = [
        (33, 263), (133, 362), (468, 473), (469, 474), (470, 475),
        (471, 476), (472, 477), (61, 291), (146, 375),
    ]
    sym_map = {}
    for left, right in pairs:
        sym_map[left] = right
        sym_map[right] = left
    return sym_map

def normalize_landmarks_2d(landmarks):
    normalized = landmarks.copy()
    x_min, x_max = float(normalized[:, 0].min()), float(normalized[:, 0].max())
    normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min + 1e-8)
    y_min, y_max = float(normalized[:, 1].min()), float(normalized[:, 1].max())
    normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min + 1e-8)
    return normalized

def normalize_landmarks(landmarks):
    return normalize_landmarks_2d(landmarks[:, :2]) if landmarks.shape[1] == 2 else landmarks
