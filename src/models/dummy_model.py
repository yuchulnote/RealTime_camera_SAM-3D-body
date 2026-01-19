"""더미 모델 (테스트용) - 실제 모델 구현 전까지 사용"""
import numpy as np
import torch
from typing import Dict, Any

from .base_model import BaseBodyEstimator


class DummyBodyEstimator(BaseBodyEstimator):
    """더미 인체 추정 모델 - 간단한 테스트 메쉬 생성"""
    
    def __init__(self, device: str = "cuda:0"):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = torch.device(device)
        print("[DummyModel] 더미 모델 초기화 완료 (실제 모델 구현 필요)")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (리사이즈 및 정규화)"""
        import cv2
        # 리사이즈
        resized = cv2.resize(image, (256, 256))
        # 정규화 (0-255 -> 0-1)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        더미 메쉬 생성 (간단한 구 형태)
        실제 구현에서는 SAM-3D-Body 또는 Anny 모델을 호출해야 합니다.
        """
        # 간단한 구 메쉬 생성 (테스트용)
        import open3d as o3d
        
        # 구 메쉬 생성
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=20)
        
        vertices = np.asarray(sphere.vertices)
        faces = np.asarray(sphere.triangles)
        
        # 키포인트 (더미)
        keypoints = np.array([
            [0, 0.5, 0],    # 머리
            [0, 0.3, 0],    # 목
            [-0.2, 0.1, 0], # 왼쪽 어깨
            [0.2, 0.1, 0],  # 오른쪽 어깨
            [0, -0.3, 0],   # 골반
            [-0.1, -0.5, 0], # 왼쪽 무릎
            [0.1, -0.5, 0],  # 오른쪽 무릎
        ])
        
        joints = {
            "head": keypoints[0],
            "neck": keypoints[1],
            "left_shoulder": keypoints[2],
            "right_shoulder": keypoints[3],
            "pelvis": keypoints[4],
            "left_knee": keypoints[5],
            "right_knee": keypoints[6],
        }
        
        return {
            "vertices": vertices.astype(np.float32),
            "faces": faces.astype(np.int32),
            "keypoints": keypoints.astype(np.float32),
            "joints": joints
        }
