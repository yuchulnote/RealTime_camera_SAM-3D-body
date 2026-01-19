"""기본 인체 추정 모델 인터페이스"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseBodyEstimator(ABC):
    """3D 인체 메쉬 추정을 위한 기본 인터페이스"""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지로부터 3D 인체 메쉬를 추정합니다.
        
        Args:
            image: RGB 이미지 (numpy array, H x W x 3)
            
        Returns:
            Dict containing:
                - vertices: (N, 3) numpy array - 메쉬 정점 좌표
                - faces: (M, 3) numpy array - 메쉬 면 인덱스
                - keypoints: (K, 3) numpy array or dict - 키포인트 위치
                - joints: Dict[str, np.ndarray] - 관절 위치 정보
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        pass
