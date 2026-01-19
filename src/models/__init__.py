"""모델 모듈"""
from .base_model import BaseBodyEstimator
from .dummy_model import DummyBodyEstimator

# SAM-3D-Body는 선택적 임포트 (설치되지 않은 경우 에러 방지)
try:
    from .sam3d_model import SAM3DBodyEstimator
    __all__ = ["BaseBodyEstimator", "DummyBodyEstimator", "SAM3DBodyEstimator"]
except ImportError:
    __all__ = ["BaseBodyEstimator", "DummyBodyEstimator"]
