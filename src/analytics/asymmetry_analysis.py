"""좌우 비대칭성 분석 모듈"""
import numpy as np
from typing import Dict, List, Tuple


def compute_bilateral_asymmetry(
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    reference_axis: str = "vertical"
) -> Dict[str, float]:
    """
    좌우 관절의 비대칭성을 계산합니다.
    
    Args:
        left_joint: 왼쪽 관절 위치 (3D)
        right_joint: 오른쪽 관절 위치 (3D)
        reference_axis: 참조 축 ("vertical", "horizontal", "depth")
        
    Returns:
        비대칭성 메트릭 딕셔너리
    """
    # 높이 차이 (수직)
    height_diff = abs(left_joint[1] - right_joint[1])
    
    # 좌우 차이 (수평)
    horizontal_diff = abs(left_joint[0] - right_joint[0])
    
    # 깊이 차이 (앞뒤)
    depth_diff = abs(left_joint[2] - right_joint[2])
    
    # 유클리드 거리
    euclidean_diff = np.linalg.norm(left_joint - right_joint)
    
    return {
        "height_diff": float(height_diff),
        "horizontal_diff": float(horizontal_diff),
        "depth_diff": float(depth_diff),
        "euclidean_diff": float(euclidean_diff),
        "symmetry_score": float(1.0 / (1.0 + euclidean_diff))  # 0-1 점수 (1이 완전 대칭)
    }


def analyze_full_body_asymmetry(joints: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    전신의 좌우 비대칭성을 분석합니다.
    
    Args:
        joints: 모든 관절 위치 딕셔너리
        
    Returns:
        각 관절 쌍에 대한 비대칭성 분석 결과
    """
    asymmetry_results = {}
    
    # 주요 관절 쌍들
    joint_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]
    
    for left_key, right_key in joint_pairs:
        if left_key in joints and right_key in joints:
            asymmetry = compute_bilateral_asymmetry(
                joints[left_key],
                joints[right_key]
            )
            asymmetry_results[f"{left_key}_{right_key}"] = asymmetry
    
    return asymmetry_results


def compute_angle_asymmetry(
    left_angle: float,
    right_angle: float
) -> Dict[str, float]:
    """
    좌우 관절 각도의 비대칭성을 계산합니다.
    
    Args:
        left_angle: 왼쪽 관절 각도
        right_angle: 오른쪽 관절 각도
        
    Returns:
        각도 비대칭성 메트릭
    """
    angle_diff = abs(left_angle - right_angle)
    angle_ratio = min(left_angle, right_angle) / max(left_angle, right_angle) if max(left_angle, right_angle) > 0 else 0.0
    
    return {
        "angle_diff": float(angle_diff),
        "angle_ratio": float(angle_ratio),
        "symmetry_score": float(1.0 - min(angle_diff / 180.0, 1.0))  # 0-1 점수
    }


def track_asymmetry_over_time(
    asymmetry_history: List[Dict[str, float]],
    joint_pair: str
) -> Dict[str, float]:
    """
    시간에 따른 비대칭성 변화를 추적합니다.
    
    Args:
        asymmetry_history: 시간별 비대칭성 결과 리스트
        joint_pair: 관절 쌍 이름
        
    Returns:
        시간 추적 통계
    """
    if not asymmetry_history or joint_pair not in asymmetry_history[0]:
        return {}
    
    symmetry_scores = [
        asymmetry_history[i][joint_pair]["symmetry_score"]
        for i in range(len(asymmetry_history))
        if joint_pair in asymmetry_history[i]
    ]
    
    if not symmetry_scores:
        return {}
    
    scores = np.array(symmetry_scores)
    
    return {
        "mean_symmetry": float(np.mean(scores)),
        "std_symmetry": float(np.std(scores)),
        "improvement": float(scores[-1] - scores[0]) if len(scores) > 1 else 0.0,
        "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable" if len(scores) > 1 else "unknown"
    }
