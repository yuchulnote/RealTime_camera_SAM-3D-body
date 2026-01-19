"""관절 각도 계산 모듈"""
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    """
    세 점으로 이루어진 각도를 계산합니다.
    
    Args:
        point_a: 첫 번째 점 (관절 시작점)
        point_b: 두 번째 점 (관절 위치)
        point_c: 세 번째 점 (관절 끝점)
        
    Returns:
        각도 (도 단위, 0-180)
    """
    # 벡터 계산
    vec_ab = point_a - point_b
    vec_cb = point_c - point_b
    
    # 벡터 정규화
    norm_ab = np.linalg.norm(vec_ab)
    norm_cb = np.linalg.norm(vec_cb)
    
    if norm_ab < 1e-8 or norm_cb < 1e-8:
        return 0.0
    
    # 코사인 값 계산
    cos_angle = np.dot(vec_ab, vec_cb) / (norm_ab * norm_cb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 라디안을 도로 변환
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def extract_joint_angles(joints: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    관절 딕셔너리로부터 주요 관절 각도를 추출합니다.
    
    Args:
        joints: 관절 위치 딕셔너리
        
    Returns:
        관절 각도 딕셔너리
    """
    angles = {}
    
    # 왼쪽 팔꿈치 각도
    if all(k in joints for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        angles["left_elbow"] = compute_angle(
            joints["left_shoulder"],
            joints["left_elbow"],
            joints["left_wrist"]
        )
    
    # 오른쪽 팔꿈치 각도
    if all(k in joints for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["right_elbow"] = compute_angle(
            joints["right_shoulder"],
            joints["right_elbow"],
            joints["right_wrist"]
        )
    
    # 왼쪽 어깨 각도
    if all(k in joints for k in ["left_hip", "left_shoulder", "left_elbow"]):
        angles["left_shoulder"] = compute_angle(
            joints["left_hip"],
            joints["left_shoulder"],
            joints["left_elbow"]
        )
    
    # 오른쪽 어깨 각도
    if all(k in joints for k in ["right_hip", "right_shoulder", "right_elbow"]):
        angles["right_shoulder"] = compute_angle(
            joints["right_hip"],
            joints["right_shoulder"],
            joints["right_elbow"]
        )
    
    # 왼쪽 무릎 각도
    if all(k in joints for k in ["left_hip", "left_knee", "left_ankle"]):
        angles["left_knee"] = compute_angle(
            joints["left_hip"],
            joints["left_knee"],
            joints["left_ankle"]
        )
    
    # 오른쪽 무릎 각도
    if all(k in joints for k in ["right_hip", "right_knee", "right_ankle"]):
        angles["right_knee"] = compute_angle(
            joints["right_hip"],
            joints["right_knee"],
            joints["right_ankle"]
        )
    
    # 왼쪽 엉덩이 각도
    if all(k in joints for k in ["left_shoulder", "left_hip", "left_knee"]):
        angles["left_hip"] = compute_angle(
            joints["left_shoulder"],
            joints["left_hip"],
            joints["left_knee"]
        )
    
    # 오른쪽 엉덩이 각도
    if all(k in joints for k in ["right_shoulder", "right_hip", "right_knee"]):
        angles["right_hip"] = compute_angle(
            joints["right_shoulder"],
            joints["right_hip"],
            joints["right_knee"]
        )
    
    return angles


def compute_range_of_motion(angle_history: List[float]) -> Dict[str, float]:
    """
    관절 움직임 범위(Range of Motion, ROM)를 계산합니다.
    
    Args:
        angle_history: 시간에 따른 각도 값 리스트
        
    Returns:
        ROM 통계 딕셔너리 (min, max, range)
    """
    if not angle_history:
        return {"min": 0.0, "max": 0.0, "range": 0.0}
    
    angles = np.array(angle_history)
    return {
        "min": float(np.min(angles)),
        "max": float(np.max(angles)),
        "range": float(np.max(angles) - np.min(angles)),
        "mean": float(np.mean(angles)),
        "std": float(np.std(angles))
    }
