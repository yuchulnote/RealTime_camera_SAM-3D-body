"""추론 워커 모듈 (AI 모델 추론)"""
import threading
import queue
import time
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
import cv2

from ..models.base_model import BaseBodyEstimator


class InferenceWorker(threading.Thread):
    """AI 모델을 사용하여 3D 메쉬를 추출하는 워커 스레드"""
    
    def __init__(
        self,
        frame_queue: queue.Queue,
        mesh_queue: queue.Queue,
        model: BaseBodyEstimator,
        device: str = "cuda:0"
    ):
        """
        Args:
            frame_queue: 입력 프레임 큐 (timestamp, frame_rgb, frame_bgr)
            mesh_queue: 출력 메쉬 큐 (timestamp, mesh_data)
            model: 3D 인체 추정 모델
            device: 사용할 디바이스 ("cuda:0", "cpu" 등)
        """
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.mesh_queue = mesh_queue
        self.model = model
        self.device = torch.device(device)
        
        self.running = False
        self.stats = {
            "processed_frames": 0,
            "total_inference_time": 0.0,
            "last_inference_time": 0.0
        }
        
    def run(self):
        """스레드 실행 (추론 루프)"""
        print(f"[InferenceWorker] 추론 워커 시작 (device: {self.device})")
        self.running = True
        
        while self.running:
            try:
                # 프레임 큐에서 데이터 가져오기 (타임아웃 1초)
                timestamp, frame_rgb, frame_bgr = self.frame_queue.get(timeout=1.0)
                if self.stats["processed_frames"] == 0:
                    print(f"[InferenceWorker] 첫 프레임 수신: {frame_rgb.shape}")
            except queue.Empty:
                if self.stats["processed_frames"] == 0:
                    print("[InferenceWorker] 경고: 프레임 큐가 비어있습니다. 웹캠이 연결되어 있는지 확인하세요.")
                continue
            
            # 추론 수행
            inference_start = time.time()
            
            try:
                with torch.no_grad():
                    # 모델 추론
                    result = self.model.predict(frame_rgb)
                    
                    # 결과 포맷팅
                    mesh_data = {
                        "vertices": result.get("vertices"),  # (N, 3) numpy array
                        "faces": result.get("faces"),        # (M, 3) numpy array
                        "keypoints": result.get("keypoints"), # (K, 3) numpy array or dict
                        "joints": result.get("joints"),      # 관절 위치
                    }
                    
                    inference_time = time.time() - inference_start
                    
                    # 통계 업데이트
                    self.stats["processed_frames"] += 1
                    self.stats["total_inference_time"] += inference_time
                    self.stats["last_inference_time"] = inference_time
                    
                    # 첫 프레임 처리 시 디버그 출력
                    if self.stats["processed_frames"] == 1:
                        print(f"[InferenceWorker] 첫 메쉬 생성 완료: "
                              f"{mesh_data['vertices'].shape[0]} 정점, "
                              f"{mesh_data['faces'].shape[0]} 면")
                    
                    # 메쉬 큐에 결과 전달
                    if self.mesh_queue.full():
                        try:
                            self.mesh_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.mesh_queue.put((timestamp, mesh_data))
                    
            except Exception as e:
                print(f"[InferenceWorker] 추론 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """추론 통계 반환"""
        if self.stats["processed_frames"] > 0:
            avg_inference_time = (
                self.stats["total_inference_time"] / self.stats["processed_frames"]
            )
            fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        else:
            avg_inference_time = 0
            fps = 0
        
        return {
            "processed_frames": self.stats["processed_frames"],
            "avg_inference_time": avg_inference_time,
            "last_inference_time": self.stats["last_inference_time"],
            "fps": fps
        }
    
    def stop(self):
        """워커 중지"""
        self.running = False
        print("[InferenceWorker] 추론 워커 중지")
