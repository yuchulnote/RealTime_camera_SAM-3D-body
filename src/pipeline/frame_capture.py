"""웹캠 프레임 캡처 모듈"""
import threading
import time
import queue
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.camera_utils import list_available_cameras, select_camera, select_camera_interactive, test_camera


class FrameCapture(threading.Thread):
    """웹캠으로부터 프레임을 캡처하는 스레드"""
    
    def __init__(
        self,
        frame_queue: queue.Queue,
        device_id: Optional[int] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        auto_select: bool = True,
        interactive: bool = False
    ):
        """
        Args:
            frame_queue: 캡처한 프레임을 전달할 큐
            device_id: 웹캠 장치 ID (None이면 자동 선택)
            width: 캡처 해상도 너비
            height: 캡처 해상도 높이
            fps: 목표 FPS
            auto_select: device_id가 None이거나 사용 불가능할 때 자동으로 다른 카메라 선택
            interactive: True이면 사용자 입력으로 카메라 선택
        """
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.device_id = device_id
        self.width = width
        self.height = height
        self.target_interval = 1.0 / fps if fps > 0 else 0.0
        self.auto_select = auto_select
        self.interactive = interactive
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        
    def initialize(self):
        """웹캠 초기화"""
        # 카메라 선택
        if self.device_id is None or self.auto_select:
            if self.device_id is None:
                if self.interactive:
                    print("[FrameCapture] 카메라 대화형 선택 모드")
                else:
                    print("[FrameCapture] 카메라 자동 선택 모드")
            else:
                print(f"[FrameCapture] 카메라 {self.device_id} 테스트 중...")
                if not test_camera(self.device_id):
                    print(f"[FrameCapture] 경고: 카메라 {self.device_id}를 사용할 수 없습니다.")
                    if self.interactive:
                        print("[FrameCapture] 대화형 선택 모드로 전환합니다.")
                    else:
                        print("[FrameCapture] 자동 선택 모드로 전환합니다.")
                    self.device_id = None
            
            if self.device_id is None:
                if self.interactive:
                    self.device_id = select_camera_interactive()
                else:
                    self.device_id = select_camera()
        
        # 웹캠 열기
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다 (device_id: {self.device_id})")
        
        # 캡처 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 실제 해상도 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[FrameCapture] 웹캠 초기화 완료: 카메라 {self.device_id}, 해상도 {actual_width}x{actual_height}")
        
    def run(self):
        """스레드 실행 (프레임 캡처 루프)"""
        self.initialize()
        self.running = True
        last_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[FrameCapture] 프레임 읽기 실패")
                time.sleep(0.1)
                continue
            
            # OpenCV는 BGR 형식으로 읽으므로 RGB로 변환
            # SAM-3D-Body는 RGB 형식을 요구함
            if frame.shape[2] == 3:  # 3채널 이미지 확인
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("[FrameCapture] 경고: 예상치 못한 이미지 형식")
                frame_rgb = frame
            
            # 타임스탬프 부여
            timestamp = time.time()
            
            # 큐에 프레임 전달 (큐가 가득 차면 기다리지 않고 최신 프레임 유지)
            if self.frame_queue.full():
                try:
                    # 오래된 프레임 제거
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # RGB 형식만 전달 (SAM-3D-Body가 RGB를 요구함)
            self.frame_queue.put((timestamp, frame_rgb, frame))  # RGB, BGR 둘 다 저장
            
            # FPS 제어
            current_time = time.time()
            elapsed = current_time - last_time
            sleep_time = max(0, self.target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()
    
    def stop(self):
        """캡처 중지"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        print("[FrameCapture] 프레임 캡처 중지")
