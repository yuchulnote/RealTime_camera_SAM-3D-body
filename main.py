"""메인 실행 파일 - 실시간 3D 인체 메쉬 복원 파이프라인"""
import sys
import time
import queue
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.pipeline import FrameCapture, InferenceWorker, Visualizer
from src.models import DummyBodyEstimator, BaseBodyEstimator
from src.analytics import extract_joint_angles


class HumanMeshPipeline:
    """실시간 3D 인체 메쉬 복원 파이프라인"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """파이프라인 초기화"""
        self.config = load_config(config_path)
        self.setup_queues()
        # 모델은 카메라 선택 후 로드 (setup_components에서)
        self.model = None
        self.frame_capture = None
        self.inference_worker = None
        self.visualizer = None
        
    def setup_queues(self):
        """큐 초기화"""
        frame_queue_size = self.config["pipeline"]["frame_queue_size"]
        mesh_queue_size = self.config["pipeline"]["mesh_queue_size"]
        
        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.mesh_queue = queue.Queue(maxsize=mesh_queue_size)
    
    def setup_model(self):
        """모델 로드 (카메라 선택 후 호출)"""
        model_config = self.config["model"]
        model_name = model_config["name"]
        device = model_config["device"]
        
        # 모델 로드
        checkpoint_path = model_config.get("checkpoint_path")
        input_resolution = tuple(model_config.get("input_resolution", [512, 512]))
        hf_repo_id = model_config.get("hf_repo_id", "facebook/sam-3d-body-dinov3")
        use_detector = model_config.get("use_detector", False)
        use_segmentor = model_config.get("use_segmentor", False)
        use_fov_estimator = model_config.get("use_fov_estimator", False)
        
        print("\n[Pipeline] 모델 로드 중...")
        
        if model_name == "dummy":
            self.model = DummyBodyEstimator(device=device)
        elif model_name == "sam3d_body":
            try:
                from src.models.sam3d_model import SAM3DBodyEstimator
                self.model = SAM3DBodyEstimator(
                    checkpoint_path=checkpoint_path,
                    hf_repo_id=hf_repo_id,
                    device=device,
                    input_resolution=input_resolution,
                    use_detector=use_detector,
                    use_segmentor=use_segmentor,
                    use_fov_estimator=use_fov_estimator
                )
                print(f"[Pipeline] SAM-3D-Body 모델 로드 완료")
            except ImportError as e:
                print(f"[Pipeline] 경고: SAM-3D-Body 모델을 로드할 수 없습니다: {e}")
                print(f"[Pipeline] 더미 모델을 사용합니다. SAM-3D-Body 설치를 참조하세요.")
                self.model = DummyBodyEstimator(device=device)
            except Exception as e:
                print(f"[Pipeline] 오류: SAM-3D-Body 모델 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                print(f"[Pipeline] 더미 모델을 사용합니다.")
                self.model = DummyBodyEstimator(device=device)
        elif model_name == "anny":
            # TODO: Anny 모델 통합
            print(f"[Pipeline] 모델 '{model_name}' 로드 예정 (현재는 더미 모델 사용)")
            self.model = DummyBodyEstimator(device=device)
        else:
            print(f"[Pipeline] 알 수 없는 모델: '{model_name}'. 더미 모델을 사용합니다.")
            self.model = DummyBodyEstimator(device=device)
        
    def setup_components(self):
        """컴포넌트 초기화 (카메라 선택 후 호출)"""
        # 모델이 아직 로드되지 않았으면 로드
        if self.model is None:
            self.setup_model()
        
        # 추론 워커
        device = self.config["model"]["device"]
        self.inference_worker = InferenceWorker(
            frame_queue=self.frame_queue,
            mesh_queue=self.mesh_queue,
            model=self.model,
            device=device
        )
        
        # 시각화
        vis_config = self.config["visualization"]
        show_camera = vis_config.get("show_camera", True)  # 기본값: True
        self.visualizer = Visualizer(
            mesh_queue=self.mesh_queue,
            frame_queue=self.frame_queue,  # 카메라 프레임 큐 전달
            window_width=vis_config["window_width"],
            window_height=vis_config["window_height"],
            show_keypoints=vis_config["show_keypoints"],
            show_skeleton=vis_config["show_skeleton"],
            show_camera=show_camera
        )
        
    def start(self):
        """파이프라인 시작"""
        print("\n" + "="*60)
        print("[Pipeline] 파이프라인 시작 중...")
        print("="*60)
        
        # 1단계: 카메라 선택 및 초기화
        camera_config = self.config["camera"]
        device_id = camera_config.get("device_id")
        interactive_camera = camera_config.get("interactive_selection", True)
        
        # device_id가 None이거나 -1이면 자동 선택
        if device_id is None or device_id == -1:
            device_id = None
        
        self.frame_capture = FrameCapture(
            frame_queue=self.frame_queue,
            device_id=device_id,
            width=camera_config["width"],
            height=camera_config["height"],
            fps=camera_config["fps"],
            auto_select=True,
            interactive=interactive_camera
        )
        
        # 카메라 초기화 (대화형 선택 포함)
        print("\n[Pipeline] 카메라 초기화 중...")
        self.frame_capture.initialize()
        print("[Pipeline] 카메라 초기화 완료\n")
        
        # 2단계: 모델 로드 (카메라 선택 완료 후)
        if self.model is None:
            self.setup_model()
        
        # 3단계: 나머지 컴포넌트 초기화
        self.setup_components()
        
        # 4단계: 스레드 시작
        print("\n[Pipeline] 스레드 시작 중...")
        self.frame_capture.start()
        time.sleep(0.5)  # 웹캠 초기화 대기
        
        self.inference_worker.start()
        time.sleep(0.5)
        
        self.visualizer.start()
        
        print("\n" + "="*60)
        print("[Pipeline] 모든 컴포넌트 시작 완료")
        print("[Pipeline] 'q' 키를 눌러 종료하세요.")
        print("="*60 + "\n")
        
    def run(self):
        """메인 실행 루프"""
        self.start()
        
        try:
            # 통계 출력 및 사용자 입력 처리 루프
            last_stats_time = time.time()
            
            while True:
                # 주기적으로 통계 출력 (5초마다)
                current_time = time.time()
                if current_time - last_stats_time > 5.0:
                    stats = self.inference_worker.get_stats()
                    print(f"[Stats] FPS: {stats['fps']:.2f}, "
                          f"처리된 프레임: {stats['processed_frames']}, "
                          f"평균 추론 시간: {stats['avg_inference_time']*1000:.2f}ms")
                    last_stats_time = current_time
                
                # 키 입력 체크 (OpenCV 윈도우는 메인 스레드에서 관리하지 않음)
                # Open3D 윈도우는 Visualizer 스레드에서 관리
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[Pipeline] 사용자에 의해 중단됨")
        finally:
            self.stop()
    
    def stop(self):
        """파이프라인 중지"""
        print("[Pipeline] 파이프라인 중지 중...")
        
        if self.frame_capture is not None:
            self.frame_capture.stop()
        if self.inference_worker is not None:
            self.inference_worker.stop()
        if self.visualizer is not None:
            self.visualizer.stop()
        
        # 스레드 종료 대기
        if self.frame_capture is not None:
            self.frame_capture.join(timeout=2.0)
        if self.inference_worker is not None:
            self.inference_worker.join(timeout=2.0)
        if self.visualizer is not None:
            self.visualizer.join(timeout=2.0)
        
        print("[Pipeline] 파이프라인 중지 완료")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="실시간 3D 인체 메쉬 복원")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="설정 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 파이프라인 생성 및 실행
    pipeline = HumanMeshPipeline(config_path=args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
