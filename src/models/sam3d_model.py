"""SAM-3D-Body 모델 래퍼"""
import numpy as np
import torch
import cv2
from typing import Dict, Any, Optional, List
import warnings
import sys
from pathlib import Path
import os

from .base_model import BaseBodyEstimator


class SAM3DBodyEstimator(BaseBodyEstimator):
    """SAM-3D-Body 인체 추정 모델 래퍼"""
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hf_repo_id: str = "facebook/sam-3d-body-dinov3",
        device: str = "cuda:0",
        input_resolution: tuple = (512, 512),
        use_detector: bool = False,
        use_segmentor: bool = False,
        use_fov_estimator: bool = False
    ):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로 (None이면 HuggingFace 캐시에서 자동 찾기)
            hf_repo_id: HuggingFace 저장소 ID
            device: 사용할 디바이스
            input_resolution: 입력 이미지 해상도 (width, height) - 참고용 (모델이 자체 해상도 사용)
            use_detector: 사람 감지기 사용 여부
            use_segmentor: 마스크 세그멘터 사용 여부
            use_fov_estimator: FOV 추정기 사용 여부
        """
        self.device = torch.device(device)
        self.input_resolution = input_resolution
        
        print(f"[SAM3DBody] 모델 초기화 중... (device: {self.device})")
        
        # SAM-3D-Body 모델 로드 시도
        try:
            self._load_model(checkpoint_path, hf_repo_id, use_detector, use_segmentor, use_fov_estimator)
        except ImportError as e:
            print(f"[SAM3DBody] 경고: SAM-3D-Body 라이브러리를 찾을 수 없습니다.")
            print(f"[SAM3DBody] 설치 방법: sam-3d-body 디렉토리가 프로젝트 루트에 있는지 확인하세요")
            raise e
        except Exception as e:
            print(f"[SAM3DBody] 모델 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        print("[SAM3DBody] 모델 초기화 완료")
    
    def _load_model(
        self, 
        checkpoint_path: Optional[str], 
        hf_repo_id: str,
        use_detector: bool,
        use_segmentor: bool,
        use_fov_estimator: bool
    ):
        """SAM-3D-Body 모델 로드"""
        # sam-3d-body 디렉토리를 경로에 추가
        project_root = Path(__file__).parent.parent.parent
        sam3d_path = project_root / "sam-3d-body"
        
        if not sam3d_path.exists():
            raise ImportError(f"SAM-3D-Body 디렉토리를 찾을 수 없습니다: {sam3d_path}")
        
        sys.path.insert(0, str(sam3d_path))
        
        # 다운로드된 모델 경로 찾기
        if checkpoint_path is None:
            # HuggingFace 캐시에서 자동으로 찾기
            from huggingface_hub import snapshot_download
            try:
                cache_dir = snapshot_download(repo_id=hf_repo_id, local_files_only=True)
                checkpoint_path = os.path.join(cache_dir, "model.ckpt")
                mhr_path = os.path.join(cache_dir, "assets", "mhr_model.pt")
                print(f"[SAM3DBody] HuggingFace 캐시에서 모델 발견: {cache_dir}")
            except Exception as e:
                print(f"[SAM3DBody] 경고: HuggingFace 캐시에서 모델을 찾을 수 없습니다: {e}")
                print(f"[SAM3DBody] hf download 명령어로 모델을 다운로드하세요")
                raise
        
        # 모델 로드
        from sam_3d_body.build_models import load_sam_3d_body
        from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator as SAM3DEstimator
        
        # 체크포인트 경로에서 mhr_model.pt 경로 찾기
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_dir = os.path.dirname(checkpoint_path)
            mhr_path = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")
            if not os.path.exists(mhr_path):
                # 상위 디렉토리에서 찾기 (snapshots 구조)
                parent_dir = os.path.dirname(checkpoint_dir)
                mhr_path = os.path.join(parent_dir, "assets", "mhr_model.pt")
                if not os.path.exists(mhr_path):
                    # HuggingFace 캐시 구조에서 찾기
                    from huggingface_hub import snapshot_download
                    try:
                        cache_dir = snapshot_download(repo_id=hf_repo_id, local_files_only=True)
                        mhr_path = os.path.join(cache_dir, "assets", "mhr_model.pt")
                    except:
                        mhr_path = ""
        else:
            mhr_path = ""
        
        print(f"[SAM3DBody] 체크포인트 로드: {checkpoint_path}")
        print(f"[SAM3DBody] MHR 모델 경로: {mhr_path}")
        
        # 모델과 설정 로드
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=str(self.device),
            mhr_path=mhr_path
        )
        
        # 선택적 컴포넌트 로드
        human_detector = None
        human_segmentor = None
        fov_estimator = None
        
        if use_detector:
            try:
                from tools.build_detector import HumanDetector
                human_detector = HumanDetector(name="vitdet", device=str(self.device))
                print("[SAM3DBody] 사람 감지기 로드 완료")
            except Exception as e:
                print(f"[SAM3DBody] 경고: 사람 감지기 로드 실패: {e}")
        
        if use_fov_estimator:
            try:
                from tools.build_fov_estimator import FOVEstimator
                fov_estimator = FOVEstimator(name="moge2", device=str(self.device))
                print("[SAM3DBody] FOV 추정기 로드 완료")
            except Exception as e:
                print(f"[SAM3DBody] 경고: FOV 추정기 로드 실패: {e}")
        
        # Estimator 생성
        self.estimator = SAM3DEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=human_segmentor,
            fov_estimator=fov_estimator
        )
        
        # Faces 저장 (메쉬 렌더링용)
        self.faces = self.estimator.faces
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (SAM-3D-Body는 자체 전처리 사용)"""
        # SAM-3D-Body는 내부에서 전처리를 수행하므로 RGB 형식만 보장
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 이미 RGB 형식인지 확인 (BGR이 아닌지)
            return image
        return image
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지로부터 3D 인체 메쉬를 추정합니다.
        
        Args:
            image: RGB 이미지 (numpy array, H x W x 3, uint8)
            
        Returns:
            Dict containing:
                - vertices: (N, 3) numpy array - 메쉬 정점 좌표
                - faces: (M, 3) numpy array - 메쉬 면 인덱스
                - keypoints: (K, 3) numpy array or dict - 키포인트 위치
                - joints: Dict[str, np.ndarray] - 관절 위치 정보
        """
        if self.estimator is None:
            raise RuntimeError("SAM-3D-Body 모델이 로드되지 않았습니다. 모델 설치를 확인하세요.")
        
        try:
            # 이미지 형식 확인 및 변환
            # SAM-3D-Body는 RGB 형식을 요구함
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # RGB 형식 확인 (3채널)
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"예상치 못한 이미지 형식: {image.shape}")
            
            # 이미 RGB 형식으로 전달됨 (FrameCapture에서 BGR->RGB 변환 완료)
            # SAM-3D-Body의 process_one_image는 numpy array를 받으면 RGB라고 가정함
            # 하지만 경고 메시지를 매번 출력하므로, 이를 억제
            
            # SAM-3D-Body 추론 (RGB 이미지 입력)
            # process_one_image는 리스트를 반환 (여러 사람 가능)
            import sys
            import io
            from contextlib import redirect_stdout
            
            # 경고 메시지 억제 (stdout 임시 리다이렉션)
            # 참고: 이 경고는 무시해도 됨 (이미 RGB 형식으로 전달됨)
            f = io.StringIO()
            with redirect_stdout(f):
                outputs_list = self.estimator.process_one_image(
                    img=image,  # RGB 형식 (numpy array, uint8)
                    inference_type="body"  # body만 사용 (빠름)
                )
            
            # 첫 번째 사람만 사용 (실시간이므로 단일 인물 가정)
            if len(outputs_list) == 0:
                # 사람이 감지되지 않음
                if not hasattr(self, '_no_person_warning_count'):
                    self._no_person_warning_count = 0
                self._no_person_warning_count += 1
                
                # 처음 몇 번만 경고 출력 (로그 스팸 방지)
                if self._no_person_warning_count <= 3:
                    print("[SAM3DBody] 사람이 감지되지 않았습니다. 카메라 앞으로 이동해주세요.")
                elif self._no_person_warning_count == 4:
                    print("[SAM3DBody] 사람이 감지되면 자동으로 메쉬가 생성됩니다. (경고 메시지 중단)")
                
                return self._get_empty_mesh()
            
            # 사람이 감지됨
            if hasattr(self, '_no_person_warning_count') and self._no_person_warning_count > 0:
                print("[SAM3DBody] 사람이 감지되었습니다! 메쉬 생성 중...")
                self._no_person_warning_count = 0  # 리셋
            
            output = outputs_list[0]
            
            # 출력 포맷 변환
            return self._format_output(output)
            
        except Exception as e:
            print(f"[SAM3DBody] 추론 오류: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 빈 메쉬 반환 (프로세스 중단 방지)
            return self._get_empty_mesh()
    
    def _format_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """SAM-3D-Body 출력을 표준 형식으로 변환"""
        result = {}
        
        # vertices 추출 (pred_vertices)
        if "pred_vertices" in output:
            vertices = output["pred_vertices"]
            if torch.is_tensor(vertices):
                vertices = vertices.cpu().numpy()
            elif isinstance(vertices, np.ndarray):
                pass
            else:
                vertices = np.array(vertices)
            result["vertices"] = vertices.astype(np.float32)
        else:
            print("[SAM3DBody] 경고: pred_vertices를 찾을 수 없습니다.")
            result["vertices"] = np.array([], dtype=np.float32).reshape(0, 3)
        
        # faces 추출 (estimator에서 가져옴)
        if hasattr(self, 'faces') and self.faces is not None:
            faces = self.faces
            if torch.is_tensor(faces):
                faces = faces.cpu().numpy()
            result["faces"] = faces.astype(np.int32)
        else:
            print("[SAM3DBody] 경고: faces를 찾을 수 없습니다.")
            result["faces"] = np.array([], dtype=np.int32).reshape(0, 3)
        
        # keypoints 추출 (pred_keypoints_3d)
        if "pred_keypoints_3d" in output:
            keypoints = output["pred_keypoints_3d"]
            if torch.is_tensor(keypoints):
                keypoints = keypoints.cpu().numpy()
            result["keypoints"] = keypoints.astype(np.float32)
        elif "pred_joint_coords" in output:
            # pred_joint_coords를 keypoints로 사용
            keypoints = output["pred_joint_coords"]
            if torch.is_tensor(keypoints):
                keypoints = keypoints.cpu().numpy()
            result["keypoints"] = keypoints.astype(np.float32)
        else:
            result["keypoints"] = np.array([], dtype=np.float32).reshape(0, 3)
        
        # joints 딕셔너리 생성 (pred_joint_coords 사용)
        joints_dict = {}
        if "pred_joint_coords" in output:
            joint_coords = output["pred_joint_coords"]
            if torch.is_tensor(joint_coords):
                joint_coords = joint_coords.cpu().numpy()
            
            # MHR70 관절 구조 (sam-3d-body의 표준)
            # 실제 관절 이름은 모델에 따라 다를 수 있음
            joint_names = [
                "pelvis", "left_hip", "right_hip",
                "spine1", "left_knee", "right_knee",
                "spine2", "left_ankle", "right_ankle",
                "spine3", "left_foot", "right_foot",
                "neck", "left_collar", "right_collar",
                "head", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
                "left_hand", "right_hand"
            ]
            
            # 관절 좌표가 2D 배열인 경우
            if joint_coords.ndim == 2 and joint_coords.shape[1] == 3:
                for i, name in enumerate(joint_names):
                    if i < len(joint_coords):
                        joints_dict[name] = joint_coords[i].astype(np.float32)
        
        result["joints"] = joints_dict
        
        # 사람 감지 여부 플래그 추가
        result["person_detected"] = True
        
        return result
    
    def _get_empty_mesh(self) -> Dict[str, Any]:
        """빈 메쉬 반환 (오류 시 사용)"""
        return {
            "vertices": np.array([], dtype=np.float32).reshape(0, 3),
            "faces": np.array([], dtype=np.int32).reshape(0, 3),
            "keypoints": np.array([], dtype=np.float32).reshape(0, 3),
            "joints": {},
            "person_detected": False  # 사람 감지 여부 플래그
        }
