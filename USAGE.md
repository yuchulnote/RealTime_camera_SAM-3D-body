# 사용 가이드

## 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성 및 활성화
conda create -n human_mesh python=3.10 -y
conda activate human_mesh

# CUDA Toolkit 설치 (CUDA 12.1 기준)
conda install -c nvidia cudatoolkit=12.1 cudnn=8.9 -y

# PyTorch 설치 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 프로젝트 의존성 설치
pip install -r requirements.txt
```

자세한 내용은 `environment_setup.md`를 참조하세요.

### 2. 실행

```bash
# 기본 설정으로 실행
python main.py

# 설정 파일 지정
python main.py --config config/config.yaml
```

### 3. 종료

- Open3D 윈도우를 닫거나
- 터미널에서 `Ctrl+C` 입력

## 프로젝트 구조

```
Sam_3D/
├── main.py                      # 메인 실행 파일
├── config/
│   └── config.yaml             # 설정 파일
├── src/
│   ├── pipeline/               # 실시간 파이프라인
│   │   ├── frame_capture.py   # 웹캠 프레임 캡처
│   │   ├── inference_worker.py # AI 모델 추론
│   │   └── visualizer.py       # 3D 메쉬 시각화
│   ├── models/                 # AI 모델 래퍼
│   │   ├── base_model.py      # 기본 인터페이스
│   │   └── dummy_model.py     # 더미 모델 (테스트용)
│   ├── analytics/              # 재활 분석 기능
│   │   ├── joint_angles.py    # 관절 각도 계산
│   │   └── asymmetry_analysis.py # 비대칭 분석
│   └── utils/                  # 유틸리티
│       └── config_loader.py   # 설정 로더
└── requirements.txt            # Python 의존성
```

## 설정 파일 (config/config.yaml)

주요 설정 항목:

- **model.name**: 사용할 모델 ("anny", "sam3d_body", "dummy")
- **camera.device_id**: 웹캠 장치 ID (기본값: 0)
- **pipeline.target_fps**: 목표 FPS
- **visualization**: 시각화 옵션

## 실제 모델 통합 방법

현재는 더미 모델이 사용됩니다. 실제 SAM-3D-Body 또는 Anny 모델을 통합하려면:

### Anny 모델 통합 예시

```python
# src/models/anny_model.py 생성

from .base_model import BaseBodyEstimator
import anny
import numpy as np
import torch

class AnnyBodyEstimator(BaseBodyEstimator):
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        # Anny 모델 로드
        self.model = anny.load('full_body')
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Anny 전처리 로직
        ...
    
    def predict(self, image: np.ndarray) -> dict:
        # Anny 추론 로직
        ...
        return {
            "vertices": vertices,
            "faces": faces,
            "keypoints": keypoints,
            "joints": joints
        }
```

그리고 `main.py`의 `setup_components()`에서:

```python
if model_name == "anny":
    from src.models.anny_model import AnnyBodyEstimator
    self.model = AnnyBodyEstimator(
        checkpoint_path=model_config["checkpoint_path"],
        device=device
    )
```

### SAM-3D-Body 모델 통합 예시

유사하게 `src/models/sam3d_model.py`를 생성하고 SAM-3D-Body의 API에 맞춰 구현하세요.

## 관절 각도 추출 사용법

```python
from src.analytics import extract_joint_angles

# joints는 모델에서 반환된 관절 딕셔너리
angles = extract_joint_angles(joints)
print(f"왼쪽 팔꿈치 각도: {angles.get('left_elbow', 'N/A')}도")
```

## 비대칭성 분석 사용법

```python
from src.analytics import analyze_full_body_asymmetry

# joints는 모델에서 반환된 관절 딕셔너리
asymmetry = analyze_full_body_asymmetry(joints)
print(f"어깨 비대칭 점수: {asymmetry['left_shoulder_right_shoulder']['symmetry_score']:.2f}")
```

## 힘판(Force Plate) 데이터 동기화

타임스탬프는 모든 데이터에 포함되어 있으므로, 힘판 데이터와 매칭 가능합니다:

```python
# 추론 결과에서 타임스탬프 추출
timestamp, mesh_data = mesh_queue.get()

# 힘판 데이터에서 가장 가까운 타임스탬프 찾기
nearest_force_data = find_nearest_timestamp(force_plate_data, timestamp)

# 동기화된 데이터 분석
analyze_synchronized_data(mesh_data, nearest_force_data)
```

## 문제 해결

### CUDA out of memory
- `config/config.yaml`에서 `input_resolution`을 낮추세요 (예: [224, 224])
- GPU 메모리가 부족하면 `device: "cpu"`로 설정 (매우 느림)

### 웹캠이 인식되지 않음
- `camera.device_id`를 1, 2 등으로 변경해보세요
- 다른 프로그램이 웹캠을 사용 중인지 확인하세요

### Open3D 윈도우가 안 뜸
- OpenGL 드라이버 업데이트 확인
- Linux: `sudo apt-get install python3-opengl`
- Windows: Visual C++ Redistributable 설치 확인

## 다음 단계

1. 실제 모델 통합 (Anny 또는 SAM-3D-Body)
2. 재활 특화 기능 추가 (`business_features.md` 참조)
3. 힘판 데이터 연동
4. 웹 대시보드 개발 (선택사항)

자세한 기능 아이디어는 `business_features.md`를 참조하세요.
