# 자주 묻는 질문 (FAQ)

## Q1: 회색 화면만 나오는데, 카메라 앞에 사람이 잡히면 3D mesh가 그려지는건가?

### 답변:

**현재 상태 (더미 모델 사용 중):**

더미 모델은 카메라 입력과 관계없이 항상 같은 구(sphere) 모양 메쉬를 생성합니다. 따라서:
- ✅ 파이프라인이 정상 작동하면 **구 모양 메쉬가 화면에 표시됩니다**
- ❌ 카메라 앞에 사람이 있어도 **더미 모델은 사람을 인식하지 않습니다**
- ❌ 사람의 3D 메쉬를 생성하지 않습니다

**실제 모델 통합 후:**

SAM-3D-Body 또는 Anny 모델을 통합하면:
- ✅ 카메라 프레임에서 **사람을 자동으로 감지**
- ✅ 감지된 사람의 **실제 3D 인체 메쉬 생성**
- ✅ 사람이 카메라 앞에 있을 때만 3D 메쉬가 업데이트됨

### 회색 화면만 나오는 경우 해결 방법:

1. **터미널 출력 확인**: 프로그램 실행 시 다음과 같은 메시지가 나타나는지 확인하세요:
   ```
   [FrameCapture] 웹캠 초기화 완료: 640x480
   [InferenceWorker] 첫 프레임 수신: (480, 640, 3)
   [InferenceWorker] 첫 메쉬 생성 완료: 400 정점, 800 면
   [Visualizer] 첫 메쉬 표시됨: 400 정점, 800 면
   ```

2. **웹캠 연결 확인**: 웹캠이 제대로 연결되어 있고 다른 프로그램이 사용 중이 아닌지 확인

3. **초기 대기 시간**: 프로그램 시작 후 1-2초 기다리면 첫 메쉬가 표시됩니다

4. **Open3D 윈도우 확인**: Open3D 윈도우가 정상적으로 열렸는지 확인 (다른 창 뒤에 있을 수 있음)

---

## Q2: 실제 모델(SAM-3D-Body 또는 Anny)을 통합하려면 어떻게 해야 하나요?

### 단계별 가이드:

#### 1단계: 모델 다운로드 및 설치

**Anny 모델 (권장 - Apache 2.0 라이선스):**
```bash
pip install anny
# 또는 GitHub에서
git clone https://github.com/naver/anny.git
cd anny
pip install -e .
```

**SAM-3D-Body 모델:**
```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -r requirements.txt
# 체크포인트 다운로드 (GitHub 릴리스 참조)
```

#### 2단계: 모델 래퍼 클래스 생성

`src/models/anny_model.py` 또는 `src/models/sam3d_model.py` 파일을 생성하고 `BaseBodyEstimator`를 상속받아 구현:

```python
from .base_model import BaseBodyEstimator
import numpy as np

class AnnyBodyEstimator(BaseBodyEstimator):
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        # Anny 모델 초기화
        ...
    
    def predict(self, image: np.ndarray) -> dict:
        # Anny 모델로 3D 메쉬 추정
        ...
        return {
            "vertices": vertices,  # (N, 3)
            "faces": faces,        # (M, 3)
            "keypoints": keypoints,
            "joints": joints
        }
```

#### 3단계: main.py 수정

`main.py`의 `setup_components()` 메서드에서:

```python
if model_name == "anny":
    from src.models.anny_model import AnnyBodyEstimator
    self.model = AnnyBodyEstimator(
        checkpoint_path=model_config["checkpoint_path"],
        device=device
    )
```

#### 4단계: config.yaml 수정

```yaml
model:
  name: "anny"  # "dummy" → "anny"로 변경
  checkpoint_path: "checkpoints/anny_model.pth"
```

---

## Q3: 파이프라인의 작동 방식은?

### 데이터 흐름:

```
[웹캠] 
  ↓ (프레임 캡처)
[FrameCapture Thread]
  ↓ (frame_queue)
[InferenceWorker Thread]
  ↓ (AI 모델 추론 → 3D 메쉬 생성)
  ↓ (mesh_queue)
[Visualizer Thread]
  ↓ (Open3D 렌더링)
[화면에 3D 메쉬 표시]
```

### 각 컴포넌트 역할:

1. **FrameCapture**: 웹캠에서 프레임을 실시간으로 캡처 (타임스탬프 포함)
2. **InferenceWorker**: AI 모델로 3D 메쉬 추정
3. **Visualizer**: Open3D로 3D 메쉬를 화면에 실시간 렌더링

모든 컴포넌트는 **스레드**로 실행되어 병렬 처리됩니다.

---

## Q4: 성능 최적화 방법은?

### GPU 메모리 부족 시:

1. **입력 해상도 낮추기** (`config/config.yaml`):
   ```yaml
   model:
     input_resolution: [224, 224]  # [256, 256] → [224, 224]
   ```

2. **배치 크기 조정**: 항상 1로 설정 (실시간 스트리밍)

3. **Mixed Precision (FP16)**: 모델이 지원하면 FP16 사용

### 추론 속도 향상:

1. **TensorRT 변환**: ONNX → TensorRT 변환으로 추론 속도 향상
2. **모델 경량화**: 경량 모델 버전 사용
3. **프레임 스킵**: 매 N번째 프레임만 처리

---

## Q5: 힘판(Force Plate) 데이터와 동기화하는 방법은?

모든 데이터에 타임스탬프가 포함되어 있으므로, 힘판 데이터와 매칭 가능:

```python
# 힘판 데이터와 3D 메쉬 데이터 동기화 예시
def synchronize_with_force_plate(mesh_timestamp, force_plate_data):
    # 가장 가까운 타임스탬프 찾기
    nearest_force = find_nearest_timestamp(
        force_plate_data, 
        mesh_timestamp,
        max_diff=0.033  # 30 FPS 기준 약 1 프레임 차이
    )
    return nearest_force
```

---

## Q6: 재활 특화 기능을 추가하려면?

`business_features.md`를 참조하여 다음 기능을 구현할 수 있습니다:

1. **비대칭성 분석**: `src/analytics/asymmetry_analysis.py` (이미 구현됨)
2. **보행 분석**: `src/analytics/gait_analysis.py` (구현 필요)
3. **자세 추적**: `src/analytics/posture_analysis.py` (구현 필요)

각 기능은 `src/analytics/` 디렉토리에 모듈로 추가하고, `InferenceWorker`에서 호출하면 됩니다.
