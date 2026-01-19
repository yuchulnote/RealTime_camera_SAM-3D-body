# SAM-3D-Body 통합 완료

## ✅ 완료된 작업

1. **모델 다운로드 완료**: HuggingFace에서 SAM-3D-Body 모델 다운로드됨
2. **코드 통합 완료**: `src/models/sam3d_model.py`에서 실제 SAM-3D-Body API 사용
3. **설정 파일 업데이트**: `config/config.yaml`에 SAM-3D-Body 설정 추가

## 🚀 실행 방법

### 1. 기본 실행

```bash
python main.py
```

### 2. 설정 확인

`config/config.yaml`에서 다음 설정을 확인하세요:

```yaml
model:
  name: "sam3d_body"  # SAM-3D-Body 사용
  checkpoint_path: null  # null이면 HuggingFace 캐시에서 자동 찾기
  hf_repo_id: "facebook/sam-3d-body-dinov3"
  device: "cuda:0"
  use_detector: false  # 사람 감지기 (true면 더 정확하지만 느림)
  use_fov_estimator: false  # FOV 추정기
```

## 📋 동작 방식

1. **모델 자동 로드**: HuggingFace 캐시에서 다운로드된 모델을 자동으로 찾습니다
   - 경로: `C:\Users\user\.cache\huggingface\hub\models--facebook--sam-3d-body-dinov3\snapshots\...`

2. **실시간 추론**: 웹캠 프레임을 받아서 SAM-3D-Body로 3D 메쉬 생성

3. **출력 형식**:
   - `vertices`: 3D 메쉬 정점 좌표
   - `faces`: 메쉬 면 인덱스
   - `keypoints`: 3D 키포인트
   - `joints`: 관절 위치 딕셔너리

## ⚙️ 성능 최적화 옵션

### 사람 감지기 사용 (더 정확하지만 느림)

```yaml
model:
  use_detector: true  # 사람 감지기 활성화
```

- 장점: 여러 사람 중 특정 사람만 추적 가능
- 단점: 추론 속도 저하 (약 2-3배 느림)

### FOV 추정기 사용 (카메라 파라미터 자동 추정)

```yaml
model:
  use_fov_estimator: true  # FOV 추정기 활성화
```

- 장점: 카메라 내부 파라미터 자동 추정으로 더 정확한 3D 복원
- 단점: 초기 추론 시간 증가

## 🔍 문제 해결

### 모델을 찾을 수 없음

에러: `HuggingFace 캐시에서 모델을 찾을 수 없습니다`

해결:
1. 모델이 다운로드되었는지 확인:
   ```bash
   hf download facebook/sam-3d-body-dinov3
   ```

2. 수동으로 경로 지정:
   ```yaml
   model:
     checkpoint_path: "C:/Users/user/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/11aaa346c7204874a1cbafe3d39a979080b2c55a/model.ckpt"
   ```

### CUDA out of memory

해결:
1. 입력 해상도 낮추기 (모델 자체 해상도 사용)
2. `use_detector: false`로 설정
3. GPU 메모리 정리 후 재시도

### 사람이 감지되지 않음

- 카메라 앞에 사람이 있는지 확인
- 조명이 충분한지 확인
- `use_detector: true`로 설정하여 사람 감지기 사용

## 📊 예상 성능

- **추론 시간**: GPU 기준 약 100-300ms (해상도 및 설정에 따라 다름)
- **FPS**: 약 3-10 FPS (실시간 처리 가능)
- **메모리**: GPU VRAM 약 3-5GB 필요

## 🎯 다음 단계

1. **실행 테스트**: `python main.py` 실행하여 카메라 앞에서 테스트
2. **성능 튜닝**: 필요에 따라 `use_detector`, `use_fov_estimator` 조정
3. **재활 기능 추가**: `src/analytics/` 모듈 활용하여 비대칭 분석 등 추가
