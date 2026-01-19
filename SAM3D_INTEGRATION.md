# SAM-3D-Body 통합 가이드

## 개요

SAM-3D-Body 모델 통합이 완료되었습니다. 이 문서는 모델을 설치하고 사용하는 방법을 설명합니다.

## 설치 방법

### 방법 1: 공식 패키지 설치 (권장)

```bash
pip install sam-3d-body
```

### 방법 2: GitHub에서 직접 설치

```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -e .
```

### 방법 3: HuggingFace에서 체크포인트 다운로드

SAM-3D-Body는 HuggingFace를 통해 체크포인트를 자동 다운로드합니다:
- 저장소 ID: `facebook/sam-3d-body-dinov3`

## 설정

`config/config.yaml` 파일에서 모델 설정:

```yaml
model:
  name: "sam3d_body"  # "dummy", "anny", 또는 "sam3d_body"
  checkpoint_path: null  # null이면 HuggingFace에서 자동 다운로드
  device: "cuda:0"
  input_resolution: [512, 512]  # 권장: 512x512 또는 640x640
```

## 사용 방법

### 1. 기본 실행

```bash
python main.py
```

### 2. 설정 파일 지정

```bash
python main.py --config config/config.yaml
```

## 동작 방식

### 파이프라인

```
[웹캠] 
  ↓ RGB 프레임 (512x512 리사이즈)
[SAM-3D-Body 모델]
  ↓ 3D 메쉬 추정 (vertices, faces, joints)
[Open3D 시각화]
  ↓ 실시간 3D 메쉬 렌더링
[화면에 표시]
```

### 특징

- ✅ **실시간 인체 감지**: 카메라 앞에 사람이 있으면 자동으로 3D 메쉬 생성
- ✅ **자동 모델 다운로드**: 체크포인트 경로가 없으면 HuggingFace에서 자동 다운로드
- ✅ **타임스탬프 동기화**: 모든 메쉬 데이터에 타임스탬프 포함
- ✅ **오류 처리**: 모델 로드 실패 시 자동으로 더미 모델로 폴백

## 문제 해결

### 1. 모델 로드 오류

**증상**: "SAM-3D-Body 라이브러리를 찾을 수 없습니다"

**해결**:
```bash
pip install sam-3d-body
# 또는
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -e .
```

### 2. CUDA out of memory

**해결**:
- `config/config.yaml`에서 `input_resolution`을 낮추세요 (예: [384, 384])
- `device: "cpu"`로 변경 (매우 느림)

### 3. 메쉬가 표시되지 않음

**확인사항**:
- 터미널에서 "[SAM3DBody] 모델 초기화 완료" 메시지 확인
- "[InferenceWorker] 첫 메쉬 생성 완료" 메시지 확인
- 카메라 앞에 사람이 있는지 확인

### 4. 추론 속도가 느림

**최적화 방법**:
- 입력 해상도 낮추기: [512, 512] → [384, 384]
- GPU 메모리가 충분한지 확인
- Mixed Precision (FP16) 사용 (모델이 지원하는 경우)

## 모델 출력 형식

SAM-3D-Body 모델은 다음 형식의 출력을 생성합니다:

```python
{
    "vertices": np.array,  # (N, 3) - 메쉬 정점 좌표
    "faces": np.array,     # (M, 3) - 메쉬 면 인덱스
    "keypoints": np.array, # (K, 3) - 키포인트 위치
    "joints": dict         # 관절 위치 딕셔너리
}
```

## 성능

- **입력 해상도**: 512x512 권장 (더 높으면 정확도 향상, 속도 저하)
- **추론 시간**: GPU 기준 약 50-200ms (해상도 및 하드웨어에 따라 다름)
- **메모리**: GPU VRAM 약 2-4GB 필요

## 참고 자료

- GitHub: https://github.com/facebookresearch/sam-3d-body
- HuggingFace: https://huggingface.co/facebook/sam-3d-body-dinov3
- 라이선스: SAM License (상업적 사용 가능, 제약 있음)
