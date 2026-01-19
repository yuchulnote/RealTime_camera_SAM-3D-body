# 환경 구성 가이드

## 하드웨어 요구사항

- NVIDIA GPU가 탑재된 Windows/Linux 노트북
- CUDA 지원 GPU (CUDA Compute Capability 7.0 이상 권장)
- 최소 8GB GPU 메모리 권장

## 단계 1: CUDA 및 드라이버 확인

### Windows
```bash
nvidia-smi
```
출력에서 CUDA Version 확인 (예: CUDA 12.1)

### Linux
```bash
nvidia-smi
nvcc --version
```

## 단계 2: Conda 환경 생성 및 CUDA 설정

### Windows/Linux 공통

```bash
# Conda 환경 생성 (Python 3.10 권장)
conda create -n human_mesh python=3.10 -y
conda activate human_mesh

# CUDA Toolkit 및 cuDNN 설치
# CUDA 12.1 기준
conda install -c nvidia cudatoolkit=12.1 cudnn=8.9 -y

# 또는 시스템 CUDA 사용 시 (이미 설치된 경우)
# CUDA 버전 확인 후 해당 버전에 맞는 PyTorch 설치
```

## 단계 3: PyTorch 설치 (CUDA 지원)

### CUDA 12.1 사용 시
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8 사용 시
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU만 사용 시 (비권장, 매우 느림)
```bash
pip install torch torchvision torchaudio
```

### 설치 확인
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## 단계 4: 핵심 의존성 라이브러리 설치

### 필수 라이브러리
```bash
# 이미지/비디오 처리
pip install opencv-python numpy pillow

# 3D 메쉬 처리 및 시각화
pip install open3d trimesh

# 추가 3D 렌더링 (선택사항)
pip install pyrender pyglet

# 수학 연산
pip install scipy scikit-learn

# 진행 상황 표시
pip install tqdm

# 설정 파일 처리
pip install pyyaml
```

### 딥러닝 관련 추가 라이브러리
```bash
# Transformer 및 Vision 모델
pip install transformers timm einops

# 메쉬 연산 (PyTorch3D)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# 또는
# conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# conda install pytorch3d -c pytorch3d

# ONNX 변환 (추론 최적화, 선택사항)
pip install onnx onnxruntime-gpu
```

## 단계 5: 모델별 의존성

### Anny 모델 (Naver Labs) - Apache 2.0 라이선스

```bash
# Anny 설치
pip install anny

# 또는 GitHub에서 직접 설치
git clone https://github.com/naver/anny.git
cd anny
pip install -e .

# Warp GPU 커널 (선택사항, 성능 향상)
# Windows: Visual Studio Build Tools 필요
# Linux: CUDA Toolkit 설치 필요
pip install warp-lang
```

### SAM-3D-Body 모델 (Meta) - SAM License

```bash
# GitHub에서 클론
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body

# 의존성 설치
pip install -r requirements.txt

# 모델 체크포인트 다운로드 (GitHub 릴리스 페이지 참조)
# 체크포인트 파일을 프로젝트의 checkpoints/ 디렉토리에 저장
```

## 단계 6: 설치 검증

```bash
# GPU 및 PyTorch 확인
python -c "import torch; import cv2; import open3d as o3d; print('✓ 모든 핵심 라이브러리 설치 완료'); print(f'✓ CUDA: {torch.cuda.is_available()}')"

# Anny 설치 확인 (선택)
python -c "import anny; print('✓ Anny 설치 확인')"
```

## 라이선스 고려사항

### SAM-3D-Body (SAM License)
- ✅ 상업적 사용 가능
- ❌ 군사/무기/핵/ITAR/제재 관련 목적 사용 금지
- ✅ 파생물 배포 시 라이선스 사본 포함 필수
- 📄 상세 내용: https://github.com/facebookresearch/sam-3d-body

### Anny (Apache 2.0)
- ✅ 완전히 자유로운 상업/비상업 활용
- ✅ 파생물 자유 사용
- 📄 라이선스: Apache License 2.0

## 문제 해결

### CUDA out of memory 오류
- 모델 입력 해상도 낮추기 (256x256 → 224x224)
- 배치 크기 1로 설정
- Mixed precision (FP16) 사용

### Open3D 시각화 윈도우가 안 뜨는 경우
```bash
# Windows
pip install pyopengl

# Linux
sudo apt-get install python3-opengl
```

### PyTorch3D 설치 오류
- CUDA 버전과 PyTorch 버전 호환성 확인
- conda를 통한 설치 권장
