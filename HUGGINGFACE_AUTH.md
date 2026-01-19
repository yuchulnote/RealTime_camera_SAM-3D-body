# HuggingFace 인증 가이드

## 문제 상황

웹사이트에서 SAM-3D-Body 모델 접근 권한을 받았지만, 코드에서 다운로드할 때 401 인증 오류가 발생합니다.

**에러 메시지:**
```
GatedRepoError: 401 Client Error
Access to model facebook/sam-3d-body-dinov3 is restricted. 
You must have access to it and be authenticated to access it. Please log in.
```

## 해결 방법

### 방법 1: HuggingFace CLI로 로그인 (권장)

1. **HuggingFace 토큰 생성**:
   - https://huggingface.co/settings/tokens 접속
   - "New token" 클릭
   - 토큰 이름 입력 (예: "sam3d-access")
   - 권한: **Read** 선택
   - "Generate token" 클릭
   - 생성된 토큰 복사 (한 번만 표시됨!)

2. **터미널에서 로그인**:
   ```bash
   # huggingface_hub 설치 (없는 경우)
   pip install huggingface_hub
   
   # 로그인
   huggingface-cli login
   ```
   
   또는 토큰을 직접 입력:
   ```bash
   huggingface-cli login --token YOUR_TOKEN_HERE
   ```

3. **확인**:
   ```bash
   huggingface-cli whoami
   ```

### 방법 2: 환경 변수로 토큰 설정

1. **토큰 생성** (방법 1과 동일)

2. **환경 변수 설정**:
   
   **Windows (CMD):**
   ```cmd
   set HF_TOKEN=your_token_here
   ```
   
   **Windows (PowerShell):**
   ```powershell
   $env:HF_TOKEN="your_token_here"
   ```
   
   **영구 설정 (Windows):**
   - 시스템 환경 변수에 `HF_TOKEN` 추가
   - 또는 사용자 환경 변수에 추가

3. **Python 코드에서 확인**:
   ```python
   import os
   print("HF_TOKEN:", os.getenv("HF_TOKEN"))
   ```

### 방법 3: Python 코드에서 직접 로그인

```python
from huggingface_hub import login

# 토큰으로 로그인
login(token="your_token_here")

# 또는 대화형 로그인
login()
```

## 확인 방법

다운로드 테스트:
```bash
huggingface-cli download facebook/sam-3d-body-dinov3 --repo-type model
```

또는 Python에서:
```python
from huggingface_hub import hf_hub_download

# 작은 파일로 테스트
hf_hub_download(
    repo_id="facebook/sam-3d-body-dinov3",
    filename="README.md",
    repo_type="model"
)
```

## 주의사항

1. **토큰 보안**: 토큰을 코드에 직접 하드코딩하지 마세요
2. **권한**: Read 권한만 있으면 충분합니다
3. **접근 권한**: 웹사이트에서 모델 접근 권한을 받았는지 확인하세요

## 문제 해결

### "Token is invalid" 오류
- 토큰을 다시 생성하세요
- 토큰이 만료되지 않았는지 확인하세요

### "Access denied" 오류
- 웹사이트에서 모델 접근 권한을 받았는지 확인
- https://huggingface.co/facebook/sam-3d-body-dinov3 접속하여 "Agree and access repository" 클릭

### "Module not found" 오류
```bash
pip install huggingface_hub
```
