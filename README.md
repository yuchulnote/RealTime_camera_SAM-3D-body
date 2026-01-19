# Hospi - 실시간 3D 인체 메쉬 복원 프로토타입

웹캠 기반 실시간 3D 인체 메쉬 복원 시스템으로, 재활/의료 기기 산업에 특화된 기능을 제공합니다.

## 데모 영상

https://github.com/user-attachments/assets/3dmeshtest.mp4

<video src="3dmeshtest.mp4" controls width="800"></video>

## 프로젝트 구조

```
Sam_3D/
├── config/                 # 설정 파일
├── src/                    # 소스 코드
│   ├── models/            # 모델 래퍼 (SAM-3D-Body, Anny)
│   ├── pipeline/          # 실시간 파이프라인 모듈
│   ├── visualization/     # 3D 시각화 모듈
│   ├── analytics/         # 재활 특화 분석 기능
│   └── utils/             # 유틸리티 함수
├── requirements.txt       # Python 의존성
├── environment_setup.md   # 환경 구성 가이드
├── setup.py              # 패키지 설정
└── main.py               # 메인 실행 파일
```

## 주요 기능

- 실시간 웹캠 기반 3D 인체 메쉬 복원
- 관절 각도 및 로봇공학적 지표 추출
- 힘판(Force Plate) 데이터 동기화 지원
- 재활 특화 분석 기능 (비대칭 분석, 보행 분석, 자세 추적)

## 빠른 시작

1. 환경 설정: `environment_setup.md` 참조
2. 의존성 설치: `pip install -r requirements.txt`
3. 실행: `python main.py`

## 라이선스

본 프로젝트는 상업적 용도로 사용 가능한 라이선스의 모델들을 활용합니다:
- SAM-3D-Body: SAM License (Meta)
- Anny: Apache 2.0 (Naver Labs)
