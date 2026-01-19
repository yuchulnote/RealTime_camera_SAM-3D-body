# Human Detector에 대한 설명

## "No human detector is used..." 메시지의 의미

### 현재 상태 (Detector 없음)

**작동 방식:**
- SAM-3D-Body는 **전체 이미지를 하나의 bounding box로 사용**합니다
- 즉, 이미지 전체를 사람이 있는 영역으로 가정하고 3D 메쉬를 생성합니다
- 코드: `boxes = np.array([0, 0, width, height]).reshape(1, 4)` (전체 이미지)

**장점:**
- ✅ 빠른 추론 속도 (detector 실행 시간 없음)
- ✅ 단일 인물 촬영 시 충분히 작동
- ✅ 추가 의존성 없음 (detectron2 등 불필요)

**단점:**
- ❌ 여러 사람이 있을 때 특정 사람을 선택할 수 없음
- ❌ 배경이 복잡할 때 정확도가 떨어질 수 있음
- ❌ 사람이 이미지 중앙에 있지 않으면 성능 저하

### Detector 사용 시

**작동 방식:**
- ViTDet 또는 SAM3 같은 사람 감지 모델이 먼저 실행됩니다
- 이미지에서 사람을 자동으로 감지하고 bounding box를 생성합니다
- 각 bounding box마다 3D 메쉬를 생성합니다

**장점:**
- ✅ 여러 사람 중 특정 사람 선택 가능
- ✅ 배경 노이즈 제거로 더 정확한 추정
- ✅ 사람이 이미지 어디에 있든 감지

**단점:**
- ❌ 추론 속도 저하 (약 2-3배 느림)
- ❌ 추가 의존성 필요 (detectron2 등)
- ❌ GPU 메모리 사용량 증가

## 현재 작동 여부

**✅ 현재도 정상 작동합니다!**

터미널 로그를 보면:
```
[InferenceWorker] 첫 메쉬 생성 완료: 18439 정점, 36874 면
```

메쉬가 생성되고 있으므로, detector 없이도 작동하고 있습니다.

## Detector 활성화 방법

`config/config.yaml`에서 설정:

```yaml
model:
  use_detector: true  # false → true로 변경
```

주의: `use_detector: true`로 설정하면 detectron2가 필요합니다.

## 권장 사항

### Detector 없이 사용 (현재 설정, 권장)
- 단일 인물 촬영 시
- 실시간 성능이 중요할 때
- 추가 의존성 설치가 어려울 때

### Detector 사용
- 여러 사람이 있는 환경
- 정확도가 더 중요할 때
- detectron2 설치가 가능할 때

## 결론

**"No human detector is used..." 메시지는 경고가 아닙니다!**

현재 설정으로도 정상 작동하며, 단일 인물 촬영 시에는 detector 없이도 충분합니다. 
메쉬가 생성되고 있다면 정상적으로 작동하고 있는 것입니다.
