# 재활 특화 기능 제안서

본 문서는 1인 개발자가 Move.ai 같은 대기업과 경쟁하기 위해, 재활/의료 특화 기능에 대한 구체적인 알고리즘 아이디어를 제시합니다.

---

## 기능 1: 좌우 비대칭성 분석 (Bilateral Asymmetry Analysis)

### 개요
재활 환자들은 뇌졸중, 편마비, 외상 등으로 인해 신체의 좌우 비대칭이 흔히 발생합니다. 이 기능은 3D 메쉬와 관절 데이터를 이용해 정량적으로 비대칭성을 측정하고 추적합니다.

### 알고리즘 아이디어

#### 1.1 공간 위치 비대칭 측정
- **입력**: 좌우 대칭 관절 쌍 (예: left_shoulder, right_shoulder)
- **계산**:
  - 높이 차이 (Y축): `|y_left - y_right|`
  - 전후 위치 차이 (Z축): `|z_left - z_right|`
  - 유클리드 거리: `||left_joint - right_joint||`
- **출력**: 각 관절 쌍별 비대칭 점수 (0-1, 1이 완전 대칭)

```python
# 구현 예시 (src/analytics/asymmetry_analysis.py 참조)
asymmetry_score = 1.0 / (1.0 + euclidean_distance)
```

#### 1.2 관절 각도 비대칭 측정
- **입력**: 좌우 관절 각도 (예: left_elbow_angle, right_elbow_angle)
- **계산**:
  - 각도 차이: `|angle_left - angle_right|`
  - 각도 비율: `min(angle_left, angle_right) / max(angle_left, right_angle)`
- **출력**: 각 관절별 각도 비대칭 메트릭

#### 1.3 시간 추적 및 재활 진전도 모니터링
- 세션별 비대칭 점수 저장 및 시각화
- 재활 전후 비교 리포트 생성
- 일일/주간/월간 트렌드 분석

### 경쟁 우위 포인트
- ✅ **정량적 데이터**: 주관적 판단이 아닌 수치 기반 평가
- ✅ **보험/임상 데이터**: 재활 효과 검증용 객관적 자료 제공
- ✅ **의료진 협업**: 치료사와 환자 간 소통 도구로 활용

### 구현 위치
- `src/analytics/asymmetry_analysis.py` (이미 구현됨)
- `src/pipeline/inference_worker.py`에서 통합

---

## 기능 2: 보행 지표 자동 추출 (Automated Gait Analysis)

### 개요
보행 재활은 뇌졸중, 척수손상, 무릎/고관절 수술 후 필수 과정입니다. 웹캠만으로 보행 패턴을 분석하여 치료사가 매번 수동 측정할 필요를 없앱니다.

### 알고리즘 아이디어

#### 2.1 보행 단계 탐지 (Gait Cycle Detection)
- **입력**: 시계열 관절 위치 (발목, 무릎, 골반)
- **알고리즘**:
  1. 발목 높이(Y축) 추적 → 최소값 = 발 바닥 접촉 지점
  2. 접촉 순간 감지 (임계값 기반)
  3. 보행 사이클 단위로 분할 (stance phase, swing phase)
- **출력**: 각 보행 단계의 시작/종료 타임스탬프

#### 2.2 보행 지표 계산
- **보폭 (Step Length)**: 같은 발의 연속 접촉 지점 간 거리
  ```python
  step_length = distance(heel_contact[n], heel_contact[n+2])  # 같은 발
  ```
  
- **보행 속도 (Cadence)**: 분당 보행 수
  ```python
  cadence = 60.0 / average_step_duration
  ```

- **양다리 지지 시간 비율**: 각 다리의 stance phase 비율
  ```python
  double_support_ratio = double_support_time / total_cycle_time
  ```

- **발각 높이 (Foot Clearance)**: 발이 지면에서 떨어지는 최소 높이
  ```python
  foot_clearance = min(ankle_height during swing_phase) - ground_level
  ```

#### 2.3 이상 보행 패턴 감지
- **트렌델렌버그 보행**: 골반이 stance phase 동안 반대편으로 기울어짐
  ```python
  pelvic_tilt = abs(left_hip_y - right_hip_y) during stance
  if pelvic_tilt > threshold:
      flag_trendelenburg = True
  ```
  
- **무릎 내반/외반 (Varus/Valgus)**: 무릎 관절 각도 이상
- **보행 불안정성**: 좌우 보폭 차이, 속도 변화량

### 힘판(Force Plate) 동기화
- 타임스탬프 기반으로 힘판 데이터와 3D 메쉬 데이터 매칭
- 발 접지 순간과 힘판 압력 피크 동기화
- 보행 분석 정확도 향상

### 경쟁 우위 포인트
- ✅ **저비용 솔루션**: 전문 보행 분석 장비(수천만원) 대비 웹캠 + 소프트웨어
- ✅ **가정/클리닉 활용**: 병원 외부에서도 모니터링 가능
- ✅ **연속 모니터링**: 운동 전/중/후 전체 과정 추적

### 구현 위치 (향후)
- `src/analytics/gait_analysis.py` (신규 생성 필요)
- 보행 감지를 위한 시계열 데이터 버퍼링 필요

---

## 기능 3: 자세 유지 및 피로 추적 (Posture Sustain & Fatigue Monitoring)

### 개요
장시간 앉거나 서 있을 때 자세 변화와 피로를 추적하여 작업장 안전, 재활 환자 모니터링, 예방 의학에 활용합니다.

### 알고리즘 아이디어

#### 3.1 자세 안정성 지표
- **상체 기울기 (Trunk Inclination)**:
  ```python
  trunk_angle = angle_between(shoulder_center - hip_center, vertical_axis)
  ```
  
- **골반 수평도 (Pelvic Tilt)**:
  ```python
  pelvic_tilt = angle_between(left_hip - right_hip, horizontal_axis)
  ```

- **어깨 수평도 (Shoulder Level)**:
  ```python
  shoulder_asymmetry = abs(left_shoulder_y - right_shoulder_y)
  ```

- **목 자세 (Neck Posture)**: 목-머리 각도 추적

#### 3.2 시간 기반 누적 지표
- **자세 점수 (Posture Score)**: 0-100 점수
  ```python
  posture_score = 100 * exp(-alpha * (deviation_from_ideal) ** 2)
  ```
  
- **피로 지표 (Fatigue Index)**:
  - 자세 변화 빈도 증가 → 피로 증가
  - 자세 각도 변화 가속도
  ```python
  fatigue = mean(abs(posture_angle_derivative)) over time_window
  ```

#### 3.3 실시간 알림 및 피드백
- 자세 이상 감지 시 경고 (예: "상체 좌측으로 5° 기울어짐 지속 2분")
- 주기적 자세 점수 표시
- 재활 운동 후 목표 자세 유지 시간 측정

#### 3.4 재활 특화 활용
- **좌우 비대칭 교정**: 일정 시간 동안 좌우 균형 유지 연습
- **강화 운동**: 특정 자세 유지 시간 측정 (예: 스쿼트 홀드)
- **치료 효과 평가**: 세션별 자세 안정성 개선도 추적

### 경쟁 우위 포인트
- ✅ **예방 의학**: 자세 문제 조기 발견
- ✅ **재활 개인화**: 환자별 맞춤 자세 목표 설정
- ✅ **데이터 기반 치료**: 치료사에게 객관적 자세 변화 데이터 제공

### 구현 위치 (향후)
- `src/analytics/posture_analysis.py` (신규 생성 필요)
- 시간 윈도우 기반 통계 계산 필요

---

## 통합 및 확장 가능성

### 공통 인프라
- **타임스탬프 동기화**: 모든 분석 기능이 동일한 타임스탬프 기준 사용
- **데이터 저장**: 세션별 결과를 JSON/CSV로 저장 → 장기 추적
- **웹 대시보드**: Flask/FastAPI로 간단한 웹 인터페이스 제공 (향후)

### 힘판(Force Plate) 연동
```python
# 힘판 데이터 수신 예시
force_data = {
    "timestamp": 1234567890.123,
    "force_left": [fx, fy, fz],
    "force_right": [fx, fy, fz],
    "cop_left": [x, y],  # Center of Pressure
    "cop_right": [x, y]
}

# 3D 메쉬 데이터와 동기화
mesh_data_timestamp = mesh_data["timestamp"]
nearest_force_data = find_nearest_timestamp(force_data_list, mesh_data_timestamp)
```

### API 설계 (향후)
```python
# REST API 예시
POST /api/analyze/asymmetry
GET /api/session/{session_id}/asymmetry
POST /api/analyze/gait
GET /api/analyze/posture/history
```

---

## 결론

위 세 가지 기능은 모두 **재활/의료 특화**이며, Move.ai와 같은 범용 모션 캡처 솔루션과 차별화됩니다:

1. **비대칭성 분석**: 정량적 재활 평가 도구
2. **보행 분석**: 저비용 보행 평가 솔루션
3. **자세 추적**: 예방 의학 및 개인화 재활

이러한 기능을 통해 "의료/재활 특화 플랫폼"으로 포지셔닝하여, 대기업과 경쟁할 수 있는 틈새 시장을 확보할 수 있습니다.
