"""카메라 유틸리티 함수"""
import cv2
import sys
import platform
from typing import List, Tuple, Optional, Dict



def list_available_cameras(max_test: int = 20) -> List[Tuple[int, str, bool]]:
    """
    사용 가능한 카메라 목록을 반환합니다.
    
    Args:
        max_test: 테스트할 최대 카메라 인덱스 (기본값: 20, Astra 등 특수 카메라 포함)
        
    Returns:
        List of (index, name, is_available) tuples
    """
    available_cameras = []
    
    print("[CameraUtils] 사용 가능한 카메라 검색 중...")
    
    # Windows에서 카메라 이름 목록 미리 가져오기 시도 (여러 방법 시도)
    all_camera_names = []  # DirectShow 인덱스 순서대로 카메라 이름
    if platform.system() == "Windows":
        # 방법 1: pygrabber 라이브러리 사용 (가장 정확)
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                all_camera_names = list(devices)
                print(f"[CameraUtils] pygrabber로 {len(all_camera_names)}개의 카메라를 발견했습니다.")
        except ImportError:
            # pygrabber가 없으면 다음 방법 시도
            pass
        except Exception:
            pass
        
        # 방법 2: Windows Registry에서 DirectShow VideoCaptureDevices 읽기
        if not all_camera_names:
            try:
                import winreg
                key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\DirectShow\VideoCaptureDevices"
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            all_camera_names.append(subkey_name)
                            i += 1
                        except OSError:
                            break
                    winreg.CloseKey(key)
                    if all_camera_names:
                        print(f"[CameraUtils] Windows Registry에서 {len(all_camera_names)}개의 카메라를 발견했습니다.")
                except (FileNotFoundError, OSError):
                    pass
                except Exception:
                    pass
            except ImportError:
                pass
            except Exception:
                pass
        
        # 방법 3: PowerShell로 DirectShow Registry 읽기
        if not all_camera_names:
            try:
                import subprocess
                ps_script = '''
                $regPath = "HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\DirectShow\\VideoCaptureDevices"
                if (Test-Path $regPath) {
                    $devices = @()
                    Get-ChildItem $regPath | ForEach-Object {
                        $devices += $_.PSChildName
                    }
                    $devices
                }
                '''
                result = subprocess.run(
                    ['powershell', '-Command', ps_script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                if result.returncode == 0 and result.stdout.strip():
                    all_camera_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
                    if all_camera_names:
                        print(f"[CameraUtils] PowerShell (Registry)로 {len(all_camera_names)}개의 카메라를 발견했습니다.")
            except Exception:
                pass
        
        # 방법 4: PowerShell로 PnP 장치에서 카메라 이름 가져오기 (최후의 수단)
        if not all_camera_names:
            try:
                import subprocess
                ps_command = """
                Get-PnpDevice -Class Camera | Where-Object {$_.Status -eq "OK"} | 
                Select-Object -ExpandProperty FriendlyName | 
                Where-Object {$_ -notmatch "printer|scanner|officejet|hp230|hp211|hp250|hp7720|hp8710"}
                """
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                if result.returncode == 0:
                    all_camera_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                    if all_camera_names:
                        print(f"[CameraUtils] PowerShell (PnP)로 {len(all_camera_names)}개의 카메라를 발견했습니다.")
            except Exception:
                pass
        
        # 발견된 카메라 이름 출력
        if all_camera_names:
            print(f"[CameraUtils] 총 {len(all_camera_names)}개의 카메라 장치를 발견했습니다:")
            for idx, name in enumerate(all_camera_names):
                print(f"  [{idx}] {name}")
    
    # 여러 백엔드 시도 (Windows)
    backends_to_try = []
    if platform.system() == "Windows":
        # DirectShow, MSMF, V4L2 순서로 시도
        backends_to_try = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Default")
        ]
    else:
        backends_to_try = [(cv2.CAP_ANY, "Default")]
    
    found_indices = set()  # 이미 찾은 인덱스 추적
    
    for i in range(max_test):
        # 이미 찾은 카메라는 스킵
        if i in found_indices:
            continue
            
        # 여러 백엔드로 시도
        for backend_id, backend_name in backends_to_try:
            try:
                if backend_id == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(i)
                else:
                    cap = cv2.VideoCapture(i, backend_id)
                
                if cap.isOpened():
                    # 실제로 프레임을 읽을 수 있는지 테스트
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_backend = cap.getBackendName()
                        
                        # PowerShell에서 가져온 카메라 이름과 매칭 시도
                        device_name = None
                        if i < len(all_camera_names):
                            device_name = all_camera_names[i]
                        
                        # 카메라 이름 구성
                        if device_name:
                            name = f"{device_name} (ID: {i}, {width}x{height}, {actual_backend})"
                        else:
                            name = f"Camera {i} ({width}x{height}, {actual_backend})"
                        
                        available_cameras.append((i, name, True))
                        found_indices.add(i)
                        print(f"  ✓ 카메라 {i}: {name}")
                        cap.release()
                        break  # 성공했으면 다음 백엔드 시도 안 함
                    else:
                        cap.release()
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                # 백엔드별 오류는 조용히 무시하고 다음 백엔드 시도
                pass
        
        # 모든 백엔드 실패 시에도 PowerShell에서 이름이 있으면 표시
        if i not in found_indices and i < len(all_camera_names):
            device_name = all_camera_names[i]
            # OpenCV로 열 수 없지만 시스템에는 등록된 카메라
            name = f"{device_name} (ID: {i}, OpenCV로 열 수 없음 - 특수 드라이버 필요)"
            available_cameras.append((i, name, False))
            print(f"  ⚠ 카메라 {i}: {name}")
            found_indices.add(i)
    
    if len(available_cameras) == 0:
        print("[CameraUtils] 경고: 사용 가능한 카메라를 찾을 수 없습니다.")
        if len(all_camera_names) > 0:
            print("[CameraUtils] 힌트: 시스템에는 카메라가 있지만 OpenCV로 접근할 수 없습니다.")
            print("[CameraUtils] 힌트: 특수 카메라(Astra 등)는 전용 SDK가 필요할 수 있습니다.")
    else:
        working_count = sum(1 for _, _, available in available_cameras if available)
        print(f"[CameraUtils] 총 {len(available_cameras)}개의 카메라를 찾았습니다. ({working_count}개 작동 가능)")
    
    return available_cameras


def select_camera_interactive(preferred_id: Optional[int] = None) -> int:
    """
    사용자 입력으로 카메라를 선택합니다.
    
    Args:
        preferred_id: 선호하는 카메라 ID (None이면 대화형 선택)
        
    Returns:
        선택된 카메라 ID
    """
    available_cameras = list_available_cameras()
    
    if len(available_cameras) == 0:
        raise RuntimeError("사용 가능한 카메라가 없습니다.")
    
    # 선호하는 카메라가 사용 가능한지 확인
    if preferred_id is not None:
        for idx, name, available in available_cameras:
            if idx == preferred_id and available:
                print(f"[CameraUtils] 선호 카메라 {preferred_id} 선택: {name}")
                return preferred_id
        print(f"[CameraUtils] 경고: 카메라 {preferred_id}를 사용할 수 없습니다. 대화형 선택으로 전환합니다.")
    
    # 대화형 선택
    print("\n" + "="*60)
    print("사용 가능한 카메라 목록:")
    print("="*60)
    for idx, name, available in available_cameras:
        status = "✓" if available else "⚠ (OpenCV로 열 수 없음)"
        print(f"  [{idx}] {name} {status}")
    print("="*60)
    
    while True:
        try:
            user_input = input(f"\n카메라 번호를 선택하세요 (0-{len(available_cameras)-1}, Enter=자동 선택): ").strip()
            
            if user_input == "":
                # Enter 키: 첫 번째 카메라 자동 선택
                selected_id, selected_name, _ = available_cameras[0]
                print(f"[CameraUtils] 자동 선택: {selected_name} (ID: {selected_id})")
                return selected_id
            
            selected_id = int(user_input)
            
            # 선택한 카메라가 사용 가능한지 확인
            for idx, name, available in available_cameras:
                if idx == selected_id:
                    if available:
                        print(f"[CameraUtils] 선택된 카메라: {name} (ID: {selected_id})")
                        return selected_id
                    else:
                        print(f"[CameraUtils] 경고: 선택한 카메라 {selected_id}는 OpenCV로 열 수 없습니다.")
                        print(f"[CameraUtils] 이 카메라는 특수 드라이버나 SDK가 필요할 수 있습니다.")
                        confirm = input("그래도 계속하시겠습니까? (y/n): ").strip().lower()
                        if confirm == 'y':
                            print(f"[CameraUtils] 선택된 카메라: {name} (ID: {selected_id})")
                            return selected_id
                        else:
                            print("[CameraUtils] 선택이 취소되었습니다. 다시 선택하세요.")
                            break
            
            print(f"[CameraUtils] 오류: 카메라 {selected_id}를 찾을 수 없습니다. 다시 선택하세요.")
            
        except ValueError:
            print("[CameraUtils] 오류: 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n[CameraUtils] 사용자에 의해 취소되었습니다.")
            raise


def select_camera(preferred_id: Optional[int] = None, interactive: bool = False) -> int:
    """
    사용 가능한 카메라를 선택합니다.
    
    Args:
        preferred_id: 선호하는 카메라 ID (None이면 첫 번째 사용 가능한 카메라)
        interactive: True이면 대화형 선택 모드
        
    Returns:
        선택된 카메라 ID
    """
    if interactive:
        return select_camera_interactive(preferred_id)
    
    available_cameras = list_available_cameras()
    
    if len(available_cameras) == 0:
        raise RuntimeError("사용 가능한 카메라가 없습니다.")
    
    # 선호하는 카메라가 사용 가능한지 확인
    if preferred_id is not None:
        for idx, name, available in available_cameras:
            if idx == preferred_id and available:
                print(f"[CameraUtils] 선호 카메라 {preferred_id} 선택: {name}")
                return preferred_id
    
    # 첫 번째 사용 가능한 카메라 선택
    selected_id, selected_name, _ = available_cameras[0]
    print(f"[CameraUtils] 카메라 자동 선택: {selected_name} (ID: {selected_id})")
    
    return selected_id


def test_camera(device_id: int) -> bool:
    """
    카메라가 정상적으로 작동하는지 테스트합니다.
    
    Args:
        device_id: 테스트할 카메라 ID
        
    Returns:
        카메라가 정상 작동하면 True
    """
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        return False
    
    # 프레임 읽기 테스트
    ret, frame = cap.read()
    cap.release()
    
    return ret and frame is not None
