"""3D 메쉬 시각화 모듈"""
import threading
import queue
import time
import numpy as np
import open3d as o3d
import cv2
from typing import Dict, Any, Optional


class Visualizer(threading.Thread):
    """Open3D를 사용하여 3D 메쉬를 시각화하는 스레드"""
    
    def __init__(
        self,
        mesh_queue: queue.Queue,
        frame_queue: Optional[queue.Queue] = None,
        window_width: int = 800,
        window_height: int = 600,
        show_keypoints: bool = True,
        show_skeleton: bool = True,
        show_camera: bool = True
    ):
        """
        Args:
            mesh_queue: 메쉬 데이터 큐 (timestamp, mesh_data)
            frame_queue: 카메라 프레임 큐 (timestamp, frame_rgb, frame_bgr) - 선택사항
            window_width: 윈도우 너비
            window_height: 윈도우 높이
            show_keypoints: 키포인트 표시 여부
            show_skeleton: 골격선 표시 여부
            show_camera: 카메라 영상 표시 여부
        """
        super().__init__(daemon=True)
        self.mesh_queue = mesh_queue
        self.frame_queue = frame_queue
        self.window_width = window_width
        self.window_height = window_height
        self.show_keypoints = show_keypoints
        self.show_skeleton = show_skeleton
        self.show_camera = show_camera
        
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.mesh_geom: Optional[o3d.geometry.TriangleMesh] = None
        self.keypoints_geom: Optional[o3d.geometry.PointCloud] = None
        self.skeleton_lines: Optional[o3d.geometry.LineSet] = None
        
        self.running = False
        self.last_timestamp = 0.0
        self.last_frame_timestamp = 0.0
        self.current_frame: Optional[np.ndarray] = None
        
    def initialize(self):
        """Open3D 시각화 윈도우 초기화"""
        # Open3D 윈도우 생성
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Hospi - 실시간 3D 인체 메쉬",
            width=self.window_width,
            height=self.window_height,
            visible=True,  # 윈도우를 즉시 표시
            left=50,  # 왼쪽에서 50픽셀 떨어진 위치
            top=50   # 위에서 50픽셀 떨어진 위치
        )
        
        # 렌더링 옵션 설정
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # 어두운 배경
        render_option.mesh_show_wireframe = False  # 와이어프레임 비활성화
        
        # 뷰 포인트 설정 (카메라 위치)
        view_control = self.vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_zoom(0.8)
        
        # 초기 렌더링 (빈 화면이라도 윈도우 표시)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # 카메라 영상 윈도우 생성 (OpenCV)
        if self.show_camera:
            cv2.namedWindow("Hospi - 카메라 영상", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Hospi - 카메라 영상", self.window_width, self.window_height)
            # Open3D 윈도우 옆에 배치 (Open3D 윈도우 너비 + 간격)
            cv2.moveWindow("Hospi - 카메라 영상", 50 + self.window_width + 20, 50)
        
        print("[Visualizer] Open3D 시각화 윈도우 초기화 완료")
        if self.show_camera:
            print("[Visualizer] 카메라 영상 윈도우 초기화 완료")
        print("[Visualizer] 3D 메쉬가 생성되면 윈도우에 표시됩니다.")
        
    def update_mesh(self, vertices: np.ndarray, faces: np.ndarray):
        """메쉬 업데이트"""
        if vertices is None or len(vertices) == 0:
            return
        
        # NumPy 배열을 Open3D 형식으로 변환
        vertices_o3d = o3d.utility.Vector3dVector(vertices)
        faces_o3d = o3d.utility.Vector3iVector(faces)
        
        if self.mesh_geom is None:
            # 첫 메쉬 생성
            self.mesh_geom = o3d.geometry.TriangleMesh(vertices_o3d, faces_o3d)
            self.mesh_geom.compute_vertex_normals()
            self.mesh_geom.paint_uniform_color([0.7, 0.7, 0.9])  # 연한 파란색
            self.vis.add_geometry(self.mesh_geom, reset_bounding_box=True)
            
            # 메쉬 중심을 원점으로 맞추고 뷰 포인트 조정
            mesh_center = self.mesh_geom.get_center()
            view_control = self.vis.get_view_control()
            view_control.set_lookat(mesh_center)
            
            # 메쉬 크기에 맞게 줌 조정
            bbox = self.mesh_geom.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            max_size = max(bbox_size)
            if max_size > 0:
                # 메쉬가 화면에 잘 보이도록 줌 조정
                zoom = 1.5 / max_size if max_size > 1.0 else 0.8
                view_control.set_zoom(zoom)
        else:
            # 기존 메쉬 업데이트
            self.mesh_geom.vertices = vertices_o3d
            self.mesh_geom.triangles = faces_o3d
            self.mesh_geom.compute_vertex_normals()
            self.vis.update_geometry(self.mesh_geom)
    
    def update_keypoints(self, keypoints: np.ndarray):
        """키포인트 업데이트"""
        if not self.show_keypoints or keypoints is None:
            return
        
        # 키포인트를 PointCloud로 변환
        keypoints_o3d = o3d.utility.Vector3dVector(keypoints)
        
        if self.keypoints_geom is None:
            self.keypoints_geom = o3d.geometry.PointCloud(keypoints_o3d)
            self.keypoints_geom.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색
            # 점 크기 설정
            self.vis.add_geometry(self.keypoints_geom, reset_bounding_box=False)
        else:
            self.keypoints_geom.points = keypoints_o3d
            self.vis.update_geometry(self.keypoints_geom)
    
    def update_skeleton(self, joints: Dict[str, np.ndarray]):
        """골격선 업데이트 (관절 간 연결)"""
        if not self.show_skeleton or joints is None:
            return
        
        # 간단한 골격 연결 예시 (모델에 따라 다름)
        # TODO: 실제 관절 구조에 맞게 수정 필요
        # 예: 왼쪽 어깨-팔꿈치-손목 연결
        
    def update_camera_view(self):
        """카메라 영상 업데이트"""
        if not self.show_camera or self.frame_queue is None:
            return
        
        try:
            # 최신 프레임 가져오기 (타임아웃 없이)
            while True:
                try:
                    timestamp, frame_rgb, frame_bgr = self.frame_queue.get_nowait()
                    # 최신 프레임만 유지
                    if timestamp >= self.last_frame_timestamp:
                        self.current_frame = frame_bgr.copy()
                        self.last_frame_timestamp = timestamp
                except queue.Empty:
                    break
            
            # 카메라 영상 표시
            if self.current_frame is not None:
                # 텍스트 오버레이 추가
                display_frame = self.current_frame.copy()
                cv2.putText(
                    display_frame,
                    "Camera Feed",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("Hospi - 카메라 영상", display_frame)
                
        except Exception as e:
            # 카메라 영상 표시 오류는 조용히 무시 (프로세스 중단 방지)
            pass
    
    def run(self):
        """스레드 실행 (시각화 루프)"""
        self.initialize()
        self.running = True
        
        mesh_count = 0
        last_log_time = time.time()
        
        while self.running:
            try:
                # 카메라 영상 업데이트 (항상 시도)
                self.update_camera_view()
                
                # OpenCV 이벤트 처리 (카메라 윈도우)
                if self.show_camera:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[Visualizer] 'q' 키 입력으로 종료합니다.")
                        self.running = False
                        break
                
                # 메쉬 큐에서 데이터 가져오기
                try:
                    timestamp, mesh_data = self.mesh_queue.get(timeout=0.05)
                except queue.Empty:
                    # 메쉬가 없어도 카메라 영상은 계속 표시
                    if self.vis is not None:
                        self.vis.poll_events()
                        self.vis.update_renderer()
                    continue
                
                # 오래된 프레임 스킵 (타임스탬프 체크)
                if timestamp < self.last_timestamp:
                    continue
                
                self.last_timestamp = timestamp
                mesh_count += 1
                
                # 사람 감지 여부 확인
                person_detected = mesh_data.get("person_detected", True)
                
                # 메쉬 업데이트
                vertices = mesh_data.get("vertices")
                faces = mesh_data.get("faces")
                
                if vertices is not None and faces is not None and len(vertices) > 0:
                    # 사람이 감지되고 메쉬가 있는 경우
                    self.update_mesh(vertices, faces)
                    if mesh_count == 1:
                        print(f"[Visualizer] 첫 메쉬 표시됨: {len(vertices)} 정점, {len(faces)} 면")
                elif not person_detected:
                    # 사람이 감지되지 않은 경우 (빈 메쉬)
                    # 기존 메쉬는 유지하되, 새로 업데이트하지 않음
                    if mesh_count == 1:
                        print("[Visualizer] 사람이 감지되지 않았습니다. 카메라 앞으로 이동해주세요.")
                else:
                    # 메쉬 데이터가 없는 경우
                    if mesh_count <= 3:  # 처음 몇 번만 경고
                        print(f"[Visualizer] 경고: vertices 또는 faces가 None입니다")
                
                # 키포인트 업데이트
                keypoints = mesh_data.get("keypoints")
                if keypoints is not None:
                    if isinstance(keypoints, dict):
                        # 딕셔너리 형태면 numpy 배열로 변환
                        keypoints_array = np.array(list(keypoints.values()))
                    else:
                        keypoints_array = keypoints
                    self.update_keypoints(keypoints_array)
                
                # 윈도우 이벤트 처리
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # 주기적으로 메쉬 수신 로그
                current_time = time.time()
                if current_time - last_log_time > 5.0:
                    print(f"[Visualizer] 메쉬 업데이트: {mesh_count}개 수신")
                    last_log_time = current_time
                
            except Exception as e:
                print(f"[Visualizer] 시각화 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def stop(self):
        """시각화 중지"""
        self.running = False
        if self.vis is not None:
            self.vis.destroy_window()
        if self.show_camera:
            cv2.destroyAllWindows()
        print("[Visualizer] 시각화 중지")
