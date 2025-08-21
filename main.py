import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import tempfile
import os

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class VideoAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hip_angles = []
        self.max_angle = 0
        self.min_angle = float('inf')
        self.fixed_head_y = None
        
        # 前方停止検知用
        self.recent_angles = []
        self.pause_detection_window = 8
        self.pause_threshold = 1.0
        self.is_paused_forward = False
        self.pause_counter = 0
    
    def calculate_angle(self, point1, point2, point3):
        """3点間の角度を計算"""
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def draw_head_horizontal_line(self, image, head_y=None):
        """固定の頭の高さに水平線を描画"""
        height, width = image.shape[:2]
        if self.fixed_head_y is None and head_y is not None:
            self.fixed_head_y = head_y
        if self.fixed_head_y is not None:
            cv2.line(image, (0, int(self.fixed_head_y)), (width, int(self.fixed_head_y)), (0, 255, 0), 3)
            cv2.putText(image, f"Head Reference: {int(self.fixed_head_y)}px", (10, int(self.fixed_head_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image
    
    def draw_hip_angle(self, image, angle, position):
        """腰の角度を描画（大きなフォント）"""
        text_size = cv2.getTextSize(f"Hip Angle: {angle:.1f} deg", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.rectangle(image, (position[0] - 10, position[1] - text_size[1] - 15),
                     (position[0] + text_size[0] + 10, position[1] + 10), (0, 0, 0), -1)
        cv2.rectangle(image, (position[0] - 10, position[1] - text_size[1] - 15),
                     (position[0] + text_size[0] + 10, position[1] + 10), (255, 255, 255), 2)
        cv2.putText(image, f"Hip Angle: {angle:.1f} deg", position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        return image
    
    def draw_eye_gaze(self, image, face_landmarks):
        """絶対に見える目線矢印表示"""
        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 3
        arrow_end_x = center_x + 80
        arrow_end_y = center_y
        gaze_status = "STRAIGHT"
        status_color = (0, 255, 0)
        
        if face_landmarks:
            try:
                nose_tip = face_landmarks.landmark[1]
                forehead = face_landmarks.landmark[10]
                chin = face_landmarks.landmark[175]
                
                nose_x = int(nose_tip.x * width)
                nose_y = int(nose_tip.y * height)
                forehead_y = int(forehead.y * height)
                chin_y = int(chin.y * height)
                
                horizontal_offset = nose_x - (width // 2)
                face_height = chin_y - forehead_y
                vertical_offset = nose_y - (forehead_y + face_height / 2)
                
                vertical_ratio = vertical_offset / (face_height * 0.3) if face_height > 0 else 0
                horizontal_ratio = horizontal_offset / (width * 0.15)
                
                h_direction = 0 if abs(horizontal_ratio) < 0.5 else (1 if horizontal_ratio > 0 else -1)
                v_direction = 0 if abs(vertical_ratio) < 0.5 else (1 if vertical_ratio > 0 else -1)
                
                if h_direction == 0 and v_direction == 0:
                    gaze_status = "STRAIGHT"
                    status_color = (0, 255, 0)
                else:
                    status_parts = []
                    if v_direction == -1: status_parts.append("UP")
                    elif v_direction == 1: status_parts.append("DOWN")
                    if h_direction == -1: status_parts.append("LEFT")
                    elif h_direction == 1: status_parts.append("RIGHT")
                    gaze_status = " + ".join(status_parts)
                    status_color = (0, 255, 255)
                
                arrow_length = 80
                arrow_end_x = center_x + (h_direction * arrow_length)
                arrow_end_y = center_y + (v_direction * arrow_length)
                
                if h_direction == 0 and v_direction == 0:
                    arrow_end_x = center_x + 50
                    arrow_end_y = center_y
            except:
                pass
        
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), (0, 0, 0), 20, tipLength=0.3)
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), (255, 255, 255), 15, tipLength=0.3)
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), status_color, 10, tipLength=0.3)
        cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), -1)
        cv2.circle(image, (center_x, center_y), 18, (255, 255, 255), 3)
        
        return image
    
    def detect_forward_pause(self, hip_angle):
        """前方停止検知（厳しい判定）"""
        self.recent_angles.append(hip_angle)
        if len(self.recent_angles) > self.pause_detection_window:
            self.recent_angles.pop(0)
        
        if len(self.recent_angles) >= self.pause_detection_window:
            angle_std = np.std(self.recent_angles)
            avg_angle = np.mean(self.recent_angles)
            
            is_forward_position = avg_angle > np.mean(self.hip_angles) + 0.25 * np.std(self.hip_angles) if len(self.hip_angles) > 10 else True
            is_stable = angle_std < self.pause_threshold
            
            if is_forward_position and is_stable:
                if not self.is_paused_forward:
                    self.is_paused_forward = True
                    self.pause_counter = 1
                else:
                    self.pause_counter += 1
            else:
                self.is_paused_forward = False
                self.pause_counter = 0
    
    def draw_pause_indicator(self, image):
        """前方停止インジケーターを描画"""
        if self.is_paused_forward and self.pause_counter > 2:
            height, width = image.shape[:2]
            
            if self.pause_counter < 10:
                color = (0, 255, 0)
                status = "GOOD PAUSE"
            elif self.pause_counter < 20:
                color = (0, 255, 255)
                status = "PAUSE OK"
            else:
                color = (0, 0, 255)
                status = "TOO LONG"
            
            cv2.rectangle(image, (width - 300, 10), (width - 10, 100), (0, 0, 0), -1)
            cv2.rectangle(image, (width - 300, 10), (width - 10, 100), color, 3)
            cv2.putText(image, "FORWARD PAUSE", (width - 290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(image, status, (width - 290, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, f"{self.pause_counter}f", (width - 290, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def process_frame(self, frame):
        """フレームを処理"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = pose_results.pose_landmarks.landmark
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            head_y = int(nose.y * frame.shape[0])
            
            frame = self.draw_head_horizontal_line(frame, head_y)
            
            hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            
            height, width = frame.shape[:2]
            hip_center = [int(left_hip[0] * width), int(left_hip[1] * height)]
            
            self.hip_angles.append(hip_angle)
            if hip_angle > self.max_angle:
                self.max_angle = hip_angle
            if hip_angle < self.min_angle:
                self.min_angle = hip_angle
            
            self.detect_forward_pause(hip_angle)
            
            frame = self.draw_hip_angle(frame, hip_angle, (int(hip_center[0]), int(hip_center[1])))
            frame = self.draw_pause_indicator(frame)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                frame = self.draw_eye_gaze(frame, face_landmarks)
        
        return frame

# Streamlit UI
st.title("🏃‍♂️ ローイング姿勢解析アプリ")
st.markdown("**骨格トレース、腰角度測定、目線検出**")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
frame_skip = st.sidebar.slider("フレームスキップ (高速化)", 1, 5, 1, 
                               help="数値が大きいほど高速処理",
                               key="main_frame_skip")

# 単体解析
st.markdown("### 📹 動画解析")

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=['mp4', 'avi', 'mov'], key="single_upload")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    analyzer = VideoAnalyzer()
    cap = cv2.VideoCapture(tfile.name)
    
    # UI要素を事前に定義
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stframe = st.empty()
        st.markdown("#### 🎯 前方停止検知")
        pause_display = st.empty()
    
    with col2:
        st.markdown("### 📈 リアルタイムデータ")
        current_display = st.empty()
        stats_display = st.empty()
    
    progress_bar = st.progress(0)
    
    if st.button("解析開始", key="single_analyze"):
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            if current_frame % frame_skip != 0:
                continue
            
            processed_frame = analyzer.process_frame(frame)
            if processed_frame is not None:
                stframe.image(processed_frame, channels="BGR")
            
            if analyzer.hip_angles:
                current_angle = analyzer.hip_angles[-1]
                
                current_display.markdown(f"""
                <div style='background-color: #1f4e79; padding: 15px; border-radius: 10px;'>
                    <h3 style='color: white; text-align: center; margin: 0;'>現在の角度</h3>
                    <h1 style='color: #00ff00; text-align: center; font-size: 36px; margin: 0;'>{current_angle:.1f}°</h1>
                </div>
                """, unsafe_allow_html=True)
                
                stats_display.markdown(f"""
                <div style='background-color: #2E2E2E; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                    <h5 style='color: #FFD700; margin: 0;'>最大: {analyzer.max_angle:.1f}°</h5>
                    <h5 style='color: #98FB98; margin: 0;'>最小: {analyzer.min_angle:.1f}°</h5>
                    <h6 style='color: #87CEEB; margin: 0;'>フレーム: {current_frame}/{frame_count}</h6>
                </div>
                """, unsafe_allow_html=True)
                
                # Forward pause display
                if analyzer.is_paused_forward and analyzer.pause_counter > 2:
                    if analyzer.pause_counter < 10:
                        status_color = "#27AE60"
                        status_text = "🟢 GOOD PAUSE"
                    elif analyzer.pause_counter < 20:
                        status_color = "#F39C12"
                        status_text = "🟡 PAUSE OK"
                    else:
                        status_color = "#E74C3C"
                        status_text = "🔴 TOO LONG"
                    
                    pause_display.markdown(f"""
                    <div style='background-color: {status_color}; padding: 20px; border-radius: 15px; text-align: center;'>
                        <h1 style='color: white; margin: 0; font-size: 42px;'>前方停止</h1>
                        <h2 style='color: white; margin: 0;'>{status_text}</h2>
                        <h3 style='color: white; margin: 0;'>継続時間: {analyzer.pause_counter} フレーム</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    pause_display.markdown("""
                    <div style='background-color: #34495E; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h3 style='color: #BDC3C7; margin: 0;'>🔍 前方停止を監視中...</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            progress_bar.progress(current_frame / frame_count)
        
        cap.release()
        os.unlink(tfile.name)
        
        if analyzer.hip_angles:
            st.success("🎉 解析完了！")
            
            avg_angle = np.mean(analyzer.hip_angles)
            std_angle = np.std(analyzer.hip_angles)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("最大角度", f"{analyzer.max_angle:.1f}°")
            with col2:
                st.metric("最小角度", f"{analyzer.min_angle:.1f}°")
            with col3:
                st.metric("平均角度", f"{avg_angle:.1f}°")
            with col4:
                st.metric("標準偏差", f"{std_angle:.1f}°")
            
            # グラフ
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(analyzer.hip_angles, color='blue', linewidth=2, label='腰角度')
            ax.axhline(y=analyzer.max_angle, color='red', linestyle='--', label=f'最大: {analyzer.max_angle:.1f}°')
            ax.axhline(y=analyzer.min_angle, color='green', linestyle='--', label=f'最小: {analyzer.min_angle:.1f}°')
            ax.axhline(y=avg_angle, color='orange', linestyle=':', label=f'平均: {avg_angle:.1f}°')
            ax.set_xlabel('フレーム数')
            ax.set_ylabel('角度 (度)')
            ax.set_title('腰角度の時系列変化')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

else:
    st.info("動画ファイルをアップロードして解析を開始してください。")
    st.markdown("""
    ### 🎯 解析機能:
    - **骨格トレース**: MediaPipeによる姿勢検出
    - **頭の基準線**: 頭の高さに固定された水平線
    - **腰角度測定**: 肩-腰-膝の角度をリアルタイム測定
    - **前方停止検知**: 前方位置での停止を検出
    - **目線検出**: 矢印による目線方向の表示
    - **統計情報**: 最大・最小・平均角度と時系列グラフ
    """)