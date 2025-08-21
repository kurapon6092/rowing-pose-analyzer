import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import tempfile
import os
import pandas as pd

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
        """腰の角度を描画"""
        text_size = cv2.getTextSize(f"Hip Angle: {angle:.1f} deg", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.rectangle(image, (position[0] - 10, position[1] - text_size[1] - 15), 
                     (position[0] + text_size[0] + 10, position[1] + 10), (0, 0, 0), -1)
        cv2.rectangle(image, (position[0] - 10, position[1] - text_size[1] - 15), 
                     (position[0] + text_size[0] + 10, position[1] + 10), (255, 255, 255), 2)
        
        cv2.putText(image, f"Hip Angle: {angle:.1f} deg", position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        return image
    
    def draw_eye_gaze(self, image, face_landmarks):
        """目線矢印表示"""
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
                
                face_center_x = width // 2
                horizontal_offset = nose_x - face_center_x
                face_center_y = height // 2
                vertical_offset = nose_y - face_center_y
                
                face_height = chin_y - forehead_y
                if face_height > 0:
                    vertical_ratio = vertical_offset / (face_height * 0.3)
                else:
                    vertical_ratio = 0
                
                horizontal_ratio = horizontal_offset / (width * 0.15)
                
                arrow_length = 80
                
                h_direction = 0 if abs(horizontal_ratio) < 0.5 else (1 if horizontal_ratio > 0 else -1)
                v_direction = 0 if abs(vertical_ratio) < 0.5 else (1 if vertical_ratio > 0 else -1)
                
                if h_direction == 0 and v_direction == 0:
                    gaze_status = "STRAIGHT"
                    status_color = (0, 255, 0)
                else:
                    status_parts = []
                    if v_direction == -1:
                        status_parts.append("UP")
                    elif v_direction == 1:
                        status_parts.append("DOWN")
                    
                    if h_direction == -1:
                        status_parts.append("LEFT")
                    elif h_direction == 1:
                        status_parts.append("RIGHT")
                    
                    gaze_status = " + ".join(status_parts)
                    status_color = (0, 255, 255)
                
                arrow_end_x = center_x + (h_direction * arrow_length)
                arrow_end_y = center_y + (v_direction * arrow_length)
                
                if h_direction == 0 and v_direction == 0:
                    arrow_end_x = center_x + 50
                    arrow_end_y = center_y
                    
            except:
                pass
        
        # 矢印描画
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), (0, 0, 0), 20, tipLength=0.3)
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), (255, 255, 255), 15, tipLength=0.3)
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), status_color, 10, tipLength=0.3)
        
        cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), -1)
        cv2.circle(image, (center_x, center_y), 18, (255, 255, 255), 3)
        
        return image
    
    def detect_forward_pause(self, current_angle):
        """前方位置での停止を検知"""
        self.recent_angles.append(current_angle)
        if len(self.recent_angles) > self.pause_detection_window:
            self.recent_angles.pop(0)
        
        if len(self.recent_angles) >= self.pause_detection_window:
            angle_variation = max(self.recent_angles) - min(self.recent_angles)
            avg_angle = sum(self.recent_angles) / len(self.recent_angles)
            
            is_forward_position = avg_angle < (self.min_angle + (self.max_angle - self.min_angle) * 0.25)
            is_stable = angle_variation < self.pause_threshold
            
            if is_forward_position and is_stable:
                if not self.is_paused_forward:
                    self.is_paused_forward = True
                    self.pause_counter = 0
                self.pause_counter += 1
            else:
                self.is_paused_forward = False
                self.pause_counter = 0
    
    def draw_pause_indicator(self, image):
        """前方停止の表示"""
        if self.is_paused_forward and self.pause_counter > 3:
            height, width = image.shape[:2]
            
            indicator_x = width - 200
            indicator_y = 50
            pause_duration = self.pause_counter
            
            if pause_duration < 15:
                bg_color = (0, 255, 0)
                status_text = "GOOD PAUSE"
            elif pause_duration < 30:
                bg_color = (0, 255, 255)
                status_text = "PAUSE OK"
            else:
                bg_color = (0, 0, 255)
                status_text = "TOO LONG"
            
            cv2.rectangle(image, (indicator_x - 10, indicator_y - 30), 
                         (indicator_x + 180, indicator_y + 20), (0, 0, 0), -1)
            cv2.rectangle(image, (indicator_x - 10, indicator_y - 30), 
                         (indicator_x + 180, indicator_y + 20), bg_color, 3)
            
            cv2.putText(image, "FORWARD PAUSE", (indicator_x, indicator_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg_color, 2)
            cv2.putText(image, status_text, (indicator_x, indicator_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_color, 2)
            
            bar_width = min(160, pause_duration * 8)
            cv2.rectangle(image, (indicator_x, indicator_y + 15), 
                         (indicator_x + bar_width, indicator_y + 20), bg_color, -1)
        
        return image
    
    def process_frame(self, frame):
        """フレームを処理"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = pose_results.pose_landmarks.landmark
            height, width = frame.shape[:2]
            
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            head_y = nose.y * height
            
            frame = self.draw_head_horizontal_line(frame, head_y)
            
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            hip_center = [(left_hip.x + right_hip.x) / 2 * width,
                         (left_hip.y + right_hip.y) / 2 * height]
            
            shoulder_center = [(left_shoulder.x + right_shoulder.x) / 2 * width,
                              (left_shoulder.y + right_shoulder.y) / 2 * height]
            
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            knee_center = [(left_knee.x + right_knee.x) / 2 * width,
                          (left_knee.y + right_knee.y) / 2 * height]
            
            hip_angle = self.calculate_angle(shoulder_center, hip_center, knee_center)
            
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
st.title("🏃‍♂️ 動画姿勢解析アプリ")
st.markdown("**骨格トレース、腰角度測定、目線検出 + 動画比較機能**")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
frame_skip = st.sidebar.slider("Frame Skip (Speed Up)", 1, 5, 1, 
                               help="Higher = Faster. 2=Half frames, 3=1/3 frames")

# タブで機能を切り替え
tab1, tab2 = st.tabs(["🔍 単体解析", "🆚 比較解析"])

with tab1:
    st.markdown("### 📹 単体動画解析")
    
    uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=['mp4', 'avi', 'mov'], key="single_video")
    
    if uploaded_file is not None:
        if st.button("解析開始", key="single_start"):
            # 解析処理
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            analyzer = VideoAnalyzer()
            cap = cv2.VideoCapture(tfile.name)
            
            # レイアウト
            col1, col2 = st.columns([3, 1])
            
            with col1:
                stframe = st.empty()
                st.markdown("#### 🎯 前方停止検知")
                pause_display = st.empty()
            
            with col2:
                st.markdown("### 📈 リアルタイム情報")
                realtime_display = st.empty()
            
            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            # フレーム処理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame += 1
                if current_frame % frame_skip != 0:
                    continue
                
                processed_frame = analyzer.process_frame(frame)
                stframe.image(processed_frame, channels="BGR")
                
                if analyzer.hip_angles:
                    current_angle = analyzer.hip_angles[-1]
                    realtime_display.markdown(f"""
                    <div style='background-color: #1f4e79; padding: 15px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>Current: {current_angle:.1f}°</h2>
                        <h4 style='color: #FFD700; margin: 0;'>Max: {analyzer.max_angle:.1f}°</h4>
                        <h4 style='color: #00FF00; margin: 0;'>Min: {analyzer.min_angle:.1f}°</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 前方停止表示
                    if analyzer.is_paused_forward and analyzer.pause_counter > 2:
                        if analyzer.pause_counter < 10:
                            color = "#27AE60"
                            status = "GOOD PAUSE"
                        elif analyzer.pause_counter < 20:
                            color = "#F39C12"
                            status = "PAUSE OK"
                        else:
                            color = "#E74C3C"
                            status = "TOO LONG"
                        
                        pause_display.markdown(f"""
                        <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                            <h1 style='color: white; margin: 0; font-size: 36px;'>🎯 FORWARD PAUSE</h1>
                            <h2 style='color: white; margin: 0;'>{status}</h2>
                            <h3 style='color: white; margin: 0;'>Duration: {analyzer.pause_counter} frames</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        pause_display.markdown("""
                        <div style='background-color: #34495E; padding: 15px; border-radius: 10px; text-align: center;'>
                            <h3 style='color: #BDC3C7; margin: 0;'>🔍 前方停止を監視中...</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                progress_bar.progress(current_frame / frame_count)
            
            cap.release()
            os.unlink(tfile.name)
            
            # 結果表示
            if analyzer.hip_angles:
                st.success("🎉 解析完了！")
                avg_angle = np.mean(analyzer.hip_angles)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最大角度", f"{analyzer.max_angle:.1f}°")
                with col2:
                    st.metric("最小角度", f"{analyzer.min_angle:.1f}°")
                with col3:
                    st.metric("平均角度", f"{avg_angle:.1f}°")
                
                # グラフ
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(analyzer.hip_angles, color='blue', linewidth=2)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Hip Angle (degrees)')
                ax.set_title('Hip Angle Analysis')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

with tab2:
    st.markdown("### 🆚 2つの動画比較解析")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("#### 🎥 動画1（基準）")
        uploaded_file1 = st.file_uploader("最初の動画ファイル", type=['mp4', 'avi', 'mov'], key="video1")
    
    with col_upload2:
        st.markdown("#### 🎥 動画2（比較対象）")
        uploaded_file2 = st.file_uploader("比較する動画ファイル", type=['mp4', 'avi', 'mov'], key="video2")
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.success("✅ 2つの動画がアップロードされました")
        
        if st.button("🔍 比較解析開始", key="comparison_start"):
            # 2つのAnalyzerを初期化
            analyzer1 = VideoAnalyzer()
            analyzer2 = VideoAnalyzer()
            
            # 一時ファイル保存
            tfile1 = tempfile.NamedTemporaryFile(delete=False)
            tfile1.write(uploaded_file1.read())
            tfile2 = tempfile.NamedTemporaryFile(delete=False)
            tfile2.write(uploaded_file2.read())
            
            # 動画読み込み
            cap1 = cv2.VideoCapture(tfile1.name)
            cap2 = cv2.VideoCapture(tfile2.name)
            
            # 比較レイアウト
            col_video1, col_video2 = st.columns(2)
            
            with col_video1:
                st.markdown("#### 🎥 動画1")
                stframe1 = st.empty()
                info1 = st.empty()
            
            with col_video2:
                st.markdown("#### 🎥 動画2")
                stframe2 = st.empty()
                info2 = st.empty()
            
            # リアルタイム比較
            comparison_display = st.empty()
            progress_bar = st.progress(0)
            
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = max(frame_count1, frame_count2)
            
            current_frame = 0
            
            # 同期処理
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 and not ret2:
                    break
                
                current_frame += 1
                if current_frame % frame_skip != 0:
                    continue
                
                # フレーム処理
                if ret1:
                    processed_frame1 = analyzer1.process_frame(frame1)
                    stframe1.image(processed_frame1, channels="BGR")
                
                if ret2:
                    processed_frame2 = analyzer2.process_frame(frame2)
                    stframe2.image(processed_frame2, channels="BGR")
                
                # 角度表示
                if analyzer1.hip_angles and analyzer2.hip_angles:
                    angle1 = analyzer1.hip_angles[-1]
                    angle2 = analyzer2.hip_angles[-1]
                    
                    info1.markdown(f"**Current: {angle1:.1f}°** | Max: {analyzer1.max_angle:.1f}°")
                    info2.markdown(f"**Current: {angle2:.1f}°** | Max: {analyzer2.max_angle:.1f}°")
                    
                    # 差分表示
                    angle_diff = angle1 - angle2
                    diff_status = "🟢 類似" if abs(angle_diff) < 5 else "🔴 差あり"
                    
                    comparison_display.markdown(f"""
                    <div style='background-color: #2C3E50; padding: 20px; border-radius: 15px; text-align: center;'>
                        <h1 style='color: white; margin: 0; font-size: 48px;'>角度差: {angle_diff:+.1f}°</h1>
                        <h2 style='color: white; margin: 0;'>{diff_status}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                progress_bar.progress(current_frame / max_frames)
            
            # 解析完了
            cap1.release()
            cap2.release()
            os.unlink(tfile1.name)
            os.unlink(tfile2.name)
            
            if analyzer1.hip_angles and analyzer2.hip_angles:
                st.success("🎉 比較解析完了！")
                
                # 比較結果
                avg1 = np.mean(analyzer1.hip_angles)
                avg2 = np.mean(analyzer2.hip_angles)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("最大角度差", f"{analyzer1.max_angle - analyzer2.max_angle:+.1f}°")
                with col2:
                    st.metric("平均角度差", f"{avg1 - avg2:+.1f}°")
                with col3:
                    better = "動画1" if np.std(analyzer1.hip_angles) < np.std(analyzer2.hip_angles) else "動画2"
                    st.metric("より安定", better)
                
                # 比較グラフ
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.plot(analyzer1.hip_angles, color='blue', linewidth=2, label='動画1', alpha=0.8)
                ax.plot(analyzer2.hip_angles, color='red', linewidth=2, label='動画2', alpha=0.8)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Hip Angle (degrees)')
                ax.set_title('Hip Angle Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    elif uploaded_file1 is not None or uploaded_file2 is not None:
        st.info("📹 もう1つの動画もアップロードして比較解析を開始してください")
    else:
        st.info("2つの動画ファイルをアップロードして比較解析を始めてください。")
        st.markdown("""
        ### 🆚 比較解析の特徴:
        - **同時解析**: 2つの動画を並行処理
        - **リアルタイム比較**: 角度差分をリアルタイム表示
        - **統計比較**: 詳細な数値比較
        - **視覚的比較**: 重ね合わせグラフ
        """)
