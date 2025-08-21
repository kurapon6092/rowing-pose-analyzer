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
        # 軽量化されたMediaPipe設定
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=軽量 (1から0に変更)
            enable_segmentation=False,
            min_detection_confidence=0.7,  # 閾値を上げて計算量削減
            min_tracking_confidence=0.7
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # Trueから変更で軽量化
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # メモリ効率化: 最新の値のみ保持
        self.current_angle = 0
        self.max_angle = 0
        self.min_angle = float('inf')
        self.fixed_head_y = None
        self.frame_count = 0
        
        # 前方停止検知用 - メモリ使用量削減
        self.recent_angles = []
        self.pause_detection_window = 5  # 8から5に削減
        self.pause_threshold = 1.5  # 少し緩めて計算負荷軽減
        self.is_paused_forward = False
        self.pause_counter = 0
        
        # 統計用の軽量データ
        self.angle_sum = 0
        self.angle_count = 0
    
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
        """腰の角度を描画（軽量化）"""
        # 軽量化: 背景矩形を簡素化
        cv2.putText(image, f"{angle:.1f}°", position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)  # 黒い縁取り
        cv2.putText(image, f"{angle:.1f}°", position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)  # メインテキスト
        return image
    
    def draw_eye_gaze(self, image, face_landmarks):
        """軽量化された目線矢印表示"""
        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 3
        arrow_end_x = center_x + 50
        arrow_end_y = center_y
        status_color = (0, 255, 0)
        
        if face_landmarks:
            try:
                # 軽量化: 基本的なランドマークのみ使用
                nose_tip = face_landmarks.landmark[1]
                nose_x = int(nose_tip.x * width)
                
                horizontal_offset = nose_x - center_x
                
                # 簡素化された方向判定
                if abs(horizontal_offset) > width * 0.1:
                    h_direction = 1 if horizontal_offset > 0 else -1
                    status_color = (0, 255, 255)
                    arrow_end_x = center_x + (h_direction * 50)
            except:
                pass
        
        # 軽量化: 単一の矢印のみ
        cv2.arrowedLine(image, (center_x, center_y), (arrow_end_x, arrow_end_y), status_color, 8, tipLength=0.3)
        cv2.circle(image, (center_x, center_y), 8, status_color, -1)
        
        return image
    
    def detect_forward_pause(self, hip_angle):
        """軽量化された前方停止検知"""
        self.recent_angles.append(hip_angle)
        if len(self.recent_angles) > self.pause_detection_window:
            self.recent_angles.pop(0)
        
        if len(self.recent_angles) >= self.pause_detection_window:
            # 軽量化: 簡素化された計算
            angle_range = max(self.recent_angles) - min(self.recent_angles)
            avg_angle = sum(self.recent_angles) / len(self.recent_angles)
            
            # 軽量化: 簡単な判定条件
            is_stable = angle_range < self.pause_threshold
            is_forward_position = avg_angle > self.current_angle * 1.1 if self.current_angle > 0 else True
            
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
        """軽量化された前方停止インジケーター"""
        if self.is_paused_forward and self.pause_counter > 2:
            height, width = image.shape[:2]
            
            # 軽量化: 色のみで判定
            if self.pause_counter < 10:
                color = (0, 255, 0)
            elif self.pause_counter < 20:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            # 軽量化: シンプルなサークル表示
            cv2.circle(image, (width - 50, 50), 30, color, -1)
            cv2.putText(image, "P", (width - 58, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return image
    
    def process_frame(self, frame):
        """軽量化されたフレーム処理"""
        # メモリ削減: フレームサイズを縮小
        height, width = frame.shape[:2]
        if width > 640:  # 大きすぎる場合は縮小
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        # 軽量化: 顔検出は5フレームに1回のみ
        face_results = None
        if self.frame_count % 5 == 0:
            face_results = self.face_mesh.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # 軽量化: 骨格線は描画しない（処理負荷軽減）
            # mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
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
            
            # メモリ効率化: 配列を使わず統計のみ保持
            self.current_angle = hip_angle
            self.frame_count += 1
            self.angle_sum += hip_angle
            self.angle_count += 1
            
            if hip_angle > self.max_angle:
                self.max_angle = hip_angle
            if hip_angle < self.min_angle:
                self.min_angle = hip_angle
            
            self.detect_forward_pause(hip_angle)
            
            frame = self.draw_hip_angle(frame, hip_angle, (int(hip_center[0]), int(hip_center[1])))
            frame = self.draw_pause_indicator(frame)
        
        # 軽量化: 目線検出は軽量化版のみ
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                frame = self.draw_eye_gaze(frame, face_landmarks)
        
        return frame
    
    def get_average_angle(self):
        """平均角度を取得"""
        return self.angle_sum / self.angle_count if self.angle_count > 0 else 0

# Streamlit UI
st.title("🏃‍♂️ ローイング姿勢解析アプリ（軽量版）")
st.markdown("**高速処理・軽量化・腰角度測定・目線検出**")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
frame_skip = st.sidebar.slider("フレームスキップ (高速化)", 1, 10, 3, 
                               help="数値が大きいほど高速処理（推奨: 3-5）",
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
            
            if analyzer.current_angle > 0:
                current_angle = analyzer.current_angle
                
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
        
        if analyzer.angle_count > 0:
            st.success("🎉 解析完了！")
            
            avg_angle = analyzer.get_average_angle()
            # 軽量化: 標準偏差の代わりに範囲を使用
            angle_range = analyzer.max_angle - analyzer.min_angle
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("最大角度", f"{analyzer.max_angle:.1f}°")
            with col2:
                st.metric("最小角度", f"{analyzer.min_angle:.1f}°")
            with col3:
                st.metric("平均角度", f"{avg_angle:.1f}°")
            with col4:
                st.metric("角度範囲", f"{angle_range:.1f}°")
            
            # 軽量化: 簡単な統計表示のみ
            st.markdown(f"""
            ### 📊 解析サマリー
            - **処理フレーム数**: {analyzer.frame_count}
            - **最大角度**: {analyzer.max_angle:.1f}°
            - **最小角度**: {analyzer.min_angle:.1f}°
            - **平均角度**: {avg_angle:.1f}°
            - **角度範囲**: {angle_range:.1f}°
            """)

else:
    st.info("動画ファイルをアップロードして解析を開始してください。")
    st.markdown("""
    ### 🎯 軽量化された解析機能:
    - **高速姿勢検出**: MediaPipe軽量モデルによる姿勢検出
    - **頭の基準線**: 頭の高さに固定された水平線
    - **腰角度測定**: 肩-腰-膝の角度をリアルタイム測定
    - **前方停止検知**: 前方位置での停止を検出（軽量版）
    - **目線検出**: 矢印による目線方向の表示（簡易版）
    - **統計情報**: 最大・最小・平均角度の表示
    

    """)
    