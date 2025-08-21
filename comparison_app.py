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
        
        # 前方停止検知用（判定を厳しく）
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
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[362]
                
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
                
                if abs(horizontal_ratio) < 0.5:
                    h_direction = 0
                elif horizontal_ratio > 0:
                    h_direction = 1
                else:
                    h_direction = -1
                
                if abs(vertical_ratio) < 0.5:
                    v_direction = 0
                elif vertical_ratio > 0:
                    v_direction = 1
                else:
                    v_direction = -1
                
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

def analyze_video_for_comparison(uploaded_file, video_name, frame_skip):
    """比較用の動画解析"""
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # VideoAnalyzerを初期化
    analyzer = VideoAnalyzer()
    
    # 動画を読み込み
    cap = cv2.VideoCapture(tfile.name)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    processed_frame_count = 0
    
    # 最後のフレームのみ表示用
    last_processed_frame = None
    
    # フレーム処理ループ
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame += 1
        
        # フレームスキップ
        if current_frame % frame_skip != 0:
            continue
        
        # フレームを処理
        processed_frame = analyzer.process_frame(frame)
        processed_frame_count += 1
        last_processed_frame = processed_frame.copy()
        
        # 進捗更新
        progress_bar.progress(current_frame / frame_count)
        status_text.text(f"{video_name} 処理中: {current_frame}/{frame_count} フレーム")
    
    cap.release()
    os.unlink(tfile.name)
    
    # 最後のフレームを表示
    if last_processed_frame is not None:
        st.image(last_processed_frame, channels="BGR", caption=f"{video_name} 最終フレーム")
    
    progress_bar.empty()
    status_text.empty()
    
    return analyzer

def display_comparison_results(analyzer1, analyzer2):
    """比較結果を表示"""
    st.markdown("## 🆚 比較結果")
    
    # 基本統計比較
    col1, col2, col3, col4 = st.columns(4)
    
    avg1 = np.mean(analyzer1.hip_angles) if analyzer1.hip_angles else 0
    avg2 = np.mean(analyzer2.hip_angles) if analyzer2.hip_angles else 0
    std1 = np.std(analyzer1.hip_angles) if analyzer1.hip_angles else 0
    std2 = np.std(analyzer2.hip_angles) if analyzer2.hip_angles else 0
    
    with col1:
        diff_max = analyzer1.max_angle - analyzer2.max_angle
        color = "#FF6B6B" if diff_max > 0 else "#4ECDC4"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>最大角度差</h4>
            <h2 style='color: white; margin: 0;'>{diff_max:+.1f}°</h2>
            <h5 style='color: white; margin: 0;'>{analyzer1.max_angle:.1f}° vs {analyzer2.max_angle:.1f}°</h5>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        diff_min = analyzer1.min_angle - analyzer2.min_angle
        color = "#4ECDC4" if diff_min < 0 else "#FF6B6B"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>最小角度差</h4>
            <h2 style='color: white; margin: 0;'>{diff_min:+.1f}°</h2>
            <h5 style='color: white; margin: 0;'>{analyzer1.min_angle:.1f}° vs {analyzer2.min_angle:.1f}°</h5>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        diff_avg = avg1 - avg2
        color = "#45B7D1" if abs(diff_avg) < 5 else "#F39C12"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>平均角度差</h4>
            <h2 style='color: white; margin: 0;'>{diff_avg:+.1f}°</h2>
            <h5 style='color: white; margin: 0;'>{avg1:.1f}° vs {avg2:.1f}°</h5>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        diff_std = std1 - std2
        color = "#27AE60" if std1 < std2 else "#E74C3C"
        better_video = "動画1" if std1 < std2 else "動画2"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>安定性</h4>
            <h2 style='color: white; margin: 0;'>{better_video}</h2>
            <h5 style='color: white; margin: 0;'>{std1:.1f}° vs {std2:.1f}°</h5>
        </div>
        """, unsafe_allow_html=True)
    
    # 比較グラフ
    st.markdown("### 📈 角度変化比較グラフ")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if analyzer1.hip_angles:
        ax.plot(analyzer1.hip_angles, color='blue', linewidth=2, label='動画1', alpha=0.8)
        ax.fill_between(range(len(analyzer1.hip_angles)), analyzer1.hip_angles, 
                       alpha=0.3, color='blue')
    
    if analyzer2.hip_angles:
        # 長さを調整（短い方に合わせる）
        min_length = min(len(analyzer1.hip_angles), len(analyzer2.hip_angles))
        if len(analyzer2.hip_angles) > min_length:
            angles2 = analyzer2.hip_angles[:min_length]
        else:
            angles2 = analyzer2.hip_angles
        
        ax.plot(angles2, color='red', linewidth=2, label='動画2', alpha=0.8)
        ax.fill_between(range(len(angles2)), angles2, alpha=0.3, color='red')
    
    ax.set_xlabel('Frame Number', fontsize=14)
    ax.set_ylabel('Hip Angle (degrees)', fontsize=14)
    ax.set_title('Hip Angle Comparison Analysis', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    st.pyplot(fig)
    
    # 詳細比較テーブル
    st.markdown("### 📋 詳細比較")
    
    comparison_data = {
        "指標": ["最大角度", "最小角度", "平均角度", "標準偏差", "角度範囲", "フレーム数"],
        "動画1": [
            f"{analyzer1.max_angle:.1f}°",
            f"{analyzer1.min_angle:.1f}°", 
            f"{avg1:.1f}°",
            f"{std1:.1f}°",
            f"{analyzer1.max_angle - analyzer1.min_angle:.1f}°",
            f"{len(analyzer1.hip_angles)}"
        ],
        "動画2": [
            f"{analyzer2.max_angle:.1f}°",
            f"{analyzer2.min_angle:.1f}°",
            f"{avg2:.1f}°", 
            f"{std2:.1f}°",
            f"{analyzer2.max_angle - analyzer2.min_angle:.1f}°",
            f"{len(analyzer2.hip_angles)}"
        ],
        "差分": [
            f"{analyzer1.max_angle - analyzer2.max_angle:+.1f}°",
            f"{analyzer1.min_angle - analyzer2.min_angle:+.1f}°",
            f"{avg1 - avg2:+.1f}°",
            f"{std1 - std2:+.1f}°",
            f"{(analyzer1.max_angle - analyzer1.min_angle) - (analyzer2.max_angle - analyzer2.min_angle):+.1f}°",
            f"{len(analyzer1.hip_angles) - len(analyzer2.hip_angles):+d}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

# メインUI
st.title("🆚 動画姿勢比較解析アプリ")
st.markdown("**2つの動画を同時に解析して比較します**")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
frame_skip = st.sidebar.slider("Frame Skip (Speed Up)", 1, 5, 1, 
                               help="Higher = Faster. 2=Half frames, 3=1/3 frames")

# 動画アップロード
st.markdown("## 📹 動画アップロード")
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("### 🎥 動画1")
    uploaded_file1 = st.file_uploader("最初の動画ファイル", type=['mp4', 'avi', 'mov'], key="video1")

with col_upload2:
    st.markdown("### 🎥 動画2")
    uploaded_file2 = st.file_uploader("比較する動画ファイル", type=['mp4', 'avi', 'mov'], key="video2")

# 比較解析の実行
if uploaded_file1 is not None and uploaded_file2 is not None:
    st.success("✅ 2つの動画がアップロードされました")
    
    if st.button("🔍 比較解析開始", key="compare_start"):
        st.markdown("## 📊 リアルタイム比較解析")
        
        # 2列レイアウトで並行表示
        col1, col2 = st.columns(2)
        
        with st.spinner("2つの動画を解析中..."):
            # 動画1の解析
            with col1:
                st.markdown("#### 🎥 動画1 解析中...")
                analyzer1 = analyze_video_for_comparison(uploaded_file1, "動画1", frame_skip)
            
            # 動画2の解析
            with col2:
                st.markdown("#### 🎥 動画2 解析中...")
                analyzer2 = analyze_video_for_comparison(uploaded_file2, "動画2", frame_skip)
        
        # 比較結果表示
        if analyzer1.hip_angles and analyzer2.hip_angles:
            display_comparison_results(analyzer1, analyzer2)

elif uploaded_file1 is not None:
    st.info("📹 動画1のみアップロード済み - もう1つの動画もアップロードして比較解析を開始してください")

elif uploaded_file2 is not None:
    st.info("📹 動画2のみアップロード済み - もう1つの動画もアップロードして比較解析を開始してください")

else:
    st.info("2つの動画ファイルをアップロードして比較解析を始めてください。")
    st.markdown("""
    ### 🆚 比較解析の特徴:
    - **同時解析**: 2つの動画を並行して処理
    - **リアルタイム比較**: 角度差分をリアルタイム表示
    - **詳細統計**: 最大・最小・平均角度の比較
    - **安定性評価**: どちらがより安定したフォームか判定
    - **視覚的比較**: 重ね合わせグラフで変化を比較
    """)
