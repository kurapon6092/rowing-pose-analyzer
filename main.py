import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

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
        self.fixed_head_y = None  # 固定の頭の高さを保存
        
        # 前方停止検知用（判定を厳しく）
        self.recent_angles = []  # 最近の角度履歴
        self.pause_detection_window = 8   # 停止判定用のフレーム数（短縮）
        self.pause_threshold = 1.0        # 角度変化の閾値（厳しく）
        self.is_paused_forward = False    # 前方での停止状態
        self.pause_counter = 0            # 停止継続カウンター
    
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
        
        # 最初のフレームで頭の位置を設定（固定）
        if self.fixed_head_y is None and head_y is not None:
            self.fixed_head_y = head_y
        
        # 固定の水平線を描画
        if self.fixed_head_y is not None:
            cv2.line(image, (0, int(self.fixed_head_y)), (width, int(self.fixed_head_y)), (0, 255, 0), 3)
            cv2.putText(image, f"Head Reference: {int(self.fixed_head_y)}px", (10, int(self.fixed_head_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def draw_hip_angle(self, image, angle, position):
        """腰の角度を描画（大きなフォント）"""
        # 背景となる矩形を描画（見やすくするため）
        text_size = cv2.getTextSize(f"腰角度: {angle:.1f}°", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        cv2.rectangle(image, (position[0] - 10, position[1] - text_size[1] - 15), 
                     (position[0] + text_size[0] + 10, position[1] + 10), (0, 0, 0), -1)
        cv2.rectangle(image, (position[0] - 10, position[1] - text_size[1] - 15), 
                     (position[0] + text_size[0] + 10, position[1] + 10), (255, 255, 255), 2)
        
        # 大きな文字で角度を表示（英語表記）
        cv2.putText(image, f"Hip Angle: {angle:.1f} deg", position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        return image
    
    def draw_angle_stats(self, image):
        """角度統計を描画（大きなフォント）"""
        if self.hip_angles:
            height, width = image.shape[:2]
            
            # 背景パネルを描画
            panel_width = 350
            panel_height = 120
            cv2.rectangle(image, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
            
            y_offset = 40
            line_spacing = 30
            
            # 現在の角度（大きく表示）
            cv2.putText(image, f"現在: {self.hip_angles[-1]:.1f}度", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # 最大・最小角度
            cv2.putText(image, f"最大: {self.max_angle:.1f}度", 
                       (20, y_offset + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, f"最小: {self.min_angle:.1f}度", 
                       (20, y_offset + line_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return image
    
    def draw_eye_gaze(self, image, face_landmarks):
        """絶対に見える目線矢印表示"""
        height, width = image.shape[:2]
        
        # 画面中央に固定位置で必ず矢印を表示（テスト用）
        center_x = width // 2
        center_y = height // 3
        
        # 常に表示される基本矢印（右向き）
        arrow_end_x = center_x + 80
        arrow_end_y = center_y
        
        gaze_status = "STRAIGHT"
        status_color = (0, 255, 0)
        
        if face_landmarks:
            try:
                # 顔の重要なランドマークを取得
                nose_tip = face_landmarks.landmark[1]    # 鼻先
                forehead = face_landmarks.landmark[10]   # 額
                chin = face_landmarks.landmark[175]      # 顎
                left_eye = face_landmarks.landmark[33]   # 左目
                right_eye = face_landmarks.landmark[362] # 右目
                
                nose_x = int(nose_tip.x * width)
                nose_y = int(nose_tip.y * height)
                forehead_y = int(forehead.y * height)
                chin_y = int(chin.y * height)
                eye_y = int((left_eye.y + right_eye.y) / 2 * height)
                
                # 水平方向の判定（左右）
                face_center_x = width // 2
                horizontal_offset = nose_x - face_center_x
                
                # 垂直方向の判定（上下）
                # 顔の基準中心を計算
                face_center_y = height // 2
                vertical_offset = nose_y - face_center_y
                
                # 額から顎までの長さで正規化
                face_height = chin_y - forehead_y
                if face_height > 0:
                    vertical_ratio = vertical_offset / (face_height * 0.3)
                else:
                    vertical_ratio = 0
                
                horizontal_ratio = horizontal_offset / (width * 0.15)
                
                # 矢印方向を決定
                arrow_length = 80
                
                # 水平方向の成分
                if abs(horizontal_ratio) < 0.5:
                    h_direction = 0
                elif horizontal_ratio > 0:
                    h_direction = 1  # 右
                else:
                    h_direction = -1  # 左
                
                # 垂直方向の成分
                if abs(vertical_ratio) < 0.5:
                    v_direction = 0  # 正面
                elif vertical_ratio > 0:
                    v_direction = 1  # 下
                else:
                    v_direction = -1  # 上
                
                # 状態判定と色決定
                if h_direction == 0 and v_direction == 0:
                    gaze_status = "STRAIGHT"
                    status_color = (0, 255, 0)  # 緑
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
                    status_color = (0, 255, 255)  # 黄
                
                # 矢印の終点を計算
                arrow_end_x = center_x + (h_direction * arrow_length)
                arrow_end_y = center_y + (v_direction * arrow_length)
                
                # 正面向きの場合は短い右向き矢印
                if h_direction == 0 and v_direction == 0:
                    arrow_end_x = center_x + 50
                    arrow_end_y = center_y
                    
            except:
                # エラーが発生した場合はデフォルト表示
                pass
        
        # 絶対に見える超巨大矢印を描画
        # 1. 黒い太い外枠
        cv2.arrowedLine(image, 
                       (center_x, center_y), 
                       (arrow_end_x, arrow_end_y), 
                       (0, 0, 0), 20, tipLength=0.3)
        
        # 2. 白い中間層
        cv2.arrowedLine(image, 
                       (center_x, center_y), 
                       (arrow_end_x, arrow_end_y), 
                       (255, 255, 255), 15, tipLength=0.3)
        
        # 3. カラーの内側
        cv2.arrowedLine(image, 
                       (center_x, center_y), 
                       (arrow_end_x, arrow_end_y), 
                       status_color, 10, tipLength=0.3)
        
        # 中心点に大きな円
        cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), -1)
        cv2.circle(image, (center_x, center_y), 18, (255, 255, 255), 3)
        
        return image
    
    def detect_forward_pause(self, current_angle):
        """前方位置での停止を検知"""
        # 最近の角度履歴を更新
        self.recent_angles.append(current_angle)
        if len(self.recent_angles) > self.pause_detection_window:
            self.recent_angles.pop(0)
        
        # 十分なデータが蓄積されてから判定
        if len(self.recent_angles) >= self.pause_detection_window:
            # 角度の変動を計算
            angle_variation = max(self.recent_angles) - min(self.recent_angles)
            avg_angle = sum(self.recent_angles) / len(self.recent_angles)
            
            # 前方位置の判定（小さい角度 = 前傾）（より厳しく）
            is_forward_position = avg_angle < (self.min_angle + (self.max_angle - self.min_angle) * 0.25)
            
            # 停止の判定（角度変動が小さい）
            is_stable = angle_variation < self.pause_threshold
            
            # 前方での停止判定
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
        if self.is_paused_forward and self.pause_counter > 3:  # 3フレーム以上継続で表示
            height, width = image.shape[:2]
            
            # 表示位置（画面右上）
            indicator_x = width - 200
            indicator_y = 50
            
            # 停止継続時間の計算
            pause_duration = self.pause_counter
            
            # 背景矩形（緑色で良い停止、赤色で長すぎる停止）
            if pause_duration < 15:  # 適切な停止時間
                bg_color = (0, 255, 0)  # 緑
                status_text = "GOOD PAUSE"
            elif pause_duration < 30:  # やや長い
                bg_color = (0, 255, 255)  # 黄
                status_text = "PAUSE OK"
            else:  # 長すぎる
                bg_color = (0, 0, 255)  # 赤
                status_text = "TOO LONG"
            
            # 背景描画
            cv2.rectangle(image, (indicator_x - 10, indicator_y - 30), 
                         (indicator_x + 180, indicator_y + 20), (0, 0, 0), -1)
            cv2.rectangle(image, (indicator_x - 10, indicator_y - 30), 
                         (indicator_x + 180, indicator_y + 20), bg_color, 3)
            
            # テキスト表示
            cv2.putText(image, "FORWARD PAUSE", (indicator_x, indicator_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg_color, 2)
            cv2.putText(image, status_text, (indicator_x, indicator_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_color, 2)
            
            # 停止時間のバー表示
            bar_width = min(160, pause_duration * 8)  # 最大160px
            cv2.rectangle(image, (indicator_x, indicator_y + 15), 
                         (indicator_x + bar_width, indicator_y + 20), bg_color, -1)
        
        return image
    
    def process_frame(self, frame):
        """フレームを処理"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿勢検出
        pose_results = self.pose.process(rgb_frame)
        # 顔のメッシュ検出
        face_results = self.face_mesh.process(rgb_frame)
        
        # 結果を描画
        if pose_results.pose_landmarks:
            # 骨格を描画
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = pose_results.pose_landmarks.landmark
            height, width = frame.shape[:2]
            
            # 頭の位置（鼻の位置を使用）
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            head_y = nose.y * height
            
            # 頭の高さに水平線を描画
            frame = self.draw_head_horizontal_line(frame, head_y)
            
            # 腰の角度計算（左腰、腰中心、右腰）
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # 腰の中心点を計算
            hip_center = [(left_hip.x + right_hip.x) / 2 * width,
                         (left_hip.y + right_hip.y) / 2 * height]
            
            # 肩の中心点を計算
            shoulder_center = [(left_shoulder.x + right_shoulder.x) / 2 * width,
                              (left_shoulder.y + right_shoulder.y) / 2 * height]
            
            # 膝の位置を取得（角度計算用）
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            knee_center = [(left_knee.x + right_knee.x) / 2 * width,
                          (left_knee.y + right_knee.y) / 2 * height]
            
            # 腰の角度を計算（肩-腰-膝）
            hip_angle = self.calculate_angle(shoulder_center, hip_center, knee_center)
            
            # 角度を記録
            self.hip_angles.append(hip_angle)
            if hip_angle > self.max_angle:
                self.max_angle = hip_angle
            if hip_angle < self.min_angle:
                self.min_angle = hip_angle
            
            # 前方停止検知
            self.detect_forward_pause(hip_angle)
            
            # 腰の角度を描画
            frame = self.draw_hip_angle(frame, hip_angle, (int(hip_center[0]), int(hip_center[1])))
            
            # 前方停止インジケーターを描画
            frame = self.draw_pause_indicator(frame)
            
            # 画面左上の角度統計表示は削除（不要）
        
        # 目線を描画
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                frame = self.draw_eye_gaze(frame, face_landmarks)
        
        return frame

# Streamlit UI
st.title("🏃‍♂️ 動画姿勢解析アプリ")
st.markdown("**骨格トレース、腰角度測定、目線検出 + 動画比較機能**")

# タブで機能を切り替え
tab1, tab2 = st.tabs(["🔍 単体解析", "🆚 比較解析"])

# サイドバーで設定（共通）
st.sidebar.header("📊 リアルタイム解析データ")

# リアルタイム角度表示用のサイドバーコンテナ
sidebar_current_angle = st.sidebar.empty()
sidebar_max_angle = st.sidebar.empty()
sidebar_min_angle = st.sidebar.empty()
sidebar_frame_count = st.sidebar.empty()
sidebar_pause_status = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 設定")

# 再生速度設定（ユニークキー付き）
frame_skip = st.sidebar.slider("Frame Skip (Speed Up)", 1, 5, 1, 
                               help="Higher = Faster. 2=Half frames, 3=1/3 frames",
                               key="global_frame_skip")
st.sidebar.text(f"Processing: 1/{frame_skip} frames")

# Tab 1: 単体解析
with tab1:
    st.markdown("### 📹 単体動画解析")
    
    # 動画アップロード
    uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=['mp4', 'avi', 'mov'], key="single_upload")

    if uploaded_file is not None:
        # 一時ファイルに保存
        import tempfile
        import os
        
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # VideoAnalyzerを初期化
        analyzer = VideoAnalyzer()
        
        # 動画を読み込み
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("解析開始", key="single_analyze"):
            # レイアウトを作成
            col1, col2 = st.columns([3, 1])
            
            with col1:
                stframe = st.empty()
                # 動画の下に前方停止表示エリアを作成
                st.markdown("#### 🎯 前方停止検知")
                pause_display_main = st.empty()
            
            with col2:
                st.markdown("### 📈 リアルタイム解析情報")
                realtime_current = st.empty()
                realtime_max = st.empty()
                realtime_min = st.empty()
                realtime_progress = st.empty()
        
        progress_bar = st.progress(0)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        processed_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            
            # フレームスキップによる高速化
            if current_frame % frame_skip != 0:
                continue
            
            # フレームを処理
            processed_frame = analyzer.process_frame(frame)
            processed_frame_count += 1
            
            # Streamlitに表示
            stframe.image(processed_frame, channels="BGR")
            
            # リアルタイム情報を更新
            if analyzer.hip_angles:
                current_angle = analyzer.hip_angles[-1]
                
                # メインエリアの横に表示
                with col2:
                    realtime_current.markdown(f"""
                    <div style='background-color: #1f4e79; padding: 15px; border-radius: 10px; margin: 5px 0;'>
                        <h2 style='color: white; margin: 0; text-align: center;'>Current Hip Angle</h2>
                        <h1 style='color: #00ff00; margin: 0; text-align: center; font-size: 48px;'>{current_angle:.1f}°</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    realtime_max.markdown(f"""
                    <div style='background-color: #8B0000; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        <h4 style='color: white; margin: 0; text-align: center;'>Max Angle: {analyzer.max_angle:.1f}°</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    realtime_min.markdown(f"""
                    <div style='background-color: #006400; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        <h4 style='color: white; margin: 0; text-align: center;'>Min Angle: {analyzer.min_angle:.1f}°</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    realtime_progress.markdown(f"""
                    <div style='background-color: #2E2E2E; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        <h5 style='color: white; margin: 0; text-align: center;'>Frame: {current_frame}/{frame_count}</h5>
                        <h5 style='color: #FFD700; margin: 0; text-align: center;'>Progress: {current_frame/frame_count*100:.1f}%</h5>
                        <h6 style='color: #87CEEB; margin: 0; text-align: center;'>Processed: {processed_frame_count} (Skip: {frame_skip})</h6>
                    </div>
                    """, unsafe_allow_html=True)
                
                # サイドバーにも表示
                sidebar_current_angle.markdown(f"""
                <div style='background-color: #1f4e79; padding: 10px; border-radius: 5px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Current Angle</h3>
                    <h2 style='color: #00ff00; margin: 0;'>{current_angle:.1f}°</h2>
                </div>
                """, unsafe_allow_html=True)
                
                sidebar_max_angle.markdown(f"**Max Angle:** {analyzer.max_angle:.1f}°")
                sidebar_min_angle.markdown(f"**Min Angle:** {analyzer.min_angle:.1f}°")
                sidebar_frame_count.markdown(f"**Frame:** {current_frame}/{frame_count} (Processed: {processed_frame_count})")
                
                # 前方停止状態の表示
                if analyzer.is_paused_forward and analyzer.pause_counter > 2:  # より早く反応
                    if analyzer.pause_counter < 10:  # より厳しい基準
                        pause_color = "🟢"
                        pause_status = "GOOD PAUSE"
                        pause_bg_color = "#27AE60"
                        border_color = "#2ECC71"
                    elif analyzer.pause_counter < 20:  # より厳しい基準
                        pause_color = "🟡"
                        pause_status = "PAUSE OK"
                        pause_bg_color = "#F39C12"
                        border_color = "#F1C40F"
                    else:
                        pause_color = "🔴"
                        pause_status = "TOO LONG"
                        pause_bg_color = "#E74C3C"
                        border_color = "#C0392B"
                    
                    # サイドバー表示（コンパクト）
                    sidebar_pause_status.markdown(f"""
                    <div style='background-color: {pause_bg_color}; padding: 8px; border-radius: 5px; text-align: center; margin: 5px 0;'>
                        <h4 style='color: white; margin: 0;'>{pause_color} PAUSE</h4>
                        <h6 style='color: white; margin: 0;'>{pause_status}</h6>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 動画の下に大きく表示
                    pause_display_main.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, {pause_bg_color} 0%, {border_color} 100%);
                        padding: 25px;
                        border-radius: 15px;
                        margin: 15px 0;
                        border: 4px solid {border_color};
                        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                        text-align: center;
                    '>
                        <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                            <h1 style='color: white; margin: 0; font-size: 48px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
                                {pause_color} FORWARD PAUSE
                            </h1>
                        </div>
                        <h2 style='color: white; margin: 10px 0; font-size: 36px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
                            {pause_status}
                        </h2>
                        <div style='background-color: rgba(255,255,255,0.2); height: 12px; border-radius: 6px; margin: 15px 0; overflow: hidden;'>
                            <div style='background-color: white; height: 12px; width: {min(100, analyzer.pause_counter * 5)}%; border-radius: 6px; transition: width 0.3s ease;'></div>
                        </div>
                        <h3 style='color: white; margin: 0; font-size: 24px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
                            継続時間: {analyzer.pause_counter} フレーム ({analyzer.pause_counter/30:.1f}秒)
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    sidebar_pause_status.markdown("**Forward Pause:** Not detected")
                    pause_display_main.markdown(f"""
                    <div style='
                        background-color: #34495E;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 15px 0;
                        text-align: center;
                        border: 2px dashed #7F8C8D;
                    '>
                        <h3 style='color: #BDC3C7; margin: 0; font-size: 24px;'>
                            🔍 前方停止を監視中...
                        </h3>
                        <p style='color: #95A5A6; margin: 10px 0; font-size: 16px;'>
                            前方位置で停止すると検知されます
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            progress_bar.progress(current_frame / frame_count)
            
            # 最高速で処理（sleepなし）
        
        cap.release()
        
        # 結果の統計を表示
        if analyzer.hip_angles:
            st.success("🎉 解析完了！")
            
            # 大きな結果表示パネル
            st.markdown("## 📊 解析結果サマリー")
            
            col1, col2, col3, col4 = st.columns(4)
            
            avg_angle = np.mean(analyzer.hip_angles)
            std_angle = np.std(analyzer.hip_angles)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: #FF6B6B; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Max Angle</h3>
                    <h1 style='color: white; margin: 0; font-size: 36px;'>{analyzer.max_angle:.1f}°</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #4ECDC4; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Min Angle</h3>
                    <h1 style='color: white; margin: 0; font-size: 36px;'>{analyzer.min_angle:.1f}°</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='background-color: #45B7D1; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Average</h3>
                    <h1 style='color: white; margin: 0; font-size: 36px;'>{avg_angle:.1f}°</h1>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div style='background-color: #F7DC6F; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: #2C3E50; margin: 0;'>Std Dev</h3>
                    <h1 style='color: #2C3E50; margin: 0; font-size: 36px;'>{std_angle:.1f}°</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 追加の統計情報
            range_angle = analyzer.max_angle - analyzer.min_angle
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.markdown(f"""
                <div style='background-color: #8E44AD; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>Range: {range_angle:.1f}°</h4>
                    <h4 style='color: white; margin: 0;'>Total Frames: {len(analyzer.hip_angles)}</h4>
                </div>
                """, unsafe_allow_html=True)
                
            with col_stats2:
                # 動きの激しさを評価
                movement_intensity = "Low" if std_angle < 5 else "Medium" if std_angle < 15 else "High"
                movement_color = "#27AE60" if std_angle < 5 else "#F39C12" if std_angle < 15 else "#E74C3C"
                
                st.markdown(f"""
                <div style='background-color: {movement_color}; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>Movement: {movement_intensity}</h4>
                    <h4 style='color: white; margin: 0;'>CV: {(std_angle/avg_angle)*100:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # 角度の変化をグラフで表示（英語表記）
            st.subheader("📈 Hip Angle Time Series")
            
            # Matplotlib設定
            plt.rcParams['font.size'] = 12
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(analyzer.hip_angles, color='blue', linewidth=3, label='Hip Angle')
            ax.axhline(y=analyzer.max_angle, color='red', linestyle='--', linewidth=2,
                      label=f'Max: {analyzer.max_angle:.1f}°')
            ax.axhline(y=analyzer.min_angle, color='green', linestyle='--', linewidth=2,
                      label=f'Min: {analyzer.min_angle:.1f}°')
            ax.axhline(y=avg_angle, color='orange', linestyle=':', linewidth=2,
                      label=f'Avg: {avg_angle:.1f}°')
            
            ax.fill_between(range(len(analyzer.hip_angles)), 
                           analyzer.hip_angles, alpha=0.3, color='lightblue')
            
            ax.set_xlabel('Frame Number', fontsize=14)
            ax.set_ylabel('Angle (degrees)', fontsize=14)
            ax.set_title('Hip Angle Time Series Analysis', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 背景を白に設定
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            st.pyplot(fig)
    
        # 一時ファイルを削除
        os.unlink(tfile.name)
    
    else:
        st.info("動画ファイルをアップロードして解析を始めてください。")
        st.markdown("""
        ### 🎯 単体解析機能:
        - **骨格トレース**: MediaPipeを使用して人体の骨格を検出・描画
        - **頭の高さ水平線**: 頭の位置に緑色の固定基準線を描画
        - **腰角度測定**: 肩-腰-膝の角度をリアルタイムで計算・表示
        - **前方停止検知**: ローイングのキャッチ位置での停止を検出
        - **目線検出**: 顔の向き（上下左右）を矢印で表示
        - **角度統計**: 動画全体での最大・最小・平均角度の記録
        - **時系列グラフ**: 腰角度の変化をグラフで可視化
        """)

# Tab 2: 比較解析
with tab2:
    st.markdown("### 🆚 2つの動画比較解析")
    
    # 比較用の動画アップロード
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("#### 🎥 動画1（基準）")
        uploaded_file1 = st.file_uploader("最初の動画ファイル", type=['mp4', 'avi', 'mov'], key="compare_video1")
    
    with col_upload2:
        st.markdown("#### 🎥 動画2（比較対象）")
        uploaded_file2 = st.file_uploader("比較する動画ファイル", type=['mp4', 'avi', 'mov'], key="compare_video2")
    
    # 比較解析の実行
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.success("✅ 2つの動画がアップロードされました")
        
        if st.button("🔍 比較解析開始", key="start_comparison"):
            # 2つのVideoAnalyzerを初期化
            analyzer1 = VideoAnalyzer()
            analyzer2 = VideoAnalyzer()
            
            # 一時ファイルに保存
            import tempfile
            import os
            
            tfile1 = tempfile.NamedTemporaryFile(delete=False)
            tfile1.write(uploaded_file1.read())
            tfile2 = tempfile.NamedTemporaryFile(delete=False)
            tfile2.write(uploaded_file2.read())
            
            # 動画を読み込み
            cap1 = cv2.VideoCapture(tfile1.name)
            cap2 = cv2.VideoCapture(tfile2.name)
            
            # 比較レイアウト
            st.markdown("## 📊 リアルタイム比較解析")
            col_video1, col_video2 = st.columns(2)
            
            with col_video1:
                st.markdown("#### 🎥 動画1（基準）")
                stframe1 = st.empty()
                angle_display1 = st.empty()
            
            with col_video2:
                st.markdown("#### 🎥 動画2（比較）")
                stframe2 = st.empty()
                angle_display2 = st.empty()
            
            # リアルタイム比較表示
            comparison_display = st.empty()
            progress_bar = st.progress(0)
            
            # フレーム数を取得
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = max(frame_count1, frame_count2)
            
            current_frame = 0
            
            # 同期フレーム処理
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 and not ret2:
                    break
                
                current_frame += 1
                
                # フレームスキップ
                if current_frame % frame_skip != 0:
                    continue
                
                # 両方のフレームを処理
                if ret1:
                    processed_frame1 = analyzer1.process_frame(frame1)
                    stframe1.image(processed_frame1, channels="BGR")
                
                if ret2:
                    processed_frame2 = analyzer2.process_frame(frame2)
                    stframe2.image(processed_frame2, channels="BGR")
                
                # 角度情報表示
                if analyzer1.hip_angles and analyzer2.hip_angles:
                    angle1 = analyzer1.hip_angles[-1]
                    angle2 = analyzer2.hip_angles[-1]
                    
                    angle_display1.markdown(f"""
                    <div style='background-color: #1f4e79; padding: 10px; border-radius: 8px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>Current: {angle1:.1f}°</h3>
                        <h5 style='color: #FFD700; margin: 0;'>Max: {analyzer1.max_angle:.1f}° | Min: {analyzer1.min_angle:.1f}°</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    angle_display2.markdown(f"""
                    <div style='background-color: #8B0000; padding: 10px; border-radius: 8px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>Current: {angle2:.1f}°</h3>
                        <h5 style='color: #FFD700; margin: 0;'>Max: {analyzer2.max_angle:.1f}° | Min: {analyzer2.min_angle:.1f}°</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # リアルタイム差分表示
                    angle_diff = angle1 - angle2
                    if abs(angle_diff) < 3:
                        diff_color = "#27AE60"
                        diff_status = "🟢 類似"
                    elif abs(angle_diff) < 8:
                        diff_color = "#F39C12"
                        diff_status = "🟡 やや差"
                    else:
                        diff_color = "#E74C3C"
                        diff_status = "🔴 大差"
                    
                    comparison_display.markdown(f"""
                    <div style='background-color: {diff_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 15px 0;'>
                        <h1 style='color: white; margin: 0; font-size: 42px;'>角度差: {angle_diff:+.1f}°</h1>
                        <h2 style='color: white; margin: 5px 0; font-size: 28px;'>{diff_status}</h2>
                        <h3 style='color: white; margin: 0; font-size: 20px;'>{angle1:.1f}° - {angle2:.1f}° = {angle_diff:+.1f}°</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 進捗表示
                progress_bar.progress(current_frame / max_frames)
            
            # 解析完了後の比較結果
            cap1.release()
            cap2.release()
            os.unlink(tfile1.name)
            os.unlink(tfile2.name)
            
            if analyzer1.hip_angles and analyzer2.hip_angles:
                st.success("🎉 比較解析完了！")
                
                # 統計計算
                avg1 = np.mean(analyzer1.hip_angles)
                avg2 = np.mean(analyzer2.hip_angles)
                std1 = np.std(analyzer1.hip_angles)
                std2 = np.std(analyzer2.hip_angles)
                
                # 比較結果サマリー
                st.markdown("## 📊 比較結果サマリー")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    diff_max = analyzer1.max_angle - analyzer2.max_angle
                    color = "#FF6B6B" if diff_max > 0 else "#4ECDC4"
                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>最大角度差</h3>
                        <h1 style='color: white; margin: 0; font-size: 32px;'>{diff_max:+.1f}°</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    diff_avg = avg1 - avg2
                    color = "#45B7D1" if abs(diff_avg) < 5 else "#F39C12"
                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>平均角度差</h3>
                        <h1 style='color: white; margin: 0; font-size: 32px;'>{diff_avg:+.1f}°</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    better_stability = "動画1" if std1 < std2 else "動画2"
                    color = "#27AE60" if std1 < std2 else "#E74C3C"
                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>より安定</h3>
                        <h1 style='color: white; margin: 0; font-size: 32px;'>{better_stability}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    frame_diff = len(analyzer1.hip_angles) - len(analyzer2.hip_angles)
                    color = "#8E44AD"
                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>フレーム差</h3>
                        <h1 style='color: white; margin: 0; font-size: 32px;'>{frame_diff:+d}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 比較グラフ
                st.markdown("### 📈 角度変化比較グラフ")
                
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # 動画1（青）
                ax.plot(analyzer1.hip_angles, color='blue', linewidth=3, label='動画1（基準）', alpha=0.8)
                ax.fill_between(range(len(analyzer1.hip_angles)), analyzer1.hip_angles, alpha=0.3, color='blue')
                
                # 動画2（赤）
                ax.plot(analyzer2.hip_angles, color='red', linewidth=3, label='動画2（比較）', alpha=0.8)
                ax.fill_between(range(len(analyzer2.hip_angles)), analyzer2.hip_angles, alpha=0.3, color='red')
                
                # 平均線
                ax.axhline(y=avg1, color='blue', linestyle=':', alpha=0.7, label=f'動画1平均: {avg1:.1f}°')
                ax.axhline(y=avg2, color='red', linestyle=':', alpha=0.7, label=f'動画2平均: {avg2:.1f}°')
                
                ax.set_xlabel('Frame Number', fontsize=14)
                ax.set_ylabel('Hip Angle (degrees)', fontsize=14)
                ax.set_title('Hip Angle Comparison Analysis', fontsize=16, fontweight='bold')
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')
                
                st.pyplot(fig)
                
                # 詳細比較テーブル
                st.markdown("### 📋 詳細比較データ")
                
                import pandas as pd
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
                
                # 推奨事項
                st.markdown("### 💡 改善推奨")
                
                recommendations = []
                
                if abs(avg1 - avg2) > 10:
                    recommendations.append("📐 平均角度に大きな差があります - フォームの一貫性を確認してください")
                
                if std1 > std2 * 1.5:
                    recommendations.append("📊 動画1の方が不安定です - 動作の滑らかさを改善してください")
                elif std2 > std1 * 1.5:
                    recommendations.append("📊 動画2の方が不安定です - 動作の滑らかさを改善してください")
                
                if abs(analyzer1.max_angle - analyzer2.max_angle) > 15:
                    recommendations.append("⚠️ 最大角度に大きな差があります - 後傾の限界を確認してください")
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.success("✅ 2つの動画は類似したフォームを示しています")
    
    elif uploaded_file1 is not None:
        st.info("📹 動画1のみアップロード済み - 動画2もアップロードして比較解析を開始してください")
    elif uploaded_file2 is not None:
        st.info("📹 動画2のみアップロード済み - 動画1もアップロードして比較解析を開始してください")
    else:
        st.info("2つの動画ファイルをアップロードして比較解析を始めてください。")
        st.markdown("""
        ### 🆚 比較解析の特徴:
        - **同時解析**: 2つの動画を並行して処理
        - **リアルタイム比較**: 角度差分をリアルタイム表示
        - **詳細統計**: 最大・最小・平均角度の詳細比較
        - **安定性評価**: どちらがより安定したフォームか自動判定
        - **視覚的比較**: 重ね合わせグラフで変化パターンを比較
        - **改善推奨**: AIによるフォーム改善アドバイス
        """)
