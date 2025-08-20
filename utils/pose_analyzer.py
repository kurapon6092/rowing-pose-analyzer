import numpy as np
import cv2
import mediapipe as mp

class PoseAnalysisUtils:
    """姿勢解析のためのユーティリティクラス"""
    
    @staticmethod
    def calculate_angle_3points(point1, point2, point3):
        """3点間の角度を計算（度数法）"""
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        
        # 角度を計算
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    @staticmethod
    def get_body_center(landmarks, image_width, image_height):
        """体の中心点を計算"""
        mp_pose = mp.solutions.pose
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4 * image_width
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4 * image_height
        
        return (int(center_x), int(center_y))
    
    @staticmethod
    def calculate_body_inclination(landmarks, image_width, image_height):
        """体の傾き角度を計算"""
        mp_pose = mp.solutions.pose
        
        # 肩の中心点
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_center = [
            (left_shoulder.x + right_shoulder.x) / 2 * image_width,
            (left_shoulder.y + right_shoulder.y) / 2 * image_height
        ]
        
        # 腰の中心点
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_center = [
            (left_hip.x + right_hip.x) / 2 * image_width,
            (left_hip.y + right_hip.y) / 2 * image_height
        ]
        
        # 垂直線との角度を計算
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        
        # 垂直からの傾き角度
        angle = np.degrees(np.arctan2(dx, dy))
        
        return angle
    
    @staticmethod
    def draw_angle_arc(image, center, start_point, end_point, radius=50, color=(0, 255, 255)):
        """角度を表す弧を描画"""
        # 開始角度と終了角度を計算
        start_vector = np.array(start_point) - np.array(center)
        end_vector = np.array(end_point) - np.array(center)
        
        start_angle = np.degrees(np.arctan2(start_vector[1], start_vector[0]))
        end_angle = np.degrees(np.arctan2(end_vector[1], end_vector[0]))
        
        # 角度を正規化
        if start_angle < 0:
            start_angle += 360
        if end_angle < 0:
            end_angle += 360
        
        # 弧を描画
        cv2.ellipse(image, tuple(map(int, center)), (radius, radius), 
                   0, start_angle, end_angle, color, 2)
        
        return image
    
    @staticmethod
    def get_pose_confidence(landmarks):
        """姿勢検出の信頼度を計算"""
        mp_pose = mp.solutions.pose
        
        # 主要な関節の可視性をチェック
        key_landmarks = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        confidence_scores = []
        for landmark_type in key_landmarks:
            landmark = landmarks[landmark_type.value]
            confidence_scores.append(landmark.visibility)
        
        average_confidence = np.mean(confidence_scores)
        return average_confidence
    
    @staticmethod
    def analyze_movement_pattern(angle_history, window_size=30):
        """動きパターンを解析"""
        if len(angle_history) < window_size:
            return {
                'pattern': 'insufficient_data',
                'frequency': 0,
                'amplitude': 0
            }
        
        # 最近のデータを取得
        recent_angles = angle_history[-window_size:]
        
        # 周波数分析（簡易的）
        fft = np.fft.fft(recent_angles)
        frequencies = np.fft.fftfreq(len(recent_angles))
        
        # 主要な周波数を検出
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        dominant_frequency = abs(frequencies[dominant_freq_idx])
        
        # 振幅を計算
        amplitude = np.std(recent_angles)
        
        # パターンを分類
        if amplitude < 5:
            pattern = 'static'  # 静的
        elif dominant_frequency > 0.1:
            pattern = 'rhythmic'  # リズミカル
        else:
            pattern = 'irregular'  # 不規則
        
        return {
            'pattern': pattern,
            'frequency': dominant_frequency,
            'amplitude': amplitude,
            'mean_angle': np.mean(recent_angles),
            'std_angle': amplitude
        }
