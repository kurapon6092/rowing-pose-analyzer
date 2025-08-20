import cv2
import numpy as np
import tempfile
import os

class VideoProcessor:
    """動画処理のためのユーティリティクラス"""
    
    def __init__(self):
        self.temp_files = []
    
    def save_uploaded_file(self, uploaded_file):
        """アップロードされたファイルを一時保存"""
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        self.temp_files.append(tfile.name)
        return tfile.name
    
    def get_video_info(self, video_path):
        """動画の基本情報を取得"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def extract_frames(self, video_path, max_frames=None, skip_frames=1):
        """動画からフレームを抽出"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                frames.append(frame.copy())
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def create_video_preview(self, video_path, num_thumbnails=6):
        """動画のプレビュー画像を作成"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        thumbnails = []
        for i in range(num_thumbnails):
            # フレーム位置を計算
            frame_pos = int((i + 1) * frame_count / (num_thumbnails + 1))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if ret:
                # サムネイルサイズにリサイズ
                thumbnail = cv2.resize(frame, (160, 120))
                thumbnails.append(thumbnail)
        
        cap.release()
        
        if thumbnails:
            # サムネイルを横に並べて1つの画像にする
            combined = np.hstack(thumbnails)
            return combined
        
        return None
    
    def apply_video_filters(self, frame, filter_type='none'):
        """動画にフィルターを適用"""
        if filter_type == 'none':
            return frame
        elif filter_type == 'grayscale':
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif filter_type == 'blur':
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif filter_type == 'edge':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'enhance':
            # コントラストと明度を調整
            alpha = 1.2  # コントラスト
            beta = 20    # 明度
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        return frame
    
    def save_processed_video(self, frames, output_path, fps=30):
        """処理済みフレームを動画として保存"""
        if not frames:
            return False
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    
    def cleanup_temp_files(self):
        """一時ファイルをクリーンアップ"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"一時ファイルの削除エラー: {e}")
        
        self.temp_files.clear()
    
    def __del__(self):
        """デストラクタで一時ファイルをクリーンアップ"""
        self.cleanup_temp_files()
