"""
Google Colab + Gradio版
このコードをGoogle Colabにコピー&ペーストするだけで公開可能
"""

# Google Colabの最初のセルで実行
"""
!pip install gradio opencv-python mediapipe numpy matplotlib

import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# 以下にVideoAnalyzerクラスをコピー
"""

# Gradio用のシンプルなインターフェース
colab_code = '''
def analyze_video(video_file):
    """動画解析のメイン関数"""
    if video_file is None:
        return None, "動画ファイルをアップロードしてください"
    
    # VideoAnalyzerを初期化
    analyzer = VideoAnalyzer()
    
    # 動画を読み込み
    cap = cv2.VideoCapture(video_file.name)
    
    processed_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, frame_count, 5):  # 5フレームごとに処理
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = analyzer.process_frame(frame)
        processed_frames.append(processed_frame)
    
    cap.release()
    
    # 結果の統計
    if analyzer.hip_angles:
        stats = f"""
        解析結果:
        - 最大角度: {analyzer.max_angle:.1f}°
        - 最小角度: {analyzer.min_angle:.1f}°
        - 平均角度: {np.mean(analyzer.hip_angles):.1f}°
        - 処理フレーム数: {len(processed_frames)}
        """
        
        # 最後のフレームを返す
        if processed_frames:
            return processed_frames[-1], stats
    
    return None, "解析に失敗しました"

# Gradioインターフェース
interface = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="動画をアップロード"),
    outputs=[
        gr.Image(label="解析結果"),
        gr.Textbox(label="統計情報")
    ],
    title="🏃‍♂️ 動画姿勢解析アプリ",
    description="動画をアップロードして姿勢解析を行います"
)

# 公開
interface.launch(share=True)  # share=Trueで公開URL生成
'''

print("Google Colab用コード:")
print("1. Google Colab (colab.research.google.com) を開く")
print("2. 新しいノートブック作成")
print("3. 上記コードをセルにコピー&ペースト")
print("4. 実行すると公開URLが生成されます！")
