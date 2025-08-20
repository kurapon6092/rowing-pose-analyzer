import cv2
import numpy as np
import os

def generate_test_video():
    """テスト用の簡単な動画を生成"""
    
    # 動画の設定
    width, height = 640, 480
    fps = 30
    duration = 5  # 5秒間
    total_frames = fps * duration
    
    # 動画ファイルの作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'test_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"テスト動画を生成中: {output_path}")
    
    for frame_num in range(total_frames):
        # 黒いフレームを作成
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 背景色を設定（薄いグレー）
        frame[:, :] = (50, 50, 50)
        
        # 簡単な人の形を描画（スティックフィギュア）
        progress = frame_num / total_frames
        
        # 中央の基本位置
        center_x = width // 2
        base_y = height - 100
        
        # 動きを追加（左右に動く）
        offset_x = int(50 * np.sin(progress * 4 * np.pi))  # 左右に振れる
        offset_y = int(20 * np.sin(progress * 6 * np.pi))  # 上下に動く
        
        person_x = center_x + offset_x
        person_y = base_y + offset_y
        
        # 頭を描画
        head_center = (person_x, person_y - 80)
        cv2.circle(frame, head_center, 15, (255, 255, 255), -1)
        
        # 胴体を描画
        body_top = (person_x, person_y - 65)
        body_bottom = (person_x, person_y - 20)
        cv2.line(frame, body_top, body_bottom, (255, 255, 255), 3)
        
        # 腕を描画（動きを追加）
        arm_angle = progress * 2 * np.pi
        arm_length = 30
        
        # 左腕
        left_shoulder = (person_x - 5, person_y - 50)
        left_arm_end = (
            int(left_shoulder[0] - arm_length * np.cos(arm_angle)),
            int(left_shoulder[1] + arm_length * np.sin(arm_angle))
        )
        cv2.line(frame, left_shoulder, left_arm_end, (255, 255, 255), 2)
        
        # 右腕
        right_shoulder = (person_x + 5, person_y - 50)
        right_arm_end = (
            int(right_shoulder[0] + arm_length * np.cos(arm_angle)),
            int(right_shoulder[1] + arm_length * np.sin(arm_angle))
        )
        cv2.line(frame, right_shoulder, right_arm_end, (255, 255, 255), 2)
        
        # 脚を描画
        leg_angle = progress * 3 * np.pi
        leg_length = 35
        
        # 左脚
        left_hip = (person_x - 5, person_y - 20)
        left_leg_end = (
            int(left_hip[0] - 15 * np.sin(leg_angle)),
            int(left_hip[1] + leg_length)
        )
        cv2.line(frame, left_hip, left_leg_end, (255, 255, 255), 2)
        
        # 右脚
        right_hip = (person_x + 5, person_y - 20)
        right_leg_end = (
            int(right_hip[0] + 15 * np.sin(leg_angle)),
            int(right_hip[1] + leg_length)
        )
        cv2.line(frame, right_hip, right_leg_end, (255, 255, 255), 2)
        
        # フレーム番号を表示
        cv2.putText(frame, f"Frame: {frame_num+1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # より詳細な目印を追加
        cv2.putText(frame, "Test Video for Pose Analysis", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # フレームを動画に追加
        out.write(frame)
    
    # リソースを解放
    out.release()
    print(f"テスト動画の生成が完了しました: {output_path}")
    print(f"動画の長さ: {duration}秒, フレーム数: {total_frames}, FPS: {fps}")
    
    return output_path

if __name__ == "__main__":
    generate_test_video()
