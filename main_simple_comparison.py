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

# 元のmain.pyからVideoAnalyzerクラスをインポート
exec(open('main.py').read().split('# Streamlit UI')[0])

# Streamlit UI
st.title("🆚 動画姿勢比較解析アプリ")
st.markdown("**2つの動画を同時に解析して比較します**")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
frame_skip = st.sidebar.slider("Frame Skip (Speed Up)", 1, 5, 1, 
                               help="Higher = Faster. 2=Half frames, 3=1/3 frames")

# 2つの動画アップロード
st.markdown("## 📹 動画アップロード")
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("### 🎥 動画1（基準）")
    uploaded_file1 = st.file_uploader("最初の動画ファイル", type=['mp4', 'avi', 'mov'], key="video1")

with col_upload2:
    st.markdown("### 🎥 動画2（比較対象）")
    uploaded_file2 = st.file_uploader("比較する動画ファイル", type=['mp4', 'avi', 'mov'], key="video2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    st.success("✅ 2つの動画がアップロードされました")
    
    if st.button("🔍 比較解析開始"):
        # 2つのAnalyzerを初期化
        analyzer1 = VideoAnalyzer()
        analyzer2 = VideoAnalyzer()
        
        # 一時ファイルに保存
        tfile1 = tempfile.NamedTemporaryFile(delete=False)
        tfile1.write(uploaded_file1.read())
        tfile2 = tempfile.NamedTemporaryFile(delete=False)
        tfile2.write(uploaded_file2.read())
        
        # 動画を読み込み
        cap1 = cv2.VideoCapture(tfile1.name)
        cap2 = cv2.VideoCapture(tfile2.name)
        
        # 比較表示レイアウト
        st.markdown("## 📊 リアルタイム比較解析")
        
        col_video1, col_video2 = st.columns(2)
        
        with col_video1:
            st.markdown("#### 🎥 動画1（基準）")
            stframe1 = st.empty()
            angle_info1 = st.empty()
        
        with col_video2:
            st.markdown("#### 🎥 動画2（比較）")
            stframe2 = st.empty()
            angle_info2 = st.empty()
        
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
                
                angle_info1.markdown(f"""
                <div style='background-color: #1f4e79; padding: 10px; border-radius: 8px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Current: {angle1:.1f}°</h3>
                    <h5 style='color: #FFD700; margin: 0;'>Max: {analyzer1.max_angle:.1f}° | Min: {analyzer1.min_angle:.1f}°</h5>
                </div>
                """, unsafe_allow_html=True)
                
                angle_info2.markdown(f"""
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
                winner = "動画1" if diff_max > 0 else "動画2"
                st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>最大角度差</h3>
                    <h1 style='color: white; margin: 0; font-size: 32px;'>{diff_max:+.1f}°</h1>
                    <h4 style='color: white; margin: 0;'>{winner} が大きい</h4>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                diff_avg = avg1 - avg2
                color = "#45B7D1" if abs(diff_avg) < 5 else "#F39C12"
                st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>平均角度差</h3>
                    <h1 style='color: white; margin: 0; font-size: 32px;'>{diff_avg:+.1f}°</h1>
                    <h4 style='color: white; margin: 0;'>差: {"小" if abs(diff_avg) < 5 else "大"}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                better_stability = "動画1" if std1 < std2 else "動画2"
                color = "#27AE60" if std1 < std2 else "#E74C3C"
                st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>より安定</h3>
                    <h1 style='color: white; margin: 0; font-size: 32px;'>{better_stability}</h1>
                    <h4 style='color: white; margin: 0;'>Std: {min(std1, std2):.1f}°</h4>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                frame_diff = len(analyzer1.hip_angles) - len(analyzer2.hip_angles)
                color = "#8E44AD"
                st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 15px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>フレーム差</h3>
                    <h1 style='color: white; margin: 0; font-size: 32px;'>{frame_diff:+d}</h1>
                    <h4 style='color: white; margin: 0;'>{len(analyzer1.hip_angles)} vs {len(analyzer2.hip_angles)}</h4>
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
