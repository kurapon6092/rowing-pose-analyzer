import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import tempfile
import os

# 元のVideoAnalyzerクラスを再利用
exec(open('main.py').read().split('# 単体動画解析関数')[0])

# 比較用の解析関数
def analyze_video_for_comparison(uploaded_file, video_name, col_position):
    """比較用の動画解析"""
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # VideoAnalyzerを初期化
    analyzer = VideoAnalyzer()
    
    # 動画を読み込み
    cap = cv2.VideoCapture(tfile.name)
    
    with col_position:
        st.markdown(f"### 📊 {video_name}")
        stframe = st.empty()
        realtime_angle = st.empty()
        pause_status = st.empty()
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    processed_frame_count = 0
    
    # フレーム処理ループ
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame += 1
        
        # フレームスキップ
        frame_skip = st.session_state.get('frame_skip', 1)
        if current_frame % frame_skip != 0:
            continue
        
        # フレームを処理
        processed_frame = analyzer.process_frame(frame)
        processed_frame_count += 1
        
        # リアルタイム表示
        stframe.image(processed_frame, channels="BGR")
        
        # 角度情報表示
        if analyzer.hip_angles:
            current_angle = analyzer.hip_angles[-1]
            realtime_angle.markdown(f"""
            <div style='background-color: #1f4e79; padding: 10px; border-radius: 8px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>Current: {current_angle:.1f}°</h3>
                <h4 style='color: #FFD700; margin: 0;'>Max: {analyzer.max_angle:.1f}°</h4>
                <h4 style='color: #00FF00; margin: 0;'>Min: {analyzer.min_angle:.1f}°</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # 前方停止状態
        if analyzer.is_paused_forward and analyzer.pause_counter > 2:
            if analyzer.pause_counter < 10:
                color = "#27AE60"
                status = "GOOD"
            elif analyzer.pause_counter < 20:
                color = "#F39C12"
                status = "OK"
            else:
                color = "#E74C3C"
                status = "LONG"
            
            pause_status.markdown(f"""
            <div style='background-color: {color}; padding: 8px; border-radius: 5px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>PAUSE: {status}</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            pause_status.markdown(f"""
            <div style='background-color: #2C3E50; padding: 8px; border-radius: 5px; text-align: center;'>
                <h5 style='color: #BDC3C7; margin: 0;'>Monitoring...</h5>
            </div>
            """, unsafe_allow_html=True)
    
    cap.release()
    os.unlink(tfile.name)
    
    return analyzer

# Streamlit UI
st.title("🏃‍♂️ 動画姿勢比較解析アプリ")
st.markdown("**2つの動画を比較して姿勢を分析できます**")

# サイドバー設定
st.sidebar.header("⚙️ 設定")

# フレームスキップ設定をセッション状態に保存
if 'frame_skip' not in st.session_state:
    st.session_state.frame_skip = 1

st.session_state.frame_skip = st.sidebar.slider("Frame Skip (Speed Up)", 1, 5, 1, 
                                               help="Higher = Faster. 2=Half frames, 3=1/3 frames")
st.sidebar.text(f"Processing: 1/{st.session_state.frame_skip} frames")

# 動画アップロード
st.markdown("## 📹 動画アップロード")
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("### 🎥 動画1")
    uploaded_file1 = st.file_uploader("最初の動画ファイル", type=['mp4', 'avi', 'mov'], key="video1")

with col_upload2:
    st.markdown("### 🎥 動画2")
    uploaded_file2 = st.file_uploader("比較する動画ファイル", type=['mp4', 'avi', 'mov'], key="video2")

# 動画処理
if uploaded_file1 is not None or uploaded_file2 is not None:
    
    # 比較モード
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.success("🔄 比較モード: 2つの動画を同時解析します")
        
        if st.button("🔍 比較解析開始", key="compare_start"):
            st.markdown("## 📊 リアルタイム比較解析")
            
            # 2列レイアウト
            col1, col2 = st.columns(2)
            
            # 2つの動画を並行解析
            with st.spinner("解析中..."):
                # セッション状態にanalyzersを保存
                if 'analyzer1' not in st.session_state:
                    st.session_state.analyzer1 = None
                if 'analyzer2' not in st.session_state:
                    st.session_state.analyzer2 = None
                
                analyzer1 = analyze_video_for_comparison(uploaded_file1, "動画1", col1)
                analyzer2 = analyze_video_for_comparison(uploaded_file2, "動画2", col2)
                
                st.session_state.analyzer1 = analyzer1
                st.session_state.analyzer2 = analyzer2
            
            # 比較結果を表示
            if st.session_state.analyzer1 and st.session_state.analyzer2:
                display_comparison_results(st.session_state.analyzer1, st.session_state.analyzer2)
    
    # 単体解析モード
    elif uploaded_file1 is not None:
        st.info("📹 動画1のみアップロード済み")
        if st.button("📊 動画1を解析", key="analyze1"):
            analyzer = analyze_video_for_comparison(uploaded_file1, "動画1", st)
            display_single_results(analyzer, "動画1")
    
    elif uploaded_file2 is not None:
        st.info("📹 動画2のみアップロード済み")
        if st.button("📊 動画2を解析", key="analyze2"):
            analyzer = analyze_video_for_comparison(uploaded_file2, "動画2", st)
            display_single_results(analyzer, "動画2")

else:
    st.info("動画ファイルをアップロードして解析を始めてください。")
    st.markdown("""
    ### 🆕 新機能: 動画比較
    - **比較解析**: 2つの動画を同時に解析・比較
    - **個別解析**: 単体動画の詳細分析
    - **統計比較**: 角度、停止時間、フォーム品質の比較
    """)

def display_comparison_results(analyzer1, analyzer2):
    """比較結果を表示"""
    st.markdown("## 🆚 比較結果")
    
    # 統計比較
    col1, col2, col3, col4 = st.columns(4)
    
    avg1 = np.mean(analyzer1.hip_angles) if analyzer1.hip_angles else 0
    avg2 = np.mean(analyzer2.hip_angles) if analyzer2.hip_angles else 0
    
    with col1:
        diff_max = analyzer1.max_angle - analyzer2.max_angle
        color = "#FF6B6B" if analyzer1.max_angle > analyzer2.max_angle else "#4ECDC4"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>最大角度差</h4>
            <h2 style='color: white; margin: 0;'>{diff_max:+.1f}°</h2>
            <h5 style='color: white; margin: 0;'>{analyzer1.max_angle:.1f}° vs {analyzer2.max_angle:.1f}°</h5>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        diff_min = analyzer1.min_angle - analyzer2.min_angle
        color = "#4ECDC4" if analyzer1.min_angle < analyzer2.min_angle else "#FF6B6B"
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
        std1 = np.std(analyzer1.hip_angles) if analyzer1.hip_angles else 0
        std2 = np.std(analyzer2.hip_angles) if analyzer2.hip_angles else 0
        diff_std = std1 - std2
        color = "#27AE60" if std1 < std2 else "#E74C3C"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>安定性</h4>
            <h2 style='color: white; margin: 0;'>{diff_std:+.1f}°</h2>
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
        ax.plot(analyzer2.hip_angles, color='red', linewidth=2, label='動画2', alpha=0.8)
        ax.fill_between(range(len(analyzer2.hip_angles)), analyzer2.hip_angles, 
                       alpha=0.3, color='red')
    
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
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

def display_single_results(analyzer, video_name):
    """単体結果表示"""
    if analyzer.hip_angles:
        st.success(f"🎉 {video_name} 解析完了！")
        
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
        
        # グラフ表示
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(analyzer.hip_angles, color='blue', linewidth=2)
        ax.axhline(y=analyzer.max_angle, color='red', linestyle='--', label=f'Max: {analyzer.max_angle:.1f}°')
        ax.axhline(y=analyzer.min_angle, color='green', linestyle='--', label=f'Min: {analyzer.min_angle:.1f}°')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Hip Angle (degrees)')
        ax.set_title(f'{video_name} Hip Angle Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
