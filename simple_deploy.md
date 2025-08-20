# 🚀 簡単Web公開方法

## Method 1: ngrok（最も簡単）

### 1. ngrokをインストール
```bash
# Homebrewの場合
brew install ngrok

# 公式サイトから直接ダウンロード
# https://ngrok.com/download
```

### 2. アプリを起動
```bash
# ターミナル1
streamlit run main.py
```

### 3. ngrokで公開
```bash
# ターミナル2（別のターミナル）
ngrok http 8501
```

### 4. 公開URL取得
ngrokが表示するURLをコピー（例：https://abc123.ngrok.io）
→ このURLを共有すれば世界中からアクセス可能！

---

## Method 2: PythonAnywhere（無料プラン）

### 1. アカウント作成
https://www.pythonanywhere.com/

### 2. ファイルアップロード
- main.py
- requirements.txt
- utils/ フォルダ

### 3. Bashで実行
```bash
pip3.11 install --user -r requirements.txt
streamlit run main.py --server.port 8000
```

---

## Method 3: Replit（最も手軽）

### 1. Replit.comでアカウント作成
### 2. 「Create Repl」→ 「Python」選択
### 3. ファイルをアップロード
### 4. Run ボタンクリック
### 5. 自動でWebURLが生成される

---

## Method 4: ローカル共有（同じWiFi内）

現在のローカルアドレスを使用：
```
http://192.168.2.100:8502
```

同じWiFiネットワークの人なら誰でもアクセス可能！

---

## 推奨順序

1. **ngrok** - 最も簡単で確実
2. **Replit** - コード共有も可能
3. **PythonAnywhere** - 長期運用向け
4. **ローカル共有** - 社内・家庭内限定
