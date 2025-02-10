import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Streamlit アプリのタイトル
st.title("ロジスティック回帰 Web アプリ")

# CSVテンプレートのダウンロード
st.markdown("### CSVテンプレートのダウンロード")
template_csv = """クラス,特徴量1,特徴量2,特徴量3
0,2.5,3.1,1.2
1,1.2,0.8,2.5
"""
st.download_button(
    label="CSVテンプレートをダウンロード",
    data=template_csv.encode('utf-8-sig'),
    file_name="template.csv",
    mime="text/csv"
)

# CSVファイルのアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### アップロードされたデータ")
    st.dataframe(df.head())
    
    # 目的変数と説明変数の選択
    target_var = st.sidebar.selectbox("目的変数（Y）を選択", df.columns)
    feature_vars = st.sidebar.multiselect("説明変数（X）を選択", [col for col in df.columns if col != target_var])
    
    if feature_vars:
        # データ前処理
        X = df[feature_vars]
        y = df[target_var]
        
        # ラベルエンコーディング（目的変数がカテゴリ変数の場合）
        if y.dtype == 'O':
            y = LabelEncoder().fit_transform(y)
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # ロジスティック回帰の実行
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # モデル評価指標
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.subheader("モデル評価")
        st.write(f"正解率 (Accuracy): {acc:.4f}")
        st.write(f"適合率 (Precision): {prec:.4f}")
        st.write(f"再現率 (Recall): {rec:.4f}")
        st.write(f"F1スコア: {f1:.4f}")
        
        # 混同行列の可視化
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("予測値")
        ax.set_ylabel("実測値")
        st.subheader("混同行列")
        st.pyplot(fig)
        
        # ROC曲線
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}')
        ax2.plot([0, 1], [0, 1], 'r--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.subheader("ROC曲線")
        st.pyplot(fig2)