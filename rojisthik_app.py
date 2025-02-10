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
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

# `y_test` に `y_train` にないラベルがあるか確認
        unknown_labels = set(y_test) - set(le.classes_)

        if unknown_labels:
            st.error(f"y_test に y_train に存在しないラベルが含まれています: {unknown_labels}")
            st.stop()  # エラーを出して実行を止める

# `y_test` も同じエンコーディングを適用
        y_test = le.transform(y_test)

        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("X_train の欠損値:\n", X_train.isnull().sum())
        st.write("y_train の欠損値:\n", y_train.isnull().sum())

# 欠損値を補完
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())
        y_train = pd.Series(y_train).fillna(y_train.mode()[0])
        y_test = pd.Series(y_test).fillna(y_test.mode()[0])


        st.write("X_train のデータ型:\n", X_train.dtypes)

# 文字列を含む列があれば、ダミー変数に変換
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        
        # 標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # ロジスティック回帰の実行
        model = LogisticRegression()
        st.write("y_train のユニーク値:", y_train.unique())
        st.write("y_train のデータ型:", y_train.dtype)

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)  # 文字列ラベルを 0/1 に変換
        y_test = le.transform(y_test)

        st.write("y_train のクラス分布:\n", pd.Series(y_train).value_counts())

# クラスが1種類しかない場合は警告
        if len(pd.Series(y_train).unique()) == 1:
           st.error("y_train に1種類のクラスしかありません。データを確認してください。")

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