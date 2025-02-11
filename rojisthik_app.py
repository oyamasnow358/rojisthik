import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

st.title("ロジスティック回帰 Web アプリ")

# CSVテンプレートのダウンロード
template_csv = """クラス,特徴量1,特徴量2,特徴量3
0,2.5,3.1,1.2
1,1.2,0.8,2.5
0,3.0,1.5,2.0
1,2.1,3.3,1.5
1,2,4,3,3,2,6
0,3.0,1.5,2.0
"""
st.download_button("CSVテンプレートをダウンロード", data=template_csv.encode('utf-8-sig'), file_name="template.csv", mime="text/csv")

# CSVファイルのアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### アップロードされたデータ")
    st.dataframe(df.head())

    target_var = st.sidebar.selectbox("目的変数（Y）を選択", df.columns)
    feature_vars = st.sidebar.multiselect("説明変数（X）を選択", [col for col in df.columns if col != target_var])

    if feature_vars:
        X = df[feature_vars]
        y = df[target_var]

        # **データ数を確認**
        if len(y) < 5:
            st.error("データ数が少なすぎます。最低でも5行以上のデータをアップロードしてください。")
            st.stop()

        # **目的変数を数値化**
        le = LabelEncoder()
        y = le.fit_transform(y)

        # **クラスごとのデータ数を確認**
        unique_classes, counts = np.unique(y, return_counts=True)
        st.write(f"クラスごとのデータ数: {dict(zip(unique_classes, counts))}")

        if len(unique_classes) < 2:
            st.error("y に1種類のクラスしかありません。ロジスティック回帰は2クラス以上の分類問題で動作します。")
            st.stop()

        # **stratify を適用するか決定**
        min_class_size = counts.min()
        stratify_option = y if min_class_size >= 2 else None

        # **テストサイズの調整**
        test_size = 0.2 if len(y) > 10 else 0.4  # データが少ない場合、テストサイズを増やす

        # **データ分割**
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify_option
            )
        except ValueError as e:
            st.error(f"train_test_split でエラー: {str(e)}")
            st.stop()

        # **訓練データのクラス数を確認**
        unique_train_classes = np.unique(y_train)
        st.write(f"y_train のユニーククラス: {unique_train_classes}")

        if len(unique_train_classes) < 2:
            st.error("y_train に1種類のクラスしかありません。データを確認してください。")
            st.stop()

        # **データの標準化**
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # **ロジスティック回帰モデルの学習**
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # **予測と評価**
        y_pred = model.predict(X_test)

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.write(f"適合率 (Precision): {prec:.4f}")
        st.write(f"再現率 (Recall): {rec:.4f}")
        st.write(f"F1スコア: {f1:.4f}")

        # **混同行列**
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("予測値")
        ax.set_ylabel("実測値")
        st.subheader("混同行列")
        st.pyplot(fig)
