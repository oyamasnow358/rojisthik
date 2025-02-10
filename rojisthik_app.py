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

        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            st.error("目的変数（Y）が単一のクラスしか含まれていません。分類できません。")
            st.stop()
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        
        if set(y_test) - set(le.classes_):
            st.error("y_test に y_train に存在しないラベルが含まれています。")
            st.stop()
        
        y_test = le.transform(y_test)
        
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())
        
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)
        
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if len(set(y_test)) > 1:
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            prec = rec = f1 = 0.0
            y_prob = np.zeros_like(y_test, dtype=float)
        
        st.write(f"適合率 (Precision): {prec:.4f}")
        st.write(f"再現率 (Recall): {rec:.4f}")
        st.write(f"F1スコア: {f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("予測値")
        ax.set_ylabel("実測値")
        st.subheader("混同行列")
        st.pyplot(fig)
        
        if len(set(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], 'r--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.subheader("ROC曲線")
            st.pyplot(fig2)
