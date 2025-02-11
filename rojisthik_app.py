import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# フォント設定
font_path = os.path.abspath("ipaexg.ttf")  # 絶対パス
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams["font.family"] = font_prop.get_name()
    plt.rc("font", family=font_prop.get_name())  # 追加
    st.write(f"✅ フォント設定: {mpl.rcParams['font.family']}")
else:
    st.error("❌ フォントファイルが見つかりません。")

st.title("ロジスティック回帰 Web アプリ")

# CSVテンプレートのダウンロード
template_csv = """クラス,特徴量1,特徴量2,特徴量3
0,2.5,3.1,1.2
1,1.2,0.8,2.5
0,3.0,1.5,2.0
1,2.1,3.3,1.5
1,2.4,3.3,2.6
0,3.0,1.5,2.0
"""
st.download_button("CSVテンプレートをダウンロード", data=template_csv.encode('utf-8-sig'), file_name="template.csv", mime="text/csv")

# 初心者向け説明の表示切り替え
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

# ボタンを押すたびにセッションステートを切り替える
if st.button("初心者向け説明を表示/非表示"):
    st.session_state.show_explanation = not st.session_state.show_explanation

# セッションステートに基づいて説明を表示
if st.session_state.show_explanation:
    st.markdown("""
    ## **クラス（目的変数）とは？**
    「クラス」は、予測したいデータのカテゴリや状態を指します。
    - **例1: スパムメール分類**
      - クラス = スパムメールかそうでないか（0 = スパムでない、1 = スパム）
    - **例2: 病気診断**
      - クラス = 病気かどうか（0 = 健康、1 = 病気）

    **ポイント**: クラスには「2つ以上のカテゴリ」が必要です。

    ## **特徴量（説明変数）とは？**
    「特徴量」は、クラスを予測するために使うデータ（情報）のことです。
    - **例1: スパムメール分類**
      - 特徴量 = メールの単語数、リンクの数、送信者の国など
    - **例2: 病気診断**
      - 特徴量 = 年齢、血圧、コレステロール値、体温など

    ## **具体例**
    ### **例: スパムメール分類**
    | クラス（目的変数） | 特徴量1（単語数） | 特徴量2（リンクの数） | 特徴量3（送信者の国） |
    |-----------------|-----------------|--------------------|-------------------|
    | 1 (スパム)      | 500             | 10                 | 1 (海外)          |
    | 0 (非スパム)    | 120             | 0                  | 0 (国内)          |
    | 1 (スパム)      | 350             | 5                  | 1 (海外)          |

    ### **例: 病気診断**
    | クラス（目的変数） | 特徴量1（年齢） | 特徴量2（血圧） | 特徴量3（体温） |
    |-----------------|--------------|--------------|-------------|
    | 1 (病気)       | 65           | 150          | 38.2        |
    | 0 (健康)       | 32           | 120          | 36.5        |
    | 1 (病気)       | 55           | 140          | 39.0        |

    ## **どうやって選べばいいの？**
    1. **クラスを決める**
       - 自分が予測したいことを考える（例: 病気の診断、購入予測、スパム判定）。
       - クラスは「答え」や「結果」となるものです。
    2. **特徴量を選ぶ**
       - クラスに関連しそうな情報を選びます。
       - 予測に役立ちそうなデータを考えてリストアップしましょう。
       - データが多すぎる場合、一部だけ選んでもOKです。
    """)




# CSVファイルのアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### アップロードされたデータ")
    st.dataframe(df.head())

    target_var = st.sidebar.selectbox("目的変数（Y）を選択", df.columns)
    feature_vars = st.sidebar.multiselect("説明変数（X）を選択", [col for col in df.columns if col != target_var])
    

    # ロジスティック回帰分析を実行するボタン
    if st.sidebar.button("ロジスティック回帰分析を実行"):
        if not feature_vars:
            st.error("エラー: 説明変数（X）を選択してください。")
            st.stop()

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

        # 訓練データのクラス数を確認
        unique_train_classes = np.unique(y_train)
        st.write(f"y_train のユニーククラス: {unique_train_classes}")

        if len(unique_train_classes) < 2:
            st.error("y_train に1種類のクラスしかありません。データのバランスを確認してください。")
            st.stop()

        # 特徴量間の相関を表示
        st.write("### 特徴量間の相関係数")
        st.dataframe(df.corr())

        # **データの標準化**
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # **ロジスティック回帰モデルの学習**
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # **モデルが学習されているか確認**
        if hasattr(model, "coef_"):
            # モデルの係数を表示
            st.write("### ロジスティック回帰の係数")
            feature_importance = pd.DataFrame(model.coef_.flatten(), index=feature_vars, columns=["係数"])
            st.dataframe(feature_importance)
        else:
            st.error("モデルの学習に失敗しました。特徴量やデータを確認してください。")
            st.stop()

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

# 日本語フォントを適用
        ax.set_xlabel("予測値", fontproperties=font_prop)
        ax.set_ylabel("実測値", fontproperties=font_prop)
        ax.set_title("混同行列", fontproperties=font_prop)

        st.subheader("混同行列")
        st.pyplot(fig)
        plt.close(fig)
        
        # 分析結果の説明
        st.write("### 分析結果の概要")
        st.write("""
ロジスティック回帰モデルを使ってデータを分類しました。この分析結果は以下のように理解できます：

1. **適合率 (Precision)**  
   モデルが「正しい」と予測した中で、本当に正しかった割合を示します。  
   例えば、メールのスパム判定の場合、適合率が高いほど「スパム」と予測したメールが本当にスパムである可能性が高くなります。

2. **再現率 (Recall)**  
   実際に正しいデータの中で、モデルが正しく予測できた割合を示します。  
   例えば、病気の診断の場合、再現率が高いと「病気を見逃さない」能力が高いことを意味します。

3. **F1スコア**  
   適合率と再現率のバランスを取った指標です。これが高いほど、モデルの予測性能が全体的に良いといえます。

### 混同行列について
上の図（混同行列）は、モデルの予測結果を表しています：

- **左上のセル**: クラス0（例えば「健常者」）を正しく予測した数
- **右下のセル**: クラス1（例えば「病気」）を正しく予測した数
- **右上のセル**: クラス0をクラス1と間違えた数
- **左下のセル**: クラス1をクラス0と間違えた数

これを見ると、モデルがどれくらい正確に予測できたか、どこで間違えたかが一目でわかります。
""")

