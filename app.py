# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ========== 页面配置 ==========
st.set_page_config(page_title="神经重症AKI风险预测器", page_icon="🧠")
st.title("🧠 神经重症急性肾损伤（AKI）风险预测计算器")
st.markdown("基于 **10项核心临床特征** 与 **集成机器学习模型**，实时预测AKI发生概率，并展示影响风险的关键因素。")

# ========== 特征名称映射 ==========
FEATURE_NAME_CN = {
    'Age': '年龄 (岁)',
    'APACHEII': 'APACHE II 评分',
    'Shock_Index': '休克指数',
    'Lactate_Max': '血乳酸峰值 (μmol/L)',
    'Glucose_CV': '血糖变异系数',
    'BUN_SCr_Ratio': '尿素/肌酐比值',
    'Lab_UricAcid_Max': '血尿酸峰值 (μmol/L)',
    'MechVent_Duration': '机械通气时长 (小时)',
    'Mannitol_ICU_Dose_g': '甘露醇累积剂量 (g)',
    'Vasopressor_Use': '血管活性药物使用'
}

# ========== 加载模型 ==========
@st.cache_resource
def load_models():
    model_ens = joblib.load('models/model_ens.pkl')      # 集成模型（用于概率预测）
    scaler = joblib.load('models/scaler.pkl')            # 标准化器
    features = joblib.load('models/feature_names.pkl')   # 特征顺序
    xgb_model = model_ens.named_estimators_['xgb']       # XGBoost 子模型（用于 SHAP）
    return model_ens, scaler, features, xgb_model

if not os.path.exists('models/model_ens.pkl'):
    st.error("❌ 模型文件未找到！请先运行 train_model.py 训练模型。")
    st.stop()

model_ens, scaler, feature_names, xgb_model = load_models()
explainer = shap.TreeExplainer(xgb_model)

# ========== 字体检测 ==========
def check_chinese_font():
    """检测系统中是否有中文字体，返回是否可用"""
    chinese_fonts = [
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',
        'SimHei',
        'Microsoft YaHei'
    ]
    for font in chinese_fonts:
        try:
            if any(f.name == font for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = font
                return True
        except:
            continue
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return False

chinese_available = check_chinese_font()
if not chinese_available:
    st.warning("⚠️ 当前环境缺少中文字体，SHAP 力图将使用英文标签显示，不影响预测结果。")
    # 使用英文特征名（即 feature_names 本身）
    label_list = feature_names
else:
    label_list = [FEATURE_NAME_CN.get(name, name) for name in feature_names]

# ========== 输入表单 ==========
with st.form("prediction_form"):
    st.subheader("📋 请输入患者临床指标")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("年龄 (岁)", min_value=18, max_value=100, value=68, step=1)
        apacheii = st.number_input("APACHE II 评分", min_value=0, max_value=50, value=19, step=1)
        shock_index = st.number_input("休克指数", min_value=0.2, max_value=2.0, value=0.7, step=0.05, format="%.2f")
        lactate_max = st.number_input("血乳酸峰值 (μmol/L)", min_value=50, max_value=3000, value=280, step=10)
        glucose_cv = st.number_input("血糖变异系数 (CV)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")

    with col2:
        bun_scr_ratio = st.number_input("尿素/肌酐比值", min_value=0.01, max_value=0.5, value=0.10, step=0.01, format="%.2f")
        uric_acid_max = st.number_input("血尿酸峰值 (μmol/L)", min_value=50, max_value=1000, value=320, step=10)
        mechvent_duration = st.number_input("机械通气时长 (小时)", min_value=0, max_value=720, value=48, step=6)
        mannitol_dose = st.number_input("甘露醇累积剂量 (g)", min_value=0, max_value=500, value=0, step=5)
        vasopressor = st.selectbox("血管活性药物使用", options=["否", "是"], index=0)

    submitted = st.form_submit_button("🔮 预测AKI风险")

# ========== 预测与 SHAP 力图 ==========
if submitted:
    vasopressor_val = 1 if vasopressor == "是" else 0
    input_dict = {
        'BUN_SCr_Ratio': bun_scr_ratio,
        'Mannitol_ICU_Dose_g': mannitol_dose,
        'MechVent_Duration': mechvent_duration,
        'Lactate_Max': lactate_max,
        'Vasopressor_Use': vasopressor_val,
        'Glucose_CV': glucose_cv,
        'Lab_UricAcid_Max': uric_acid_max,
        'Shock_Index': shock_index,
        'APACHEII': apacheii,
        'Age': age
    }
    input_df = pd.DataFrame([input_dict])[feature_names]  # 按模型要求顺序排列

    # 标准化与预测
    input_scaled = scaler.transform(input_df)
    prob = model_ens.predict_proba(input_scaled)[0, 1]
    prob_percent = prob * 100

    st.subheader("📊 预测结果")
    st.metric("AKI发生概率", f"{prob_percent:.1f}%")
    if prob < 0.2:
        st.success("低风险 (概率 < 20%)")
    elif prob < 0.4:
        st.info("中风险 (20% ~ 40%)")
    else:
        st.error("高风险 (概率 ≥ 40%)")

    # SHAP 力图
    st.subheader("🔍 影响风险的关键因素（SHAP 力图）")
    st.markdown("下图展示了每个特征对当前患者 AKI 风险的贡献：**红色条表示推高风险**，**蓝色条表示降低风险**。")

    shap_values = explainer.shap_values(input_scaled)

    # 创建大尺寸、高分辨率图形
    plt.figure(figsize=(18, 6), dpi=150)
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0],
        features=input_df.iloc[0].values,
        feature_names=label_list,            # 根据字体可用性选择标签
        matplotlib=True,
        show=False
    )

    # 强制重新设置图形尺寸（防止被覆盖）
    fig = plt.gcf()
    fig.set_size_inches(18, 6)

    # 调整 x 轴标签旋转与对齐
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # 紧凑布局，但保留底部空间给标签
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 底部留出 5% 空间

    st.pyplot(fig, use_container_width=False)  # 禁用自动缩放，保持原始尺寸
    plt.close()

    st.caption("注：本预测结果基于回顾性研究模型，仅供参考，不能替代临床医生判断。")
    if not chinese_available:
        st.caption("（当前使用英文标签，因云端环境缺少中文字体。如需中文，可考虑部署至支持中文的环境。）")
