import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(page_title="神经重症 AKI 风险预测器", page_icon="🏥", layout="wide")
st.title("🏥 神经重症患者 AKI 风险实时预测系统")
st.markdown("---")

# ==========================================
# 2. 加载模型与预处理工具 (开启缓存加速)
# ==========================================
@st.cache_resource
def load_models():
    # 注意：这里直接读取根目录的文件，没有 models/ 前缀
    model_ens = joblib.load('model_ens.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model_ens, scaler, feature_names

try:
    model, scaler, feature_names = load_models()
except FileNotFoundError as e:
    st.error(f"⚠️ 文件加载失败，请确保 GitHub 仓库根目录下存在对应的 pkl 文件。\n报错信息: {e}")
    st.stop()

# ==========================================
# 3. 初始化 SHAP 解释器 (开启缓存加速)
# ==========================================
@st.cache_resource
def create_shap_explainer(_model):
    # 构建 SHAP 解释器
    return shap.Explainer(_model)

explainer = create_shap_explainer(model)

# ==========================================
# 4. 侧边栏：动态生成患者数据输入面板
# ==========================================
st.sidebar.header("📋 输入患者生理指标")
st.sidebar.info("请在下方输入患者的各项数据，系统将实时计算 AKI 风险。")

patient_data = {}
# 根据你的 feature_names.pkl 自动生成对应数量的输入框
for feature in feature_names:
    # 默认值设为 0.0，你可以后续根据具体的生理指标（如年龄、肌酐等）在此处硬编码修改默认值
    patient_data[feature] = st.sidebar.number_input(f"请输入 {feature}", value=0.0, step=0.1)

# ==========================================
# 5. 核心预测与可视化逻辑
# ==========================================
if st.button("🚀 开始预测计算", type="primary"):
    
    # 将用户输入转化为 DataFrame，并强制约束列的顺序与训练时一致
    patient_df = pd.DataFrame([patient_data])[feature_names]
    
    # 执行数据标准化 (极为关键的一步)
    patient_df_scaled = scaler.transform(patient_df)
    
    # 计算风险概率 (提取并发 AKI [类别1] 的概率)
    try:
        prob = model.predict_proba(patient_df_scaled)[0][1]
    except AttributeError:
        # 兼容部分不支持 predict_proba 的模型
        prob = model.predict(patient_df_scaled)[0]

    # --- 界面展示：预测结果 ---
    st.subheader("📊 预测结果")
    
    # 动态颜色提示
    if prob >= 0.5:
        st.error(f"**高风险预警**：该患者并发 AKI 的概率为 **{prob:.2%}**")
    elif prob >= 0.2:
        st.warning(f"**中等风险**：该患者并发 AKI 的概率为 **{prob:.2%}**")
    else:
        st.success(f"**低风险**：该患者并发 AKI 的概率为 **{prob:.2%}**")
        
    st.progress(float(prob))
    
    # --- 界面展示：SHAP 归因分析 ---
    st.markdown("---")
    st.subheader("🔍 特定患者风险驱动因素分析 (SHAP Force Plot)")
    st.markdown("""
    **图表解读指南：**
    * **基准值 (Base Value)**: 模型在所有患者上的平均预测概率。
    * <span style='color: #ff0051; font-weight:bold;'>红色条形 (推高风险)</span>: 此类生理指标正在**推高**当前患者的 AKI 风险。
    * <span style='color: #008bfb; font-weight:bold;'>蓝色条形 (降低风险)</span>: 此类生理指标正在**降低**当前患者的 AKI 风险。
    * 箭头越宽，代表该指标对最终结果的影响力越大。
    """, unsafe_allow_html=True)

    with st.spinner('正在进行特征归因运算，请稍候...'):
        # 计算当前患者的 SHAP 值 (必须传入标准化后的数据)
        shap_values = explainer(patient_df_scaled)
        
        # 兼容性处理：随机森林输出的是三维数组，XGBoost 输出的是二维数组
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[1]
            if len(shap_values.values.shape) == 3:
                shap_vals_patient = shap_values.values[0, :, 1]
            else:
                shap_vals_patient = shap_values.values[0]
        else:
            base_value = explainer.expected_value
            shap_vals_patient = shap_values.values[0]
            
        # 绘制 SHAP 力图 (传入患者的原始数据以供图表展示具体数值)
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            base_value, 
            shap_vals_patient, 
            patient_df.iloc[0], 
            matplotlib=True, 
            show=False,
            feature_names=feature_names,
            figsize=(12, 3),
            text_rotation=15 # 防止特征名称重叠
        )
        
        # 将生成的图表渲染至 Streamlit 页面
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf() # 清空内存，防止下次预测时图表重叠

st.markdown("---")
st.caption("⚠️ **免责声明**：本计算器基于机器学习算法构建，仅供神经重症临床研究与决策辅助参考，不能替代具有专业执照医师的临床诊断与治疗建议。")
