# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import io
from matplotlib import lines
from matplotlib.patches import PathPatch
from matplotlib.path import Path

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
    model_ens = joblib.load('models/model_ens.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/feature_names.pkl')
    xgb_model = model_ens.named_estimators_['xgb']
    return model_ens, scaler, features, xgb_model

if not os.path.exists('models/model_ens.pkl'):
    st.error("❌ 模型文件未找到！请先运行 train_model.py 训练模型。")
    st.stop()

model_ens, scaler, feature_names, xgb_model = load_models()
explainer = shap.TreeExplainer(xgb_model)

# ========== 字体检测 ==========
def check_chinese_font():
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
    plt.rcParams['font.sans-serif'] = ['Arial']
    return False

chinese_available = check_chinese_font()
if not chinese_available:
    st.warning("⚠️ 当前环境缺少中文字体，SHAP 力图将使用英文标签显示，不影响预测结果。")
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

# ========== 自定义 SHAP 力图绘制函数 ==========
def draw_custom_force_plot(
    base_value,
    shap_values,
    features,
    feature_names,
    figsize=None,
    text_rotation=30,
    min_perc=0.02,
    offset_factor=0.04,
    dpi=150,
):
    """
    自定义绘制 SHAP 力图，基于 SHAP 库底层代码优化。
    参数：
        base_value: 基准值（explainer.expected_value）
        shap_values: 单个样本的 SHAP 值数组
        features: 单个样本的特征值数组
        feature_names: 特征名称列表
        figsize: 图形尺寸，若为 None 则根据特征数量自适应
        text_rotation: 标签旋转角度
        min_perc: 最小贡献百分比，小于此值的特征不显示标签
        offset_factor: 标签与条形的水平偏移系数
        dpi: 图形分辨率
    """
    # 准备数据字典（格式与 SHAP 库内部一致）
    data = {
        "outValue": base_value + shap_values.sum(),
        "baseValue": base_value,
        "features": {},
        "featureNames": feature_names,
        "outNames": ["f(x)"],
        "link": "identity",
    }
    # 填充特征数据
    for i, (name, val, shap_val) in enumerate(zip(feature_names, features, shap_values)):
        data["features"][i] = {"effect": shap_val, "value": f"{val:.2f}"}
    # 格式化数据（排序、转换）
    def format_data(data):
        # 分离正负特征
        neg_features = []
        pos_features = []
        for i, f in data["features"].items():
            effect = f["effect"]
            if effect < 0:
                neg_features.append([effect, f["value"], data["featureNames"][int(i)]])
            else:
                pos_features.append([effect, f["value"], data["featureNames"][int(i)]])
        neg_features = np.array(sorted(neg_features, key=lambda x: float(x[0]), reverse=False))
        pos_features = np.array(sorted(pos_features, key=lambda x: float(x[0]), reverse=True))
        # 转换函数（这里用 identity）
        convert = lambda x: x
        # 转换负特征
        neg_val = data["outValue"]
        for i in neg_features:
            val = float(i[0])
            neg_val = neg_val + np.abs(val)
            i[0] = convert(neg_val)
        total_neg = np.max(neg_features[:, 0].astype(float)) - np.min(neg_features[:, 0].astype(float)) if len(neg_features) > 0 else 0
        # 转换正特征
        pos_val = data["outValue"]
        for i in pos_features:
            val = float(i[0])
            pos_val = pos_val - np.abs(val)
            i[0] = convert(pos_val)
        total_pos = np.max(pos_features[:, 0].astype(float)) - np.min(pos_features[:, 0].astype(float)) if len(pos_features) > 0 else 0
        # 转换基准值和输出值
        data["outValue"] = convert(data["outValue"])
        data["baseValue"] = convert(data["baseValue"])
        return neg_features, total_neg, pos_features, total_pos

    neg_features, total_neg, pos_features, total_pos = format_data(data)
    base_value = data["baseValue"]
    out_value = data["outValue"]
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * offset_factor

    # 创建图形
    if figsize is None:
        n_features = len(feature_names)
        figsize = (max(16, n_features * 1.8), 6)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 设置坐标轴范围
    padding = max(np.abs(total_pos) * 0.2, np.abs(total_neg) * 0.2)
    min_x = min(np.min(pos_features[:, 0].astype(float)) if len(pos_features) > 0 else out_value, base_value) - padding
    max_x = max(np.max(neg_features[:, 0].astype(float)) if len(neg_features) > 0 else out_value, base_value) + padding
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-0.5, 0.15)
    ax.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False, labeltop=True, labelbottom=False)
    plt.locator_params(axis="x", nbins=12)

    # 定义条形参数
    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    # 绘制负贡献条
    def draw_bars(out_value, features, feature_type, width_separators, width_bar):
        rects, segs = [], []
        pre_val = out_value
        for idx, feat in enumerate(features):
            if feature_type == "positive":
                left = float(feat[0])
                right = pre_val
                pre_val = left
                sep_indent = np.abs(width_separators)
                sep_pos = left
                colors = ["#FF0D57", "#FFC3D5"]
            else:
                left = pre_val
                right = float(feat[0])
                pre_val = right
                sep_indent = -np.abs(width_separators)
                sep_pos = right
                colors = ["#1E88E5", "#D1E6FA"]
            # 条形多边形
            if idx == 0:
                points = [
                    [left, 0], [right, 0], [right, width_bar], [left, width_bar],
                    [left + sep_indent, width_bar / 2],
                ]
            else:
                points = [
                    [left, 0], [right, 0], [right + sep_indent * 0.9, width_bar / 2],
                    [right, width_bar], [left, width_bar], [left + sep_indent * 0.9, width_bar / 2],
                ]
            poly = plt.Polygon(points, closed=True, facecolor=colors[0], linewidth=0)
            rects.append(poly)
            # 分隔线
            sep_points = [[sep_pos, 0], [sep_pos + sep_indent, width_bar / 2], [sep_pos, width_bar]]
            sep_line = plt.Polygon(sep_points, closed=None, fill=None, edgecolor=colors[1], lw=3)
            segs.append(sep_line)
        return rects, segs

    # 负特征
    rects_neg, segs_neg = draw_bars(out_value, neg_features, "negative", width_separators, width_bar)
    for p in rects_neg + segs_neg:
        ax.add_patch(p)
    # 正特征
    rects_pos, segs_pos = draw_bars(out_value, pos_features, "positive", width_separators, width_bar)
    for p in rects_pos + segs_pos:
        ax.add_patch(p)

    # 绘制标签
    def draw_labels(features, feature_type):
        start_text = out_value
        pre_val = out_value
        colors = ["#FF0D57", "#FFC3D5"] if feature_type == "positive" else ["#1E88E5", "#D1E6FA"]
        alignment = "right" if feature_type == "positive" else "left"
        sign = 1 if feature_type == "positive" else -1

        # 初始垂直线
        if feature_type == "positive":
            line = lines.Line2D([pre_val, pre_val], [0, -0.18], lw=1, alpha=0.5, color=colors[0])
            ax.add_line(line)

        total_effect = np.abs(total_neg) + total_pos
        box_end = out_value
        for feat in features:
            contr = np.abs(float(feat[0]) - pre_val) / total_effect
            if contr < min_perc:
                break
            val = float(feat[0])
            # 构建标签文本
            if feat[1] == "":
                text = feat[2]
            else:
                text = f"{feat[2]} = {feat[1]}"
            # 添加标签
            va = "top" if text_rotation != 0 else "baseline"
            label = ax.text(
                start_text - sign * offset_text,
                -0.15,
                text,
                fontsize=10,
                color=colors[0],
                ha=alignment,
                va=va,
                rotation=text_rotation,
            )
            # 获取文本尺寸（需立即渲染）
            fig.canvas.draw()
            bbox = label.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax.transData.inverted())
            if feature_type == "positive":
                box_end_ = bbox_data.get_points()[0][0]
            else:
                box_end_ = bbox_data.get_points()[1][0]

            # 绘制连接线
            if (sign * box_end_) > (sign * val):
                # 标签位置合适，直接连接
                line = lines.Line2D([val, val], [0, -0.18], lw=1, alpha=0.5, color=colors[0])
                ax.add_line(line)
                start_text = val
                box_end = val
            else:
                # 需要折线
                line = lines.Line2D([val, box_end_, box_end_], [0, -0.08, -0.18], lw=1, alpha=0.5, color=colors[0])
                ax.add_line(line)
                start_text = box_end_
                box_end = box_end_
            pre_val = val

        # 添加底纹区域
        path_points = [[out_value, 0], [pre_val, 0], [box_end, -0.08], [box_end, -0.2], [out_value, -0.2], [out_value, 0]]
        path = Path(path_points)
        patch = PathPatch(path, facecolor="none", edgecolor="none")
        ax.add_patch(patch)
        # 创建渐变色
        if feature_type == "positive":
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", np.array([(255,13,87), (255,255,255)])/255.0)
        else:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", np.array([(30,136,229), (255,255,255)])/255.0)
        extent = [out_value, box_end, 0, -0.31]
        _, Z2 = np.meshgrid(np.linspace(0,10), np.linspace(-10,10))
        im = ax.imshow(Z2, interpolation="quadric", cmap=cmap, vmax=0.01, alpha=0.3,
                       origin="lower", extent=extent, clip_path=patch, clip_on=True, aspect="auto")
        im.set_clip_path(patch)

    if len(neg_features) > 0:
        draw_labels(neg_features, "negative")
    if len(pos_features) > 0:
        draw_labels(pos_features, "positive")

    # 绘制基准值和输出值
    # 基准值
    line = lines.Line2D([base_value, base_value], [0.13, 0.25], lw=2, color="#F2F2F2")
    ax.add_line(line)
    ax.text(base_value, 0.33, "base value", fontsize=10, alpha=0.5, ha="center")
    # 输出值
    line = lines.Line2D([out_value, out_value], [0, 0.24], lw=2, color="#F2F2F2")
    ax.add_line(line)
    ax.text(out_value, 0.25, f"{out_value:.2f}", fontsize=12, weight="bold", ha="center", bbox=dict(facecolor="white", edgecolor="white"))
    ax.text(out_value, 0.33, "f(x)", fontsize=10, alpha=0.5, ha="center")
    # higher/lower 指示
    ax.text(out_value - offset_text, 0.405, "higher", fontsize=11, color="#FF0D57", ha="right")
    ax.text(out_value + offset_text, 0.405, "lower", fontsize=11, color="#1E88E5", ha="left")
    ax.text(out_value, 0.4, r"$\leftarrow$", fontsize=11, color="#1E88E5", ha="center")
    ax.text(out_value, 0.425, r"$\rightarrow$", fontsize=11, color="#FF0D57", ha="center")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

# ========== 预测与绘图 ==========
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
    input_df = pd.DataFrame([input_dict])[feature_names]
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

    # 计算 SHAP 值
    shap_values = explainer.shap_values(input_scaled)[0]

    # 使用自定义力图绘制函数
    st.subheader("🔍 影响风险的关键因素（SHAP 力图）")
    st.markdown("下图展示了每个特征对当前患者 AKI 风险的贡献：**红色条表示推高风险**，**蓝色条表示降低风险**。")

    # 可调整参数：min_perc 控制显示的特征数量，offset_factor 控制标签间距，text_rotation 控制旋转角度
    fig = draw_custom_force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values,
        features=input_df.iloc[0].values,
        feature_names=label_list,
        text_rotation=30,
        min_perc=0.02,
        offset_factor=0.04,
        dpi=150,
    )

    # 保存为 PNG 并显示
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)

    st.caption("注：本预测结果基于回顾性研究模型，仅供参考，不能替代临床医生判断。")
    if not chinese_available:
        st.caption("（当前使用英文标签，因云端环境缺少中文字体。如需中文，可考虑部署至支持中文的环境。）")
