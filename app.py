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

# ========== 以下是您提供的 SHAP 绘图函数 ==========
# （为节省篇幅，这里只列出修改过的关键函数，其余与您提供的完全相同）
# 注意：draw_output_element 已去除白色背景框

def draw_output_element(out_name, out_value, ax):
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2.0, color="#F2F2F2")
    line.set_clip_on(False)
    ax.add_line(line)
    # 输出数值，无背景框
    plt.text(out_value, 0.25, f"{out_value:.2f}", fontsize=12, weight="bold", ha="center")
    plt.text(out_value, 0.33, out_name, fontsize=10, alpha=0.5, ha="center")

def draw_base_element(base_value, ax):
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2.0, color="#F2F2F2")
    line.set_clip_on(False)
    ax.add_line(line)
    plt.text(base_value, 0.33, "base value", fontsize=10, alpha=0.5, ha="center")

def draw_higher_lower_element(out_value, offset_text):
    plt.text(out_value - offset_text, 0.405, "higher", fontsize=11, color="#FF0D57", ha="right")
    plt.text(out_value + offset_text, 0.405, "lower", fontsize=11, color="#1E88E5", ha="left")
    plt.text(out_value, 0.4, r"$\leftarrow$", fontsize=11, color="#1E88E5", ha="center")
    plt.text(out_value, 0.425, r"$\rightarrow$", fontsize=11, color="#FF0D57", ha="center")

def update_axis_limits(ax, total_pos, pos_features, total_neg, neg_features, base_value, out_value):
    ax.set_ylim(-0.5, 0.15)
    padding = np.max([np.abs(total_pos) * 0.2, np.abs(total_neg) * 0.2])
    if len(pos_features) > 0:
        min_x = min(np.min(pos_features[:, 0].astype(float)), base_value) - padding
    else:
        min_x = out_value - padding
    if len(neg_features) > 0:
        max_x = max(np.max(neg_features[:, 0].astype(float)), base_value) + padding
    else:
        max_x = out_value + padding
    ax.set_xlim(min_x, max_x)
    plt.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False, labeltop=True, labelbottom=False)
    plt.locator_params(axis="x", nbins=12)
    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
        if key != "top":
            spine.set_visible(False)

def draw_bars(out_value, features, feature_type, width_separators, width_bar):
    # 与您提供的代码完全相同，此处省略以节省篇幅
    rectangle_list = []
    separator_list = []
    pre_val = out_value
    for index, feature_values in enumerate(features):
        if feature_type == "positive":
            left_bound = float(feature_values[0])
            right_bound = pre_val
            pre_val = left_bound
            separator_indent = np.abs(width_separators)
            separator_pos = left_bound
            colors = ["#FF0D57", "#FFC3D5"]
        else:
            left_bound = pre_val
            right_bound = float(feature_values[0])
            pre_val = right_bound
            separator_indent = -np.abs(width_separators)
            separator_pos = right_bound
            colors = ["#1E88E5", "#D1E6FA"]
        if index == 0:
            if feature_type == "positive":
                points_rectangle = [
                    [left_bound, 0],
                    [right_bound, 0],
                    [right_bound, width_bar],
                    [left_bound, width_bar],
                    [left_bound + separator_indent, (width_bar / 2)],
                ]
            else:
                points_rectangle = [
                    [right_bound, 0],
                    [left_bound, 0],
                    [left_bound, width_bar],
                    [right_bound, width_bar],
                    [right_bound + separator_indent, (width_bar / 2)],
                ]
        else:
            points_rectangle = [
                [left_bound, 0],
                [right_bound, 0],
                [right_bound + separator_indent * 0.90, (width_bar / 2)],
                [right_bound, width_bar],
                [left_bound, width_bar],
                [left_bound + separator_indent * 0.90, (width_bar / 2)],
            ]
        poly = plt.Polygon(points_rectangle, closed=True, fill=True, facecolor=colors[0], linewidth=0)
        rectangle_list.append(poly)
        points_separator = [
            [separator_pos, 0],
            [separator_pos + separator_indent, (width_bar / 2)],
            [separator_pos, width_bar],
        ]
        sep_line = plt.Polygon(points_separator, closed=None, fill=None, edgecolor=colors[1], lw=3)
        separator_list.append(sep_line)
    return rectangle_list, separator_list

def draw_labels(fig, ax, out_value, features, feature_type, offset_text, total_effect=0, min_perc=0.05, text_rotation=0):
    start_text = out_value
    pre_val = out_value
    if feature_type == "positive":
        colors = ["#FF0D57", "#FFC3D5"]
        alignment = "right"
        sign = 1
    else:
        colors = ["#1E88E5", "#D1E6FA"]
        alignment = "left"
        sign = -1
    if feature_type == "positive":
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1.0, alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val
    box_end = out_value
    val = out_value
    for feature in features:
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break
        val = float(feature[0])
        if feature[1] == "":
            text = feature[2]
        else:
            text = feature[2] + " = " + feature[1]
        if text_rotation != 0:
            va_alignment = "top"
        else:
            va_alignment = "baseline"
        text_out_val = plt.text(
            start_text - sign * offset_text,
            -0.15,
            text,
            fontsize=10,
            color=colors[0],
            horizontalalignment=alignment,
            va=va_alignment,
            rotation=text_rotation,
        )
        # 获取文本尺寸
        fig.canvas.draw()
        box_size = text_out_val.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        if feature_type == "positive":
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, -0.18]])
            line = lines.Line2D(x, y, lw=1.0, alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val
        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end], [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1.0, alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end
        pre_val = val
    # 添加底纹
    extent_shading = [out_value, box_end, 0, -0.31]
    path = [[out_value, 0], [pre_val, 0], [box_end, -0.08], [box_end, -0.2], [out_value, -0.2], [out_value, 0]]
    path = Path(path)
    patch = PathPatch(path, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    if feature_type == "positive":
        colors_cmap = np.array([(255, 13, 87), (255, 255, 255)]) / 255.0
    else:
        colors_cmap = np.array([(30, 136, 229), (255, 255, 255)]) / 255.0
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("cm", colors_cmap)
    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = ax.imshow(
        Z2,
        interpolation="quadric",
        cmap=cm,
        vmax=0.01,
        alpha=0.3,
        origin="lower",
        extent=extent_shading,
        clip_path=patch,
        clip_on=True,
        aspect="auto",
    )
    im.set_clip_path(patch)
    return fig, ax

def format_data(data):
    # 与您提供的代码完全相同
    neg_features = np.array(
        [
            [data["features"][x]["effect"], data["features"][x]["value"], data["featureNames"][x]]
            for x in data["features"].keys()
            if data["features"][x]["effect"] < 0
        ]
    )
    neg_features = np.array(sorted(neg_features, key=lambda x: float(x[0]), reverse=False))
    pos_features = np.array(
        [
            [data["features"][x]["effect"], data["features"][x]["value"], data["featureNames"][x]]
            for x in data["features"].keys()
            if data["features"][x]["effect"] >= 0
        ]
    )
    pos_features = np.array(sorted(pos_features, key=lambda x: float(x[0]), reverse=True))
    if data["link"] == "identity":
        def convert_func(x): return x
    elif data["link"] == "logit":
        def convert_func(x): return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f"Unrecognized link function: {data['link']}")
    neg_val = data["outValue"]
    for i in neg_features:
        val = float(i[0])
        neg_val = neg_val + np.abs(val)
        i[0] = convert_func(neg_val)
    total_neg = np.max(neg_features[:, 0].astype(float)) - np.min(neg_features[:, 0].astype(float)) if len(neg_features) > 0 else 0
    pos_val = data["outValue"]
    for i in pos_features:
        val = float(i[0])
        pos_val = pos_val - np.abs(val)
        i[0] = convert_func(pos_val)
    total_pos = np.max(pos_features[:, 0].astype(float)) - np.min(pos_features[:, 0].astype(float)) if len(pos_features) > 0 else 0
    data["outValue"] = convert_func(data["outValue"])
    data["baseValue"] = convert_func(data["baseValue"])
    return neg_features, total_neg, pos_features, total_pos

def draw_additive_plot(data, figsize, show, text_rotation=0, min_perc=0.05):
    """整合后的绘图主函数，已修改 offset_text 系数为 0.08 以延长连接线"""
    if show is False:
        plt.ioff()
    neg_features, total_neg, pos_features, total_pos = format_data(data)
    base_value = data["baseValue"]
    out_value = data["outValue"]
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.08  # 原为0.04，增大为0.08
    fig, ax = plt.subplots(figsize=figsize)
    update_axis_limits(ax, total_pos, pos_features, total_neg, neg_features, base_value, out_value)
    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200
    # 负贡献条
    rects_neg, segs_neg = draw_bars(out_value, neg_features, "negative", width_separators, width_bar)
    for p in rects_neg + segs_neg:
        ax.add_patch(p)
    # 正贡献条
    rects_pos, segs_pos = draw_bars(out_value, pos_features, "positive", width_separators, width_bar)
    for p in rects_pos + segs_pos:
        ax.add_patch(p)
    total_effect = np.abs(total_neg) + total_pos
    # 负标签
    fig, ax = draw_labels(fig, ax, out_value, neg_features, "negative", offset_text,
                          total_effect, min_perc=min_perc, text_rotation=text_rotation)
    # 正标签
    fig, ax = draw_labels(fig, ax, out_value, pos_features, "positive", offset_text,
                          total_effect, min_perc=min_perc, text_rotation=text_rotation)
    # 图例、基准值、输出值
    draw_higher_lower_element(out_value, offset_text)
    draw_base_element(base_value, ax)
    draw_output_element(data["outNames"][0], out_value, ax)
    if data["link"] == "logit":
        plt.xscale("logit")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(style="plain")
    if show:
        plt.show()
    else:
        return plt.gcf()

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

    # 构造 data 字典供 draw_additive_plot 使用
    data = {
        "outValue": float(explainer.expected_value + shap_values.sum()),  # 转换为 Python float
        "baseValue": float(explainer.expected_value),
        "features": {},
        "featureNames": label_list,
        "outNames": ["f(x)"],
        "link": "identity"
    }
    # 填充特征数据（特征值已格式化为两位小数）
    for i, (name, val, shap_val) in enumerate(zip(label_list, input_df.iloc[0].values, shap_values)):
        data["features"][i] = {
            "effect": float(shap_val),
            "value": f"{val:.2f}"   # 保留两位小数
        }

    # 计算图形尺寸：根据特征数量自适应宽度
    n_features = len(label_list)
    figsize = (max(16, n_features * 1.8), 6)

    # 绘制力图（text_rotation=30, min_perc=0.02）
    fig = draw_additive_plot(
        data=data,
        figsize=figsize,
        show=False,
        text_rotation=30,
        min_perc=0.02
    )

    # 在 Streamlit 中显示图形
    st.subheader("🔍 影响风险的关键因素（SHAP 力图）")
    st.markdown("下图展示了每个特征对当前患者 AKI 风险的贡献：**红色条表示推高风险**，**蓝色条表示降低风险**。")
    st.pyplot(fig)
    plt.close(fig)

    st.caption("注：本预测结果基于回顾性研究模型，仅供参考，不能替代临床医生判断。")
    if not chinese_available:
        st.caption("（当前使用英文标签，因云端环境缺少中文字体。如需中文，可考虑部署至支持中文的环境。）")
