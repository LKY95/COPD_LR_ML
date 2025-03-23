# main.py
import streamlit as st
import pandas as pd
import os
from joblib import load

# 加载模型、标准化器和特征名称
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载必要的文件
try:
    model = load(os.path.join(current_dir, 'logistic_regression_model.pkl'))
    scaler = load(os.path.join(current_dir, 'standard_scaler.pkl'))
    feature_names = load(os.path.join(current_dir, 'feature_names.pkl'))
except Exception as e:
    st.error(f"初始化失败: {str(e)}")
    st.stop()

# 配置页面
st.set_page_config(page_title="COPD预测系统", layout="wide")
st.title("COPD风险预测系统")
st.markdown("""
**说明**：本系统通过呼吸模式特征预测慢性阻塞性肺疾病（COPD）风险  
输入患者特征后，点击预测按钮获取结果
""")

# 侧边栏输入
with st.sidebar:
    st.header("患者基本信息")
    # 基本信息
    age = st.number_input("年龄 (age)",
                          min_value=18,
                          max_value=89,
                          value=59,
                          help="患者年龄，范围：18-89岁")

    gender = st.radio("性别",
                      options=[("女性", 0), ("男性", 1)],
                      format_func=lambda x: x[0])[1]

    smoke = st.radio("吸烟史",
                     options=[("无", 0), ("有", 1)],
                     format_func=lambda x: x[0])[1]

    # 坐姿呼吸特征
    st.header("静坐呼吸特征")
    with st.expander("展开坐姿参数"):
        # 修改后的连续变量参数
        r1 = st.slider("吸呼比 (r1)",
                       min_value=37.27,
                       max_value=82.54,
                       value=60.0,
                       help="静坐吸呼比 范围：37.27-82.54")

        tt1 = st.slider("呼气时间比 (tt1)",
                        min_value=3.77,
                        max_value=23.87,
                        value=10.81,
                        help="静坐平均呼气峰流量时间比 范围：3.77-23.87")

        nl12 = st.slider("第二模态指数 (nl12)",
                         min_value=1.74,
                         max_value=21.02,
                         value=5.01,
                         step=0.1,
                         format="%.2f",
                         help="静坐第二模态非线性指数 范围：1.74-21.02")

    # 站立呼吸特征
    st.header("静站呼吸特征")
    with st.expander("展开站姿参数"):
        f2 = st.slider("呼吸频率 (f2)",
                       min_value=9.63,
                       max_value=29.66,
                       value=20.19,
                       help="静站平均呼吸频率 范围：9.63-29.66")

        r2 = st.slider("吸呼比 (r2)",
                       min_value=40.15,
                       max_value=87.38,
                       value=60.92,
                       help="静站平均吸呼比 范围：40.15-87.38")

        nl21 = st.slider("第一模态指数 (nl21)",
                         min_value=0.66,
                         max_value=26.49,
                         value=6.60,
                         step=0.1,
                         format="%.2f",
                         help="静站第一模态非线性指数 范围：0.66-26.49")

        omega_mean21 = st.slider("第一主频 (omega_mean21)",
                                 min_value=13.91,
                                 max_value=57.83,
                                 value=31.47,
                                 step=0.1,
                                 format="%.2f",
                                 help="静站第一模态主频 范围：13.91-57.83")

# 构建特征数据框
input_data = pd.DataFrame([{
    'smoke': smoke,
    'age': age,
    'r2': r2,
    'tt1': tt1,
    'gender': gender,
    'nl21': nl21,
    'nl12': nl12,
    'r1': r1,
    'omega_mean21': omega_mean21,
    'f2': f2
}], columns=feature_names)

# 数据标准化
input_data_scaled = scaler.transform(input_data)

# 预测和展示
if st.button("开始预测"):
    try:
        proba = model.predict_proba(input_data_scaled)[0][1]
        prediction = "高危 (COPD)" if proba >= 0.5 else "低危 (非COPD)"

        st.subheader("预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("风险评估",
                      f"{prediction}",
                      f"{proba * 100:.1f}% 概率")
        with col2:
            st.write("""
            **结果解读**：  
            - 高风险 (≥50%概率): 建议进行肺功能检查  
            - 低风险 (<50%概率): 建议保持健康监测
            """)
    except Exception as e:
        st.error(f"预测失败: {str(e)}")

# 注意事项
st.markdown("---")
st.warning("""
**临床使用注意事项**：  
1. 本预测结果仅供参考，不能替代专业医疗诊断  
2. 高风险患者建议结合肺功能检查确认  
3. 系统预测准确率：验证集AUC=0.80
""")