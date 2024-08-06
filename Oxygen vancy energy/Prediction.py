import pandas as pd
import joblib
from GBDT import chemical_formula_to_feature_vector, element_properties

# 加载模型
gbdt_1 = joblib.load('trained_model_point_2.pkl')
gbdt_2 = joblib.load('trained_model_point_3.pkl')

# 读取新的化学式数据
df_new = pd.read_excel('ZnABFe2O4.xlsx', usecols=[1])

# 特征工程：化学式转换为特征向量
features_new = [chemical_formula_to_feature_vector(f, element_properties) for f in df_new.iloc[:, 0]]

# 预测新的化学式的两个位点的氧空位形成能
predictions_new_1 = gbdt_1.predict(features_new)
predictions_new_2 = gbdt_2.predict(features_new)

# 将预测结果保存到预测结果.xlsx文件中
df_result = pd.DataFrame({
    'Chemical_Formula': df_new.iloc[:, 0],
    'Prediction_Point_1': predictions_new_1,
    'Prediction_Point_2': predictions_new_2
})
df_result.to_excel('氧空位形成能ZnABFe2O4.xlsx', index=False)
