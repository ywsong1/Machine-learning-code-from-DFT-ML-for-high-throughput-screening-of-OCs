import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import joblib

element_properties = {
    'Cu': {'atomic_number': 29, 'electronegativity': 1.9, 'atomic_radius': 135, 'electron_affinity': 1.3, 'chemical_potential': -3.71},
    'Mg': {'atomic_number': 12, 'electronegativity': 1.31, 'atomic_radius': 150, 'electron_affinity': 0.0, 'chemical_potential': -1.417},
    'Al': {'atomic_number': 13, 'electronegativity': 1.61, 'atomic_radius': 125, 'electron_affinity': 0.43, 'chemical_potential': -3.66},
    'Co': {'atomic_number': 27, 'electronegativity': 1.88, 'atomic_radius': 135, 'electron_affinity': 0.66, 'chemical_potential': -7.078},
    'Ca': {'atomic_number': 20, 'electronegativity': 1.00, 'atomic_radius': 180, 'electron_affinity': 0.02, 'chemical_potential': -2.902},
    'Fe': {'atomic_number': 26, 'electronegativity': 1.83, 'atomic_radius': 140, 'electron_affinity': 0.16, 'chemical_potential': -8.499},
    'Mn': {'atomic_number': 25, 'electronegativity': 1.55, 'atomic_radius': 140, 'electron_affinity': -0.5, 'chemical_potential': -8.898},
    'Ni': {'atomic_number': 28, 'electronegativity': 1.91, 'atomic_radius': 135, 'electron_affinity': 1.16, 'chemical_potential': -5.587},
    'Cd': {'atomic_number': 48, 'electronegativity': 1.69, 'atomic_radius': 155, 'electron_affinity': 0.0, 'chemical_potential': -0.861},
    'V': {'atomic_number': 23, 'electronegativity': 1.63, 'atomic_radius': 135, 'electron_affinity': 0.53, 'chemical_potential': -8.898},
    'O': {'atomic_number': 8, 'electronegativity': 3.44, 'atomic_radius': 60, 'electron_affinity': 1.46, 'chemical_potential': -4.523},
    'In': {'atomic_number': 49, 'electronegativity': 1.78, 'atomic_radius': 155, 'electron_affinity': 0.0, 'chemical_potential': -2.609},
    'Zn': {'atomic_number': 30, 'electronegativity': 1.65, 'atomic_radius': 135, 'electron_affinity': 0.0, 'chemical_potential': -1.157},
    'Ti': {'atomic_number': 22, 'electronegativity': 1.54, 'atomic_radius': 140, 'electron_affinity': 0.08, 'chemical_potential': -7.702},
    'Cr': {'atomic_number': 24, 'electronegativity': 1.66, 'atomic_radius': 140, 'electron_affinity': 0.67, 'chemical_potential': -9.463},
    'Ga': {'atomic_number': 31, 'electronegativity': 1.81, 'atomic_radius': 135, 'electron_affinity': 0.41, 'chemical_potential': -2.902},
    # Add other elements if necessary...
}


# 化学式解析函数
def parse_chemical_formula(formula):
    """
    解析化学式，提取元素及其配比
    """
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    pairs = re.findall(pattern, formula)
    elements = {}
    for element, amt in pairs:
        elements[element] = float(amt) if amt else 1.0  # 若未指定数量，默认为1
    return elements

# 特征向量化函数
def chemical_formula_to_feature_vector(formula, element_properties):
    """
    将化学式转换为特征向量
    """
    elements = parse_chemical_formula(formula)
    feature_vector = np.zeros(5)  # 根据实际属性数量调整
    total_weight = sum(elements.values())
    for element, amt in elements.items():
        props = element_properties.get(element, {})
        if props and len(props.values()) == 5:  # 确保每个元素有5个属性
            feature_vector += np.array(list(props.values())) * (amt / total_weight)
        else:
            print(f"Warning: Properties for element {element} not found or incomplete.")
    return feature_vector

# 加载数据
df = pd.read_excel('dataset oxygen vancy.xlsx', skiprows=1)  # 跳过第一行标签

# 特征工程：化学式转换为特征向量
features = [chemical_formula_to_feature_vector(f, element_properties) for f in df.iloc[:, 1]]

# 创建 DataFrame 用于保存原始结果和预测结果
df_results = pd.DataFrame()

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(features, df.iloc[:, 2:4].values, test_size=0.25, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义神经网络模型的构建函数
def create_nn_model(hidden_layers=(32,), optimizer='adam'):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=len(X_train[0]), activation='relu'))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(2))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 创建 KerasRegressor 包装器用于 GridSearchCV
nn_model = KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=32, verbose=0)

# 定义超参数的网格，包括隐藏层神经元个数的搜索范围
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'batch_size': [16, 32],
    'epochs': [50, 100],
    'hidden_layers': [(i,) for i in range(2, 11)] + [(i, i) for i in range(2, 11)],
}



# 5折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索初始化
grid_search = GridSearchCV(nn_model, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid_result = grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数
print(f"Best Hyperparameters: {grid_result.best_params_}")

# 保存最佳模型
best_model = grid_result.best_estimator_.model
best_model.save('best_nn_model.h5')

# 预测
y_pred = best_model.predict(X_test_scaled)

# 评估指标计算
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())

# 输出模型的评估指标
print(f"R2 Score: {r2}, "
      f"MAE: {mae}, "
      f"Pearson Correlation: {pearson_corr}")

# 将原始数据和预测结果添加到 DataFrame
df_results = pd.DataFrame()
df_results['Original_Target_1'] = y_test[:, 0]
df_results['Predicted_Target_1'] = y_pred[:, 0]
df_results['Original_Target_2'] = y_test[:, 1]
df_results['Predicted_Target_2'] = y_pred[:, 1]

# 将结果保存到 Excel 表
df_results.to_excel('nn_prediction_results.xlsx', index=False)
