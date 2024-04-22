import numpy as np
import pandas as pd
from pyDOE2 import doe_lhs
from sklearn.preprocessing import MinMaxScaler

# 파라미터 설정
num_vars = 8  # 변수의 수
num_experiments = 200  # 실험 횟수
lower_bounds = 1
upper_bounds = 1000

# 라틴 하이퍼큐브 샘플링으로 초기 설계 생성
lhs_design = doe_lhs.lhs(n=num_vars, samples=num_experiments, criterion='maximin')

# 스케일러를 사용하여 범위 1 ~ 1000로 조정
scaler = MinMaxScaler(feature_range=(lower_bounds, upper_bounds))
scaled_design = scaler.fit_transform(lhs_design)

# 결과를 DataFrame으로 변환
column_names = [f"Variable_{i+1}" for i in range(num_vars)]
experiment_design = pd.DataFrame(scaled_design, columns=column_names).round()
experiment_design.head()
experiment_design.to_clipboard()
