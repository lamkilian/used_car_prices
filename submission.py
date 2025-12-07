import xgboost as xgb
import numpy as np
import pandas as pd

model = xgb.XGBRegressor()

model.load_model("model.json")

X_test = pd.read_csv('data/data_final_test.csv')
orig_data = pd.read_csv('sample_submission.csv')['id'].values

print(orig_data[:5])

y_pred = model.predict(X_test)
print(y_pred.shape)

submission = pd.DataFrame(zip(orig_data, y_pred), columns=['id', 'price'])
print(submission.shape)

submission.to_csv('submission.csv', index=False)
