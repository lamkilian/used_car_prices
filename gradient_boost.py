import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# This file will contain test runs for xgboost
# we try this out since it was presented in homework 5 and it might give some better results

data = pd.read_csv('data/data_final.csv')
print(data.shape)

for col in data.columns:
  if np.sum(data[col].isna()) > 0:
    print(col, np.sum(data[col].isna()))
# For the first tests we will just try to make the boolean decision whether the train was delayed or not
# So we will create a new column from dst_arrival_delay that just checks if the value is larger than 0 or not
X_train_all = data.drop('price', axis=1)
y_train_all = data['price']

X_train_all, X_test, y_train_all, y_test = train_test_split(X_train_all, y_train_all, random_state = 13, test_size = 0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, random_state = 13, test_size = 0.20)

# Since this is just some basic testing we won't use any cross validation

print(X_val.shape)
print(X_train.shape)
print(y_val.shape)
print(y_train.shape)

dtrain_all = xgb.DMatrix(X_train_all, label=y_train_all)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

def xgb_rmse(preds, labels, threshold=0.5):
  # This function will be used so the model actually optimizes the f-score and not the accuracy, as our real data set is heavily imbalanced
  prices = labels.get_label()
  #for ind, pred in enumerate(preds):
  #  print(pred, prices[ind])
  return 'rmse', root_mean_squared_error(prices, preds)

optimal_params = 0
best_f_score = np.inf 
best_iters = 0

for max_depth in range(3,10):
  for iters in range(100, 10000, 200):
    params = {
      "booster": 'gbtree', 
      "eta": 0.3,
      "objective": 'reg:squarederror',
      "seed":111,
      "max_depth": max_depth,
      "sub_sampling": 0.9
    }

    eval_list = [(dtrain, 'train'), (dval, 'eval')]

    bst = xgb.train(params, dtrain, iters, evals=eval_list, custom_metric=xgb_rmse, maximize=False)

    preds = (bst.predict(dtest) > 0.5).astype(int) # We could also experiment with different cutoff points
    current_score = root_mean_squared_error(y_test, preds) 
    print(current_score)
    if current_score < best_f_score:
      bst.save_model('model.json')
      optimal_params = params
      best_f_score = current_score
      best_iters = iters


eval_list = [(dtrain_all, 'train'), (dtest, 'eval')]
bst = xgb.train(optimal_params, dtrain_all, best_iters, evals=eval_list, custom_metric=xdb_rmse, maximize=False)
bst.save_model('model.json')
