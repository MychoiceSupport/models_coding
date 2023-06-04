from utils import *

train_len = int(0.8 * len(X))
test_len= int(0.1 * len(X))
valid_len= len(X) - train_len - test_len
current_data = X[train_len: train_len + valid_len]
current_label = y[train_len: train_len + valid_len]
predict_label = X[train_len: train_len + valid_len, 5, :]

get_metrics(current_label, predict_label)
