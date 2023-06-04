import logging

import xgboost
from math import sin, asin, cos, radians, fabs, sqrt
from traj_dataset import *
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import joblib
# from geopy.distance import geodesic
from sklearn.metrics import make_scorer
import warnings
from sklearn.ensemble import RandomForestRegressor
import sklearn
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor
rng = np.random.RandomState(1)
X = np.load('data/all_data_X_origin.npy',allow_pickle=True)
y = np.load('data/all_data_y_origin.npy',allow_pickle=True)
# X, y = generate_raw_data()
X, y, scaler = generate_training_info(X, y)
EARTH_RADIUS = 6370860
def hav(theta):
    s = sin(theta / 2)
    return s * s


def geodesic(pos1, pos2):
    """用haversine公式计算球面两点间的距离。"""
    # 经纬度转换成弧度
    lat0 = radians(pos1[0])
    lat1 = radians(pos2[0])
    lng0 = radians(pos1[1])
    lng1 = radians(pos2[1])

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance

def geodis(y_true, y_pred):
    y_pred = scaler.inverse_transform(y_pred)
    dis = []
    for i in range(len(y_true)):
        dis.append(geodesic(y_true[i],y_pred[i]))
    return np.mean(dis)

geo_dis = make_scorer(geodis, greater_is_better=False)

def split_train_val_test(X, y, train_ratio = 0.8, test_ratio = 0.1):
    train_len = int(train_ratio * len(X))
    test_len = int(test_ratio * len(X))
    val_len = len(X) - train_len - test_len
    train_X = X[:train_len]
    train_y = y[:train_len]
    test_X = X[-test_len:]
    test_y = y[-test_len:]
    val_X = X[train_len: train_len + val_len]
    val_y = y[train_len: train_len + val_len]
    return train_X, train_y, val_X, val_y, test_X, test_y

def train(X, y, model_name):
    if model_name == 'xgboost':
        model = xgboost.XGBRegressor()
        Grid_model = GridSearchCV(
        model,
        {
            'max_depth': [1, 2, 5, 10, 20],
            'C': [20, 30, 50, 70, 100],
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring=geo_dis
        )
    elif model_name == 'SVM':
        model = SVR()
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        Grid_model = GridSearchCV(
            model,tuned_parameters,
            cv=3,
            verbose=1,
            n_jobs=-1,
            scoring=geo_dis
        )
    elif model_name == 'mlp':
        Grid_model = MLPRegressor(
            hidden_layer_sizes=(22,11), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
            random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif model_name == 'rf':
        model = RandomForestRegressor()
        params_grid = {'max_depth': np.arange(1, 20),
                      'n_estimators': np.arange(50, 1000, 50)}
        Grid_model = GridSearchCV(model, params_grid, cv=5)
    else:
        raise Exception('Please input the availble name.')
    Grid_model.fit(X.reshape(len(X),-1), y)

    print(y.shape)
    save_model(Grid_model,'data/{}.pkl'.format(model_name))
    # print('Best: %f using %s' % (Grid_model.best_score_, Grid_model.best_params_))
    return Grid_model

def save_model(model, filename):
    return joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def get_the_metrics(model, test_X, groundtruth):
    # (test_X, groundtruth, _) = batch_x
    # model = load_model(model_name)
    pred_data = model.predict(test_X.reshape(len(test_X), -1))
    # groundtruth = scaler.inverse_transform(pred_data)

    predict_data = scaler.inverse_transform(pred_data)
    print(predict_data, groundtruth)
    dis = []
    count = 0
    for i in range(len(groundtruth)):
        tmp_dis = geodesic(predict_data[i], groundtruth[i])
        dis.append(tmp_dis)
        if tmp_dis <= 150:
            count += 1
    rmse = np.sqrt(mean_squared_error(dis, np.zeros_like(dis)))
    mae = mean_absolute_error(dis, np.zeros_like(dis))
    rate_250 = count / len(test_X) * 100
    print('get the metrics:{},{},{}'.format(rmse, mae, rate_250))



if __name__ == '__main__':
    train_X, train_y, val_X, val_y, test_X, test_y = split_train_val_test(X, y)
    model = train(train_X,train_y,'rf')
    get_the_metrics(model, test_X, test_y)

