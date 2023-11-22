from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Pipeline():
    def __init__(self) -> None:
        self.load_data()
        self.preprocess()

    def filter_by_freq(self, data, col, min_freq=0.2):
        counts = data[col].value_counts()
        big_counts = counts[counts > counts.quantile(min_freq)].index
        return data[data[col].isin(big_counts)]

    def load_data(self):
        data = pd.read_csv('./app/data/car_price_prediction.csv', index_col='ID')
        data.drop(['Levy', 'Model'], axis=1, inplace=True)
        self.data = data

    def preprocess(self):
        data = self.data.copy()
        target = 'Manufacturer'

        data.Mileage = data.Mileage.replace('km', '', regex=True).astype(int)
        
        cat_cols = [col for col in data.columns if data[col].dtype in [object, str]]
        num_cols = [col for col in data.columns if col not in cat_cols]

        for col in cat_cols:
            data = self.filter_by_freq(data, col)
        data = self.filter_by_freq(data, target, 0.5)

        cat_cols.remove(target)
        data = pd.get_dummies(data, columns=cat_cols)

        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.X = data.drop(target, axis=1).to_numpy()
        self.y = data[target].to_numpy()

    def train(self, estimator, test_size=0.2):      
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            random_state=92)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        estimator.fit(X_train, y_train)

        print('Train score', estimator.score(X_train, y_train))
        print('Test score', estimator.score(X_test, y_test))

        self.scaler = scaler
        self.estimator = estimator

    def predict(self, X):
        assert self.estimator, 'pipeline não foi treinada'

        X = self.scaler.transform(X)
        return self.estimator.predict(X)
    
    def score(self, test_size=0.2):
        _, X_test, _, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            random_state=92)
        preds = self.predict(X_test)
        acc = round(accuracy_score(preds, y_test), 4)
        macro_pre = round(precision_score(preds, y_test, average='macro'), 4)
        macro_rec = round(recall_score(preds, y_test, average='macro'), 4)
        macro_f1 = round(f1_score(preds, y_test, average='macro'), 4)
        micro_f1 = round(f1_score(preds, y_test, average='micro'), 4)

        cm = confusion_matrix(y_test, preds, labels=self.estimator.classes_)
        plt.figure(figsize=(20,15))
        sns.heatmap(cm, annot=True)
        plt.savefig('app/static/images/cm.png')

        return acc, macro_pre, macro_rec, macro_f1, micro_f1
    
def fit_knn(n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors)
    pipe = Pipeline()
    pipe.train(knn)
    
    return pipe.score()

def fit_mlp(hidden_sizes=(100,), activation='relu', lr=0.001):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_sizes,
                        activation=activation,
                        learning_rate_init=lr)
    pipe = Pipeline()
    pipe.train(mlp)
    return pipe.score()

def fit_rf(n_estimators=100, max_depth=None):
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth)
    pipe = Pipeline()
    pipe.train(rf)
    return pipe.score()

def fit_svm(C=1.0, kernel='rbf', poly_degree=3):
    svm = SVC(C=C,
             kernel=kernel,
             degree=poly_degree)
    pipe = Pipeline()
    pipe.train(svm)
    return pipe.score()