import numpy as np # 수학 연산 수행을 위한 모듈
import pandas as pd # 데이터 처리를 위한 모듈
import seaborn as sns # 데이터 시각화 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈

# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

class MyClassifier:
    df = 0

    rate_svm = 0
    rate_l_l = 0
    rate_n_c = 0
    rate_d_t_c = 0

    # csv 파일을 로드함. 예)df = read("a.csv")
    def read(self, fn):
        self.df = pd.read_csv(fn)

    #
    def show(self):
        print(self.df.info())
        print(self.df.head(5))
        print(self.df.shape)

    # 예)히트맵으로 성별과 가장 상관관계가 높은 필드(발크기,
    # 몸무게, 키 등)를 알 수 있음.
    def heatmap(self):
        plt.figure(figsize=(14, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='cubehelix_r')
        plt.show()

    def ignore_warning(self):
        import warnings
        warnings.filterwarnings('ignore')

    def run_svm(self, input_cols, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = svm.SVC()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_svm = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_svm)

    def run_logistic_regression(self, input_cols, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = LogisticRegression()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_l_l = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_l_l)

    def run_neighbor_classifier(self, input_cols, target, num):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = KNeighborsClassifier(n_neighbors=num)  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_n_c = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_n_c)

    def run_decision_tree_classifier(self, input_cols, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = DecisionTreeClassifier()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_d_t_c = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_d_t_c)

    def run_all(self, input_cols, target, neighbor_num):
        self.run_logistic_regression(input_cols, target)
        self.run_decision_tree_classifier(input_cols, target)
        self.run_svm(input_cols, target)
        self.run_neighbor_classifier(input_cols, target, neighbor_num)

    def draw_4_accuracy(self):
        plt.figure(figsize=(8, 5))
        plt.title(' ')
        plt.plot(['SVM', 'Logistic_L', 'Neighbor_C', 'Decision_T_C'],
                 [self.rate_svm, self.rate_l_l, self.rate_n_c, self.rate_d_t_c],
                 label='Accuracy')
        plt.legend()
        plt.xlabel('')
        plt.ylabel('Accuracy')
        plt.show()

