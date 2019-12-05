from ai_prediction_util import MyClassifier

# 예측 알고리즘 함수가 들어가 있는 클래스 객체 생성
gildong = MyClassifier()

# csv파일을 읽는 함수
gildong.read('output.csv')

# 읽은 csv 파일을 보여주는 함수
gildong.show()

# 가장 상관관계가 높은 요소를 확인하기 위한 히트맵 출력 함수
gildong.heatmap()

# 실행에 있어서 모든 경고를 무시
gildong.ignore_warning()

# logistic_regression, decision_tree_classifier, svm, neighbor_classifier 를 순차적으로 실행, 넘겨주는 숫자는 neighbor_classifier에서 사용
gildong.run_all(['Money','Day2'],'Place', 3)
# 각 알고리즘의 인식률 비교를 표로 보여주는 함수
gildong.draw_4_accuracy()
