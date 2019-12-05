from classification_util import ClassificationUtil

gildong = ClassificationUtil()

gildong.read('output.csv')
gildong.show()
#gildong.myplot('Day2', 'Time', 'Place')
gildong.run_svm(['Day2', 'Time'], 'Place')
