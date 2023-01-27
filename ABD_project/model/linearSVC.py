from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def linearSVC(data):
  lsvc = LinearSVC(labelCol="label")
  
  paramGrid = ParamGridBuilder().addGrid(lsvc.maxIter, [1, 5, 10, 20, 50, 100]) \
                                  .addGrid(lsvc.regParam, [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) \
                                  .build() 
  crossval = CrossValidator(estimator=lsvc,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                            numFolds=3) 
  lsvcModel = crossval.fit(data)
  bestModel = lsvcModel.bestModel

  model = LinearSVC(labelCol="label",regParam=bestModel._java_obj.getRegParam(), maxIter = bestModel._java_obj.getMaxIter())\
    .fit(data)
  return model
