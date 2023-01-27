from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def decisionTreeClassifier(data):
  dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")
  
  paramGrid = ParamGridBuilder().addGrid(dtc.maxDepth, [2, 5, 10, 20, 30]) \
                                  .addGrid(dtc.maxBins, [10, 20, 40, 80, 100]) \
                                  .build() 
  crossval = CrossValidator(estimator=dtc,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                            numFolds=3) 
  dtcModel = crossval.fit(data)
  bestModel = dtcModel.bestModel

  model = DecisionTreeClassifier(featuresCol="features", labelCol="label",maxDepth=bestModel._java_obj.getMaxDepth(), maxBins = bestModel._java_obj.getMaxBins()).fit(data)
  return model
