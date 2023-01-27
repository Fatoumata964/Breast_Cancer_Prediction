from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def randomForestClassifier(data):
  rfc = RandomForestClassifier(featuresCol="features", labelCol="label")

  paramGrid = ParamGridBuilder().addGrid(rfc.maxDepth, [2, 5, 10, 20, 30]) \
                                  .addGrid(rfc.maxBins, [10, 20, 40, 80, 100]) \
                                  .addGrid(rfc.numTrees, [5, 20, 50, 100, 500]) \
                                  .build() 
  crossval = CrossValidator(estimator=rfc,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                            numFolds=3) 
  rfcModel = crossval.fit(data)
  bestModel = rfcModel.bestModel

  model = RandomForestClassifier(featuresCol="features", labelCol="label",maxDepth=bestModel._java_obj.getMaxDepth(), maxBins = bestModel._java_obj.getMaxBins(),\
     numTrees = bestModel._java_obj.getNumTrees()).fit(data)
  return model
