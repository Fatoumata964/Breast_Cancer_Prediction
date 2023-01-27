from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

def logisticRegress(data):
  lr = LogisticRegression()  
  paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [1, 5, 10, 20, 50]) \
                                  .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0, 2.0]) \
                                  .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0]) \
                                  .build() 
  crossval = CrossValidator(estimator=lr,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                            numFolds=3) 

  cvModel = crossval.fit(data)
  bestModel = cvModel.bestModel

  model = LogisticRegression(regParam=bestModel._java_obj.getRegParam(), maxIter = bestModel._java_obj.getMaxIter(), \
                                      elasticNetParam = bestModel._java_obj.getElasticNetParam()).fit(data)
  return model
