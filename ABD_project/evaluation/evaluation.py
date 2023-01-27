from sklearn.metrics import confusion_matrix
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate(pred):
  evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR")
  acc = evaluator.evaluate(pred)
  
  print("Prediction Accuracy: ", acc)

  y_pred=pred.select("prediction").collect()
  y_orig=pred.select("label").collect()

  cm = confusion_matrix(y_orig, y_pred)
  print("Confusion Matrix:")
  print(cm) 
