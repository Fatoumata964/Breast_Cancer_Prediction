import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature  import VectorAssembler
from pyspark.ml.feature import (OneHotEncoder, StringIndexer)
from pyspark.ml import Pipeline

def process1(spark, path):
    df = spark.read.csv(path, inferSchema = True, header = True)
    df = df.drop("_c32")
    train, test = df.randomSplit([0.7, 0.3], seed=7)
    catCols = [x for (x, dataType) in train.dtypes if dataType == "string"]
    numCols = [x for (x, dataType) in train.dtypes if dataType == "double"]
    string_indexer = [StringIndexer(inputCol = x, outputCol=x+"_StringIndexer", handleInvalid="skip")
                    for x in catCols]
    assemblerInput = [x for x in numCols]
    vector_assembler = VectorAssembler(inputCols=assemblerInput, outputCol="VectorAssembler_features")
    stages = []
    stages += string_indexer
    stages += [vector_assembler]
    pipeline = Pipeline().setStages(stages)
    model = pipeline.fit(train)

    pp_df =model.transform(train)
    pp_dftest =model.transform(test)
    data = pp_df.select(F.col("VectorAssembler_features").alias("features"),
                    F.col("diagnosis_StringIndexer").alias("label"))

    datatest = pp_dftest.select(F.col("VectorAssembler_features").alias("features"),
                    F.col("diagnosis_StringIndexer").alias("label"))
    return data, datatest
