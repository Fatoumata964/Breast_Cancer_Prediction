from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
from pyspark.ml.feature  import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns

def pca_viz(df):
    df = df.drop("_c32")
    inputs = df.drop('diagnosis')
    inp_cols = [x for (x, dataType) in inputs.dtypes]
    assembler = VectorAssembler( inputCols=inp_cols, outputCol='features')
    df1 = assembler.transform(df)
    pca = PCA(k=2, inputCol='features', outputCol='pcaFeature')
    model = pca.fit(df1)
    result= model.transform(df1).select("pcaFeature")
    pandasDF = result.toPandas()
    dataX = []
    dataY = []
    for vec in pandasDF.values:
      dataX.extend([vec[0][0]])
      dataY.extend([vec[0][1]])
    plt.scatter(dataX, dataY)
    plt.show

def plot_corr_matrix(df):
    df = df.drop("_c32")
    inputs = df.drop('diagnosis')
    inp_cols = [x for (x, dataType) in inputs.dtypes]
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=inputs.columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    corrmatrix = matrix.toArray().tolist()
    fig=plt.figure(234)
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    ax.set_xticklabels(['']+inp_cols)
    ax.set_yticklabels(['']+inp_cols)
    cax=ax.matshow(corrmatrix,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.show()
