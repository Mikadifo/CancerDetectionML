import os
import shutil
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import GradientBoostedTrees, RandomForest
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("project3").getOrCreate()

df = spark.read.csv("data/data.csv", header=True, inferSchema=True)
columns = df.columns

def mapDiagnosisFeature(row):
    label = 1.0 if row.diagnosis == "M" else 0.0
    features = [row[col] for col in columns if col != "diagnosis"]

    return LabeledPoint(label, features)


def randomForest():
    modelPath = "models/randomClassificationModel"
    data = df.rdd.map(mapDiagnosisFeature)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=40)
    model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=50, featureSubsetStrategy="auto", impurity='gini', maxDepth=5, maxBins=32)

    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    correct = labelsAndPredictions.filter(lambda pl: pl[0] == pl[1]).count()
    incorrect = labelsAndPredictions.filter(lambda pl: pl[0] != pl[1]).count()
    testError = incorrect / float(testData.count())
    metrics = MulticlassMetrics(labelsAndPredictions)

    print("\nTest Error = " + str(testError))
    print("\nCorrect predictions = " + str(correct))
    print("\nIncorrect predictions = " + str(incorrect))
    print("\nConfusion Matrix = ", metrics.confusionMatrix().toArray())
    print("\nF1 Score = ", metrics.fMeasure(1.0))
    print("\nPrecision = ", metrics.precision(1.0))
    print("\nRecall = ", metrics.recall(1.0))
    print("\nAccuracy = ", str((1 - testError) * 100))
    # print("Learned classification forest model:")
    # print(model.toDebugString())

    if os.path.exists(modelPath):
        print("Overwriting previous model file...")
        shutil.rmtree(modelPath)

    model.save(spark.sparkContext, modelPath)

def gradientBoosting():
    modelPath = "models/gradientBoostingModel"
    data = df.rdd.map(mapDiagnosisFeature)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)
    model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, numIterations=50, maxDepth=5)

    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    correct = labelsAndPredictions.filter(lambda pl: pl[0] == pl[1]).count()
    incorrect = labelsAndPredictions.filter(lambda pl: pl[0] != pl[1]).count()
    testError = incorrect / float(testData.count())
    metrics = MulticlassMetrics(labelsAndPredictions)

    print("\nTest Error = " + str(testError))
    print("\nCorrect predictions = " + str(correct))
    print("\nIncorrect predictions = " + str(incorrect))
    print("\nConfusion Matrix = ", metrics.confusionMatrix().toArray())
    print("\nF1 Score = ", metrics.fMeasure(1.0))
    print("\nPrecision = ", metrics.precision(1.0))
    print("\nRecall = ", metrics.recall(1.0))
    print("\nAccuracy = ", str((1 - testError) * 100))
    # print("Learned classification forest model:")
    # print(model.toDebugString())

    if os.path.exists(modelPath):
        print("Overwriting previous model file...")
        shutil.rmtree(modelPath)

    model.save(spark.sparkContext, modelPath)

selection = 0
while selection != 1 or selection != 2:
    print("\n----------------------------------\n")
    print("1. Random Forest")
    print("2. Gradient Boosting")
    print("3. Exit")
    print("\n----------------------------------\n")
    selection = int(input("Select which model you watnt to use:  "))

    if selection == 3: break

    if selection == 1:
        randomForest()
    else:
        gradientBoosting()

    selection = 0
