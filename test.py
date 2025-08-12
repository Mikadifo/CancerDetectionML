from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import GradientBoostedTreesModel, RandomForestModel
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("project3RFModelTesting").getOrCreate()

RFmodel = RandomForestModel.load(spark.sparkContext, "models/randomClassificationModel")
GBmodel = GradientBoostedTreesModel.load(spark.sparkContext, "models/gradientBoostingModel")

correctValue = 'M'

sampleFeatures = [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

features = spark.sparkContext.parallelize([Vectors.dense(sampleFeatures)])
predictions = GBmodel.predict(features).collect()

for prediction in predictions:
    predictedValue = ""

    if prediction == 1.0:
        print("\nPrediction: Malignant Tumor (M)")
        predictedValue = "M"
    else:
        print("\nPrediction: Benign Tumor (B)")
        predictedValue = "B"

    if predictedValue == correctValue:
        print("\nPrediction is correct!")
    else:
        print("\nPrediction was incorrect")
