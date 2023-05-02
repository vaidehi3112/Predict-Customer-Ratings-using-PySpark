from pyspark.sql import SparkSession
import time

from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import regexp_replace,col,substring,length,abs

# Create a sample DataFrame with text data and labels
spark = SparkSession.builder \
     .master('local[7]') \
     .appName('TFIDF') \
     .config('spark.jars.packages', 
             'com.johnsnowlabs.nlp:spark-nlp_2.11:2.3.5') \
     .getOrCreate()

df = spark.read.csv("amazon_reviews_multilingual_US_v1_00.tsv", sep="\t", header=True, inferSchema=True).limit(200000).repartition(7)
df = df.select( col("review_body"), col("star_rating").cast('int').alias("label"))

df = df.where("star_rating = 1").limit(23000).union(
            df.where("star_rating = 2").limit(23000)).union(
            df.where("star_rating = 3").limit(23000)).union(
            df.where("star_rating = 4").limit(23000)).union(
            df.where("star_rating = 5").limit(23000))

# Remove HTML tags
df = df.withColumn("review_body", regexp_replace("review_body", "<.*?>", ""))

# Truncate review body at 2,000 characters
df = df.withColumn("review_body", substring("review_body", 0, 2000))

df = df.filter(length(col("review_body")) >= 20)

start_time = time.time()
# Tokenize the text
tokenizer = Tokenizer(inputCol="review_body", outputCol="words")
df = tokenizer.transform(df)

# Calculate term frequency (TF)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=200)
df = hashingTF.transform(df)

# Calculate inverse document frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(df)
df = idfModel.transform(df)

end_time = time.time()
print("Runtime for preprocessing: {:.2f} seconds".format(end_time-start_time))

start_time = time.time()

# Define the neural network architecture
layers = [df.select("features").first()[0].size, 200, 100, 50, 10]

# Create the neural network classifier
classifier = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# Define the hyperparameter grid for cross-validation
param_grid = ParamGridBuilder() \
    .addGrid(classifier.maxIter, [50, 100, 150, 200]) \
    .addGrid(classifier.blockSize, [64, 128, 256]) \
    .addGrid(classifier.stepSize, [0.1, 0.01, 0.02, 0.05]) \
    .build()

# Create the cross-validator
cross_validator = CrossValidator(estimator=classifier,
                                 estimatorParamMaps=param_grid,
                                 evaluator=MulticlassClassificationEvaluator(metricName="accuracy"),
                                 numFolds=5,
                                 seed=1234)

# Split the data into training and test sets
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed=1234)

# Run cross-validation to find the best model
cv_model = cross_validator.fit(trainingData)

end_time = time.time()
print("Runtime for training: {:.2f} seconds".format(end_time-start_time))

start_time = time.time()
# Make predictions on the testing data using the best model
predictions = cv_model.transform(testData)
end_time = time.time()
print("Runtime for validation: {:.2f} seconds".format(end_time-start_time))

# Evaluate the performance of the classifier
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions,{evaluator.metricName: "accuracy"})
print("Accuracy = %g" % (accuracy))
f1score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("F1 score = %g"  % (f1score))
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
print("MAE = %g" % (mae))
# Calculate absolute error
predictions = predictions.withColumn("abs_error", abs(col("prediction") - col("label")))

# Calculate mean absolute percentage error (MAPE)
mape = predictions.select((col("abs_error") / col("label")).alias("mape")) \
                .agg({"mape": "mean"}) \
                .collect()[0][0]

print("Mean Absolute Percentage Error (MAPE): {:.4f}".format(mape * 100))  # MAPE as percentage
