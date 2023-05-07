import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
from pyspark.sql.functions import array_union, explode, array, flatten, split, col, sum as pyspark_sum, log, concat_ws


spark = SparkSession.builder.appName('Spark DF practice').master('yarn').getOrCreate()


from pyspark.sql.types import *
schema = StructType(fields=[
    StructField("user", StringType()),
    StructField("follower", StringType())
])

forward_edges = spark.read.csv("/data/twitter/twitter_sample.txt", sep="\t", schema=schema)

x = 12
d = 0
distances = spark.createDataFrame([
    (12, 0, [12])
], schema = StructType(fields=[
    StructField("vertex", StringType()),
    StructField("distance", IntegerType()),
    StructField("path", ArrayType(StringType()))
]))
i = 0
keys = []
while True:
  i += 1
  candidates = distances.join(forward_edges, (forward_edges['follower'] == distances['vertex'])).select(col("user").alias("vertex"), (col("distance") + 1).alias("distance"), array_union("path", array("follower")).alias("path"))
  
  count = candidates.where("vertex == 34").count()
  if count == 0:
    keys = list(set([str(row['vertex']) for row in distances.collect()]))
    forward_edges = forward_edges.join(~forward_edges["user"].isin(keys) & ~forward_edges["follower"].isin(keys))
    distances = candidates.filter("distance > " + str(d-1))
    d += 1
  else:
    break


path = candidates.where("vertex == 34").limit(1).collect()[0]["path"] + [34]

print(",".join(map(str, path)))