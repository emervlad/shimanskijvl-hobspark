import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
from pyspark.sql.functions import explode
from pyspark.sql.functions import array


spark = SparkSession.builder.appName('Spark DF practice').master('yarn').getOrCreate()

from pyspark.sql.types import *
schema = StructType(fields=[
    StructField("id", StringType()),
    StructField("title", StringType()),
    StructField("text", ArrayType(StringType()))
])

stop_words =  spark.sparkContext.textFile("/data/wiki/stop_words_en-xpo6.txt").collect()

def cleaning(text):
    arr = re.split("\t|\s", text, 2)
    arr[2] = re.sub("[^a-zA-Z\s\d]+", "", arr[2].lower())#.split()
    arr[2] = [word for word in arr[2].split() if word not in stop_words]
    return arr


rdd =  spark.sparkContext.textFile("/data/wiki/en_articles_part").map(cleaning)

df = spark.createDataFrame(rdd, schema=schema)
#df = df.withColumn("text", array(df["text"]))

ngram_df = NGram(n=2, inputCol="text", outputCol="bigrams").transform(df)

print(ngram_df.select(explode("bigrams").alias("bigram")).groupBy("bigram").count().where("count >= 500").orderBy("count", ascending=False).show(39, truncate=False))