import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
from pyspark.sql.functions import explode, array, flatten, split, col, sum as pyspark_sum, log, concat_ws


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

ngram_df = NGram(n=2, inputCol="text", outputCol="bigrams").transform(df)

bigrams_count = ngram_df.select(explode("bigrams").alias("bigram")).groupBy("bigram").count()

sum_all_pairs = bigrams_count.select(pyspark_sum('count')).collect()[0][0]

bigrams_count = bigrams_count.where("count >= 500").orderBy("count", ascending=False)

required_words = bigrams_count.select(explode(split(bigrams_count["bigram"], ' ')).alias('s')).distinct()#NGram(n=1, inputCol="bigram", outputCol="words").transform(bigrams_count)

words = NGram(n=1, inputCol="text", outputCol="s").transform(df)

words_count = words.select(explode("s").alias("s")).groupBy("s").count().join(required_words, 's')

sum_all_words = words_count.select(pyspark_sum('count')).collect()[0][0]

required_words = bigrams_count.select(split(bigrams_count["bigram"], ' ').alias('s'), bigrams_count["count"].alias("count1"),).select(col("s")[0].alias("s1"), col("s")[1].alias("s2"), col("count1"))

df2 = required_words.join(words_count, (required_words["s1"] == words_count["s"])).select(col('s1'), col('s2'), col('count1').alias('column_bi'), col('count').alias('count1'))

df3 = df2.join(words_count, ((df2["s2"] == words_count["s"]))).select(concat_ws('_', col('s1'), col('s2')).alias('con'), (-log(col('column_bi') * sum_all_words**2 / (col('count1') * col('count') * sum_all_pairs)) / log(col('column_bi') / sum_all_pairs)).alias('npmi')).orderBy("npmi", ascending=False).limit(39)

for pair in [str(row['con']) for row in df3.collect()]:
    print(pair)