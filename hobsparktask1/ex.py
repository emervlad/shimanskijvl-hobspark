from pyspark import SparkContext, SparkConf

config = SparkConf().setAppName("my_super_app").setMaster("yarn") 
sc = SparkContext(conf=config)
#sc.setLogLevel("ERROR")

def parse_edge(s):
  user, follower = s.split("\t")
  return (int(user), int(follower))

def step(item):
  prev_v, prev_d, next_v, array = item[0], item[1][0][0], item[1][1], item[1][0][1]
  return (next_v, (prev_d + 1, array + [prev_v]))

def complete(item):
  v, old_d, new_d = item[0], item[1][0], item[1][1]
  return (v, old_d if old_d is not None else new_d)

n = 20  # number of partitions
edges = sc.textFile("/data/twitter/twitter_sample.txt").map(parse_edge)
forward_edges = edges.map(lambda e: (e[1], e[0])).partitionBy(n).persist()

x = 12
d = 0
distances = sc.parallelize([(x, (d, []))]).partitionBy(n)

while True:
  candidates = distances.join(forward_edges, n).map(step)

  keys = distances.keys().collect()

  forward_edges = forward_edges.filter(lambda i: i[0] not in keys and i[1] not in keys)
  new_distances = distances.fullOuterJoin(candidates, n).map(complete, True).persist()
  count = new_distances.filter(lambda i: i[0] == 34).count()
  if count == 0:
    distances = new_distances.filter(lambda i: i[1][0] > d - 1)
    d += 1
  else:
    break

path = new_distances.filter(lambda i: i[0] == 34).take(1)[0][1][1] + [34]

print(",".join(map(str, path)))