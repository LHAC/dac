# Distributed L-BFGS on Apache Spark
by Xiaocheng Tang [http://goo.gl/6QuMl]

This code implements distributed L-BFGS using [Apache Spark](http://spark.apache.org). Compared with the L-BFGS implementation included in Spark, this code also distributes the storage of L-BFGS history and the computations of descent direction, based upon the idea proposed in this [2014 NIPS paper](http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf).

The present code trains l2-regularized logistic regression.

## Getting Started
How to run the code locally:

```bash
sbt/sbt package
# need to update the path to spark-submit
./run-demo-local.sh
```



