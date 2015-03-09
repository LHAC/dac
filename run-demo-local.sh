# point to where spark-submit is installed
SPARK_SUBMIT=/usr/local/bin/spark-submit

$SPARK_SUBMIT \
--master local[5] --class dac.dacer \
target/scala-*/dac_2.10-*.jar \
--trainFile=data/train_dac \
--testFile=data/train_dac \
--numIters=100 \
--numBlocks=5 \
--lambda=.001 \
--convergenceTol=1e-10 \
--numCorrections=5

