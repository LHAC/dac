package dac

import dac.opt.{LogisticGradient => DacLogGrad, lhac}
import dac.utils.OptUtils
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by xiaochengtang on 3/6/15.
 */
object dacer {

  def main(args: Array[String]) {
    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // read in inputs
    val master = options.getOrElse("master", "local[5]")
    val trainFile = options.getOrElse("trainFile", "data/train_dac")
    val testFile = options.getOrElse("testFile", "data/train_dac")
    val numIters = options.getOrElse("numIters","20").toInt
    val numBlocks = options.getOrElse("numBlocks","5").toInt
    val numCorrections = options.getOrElse("numCorrections","10").toInt
    val convergenceTol = options.getOrElse("convergenceTol", "1e-10").toDouble
    val lambda = options.getOrElse("lambda", "0.01").toDouble // regularization parameter

    // print out inputs
    println("trainFile:    " + trainFile);
    println("testfile:     " + testFile);
    println("numBlocks:    " + numBlocks);
    println("numIters:    " + numIters);
    println("lambda:       " + lambda);
    println("convergenceTol:    " + convergenceTol);
    println("numCorrections:    " + numCorrections);

    val conf = new SparkConf()
      .setMaster(master)
      .setAppName("lhac")
    val sc = new SparkContext(conf)

    val (training, test) = {
      if (trainFile == testFile) {
        val data = OptUtils.loadLibSVMFile(sc, trainFile)
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
        (splits(0).cache(), splits(1))
      } else {
        val tr = OptUtils.loadLibSVMFile(sc, trainFile)
        val te = OptUtils.loadLibSVMFile(sc, testFile)
        (tr, te)
      }
    }

    val numFeatures = training.first()._2.length
    val initialWeights = Array.fill(numFeatures)(0.0)
    val weights =  OptUtils.time(lhac.runLBFGS(numCorrections, convergenceTol,
      numIters, training, new DacLogGrad(),
      lambda, numBlocks, initialWeights))

    val model = new LogisticRegressionModel(Vectors.dense(weights), 0)
    // Clear the default threshold.
    model.clearThreshold()
    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { case(label, features) =>
      val score = model.predict(Vectors.dense(features.toArray))
      (score, label)
    }
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)
  }
}
