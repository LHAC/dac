package dac.utils

import breeze.linalg.{SparseVector => BSV, Vector => BVC}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Created by xiaochengtang on 3/5/15.
 */

import org.slf4j.{Logger, LoggerFactory}

/**
 * A logger that only evaluates parameters lazily if the corresponding log level is enabled.
 */
class LazyLogger(log: Logger) extends Serializable {

  def info(msg: => String) {
    if (log.isInfoEnabled) log.info(msg)
  }

  def debug(msg: => String) {
    if (log.isDebugEnabled) log.debug(msg)
  }

  def trace(msg: => String) {
    if (log.isTraceEnabled) log.trace(msg)
  }

  def warn(msg: => String) {
    if (log.isWarnEnabled) log.warn(msg)
  }

  def error(msg: => String) {
    if (log.isErrorEnabled) log.error(msg)
  }

  def info(msg: => String, throwable: Throwable) {
    if (log.isInfoEnabled) log.info(msg, throwable)
  }

  def debug(msg: => String, throwable: Throwable) {
    if (log.isDebugEnabled) log.debug(msg, throwable)
  }

  def trace(msg: => String, throwable: Throwable) {
    if (log.isTraceEnabled) log.trace(msg, throwable)
  }

  def warn(msg: => String, throwable: Throwable) {
    if (log.isWarnEnabled) log.warn(msg, throwable)
  }

  def error(msg: => String, throwable: Throwable) {
    if (log.isErrorEnabled) log.error(msg, throwable)
  }
}

/**
 * Stupid Typesafe logging lib trait isn't serializable. This is just a better version.
 *
 * @author dlwh
 **/
trait SerializableLogging extends Serializable {
  @transient @volatile
  private var _the_logger: LazyLogger = null

  protected def logger: LazyLogger = {
    var logger = _the_logger
    if(logger eq null) {
      synchronized {
        logger = _the_logger
        if(logger eq null) {
          val ll = new LazyLogger(LoggerFactory.getLogger(this.getClass))
          _the_logger = ll
          logger = ll
        }
      }
    }
    logger
  }
}

sealed class FirstOrderException(msg: String="") extends RuntimeException(msg)
class NaNHistory(sy: Double) extends FirstOrderException("sy too small or negative: %.4f".format(sy))
class StepSizeUnderflow extends FirstOrderException("Step Size Underflow!")
class StepSizeOverflow extends FirstOrderException
class LineSearchFailed(delta: Double) extends FirstOrderException("Not Descent Direction: %.4f".format(delta))

object OptUtils {

  // original requires that labels have to be 0 or 1 (no -1)
  def loadLibSVMFile(
                      sc: SparkContext,
                      path: String,
                      numFeatures: Int = -1,
                      minPartitions: Int = -1): RDD[(Double, BVC[Double])] = {
    val numPartitions = if (minPartitions < 0) sc.defaultMinPartitions else minPartitions
    val parsed = sc.textFile(path, numPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      val items = line.split(' ')
      val label_ = items.head.toDouble
      val label = if (label_ <= 0) 0.0 else 1.0
      val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
        val indexAndValue = item.split(':')
        val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
      val value = indexAndValue(1).toDouble
        (index, value)
      }.unzip
      (label, indices.toArray, values.toArray)
    }

    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }

    parsed.map { case (label, indices, values) =>
      (label, new BSV[Double](indices, values, d))
    }
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    val res = t1 - t0
    println(f"Elapsed time: $res%2d ns")
    result
  }
}
