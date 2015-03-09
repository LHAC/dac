package dac.linalg

/**
 * Created by xiaochengtang on 2/28/15.
 */

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BVC, axpy => Baxpy, norm => Bnorm}
import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.rdd.RDD


class LMatrix(val data: BDM[Double]) {
  def this() = this(BDM.zeros[Double](0, 0))
  def apply(row: Int, col: Int): Double = data(row, col)
  def +(c: Array[Double]): LMatrix = this.appendCol(c)
  def +(c: Double): LMatrix = this.appendCol(c)
  def appendCol(c: Array[Double]): LMatrix = {
    if (data.rows == 0) {
      val _data = BDM.zeros[Double](c.length, 1)
      _data(::, 0) := BDV(c)
      new LMatrix(_data)
    } else {
      new LMatrix(BDM.horzcat(data, BDM(c).t))
    }
  }
  def appendCol(z: Double): LMatrix = {
    if (data.rows == 0) {
      val _data = BDM.zeros[Double](1, 1)
      _data(::, 0) := z
      new LMatrix(_data)
    } else {
      new LMatrix(BDM.horzcat(data, BDM(Array.fill[Double](data.rows)(z)).t))
    }
  }
  // the first column
  def deleteCol(): LMatrix = {
    if (data.cols != 0) {
      new LMatrix(data(::, 1 to data.cols-1))
    } else {
      this
    }
  }
  def appendRow(c: Array[Double]): LMatrix = {
    if (data.cols == 0) {
      new LMatrix(BDM.zeros[Double](0, c.length))
    }
    new LMatrix(BDM.vertcat(data, BDM(c)))
  }
  // the first row
  def deleteRow(): LMatrix = {
    if (data.rows != 0) {
      new LMatrix(data(1 to data.rows-1, ::))
    } else {
      this
    }
  }
}


object BLAS extends Serializable with Logging {
  /**
   * y += a * x
   */
  def axpy(a: Double, x: BVC[Double], y: BVC[Double]): Unit = {
    require(x.size == y.size)
    y match {
      case dy: BDV[Double] =>
        x match {
          case sx: BSV[Double] =>
            Baxpy(a, sx, dy)
          case dx: BDV[Double] =>
            Baxpy(a, dx, dy)
          case _ =>
            throw new UnsupportedOperationException(
              s"axpy doesn't support x type ${x.getClass}.")
        }
      case _ =>
        throw new IllegalArgumentException(
          s"axpy only supports adding to a dense vector but got type ${y.getClass}.")
    }
  }

//  def axpy(a: Double, x: Array[Double], y: Array[Double]) = {
//    for (((x, y), i) <- y.zip(x.map(_*a)).view.zipWithIndex) y(i) = x + y
//  }

  def dot(x: BVC[Double], y: BVC[Double]): Double = x.dot(y)

  def dot(x: Array[Double], y: Array[Double]): Double = x.zip(y).map{case(a, b)=>a*b}.sum

  /**
   * x = a * x
   */
  def scal(a: Double, x: BVC[Double]): Unit = x *= a

  def norm2(x: BVC[Double]): Double = Bnorm(x, 2.0)
}



class RowDistributedLMatrix(val p: Long, val data: RDD[(Int, Array[Double])]) extends Serializable {
  val sc = data.context
  def cols(): Int = data.first._2.length
  def rows(): Long = p
  def appendCol(c: Array[Double]): RowDistributedLMatrix = {
    require(c.length == p)
    val bc = sc.broadcast(c)
    new RowDistributedLMatrix(p, data.mapPartitions(x =>
      x.map{ case(k, a) => (k, a :+ bc.value(k)) },
      preservesPartitioning = true))
  }
  // first column
  def deleteCol(): RowDistributedLMatrix =  new RowDistributedLMatrix(p, data.mapValues(_.tail))
  // vec.t * this
  def *:(r: RDD[(Int, Array[Double])]): Array[Double] = {
    data.join(r).mapValues{ case(a1, a2) => a1.map(_*a2(0)) }
      .treeReduce{ case((x1, a1), (x2, a2)) => (0, (a1, a2).zipped.map(_+_)) }._2
  }
  // this * vec
  def :*(l: Array[Double]): RowDistributedLMatrix = {
    require(this.cols() == l.length)
    new RowDistributedLMatrix(p, data.mapValues(x => Array(x.zip(l).map{case(a, b) => a*b}.sum)))
  }
  // horizontal concatenate two RowDistributedLMatrix
  def ++(that: RowDistributedLMatrix): RowDistributedLMatrix = {
    require(p == that.p)
    new RowDistributedLMatrix(p, data.join(that.data).mapValues{ case(a1, a2) => a1 ++ a2 })
  }
  def +(c: Array[Double]): RowDistributedLMatrix = this.appendCol(c)
  def innerProduct(): BDM[Double] = {
    val m: RDD[BDM[Double]] = data.mapPartitions{x =>
      val xs = x.map(_._2).toArray
      val rows = xs.length
      val cols = xs.head.length
      Iterator(new BDM(rows, cols, xs.reduce(_++_), 0, cols, true))
    }.map(x => x.t*x)
    m.treeReduce(_+_)
  }
  def column(i: Int): RDD[(Int, Array[Double])] = {
    require(i < data.first._2.length)
    data.mapValues(a => Array(a(i)))
  }
  def lastCol(): RDD[(Int, Array[Double])] = data.mapValues(a => Array(a.last))
  def toArray(): Array[Double] = {
    require(this.cols() == 1)
    data.collect().sortBy(_._1).flatMap(_._2)
  }
  def toDenseMatrix(): BDM[Double] = {
    val cols = data.first._2.length
    val rows = p.toInt
    val v: Array[Double] = data.collect().sortBy(_._1).flatMap(_._2)
    // istranspose == true so majorstrides should be equal to cols
    new BDM(rows, cols, v, 0, cols, true)
  }
  def print() = {
    println("num partition size: " + data.partitions.size)
    data.foreach{ case(i, a) => println(i + "\t" + a.deep.mkString(","))}
  }
}

// companion object
object RowDistributedLMatrix {
  def apply(data: RDD[(Int, Array[Double])]) = {
    val p = data.count()
    new RowDistributedLMatrix(p, data)
  }
}

class DoubleArray(arr: Array[Double]){
  def plus(plusArr: Array[Double]) : Array[Double] = {
    (0 to plusArr.length-1).map( i => this.arr(i) + plusArr(i)).toArray
  }
  def times(c: Double) : Array[Double] = this.arr.map(x => x*c)

  def minus(that: Array[Double]): Array[Double] = this plus (new DoubleArray(that) times -1.0)
}

object Implicits{
  implicit def arraytoDoubleArray(arr: Array[Double]) = new DoubleArray(arr)
}
