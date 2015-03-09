package dac.opt

/**
 * Created by xiaochengtang on 2/28/15.
 */

import dac.linalg._
import dac.linalg.Implicits._
import dac.utils._
import org.apache.spark.{SparkContext, HashPartitioner}
import org.apache.spark.SparkContext._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BVC}
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
 *
 * @param x - iterate
 * @param df - gradient on x
 * @param f - objective value on x
 * @param a - step size
 * @param iter - iteration number
 * @param isfailed - error indicator
 * @tparam T
 */
case class State[T](x: T, df: T, f: Double,
                    improve: Double = 1e15,
                    a: Double = 1.0,
                    iter: Int = 0,
                    isfailed: Boolean = false) {
  def this(x: T, cb: (Double, T)) = this(x, cb._2, cb._1)
  def this(x: T, cb: (Double, T), a: Double) = this(x, cb._2, cb._1, a)

}

object State {
  def apply(x: Array[Double], cb: (Double, Array[Double])) = new State(x, cb)
  def apply(x: Array[Double], cb: (Double, Array[Double]), a: Double) = new State(x, cb, a)
}



/**
 * Run Limited-memory BFGS (L-BFGS) in parallel.
 * distribute gradient evaluation and direction computation
 *
 *
 * @return double array containing weights for each feature.
 */
object lhac extends SerializableLogging {
  /**
   *
   * @param numCorrections - The number of corrections used in the L-BFGS update.
   * @param convergenceTol - The convergence tolerance of iterations for L-BFGS
   * @param numIters - Maximal number of iterations that L-BFGS can be run.
   * @param data - Input data for L-BFGS. RDD of the set of data examples, each of
   *               the form (label, [feature values]).
   * @param gradient - Gradient object (used to compute the gradient of the loss function of
   *                   one single data example)
   * @param regParam - Regularization parameter
   * @param numBlocks - to divide {s.y} pairs in lbfgs
   * @param initialWeights
   * @return
   */
  def runLBFGS(numCorrections: Int = 10,
                convergenceTol: Double = 1E-15,
                numIters: Int = 10,
                data: RDD[(Double, BVC[Double])],
                gradient: Gradient,
                regParam: Double,
                numBlocks: Int,
                initialWeights: Array[Double]): Array[Double] = {


    // warm start with steepest descent
    val SWITCH = 3
    val N = data.count()
    val p = data.first._2.length
    val sc = data.context

    val costFun =
      new CostFun(data, gradient, regParam, N)

    val lbfgs =
      new HessianApproximation(sc, numCorrections, p, numBlocks)

    print("#iter ")
    print("fval ")
    print("imprv ")
    print("step ")
    print("\n")
    val s0 = State(initialWeights, costFun.calculate(initialWeights))
    val states = Iterator.iterate(s0){ s: State[Array[Double]] => try {
      val d = if (s.iter < SWITCH) s.df times -0.5 else lbfgs.descentDirection(s.df)
      val s_ = lineSearch(costFun, d, s).copy[Array[Double]](iter = s.iter + 1)
      lbfgs.update(s_.x minus s.x, s_.df minus s.df)
//      logger.debug(s_.iter + " ")
      val delta = s.f - s_.f
      print(s_.iter + " ")
      print(s_.f + " ")
      print(delta + " ")
      print(s_.a + " ")
//      print(BLAS.norm2(BDV(s_.df)) + " ")
//      print(BLAS.dot(s_.df, d) + " ")
//      print(BLAS.dot(s_.x minus s.x, s_.df minus s.df) + " ")
      print("\n")
      s_.copy(improve = delta)
    } catch {
      case x: FirstOrderException =>
        logger.error("Failure!: " + x)
        s.copy(isfailed = true)
    }
    }.takeWhile(s => s.iter < numIters && s.improve >= convergenceTol && !s.isfailed)

    var state = states.next()
    while(states.hasNext) {
      state = states.next()
    }
    state.x
  }

  private def lineSearch(costFun: CostFun,
                          d: Array[Double],
                          s: State[Array[Double]]): State[Array[Double]] = {
    val delta = BLAS.dot(s.df, d)
    if (delta >= 0)
      throw new LineSearchFailed(delta)
    val c = 1e-5
    val rho = 0.5
    val maxInnerIters = 100
    // s.a default to 1
    val a0 = 1.0
    var x = s.x plus (d times a0)

    val cachedFun = costFun.cached()

    /**
     * a lot iterations need to be taken at first
     * reusing memory is critical
     */
    val (f, a) = Iterator.iterate((cachedFun.valueAt(x), a0))
    {case(f, a) =>
      val a_ = rho*a
      if(a < 1E-20)
        throw new StepSizeUnderflow
      //      val x = s.x plus (d times a_)
      for (i <- 0 until s.x.length)
        x(i) = s.x(i) + d(i)*a_
      val f_ = cachedFun.valueAt(x)
      (f_, a_)
    }.dropWhile{case(f, a) =>
      f >= s.f + a*c*delta
    }.take(1).next()
//    for (i <- 0 until s.x.length)
//      x(i) = s.x(i) + d(i)*a
    State(x, cachedFun.gradAt(), f)
      .copy[Array[Double]](a=a, iter=s.iter)


//    Iterator.iterate(State(x, costFun.calculate(x))
//      .copy[Array[Double]](iter = s.iter)) { state =>
//      val a_ = rho*state.a
//      if(a_ < 1E-20)
//        throw new StepSizeUnderflow
////      val x = s.x plus (d times a_)
//      for (i <- 0 until s.x.length)
//        x(i) = s.x(i) + d(i)*a_
//      val s_ = State(x, costFun.calculate(x))
//        .copy[Array[Double]](a = a_, iter = s.iter)
////      println(s_.f)
//      s_
//    }.zipWithIndex.dropWhile{case(state, i) =>
//      state.f >= s.f + state.a*c*delta
//    }.take(1).next()._1
  }

  private[opt] class HessianApproximation(val sc: SparkContext,
                                     val memory: Int,
                                     val dim: Int,
                                     val numBlocks: Int) {
    val data1 = sc.parallelize(0 until dim).map(x => (x, Array[Double]()))
      .partitionBy(new HashPartitioner(numBlocks)).cache()
    val data2 = sc.parallelize(0 until dim).map(x => (x, Array[Double]()))
      .partitionBy(new HashPartitioner(numBlocks)).cache()
    var S = RowDistributedLMatrix(data1)
    var T = RowDistributedLMatrix(data2)
    var L = new LMatrix()
    var D = List[Double]()
    var buffer = Array[Double]()

    def descentDirection(df: Array[Double]): Array[Double] = {
      val Q = S ++ T + df
      val b = Q.innerProduct()
      val m0 = S.cols()
      val a = BDV.vertcat(BDV.zeros[Double](2*m0), BDV.fill(1){-1.0})
      val c = Array.fill(m0)(0.0)
      for (jj <- m0-1 to 0 by -1) {
        c(jj) = (a.t * b(::, jj)) / b(jj, m0+jj)
        a(m0+jj) -= c(jj)
      }
      a *= (b(m0-1, 2*m0-1) / b(2*m0-1, 2*m0-1))
      for (jj <- 0 to m0-1) {
        a(jj) += (c(jj) - a.t*b(m0+jj, ::).t / b(jj, m0+jj))
      }
      (Q :* a.toArray).toArray
    }

    def update(s: Array[Double], t: Array[Double]) = {
      val m0 = S.cols()
      if (m0 < memory) {
        S = S + s
        T = T + t
        buffer = S.lastCol() *: T
        L = L.appendRow(buffer.dropRight(1)).appendCol(0.0)
        D = D :+ buffer.last
      } else {
        S = S.deleteCol() + s
        T = T.deleteCol() + t
        L = L.deleteRow().deleteCol()
        buffer = S.lastCol() *: T
        L = L.appendRow(buffer.dropRight(1)).appendCol(0.0)
        D = D.tail :+ buffer.last
      }
      if(buffer.last < 0 || buffer.last.isNaN)
        throw new NaNHistory(buffer.last)

    }
  }

  /**
   * data features use breeze vector to allow for sparse representation
   * private scope is set to opt package so that testing is possible
   */
  private[opt] class CostFun(
                         data: RDD[(Double, BVC[Double])],
                         gradient: Gradient,
                         regParam: Double,
                         numExamples: Long) {
    private var cachedFun: cachedCostFun = null

    class cachedCostFun {
      private var cachedGrad: Array[Double] = null
      def valueAt(w: Array[Double]): Double = {
        val updates = calculate(w)
        cachedGrad = updates._2
        updates._1
      }

      def gradAt(): Array[Double] = cachedGrad
    }

    def cached(): cachedCostFun = {
      if (cachedFun == null) {
        cachedFun = new cachedCostFun
      }
      cachedFun
    }

    def calculate(w: Array[Double]): (Double, Array[Double]) = {
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val n = w.size
      val bcW = data.context.broadcast(w)
      val localGradient = gradient

      val (gradientSum, lossSum) = data.treeAggregate((BVC.fill(n)(0.0), 0.0))(
        seqOp = (c, v) => (c, v) match { case ((grad, loss), (label, features)) =>
          val l = localGradient.compute(
            features, label, BVC(bcW.value), grad)
          (grad, loss + l)
        },
        combOp = (c1, c2) => (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
          BLAS.axpy(1.0, grad2, grad1)
          (grad1, loss1 + loss2)
        })


      // regVal is sum of weight squares in the case of L2 updater;
      val norm = BLAS.norm2(BVC(w))
      val regVal = 0.5 * regParam * norm * norm
      val loss = lossSum / numExamples + regVal

      BLAS.scal(1.0 / numExamples, gradientSum)
      BLAS.axpy(regParam, BVC(w), gradientSum)

      (loss, gradientSum.toArray)
    }
  }
}

sealed trait ConvergenceReason {
  def reason: String
}
case object MaxIterations extends ConvergenceReason {
  override def reason: String = "max iterations reached"
}
case object FunctionValuesConverged extends ConvergenceReason {
  override def reason: String = "function values converged"
}
case object GradientConverged extends ConvergenceReason {
  override def reason: String = "gradient converged"
}
case object SearchFailed extends ConvergenceReason {
  override def reason: String = "line search failed!"
}
case object ObjectiveNotImproving extends ConvergenceReason {
  override def reason: String = "objective is not improving"
}

object testLHAC {
  def testHessianApproximation(sc: SparkContext,
                               memory: Int,
                               dim: Int,
                               numBlocks: Int): lhac.HessianApproximation = {
    new lhac.HessianApproximation(sc, memory, dim, numBlocks)
  }

  def testCostFun(data: RDD[(Double, BVC[Double])],
                  gradient: Gradient,
                  regParam: Double,
                  numExamples: Long): lhac.CostFun = {
    new lhac.CostFun(data, gradient, regParam, numExamples)
  }

}
















