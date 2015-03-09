scalaVersion := "2.10.4"

name := "dac"

organization := "ise.lehigh.edu"

parallelExecution in Test := false

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.2.1",
  "org.apache.spark" % "spark-mllib_2.10" % "1.2.1",
  "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test",
  "org.jblas" % "jblas" % "1.2.3"
)

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Spray" at "http://repo.spray.cc"
)
