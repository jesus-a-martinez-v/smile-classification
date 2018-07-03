package knn

import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.classification.{KNN, knn}
import smile.validation.accuracy

object KNN extends App {

  val attributes = new Array[Attribute](8)

  attributes(0) = new NominalAttribute("0")
  attributes(1) = new NumericAttribute("1")
  attributes(2) = new NumericAttribute("2")
  attributes(3) = new NumericAttribute("3")
  attributes(4) = new NumericAttribute("4")
  attributes(5) = new NumericAttribute("5")
  attributes(6) = new NumericAttribute("6")
  attributes(7) = new NumericAttribute("7")

  val label = new NominalAttribute("8")

  val data: AttributeDataset = read.csv(
    this.getClass.getClassLoader.getResource("knn/ecoli.csv").toURI.getPath,
    attributes = attributes,
    response = Some((label, 8)))

  val model: KNN[Array[Double]] = knn(data.x(), data.labels(), 3)

  println(accuracy(data.labels(), model.predict(data.x())))

}
