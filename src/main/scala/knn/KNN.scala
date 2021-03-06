package knn

import smile.classification.{KNN, knn}
import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.validation.accuracy

object KNN extends App {

  val attributes = new Array[Attribute](8)

  attributes(0) = new NominalAttribute("Sequence Name")
  attributes(1) = new NumericAttribute("mcg")
  attributes(2) = new NumericAttribute("gvh")
  attributes(3) = new NumericAttribute("lip")
  attributes(4) = new NumericAttribute("chg")
  attributes(5) = new NumericAttribute("aac")
  attributes(6) = new NumericAttribute("alm1")
  attributes(7) = new NumericAttribute("alm2")

  val label = new NominalAttribute("class")

  val dataFileUri = this.getClass.getClassLoader.getResource("knn/ecoli.csv").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 8)))

  println("Sneak peek of the data:")
  println(data)

  val model: KNN[Array[Double]] = knn(data.x(), data.labels(), 3)
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}
