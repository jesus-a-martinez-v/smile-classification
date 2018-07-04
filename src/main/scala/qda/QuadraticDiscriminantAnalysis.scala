package qda

import smile.classification.qda
import smile.data.{Attribute, AttributeDataset, NominalAttribute}
import smile.read
import smile.validation.accuracy

object QuadraticDiscriminantAnalysis extends App {
  val attributes = new Array[Attribute](4)

  attributes(0) = new NominalAttribute("Left-Weight")
  attributes(1) = new NominalAttribute("Left-Distance")
  attributes(2) = new NominalAttribute("Right-Weight")
  attributes(3) = new NominalAttribute("Right-Distance")

  val label = new NominalAttribute("Class Name")

  val dataFileUri = this.getClass.getClassLoader.getResource("qda/balance-scale.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 0)))

  println("Sneak peek of the data:")
  println(data)

  val model = qda(data.x(), data.labels())
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")

}
