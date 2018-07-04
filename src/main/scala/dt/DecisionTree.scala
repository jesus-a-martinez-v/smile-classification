package dt

import smile.classification.{DecisionTree, cart}
import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.validation.accuracy

object DecisionTree extends App {
  val attributes = new Array[Attribute](4)

  attributes(0) = new NumericAttribute("sepal length in cm")
  attributes(1) = new NumericAttribute("sepal width in cm")
  attributes(2) = new NumericAttribute("petal length in cm")
  attributes(3) = new NumericAttribute("petal width in cm")

  val label = new NominalAttribute("class")

  val dataFileUri = this.getClass.getClassLoader.getResource("dt/iris.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 4)), header = true)

  println("Sneak peek of the data:")
  println(data)

  val model: DecisionTree = cart(data.x(), data.labels(), maxNodes = 5)
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}
