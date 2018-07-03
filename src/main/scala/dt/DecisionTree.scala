package dt

import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.classification.{DecisionTree, KNN, cart}
import smile.validation.accuracy

object DecisionTree extends App {
  val attributes = new Array[Attribute](4)

  attributes(0) = new NumericAttribute("Recency (months)")
  attributes(1) = new NumericAttribute("Frequency (times)")
  attributes(2) = new NumericAttribute("Monetary (c.c. blood)")
  attributes(3) = new NumericAttribute("Time (months)")

  val label = new NominalAttribute("whether he/she donated blood in March 2007")

  val dataFileUri = this.getClass.getClassLoader.getResource("dt/transfusion.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 4)), header = true)

  println("Sneak peek of the data:")
  println(data)

  val model: DecisionTree = cart(data.x(), data.labels(), maxNodes = 256)
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}
