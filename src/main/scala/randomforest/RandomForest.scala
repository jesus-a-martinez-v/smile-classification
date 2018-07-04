package randomforest

import smile.classification.randomForest
import smile.data.{Attribute, AttributeDataset, NominalAttribute}
import smile.read
import smile.validation.accuracy

object RandomForest extends App {
  val attributes = new Array[Attribute](6)

  attributes(0) = new NominalAttribute("buying")
  attributes(1) = new NominalAttribute("maint")
  attributes(2) = new NominalAttribute("doors")
  attributes(3) = new NominalAttribute("persons")
  attributes(4) = new NominalAttribute("lug_boot")
  attributes(5) = new NominalAttribute("safety")

  val label = new NominalAttribute("Class")

  val dataFileUri = this.getClass.getClassLoader.getResource("randomforest/car.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 6)))

  println("Sneak peek of the data:")
  println(data)

  val model = randomForest(data.x(), data.labels())
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}
