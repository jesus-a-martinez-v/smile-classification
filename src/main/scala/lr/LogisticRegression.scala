package lr

import smile.classification.logit
import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.validation.accuracy

object LogisticRegression extends App {
  val attributes = new Array[Attribute](4)

  attributes(0) = new NumericAttribute("Recency (months)")
  attributes(1) = new NumericAttribute("Frequency (times)")
  attributes(2) = new NumericAttribute("Monetary (c.c. blood)")
  attributes(3) = new NumericAttribute("Time (months)")

  val label = new NominalAttribute("whether he/she donated blood in March 2007")

  val dataFileUri = this.getClass.getClassLoader.getResource("lr/transfusion.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 4)), header = true)

  println("Sneak peek of the data:")
  println(data)

  val model = logit(data.x(), data.labels())
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}