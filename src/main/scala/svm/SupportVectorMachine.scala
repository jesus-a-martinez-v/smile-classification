package svm

import smile.classification.svm
import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.math.kernel.GaussianKernel
import smile.read
import smile.validation.accuracy

object SupportVectorMachine extends App {
  val attributes = new Array[Attribute](5)

  attributes(0) = new NominalAttribute("BI-RADS assessment")
  attributes(1) = new NumericAttribute("Age")
  attributes(2) = new NominalAttribute("Shape")
  attributes(3) = new NominalAttribute("Margin")
  attributes(4) = new NominalAttribute("Density")

  val label = new NominalAttribute("Severity")

  val dataFileUri = this.getClass.getClassLoader.getResource("svm/mammographic_masses.data").toURI.getPath
  val data: AttributeDataset =
    read.csv(
      dataFileUri,
      attributes = attributes,
      response = Some((label, 5)),
      header = false,
      missing = "?")

  println("Sneak peek of the data:")
  println(data)

  val kernel = new GaussianKernel(1.0)
  val model = svm(data.x(), data.labels(), kernel = kernel, C = 1.0)  // TODO Fix this. It throws java.lang.ArrayIndexOutOfBoundsException: 16
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}
