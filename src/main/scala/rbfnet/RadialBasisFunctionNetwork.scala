package rbfnet

import smile.classification.rbfnet
import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.math.distance.EuclideanDistance
import smile.read
import smile.util.gaussrbf
import smile.validation.accuracy

object RadialBasisFunctionNetwork extends App {
  val attributes = new Array[Attribute](9)

  attributes(0) = new NumericAttribute("Wife's age")
  attributes(1) = new NominalAttribute("Wife's education")
  attributes(2) = new NominalAttribute("Husband's education")
  attributes(3) = new NumericAttribute("Number of children ever born")
  attributes(4) = new NominalAttribute("Wife's religion")
  attributes(5) = new NominalAttribute("Wife's now working?")
  attributes(6) = new NominalAttribute("Husband's occupation")
  attributes(7) = new NominalAttribute("Standard-of-living index")
  attributes(8) = new NominalAttribute("Media exposure")

  val label = new NominalAttribute("Contraceptive method used")

  val dataFileUri = this.getClass.getClassLoader.getResource("rbfnet/cmc.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((label, 9)))

  println("Sneak peek of the data:")
  println(data)

  val centers = new Array[Array[Double]](50)
  val basis = gaussrbf(data.x(), centers)
  val model = rbfnet(
    data.x(),
    data.labels(),
    new EuclideanDistance,
    basis,
    centers)
  val predictions = model.predict(data.x())

  println(s"Model's accuracy on the training set: ${accuracy(data.labels(), predictions)}")
}
