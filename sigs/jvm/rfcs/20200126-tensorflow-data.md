# TensorFlow Data (Java)

| Status        |Proposed     |
:-------------- |:---------------------------------------------------- |
| **Current Implementation** | [dhruvrajan/tensorflow-java](https://github.com/dhruvrajan/tensorflow-java/tree/tf-data-cherrypick/tensorflow-frameworks/tensorflow-data) |
| **Author(s)** | Dhruv Rajan (dhruv@krishnaprem.com) |
| **Sponsor**   | Karl Lessard (karl.lessard@gmail.com)                 |
| **Updated**   | 2020-01-26                                  |

## Objective

To provide simple APIs for loading data of various formats, and preparing
datasets for use in training deep learning models with TensorFlow.

The design of the API and summary written here is based on
the open source code and documentation of 
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data).

## Motivation

A large part of the standard deep-learning workflow is managing datasets:
configuring iteration and batching over a sequence of elements,
and performing transformations on individual or grouped dataset elements.

The goal of this project is to provide a high-level abstraction which simplifies
this workfow, based on TensorFlow data ops, with syntax closely mirorring
Python's [tf.data](https://www.tensorflow.org/api_docs/python/tf/data).

## User Benefit

Users of TensorFlow Java will be able to use this framework to easily
load, transform, and iterate over datasets.

Additionally this framework will be used at the core of the automated
training process in TensorFlow Java's port of [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras).

## Design Proposal

The `Dataset` class provides the main interface for creating and transforming datasets. A dataset represents a sequence of elements, where each element is a collection of tensor components. 

### Construction

Datasets can be constructed either directoy from a data source (e.g. a list of tensors representing the components of the dataset), or as a transformation on an existing dataset.

#### From Data Source

To construct a dataset from a list of tensor components, use 
`Dataset.fromTensorSlices( ... )`. For example, say we are working
with a standard feature-label dataset consisting of 4 elements.

```java
float[][] features = new float[][] {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12}
}

float[] labels = new float[] {
    0,
    1,
    1,
    0
}
```


A dataset can be constructed from a list of the constant `Operand`s generated
from this dataset, and a list of `DataType` objects corresponding
to the type of each component:

(NOTE: each of the input components must share the same first "batch" dimension.)

```java
Ops tf = // ... TensorFlow Ops accessor (either graph or eager)
Dataset dataset = Dataset.fromTensorSlices(
    Arrays.asList(tf.constant(features), tf.constant(labels)),
    Arrays.asList(TInt32.DTYPE, TInt32.DTYPE)
);
```


Other data sources are also possible, using `tf.data` ops. Datasets can also be constructed from 

#### Transformations on Existing Datasets

Transformed datasets can be created from existing datasets. For example, to group elements in the above dataset into batches of size `2`, use `Dataset.batch(int batchSize)`:

```java
dataset = dataset.batch(2)
```
The call to `.batch` (as with the other transformations) returns a new dataset
object, to allow chaining of these transformations.

Similar transformations include `.skip`, `.take`, `.map`, `.filter`, etc.


### Iterating through Dataset Elements

The primary use of a dataset is for iteration through its elements.
Each row (or batch) element is represented as a list of tensor components, with
type `List<Output<?>>`. The tensor components of this elements can be accessed using `List.get(int index)`.

It is recommended to use `Tensor.expect(DataType<?> dtype)` to restore types
to the retrieved tensors.


#### Eager Mode: Iterable
The `Dataset` class implements the `Iterable` interface, so in
eager mode, iteration through dataset elements is possible using a standard for-each loop.

Using the same example dataset from above, dataset elements can be extracted and
used as follows:
```java
// Use default EagerSession
Ops tf = Ops.create()

// Dataset of (features, labels) from above
Dataset dataset = Dataset.fromTensorSlices(tf, ... );

// batch dataset elements into batches of size 2
dataset = dataset.batch(2); 

for (List<Output<?>> batch : dataset) {
    Tensor<TInt32> featureBatch = element.get(0).tensor().expect(TInt32.DTYPE);
    Tensor<TInt32> labelBatch = element.get(1).tensor().expect(TInt32.DTYPE);

    // Perform batch-wise computations on featreBatch and labelBatch
}

```


#### Graph Mode: OneShotIterator

The above code will not work in graph mode, which requires the use of `Session`s
to run the computations. In graph mode, datasets can be iterated through using the `OneShotIterator` abstraction, and a while loop.

`OneShotIterator` allows iteration through a dataset's elements "one time", using
repeated calls to `Session.run`. 

The `OneShotIterator` object has two fields:
- `OneShotIterator.getMakeIteratorOp()` returns a "make-iterator" `Operation` which is run to initialize this iterator
- `OneShotIterator.getComponents()` returns a list of tensor components which can be used / fetched in a session, corresponding
to the current set of elements during iteration.

Once the iterator is initialized, repeated calls to `Session.run` will populate the components with new values, until all elements have
been retrieved. After this, `Session.run` will result in an `IndexOutOfBounds` exception.

Note that the make-iterator operation can be re-run to re-initialize
the iterator, to iterate through the dataset a second time.

```java

try (Graph graph = new Graph()) {
    // Graph mode Ops accessor
    Ops tf = Ops.create(graph)

    // Dataset of (features, labels) from above
    Dataset dataset = Dataset.fromTensorSlices(tf, ... );

    // batch dataset elements into batches of size 2
    dataset = dataset.batch(2); 

    // Get OneShotIterator components
    OneShotIterator oneShotIterator = dataset.makeOneShotIterator();
    Operation makeIteratorOp = oneShotIterator.getMakeIteratorOp();
    List<Output<?>> batch = oneShotIterator.getComponents();

    // instantiate graph-mode session
    try (Session session = new Session(graph)) {
        // Initialize iterator
        session.runner()
            .addTarget(makeIteratorOp)
            .run();

        // Iterate through dataset elements
        while (true) {
            try {
                List<Tensor<?>> outputs = session.runner()
                    .fetch(batch.get(0))
                    .fetch(batch.get(1))
                    .run();

                Tensor<TInt32> featureBatch = outputs.get(0).expect(TInt32.DTYPE);
                Tensor<TInt32> labelBatch = outputs.get(1).expect(TInt32.DTYPE);

                // Perform batch-wise computations on featreBatch and labelBatch
            } catch (IndexOutOfBoundsException e) {
                // Finished iterating
                break;
            }
        }
    }
}

```

## Questions and Discussion Topics

Se [MNISTBasicEagerClassifier.java](https://github.com/dhruvrajan/tensorflow-java/blob/tensorflow-keras-dev/tensorflow-frameworks/tensorflow-keras/src/main/java/org/tensorflow/keras/examples/mnist/MNISTBasicEagerClassifier.java) and [MNISTBasicGraphClassifier.java](https://github.com/dhruvrajan/tensorflow-java/blob/tensorflow-keras-dev/tensorflow-frameworks/tensorflow-keras/src/main/java/org/tensorflow/keras/examples/mnist/MNISTBasicGraphClassifier.java) as examples using the described API
for training a simple MNIST classifier.