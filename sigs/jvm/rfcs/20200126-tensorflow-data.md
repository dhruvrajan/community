# TensorFlow Data (Java)

| Status        |Proposed     |
:-------------- |:---------------------------------------------------- |
| **Current Implementation** | [dhruvrajan/tensorflow-java](https://github.com/dhruvrajan/tensorflow-java) |
| **Author(s)** | Dhruv Rajan (dhruv@krishnaprem.com) |
| **Sponsor**   | Karl Lessard (karl.lessard@gmail.com)                 |
| **Updated**   | 2020-01-26                                  |

## Objective

To create a high-level API wrapping TensorflowData ops for batching
and performing various transformations on datasets.

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

The `Dataset` class provides the main interface for creating and transforming datasets. A dataset
represents a sequence of elements, where each element is a collection of tensor components.

### Construction

A Dataset can be constructed from a list of tensors ("components") which share the same first, or "batch" dimension:

```java
Ops tf = // ... TensorFlow Ops accessor (either graph or eager)

// Say we have a standard dataset of features and labels:
Constant<TInt64> features = tf.constant(
    new float[][] {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    }
)

Constant<TInt64> labels = tf.constant(
    new float[] {
        0,
        1,
        1,
        0
    }
)

// These two "components" can be used to create a dataset:
Dataset dataset = Dataset.fromTensorSlices(tf, 
    Arrays.asList(features, labels),
    Arrays.asList(TInt64.DTYPE, TInt64.DTYPE)
);
```

### Iteration: Eager Mode

In eager mode, datasets can be iterated through using
a standard for-each loop.

```java
for (List<Output<?>> components : dataset) {
    Tensor<TInt64> X = components.get(0).tensor().expect(TInt64.DTYPE);
    Tensor<TInt64> y = components.get(1).tensor().expect(TInt64.DTYPE);

    // ... Use X and y in some computation
}
```

### Iteration: Graph Mode
In graph mode, datasets can be iterated through using the `OneShotIterator`
abstraction, and a while loop.

```java
OneShotIterator oneShotIterator = dataset.makeOneShotIterator();
Operation makeIterator = oneShotIterator.getMakeIteratorOp();
List<Output<?>> components = oneShotIterator.getComponents();

try (Session session = new Session(graph)) {
    // Run MakeIterator Op to set iterator position
    session.runner()
        .addTarget(makeIterator)
        .run();
    
    while (true) {
        try {
            List<Tensor<?>> outputs = session.runner()
                .fetch(components.get(0))
                .fetch(components.get(1))
                .run();

            Tensor<TInt64> X = outputs.get(0).expect(TInt64.DTYPE);
            Tensor<TInt64> y = outputs.get(1).expect(TInt64.DTYPE);

            // ... Use X and y in some computation
        } catch (IndexOutOfBoundsException e) {
            // Finished iterating
            break;
        }
    }
}
```

### Dataset Transformations

Datasets can be transformed in a variety of ways; one of the most
common transformations is "batching":

```java
Dataset batchedDataset = Dataset.fromTensorSlices( ... ).batch(BATCH_SIZE);
```

The `Dataset` objects are meant to be immutable; each call to a transformation
such as `.batch( ... )` returns a new `Dataset` object, wrapping a new `tf.data`
op.

The currently supported transformations include:
- `Dataset.batch`
- `Dataset.take`
- `Dataset.skip`

The goal is to grow this implementation to support all C++ `tf.data` ops.


## Questions and Discussion Topics

The initially supported features will be:

- Dataset creation from tensors with `Dataset.fromTensorSlices`
- Eager mode iteration over elements with Java `Iterable` (foreach) syntax
- Graph mode iteration over elements using the `OneShotIterator` abstraction
- Transformations: `Dataset.batch`, `Dataset.take`, `Dataset.skip`

### Mapping and Filtering
Mapping over and filtering elements within datasets is a very commonly
used feature. There don't seem to be `MapDataset` or `FilterDataset` ops
currently usable. Serializing a function type for `mapping` and `filter`
functions is an interesting issue.

A proposal for this could be to use the signatures

```java
Dataset.map(Function<List<Output<?>>, List<Output<?>>)
Dataset.filter(Function<List<Output<?>>, Boolean>)
```

and implement map and filter in java, on top of the iteration
schemes defined above. Would love to hear thoughts on this.