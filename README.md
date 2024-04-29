## Distributed Data Parallel MNIST Training with TensorFlow 2 and SageMaker Distributed

This guide demonstrates how to use SageMaker's distributed data parallel library (`smdistributed.dataparallel`) to train a TensorFlow 2 model on the MNIST dataset.

**What is SageMaker Distributed Data Parallel Library?**

Amazon SageMaker's distributed library allows you to train deep learning models faster and more efficiently. The `smdistributed.dataparallel` feature offers a distributed data parallel training framework for PyTorch, TensorFlow, and MXNet.

**What this guide covers:**

1. **Introduction:** Briefly explains SageMaker's distributed data parallel library and its benefits.
2. **Dataset:** Introduces the MNIST dataset, a popular benchmark for handwritten digit classification.
3. **SageMaker Role:** Defines the IAM role required to create and run SageMaker training and hosting jobs.
4. **Model Training Script:** Explains the Python script (`train_tensorflow_smdataparallel_mnist.py`) used for training the TensorFlow model with `smdistributed.dataparallel`.
5. **SageMaker TensorFlow Estimator:** Demonstrates how to configure a SageMaker TensorFlow Estimator object, including:
   - Specifying the training script, instance type, instance count, and SageMaker session.
   - Setting the `distribution` strategy to use `smdistributed.dataparallel`.
6. **Training the Model:** Trains the TensorFlow model using the SageMaker Estimator.
7. **Model Deployment:** Deploys the trained model as an endpoint for real-time predictions.
8. **Inference:** Shows how to use the deployed endpoint for making predictions on new data.
9. **Cleanup:** Guides you on how to delete the endpoint if you don't intend to use it further.

**Additional Notes**

- The guide includes CI test results for different regions.
- For best performance, it's recommended to use instance types that support Amazon Elastic Fabric Adapter (e.g., ml.p3dn.24xlarge, ml.p4d.24xlarge) when training with `smdistributed.dataparallel`.

**Further Resources**

- TensorFlow in SageMaker: [https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html](https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html)
- SageMaker distributed data parallel API Specification: [https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html)
- SageMaker's Distributed Data Parallel Library: [https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html)
- Modify a TensorFlow 2.x Training Script Using SMD Data Parallel: [https://aws.amazon.com/tensorflow/](https://aws.amazon.com/tensorflow/)
