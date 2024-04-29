import os
import tensorflow as tf
from sagemaker.tensorflow import TensorFlow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

# Step 1: Define the TensorFlow script file containing your training code
script_file = 'train.py'

# Step 2: Write your TensorFlow training code into the script file
with open(script_file, 'w') as f:
    f.write("""
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

# Step 2: Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 3: Preprocessing the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Step 4: Define the TensorFlow model
def create_model():
    model = Sequential([
        Reshape(target_shape=(28, 28, 1), input_shape=(28, 28, 1)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 5: Train the model
model = create_model()
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
""")

# Step 3: Specify the role for your SageMaker training job
role = 'arn:aws:iam::620527015753:role/AmazonSageMaker-ExecutionRole'

# Step 4: Specify the number of instances for distributed training
instance_count = 2  # You can adjust this based on your requirements

# Step 5: Specify the instance type for distributed training
#instance_type = 'ml.p3.8xlarge'  Or any other instance type that supports multiple GPUs
instance_type = 'ml.m5.2xlarge'
# Step 6: Specify the URI of the pre-built Docker image for TensorFlow training
image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.1-cpu-py39-ubuntu20.04'

# Step 7: Create a TensorFlow Estimator with the pre-built image
estimator = TensorFlow(entry_point=script_file,
                       role=role,
                       instance_count=instance_count,
                       instance_type=instance_type,
                       image_uri=image_uri)

# Step 8: Launch the distributed training job
estimator.fit()