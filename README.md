# TensorFlow-Based Currency Detection
This project utilizes a TensorFlow model (Model.keras) for real-time and static image-based currency recognition. The implementation consists of two primary scripts: tensor.py and main.py.

# Implementation Overview
The tensor.py script contains core functions, including:

LoadModel(): Loads a pre-trained deep learning model capable of recognizing various currencies.
cameraPredict(): Uses a webcam feed to perform real-time currency detection.
Predict(image): Allows for currency detection using static image files.
In main.py, we import LoadModel and cameraPredict from tensor.py to handle model loading and real-time currency detection. Additionally, Predict() can be used to classify currency from an image file.

# Setup and Execution
Ensure TensorFlow, OpenCV, and NumPy are installed as dependencies.
Run main.py to initialize the model and start real-time detection.
The camera feed will activate automatically.
Press 'q' to exit the webcam-based detection.
To classify a static image, use the Predict(image) function.
Sample images are provided for testing.
YOLOv5-Based Object Detection for USD Banknotes
This project utilizes YOLOv5, a state-of-the-art object detection model, for recognizing the front and back of U.S. Federal Reserve Notes. The model was trained using 350 epochs with the YOLOv5s.pt pre-trained weights.

# Implementation Overview
The two main components are:

train.py: Used for training the detection model.
detect.py: Runs inference using a trained model.
The final trained model, 350epoch.pt, was optimized for detecting the front and back of $1, $5, $10, $20, and $50 USD bills.

# Challenges and Dataset Considerations
Initially, the goal was to detect multiple international currencies, but dataset limitations led to a focus on USD banknotes.
The training dataset consisted of 2,400 labeled images.
The model achieves 91% accuracy in detecting banknotes.
Training and Detection Commands
To train the model efficiently, an NVIDIA CUDA-enabled GPU is required.

# Training Command:
bash
Copy
Edit
python train.py --img 640 --epochs 350 --data data.yaml --weights yolov5s.pt
--img 640: Sets the image input size.
--epochs 350: Trains for 350 epochs.
--data data.yaml: Specifies the dataset configuration.
--weights yolov5s.pt: Uses YOLOv5's small pre-trained model as a base.
# Running Detection:
bash
Copy
Edit
python detect.py --source 0 --weights exp21/350epoch.pt
--source 0: Uses the default webcam as the input.
--weights exp21/350epoch.pt: Loads the trained model for inference.
Model and Dataset Download
Due to the size of the Model.keras file, it must be downloaded separately from Google Drive:

Download Model.keras

The dataset is included in the project repository. Additionally, the MakeModelAndTrain() function in main.py allows training from scratch if desired.

Training the TensorFlow Model
Instead of loading a pre-trained model, the MakeModelAndTrain() function can be used for training. It accepts the following parameters:

training_images
validation_images
class_names (from ImageData())
epochs (default: 15)
Once training completes, cameraPredict() is called to activate real-time detection.

# Prediction Using a Pre-Trained Model
To classify a single image, use:

python
Copy
Edit
Predict(model, image_dir, class_names)
Where:

model: The trained TensorFlow model.
image_dir: Path to the input image.
class_names: List of class labels.
Additional Functions
FourImagesWithPrediction(model, class_names, testData): Displays predictions for multiple test images.
SaveModel(model): Saves the trained model as Model.keras.
LoadModel(file): Loads a pre-trained model from .keras or .h5 format.
trainExistingModel(model, trainData, valData, epoch=15): Continues training an existing model.
Performance Visualization
After training, a plot displaying validation loss (Val_Loss) and training loss (Train_Loss) will appear. Close the window to proceed.
