# Emotion Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify emotions from facial expressions. The model can be trained on a dataset of facial images and can detect seven different emotions in real-time using a webcam.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-detection.git
   cd emotion-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

There are two modes for running the code:

1. **Training mode**: To train the model:
   ```bash
   python src/emotion_detector.py train
   ```

2. **Display mode**: To start real-time emotion detection using a webcam:
   ```bash
   python src/emotion_detector.py display
   ```

## License
This project is licensed under the MIT License.
