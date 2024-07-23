# Animal Image Classifier with Custom AlexNet

This project implements a custom AlexNet model to classify images of various animals. It includes functionality to download the dataset, preprocess it, train the model, evaluate its performance, and visualize the CNN layers and weights.

## Overview

1. **Download Dataset**: The dataset is downloaded from Kaggle using the Kaggle API.
2. **Load and Split Data**: The dataset is split into training and testing sets.
3. **Train Model**: The custom AlexNet model is trained on the training data with progress bars indicating the training process.
4. **Evaluate Model**: The trained model is evaluated on the testing data, and performance metrics are printed.
5. **Visualize Layers**: The CNN layers and weights are visualized using Matplotlib.

## Requirements

Ensure you have the following Python packages installed:

- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `kaggle`
- `tqdm`

You can install all required packages using:

```bash
pip install -r requirements.txt
```

## Files

- `animal_classifier.py`: Main script for downloading the dataset, training the model, and evaluating it.
- `requirements.txt`: List of dependencies required for the project.

## Usage

1. **Download Dataset**: The script will automatically download and unzip the dataset from Kaggle.
2. **Run Training**: Execute the script to start training and evaluating the model:

   ```bash
   python animal_classifier.py
   ```

3. **View Results**: The script will display the progress of training and evaluation, save the model, and visualize the CNN layers and weights.

## Google Colab

You can run this project on Google Colab using the following link:

[Google Colab Notebook](https://colab.research.google.com/drive/1GCeGeZtOLlDZUI-VjBq_f3JqgnRk0-k0?usp=sharing)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The custom AlexNet architecture is adapted from the original AlexNet paper.
- Dataset obtained from Kaggle.
- TQDM for progress bars and visualization tools.
