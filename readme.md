# Lung Cancer Detection from ECG Data

This repository contains the code and documentation for a machine learning project aimed at detecting lung cancer from ECG data. The project utilizes Convolutional Neural Networks (CNNs) to analyze ECG signals and identify potential indicators of lung cancer.

## Project Objective

The goal of this project is to develop a predictive model that can assist medical professionals by providing preliminary analysis of ECG data to detect lung cancer. This tool aims to enhance the accuracy and speed of lung cancer diagnosis.

## Dataset

This project uses the IQ-OTH/NCCD Lung Cancer Dataset (Augmented). Due to privacy and size considerations, the dataset is not included in this repository. 

### Data Description

The dataset comprises augmented chest CT scans labeled with indications of lung cancer presence. Each record includes:

- CT scan images
- Binary labels indicating the presence of lung cancer

## Installation

To set up a local development environment, follow these steps:

```bash
git clone https://github.com/Issac-Kondreddy/Lung-Cancer-Detection.git
cd Lung-Cancer-Detection
pip install -r requirements.txt
```

## Usage
To run the model training script, use:
```bash
python train_model.py
```
For model evaluation, use:
```bash
python evaluate_model.py
```
## Model
The model is based on a Convolutional Neural Network (CNN) architecture, which is particularly effective for image-based data like CT scans. Details about the model architecture, training process, and evaluation metrics are documented in the model documentation.

## Contributing
Contributions to this project are welcome! Please refer to CONTRIBUTING.md for details on how to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.


## Contact
For any questions or concerns, please contact at issackondreddy@gmail.com

## Acknowledgments
Thanks to ```Hamdalla F. Al-Yasriy``` for providing the dataset.
Acknowledge any other support or data sources here.

