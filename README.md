# License Plate Recognition (LPR) System in Python

This project implements a License Plate Recognition (LPR) system in Python.
The system uses **image processing** + **machine learning** to detect license plates from vehicle images, segment characters, and recognize them.

## Overview

This LPR system takes an input image of a vehicle and outputs the text on its license plate.  
It is composed of three core stages:

1. **License Plate Detection**: find the region of the license plate in the image.
2. **Character Segmentation**: segment each character from the detected plate.
3. **Character Recognition**: use a machine learning classifier to identify each character.

## Pipeline

1. **Preprocess image** — convert to grayscale, binarize. (**localization.py**)
2. **Connected Component Analysis (CCA)** — find candidate regions via connected regions labeling. (**cca.py**)
3. **Filter regions** — apply heuristics (e.g., aspect ratio, relative size) to pick plate-like regions. (**cca2.py**)
4. **Vertical projection** — further validate plate region based on pixel intensity sums across columns.
5. **Character segmentation** — apply CCA again on the plate region, isolate individual characters, resize them (20×20 px). (**segmentation.py**)
6. **Sorting** — sort segmented character images by their x-locations to recover the correct order.
7. **Character recognition** — train a classifier (SVM / SVC) to predict character labels. (**machine_train.py**)
8. **Inference** — apply the trained model to recognize characters on new images. (**prediction.py**)

## Outputs

## Dependencies

Here are some of the key Python packages used in the project (per original tutorial):

- `scikit-image` for image processing
- `numpy` a python package that helps in handling n-dimensional arrays and matrices
- `scipy` for scientific python
- `matplotlib` for visualization
- `Pillow` for image I/O
- `scikit-learn` for the machine learning classifier (SVC)

## Setup

```bash
# Create and activate a virtual environment
pip install virtualenv
virtualenv lpr
source lpr/bin/activate

# Install dependencies
pip install -r requirements.txt

# To run
# upload car picture in directory
# change car image in localization.py
# run all python files accoding to pipeline steps
python localization.py # etc...
```
