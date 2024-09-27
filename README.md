# Nutritional Information Extraction from Product Images

## Overview

This project aims to automate the extraction of nutritional information from product images. The pipeline includes image preprocessing, optical character recognition (OCR), data preparation for BERT (Bidirectional Encoder Representations from Transformers), and model training and validation.

## Features

- **Image Preprocessing**: Resize and normalize images for improved OCR performance.
- **OCR**: Extract text from images using OCR techniques.
- **Data Preparation**: Prepare extracted text for BERT, including mapping nutritional values to labels.
- **Model Training**: Train a BERT-based model for entity tagging and classification.
- **Validation**: Evaluate the trained model to ensure its accuracy and performance.

## Folder Structure

```plaintext
/project
├── /data
│   └── (place your images and other raw data files here)
├── /scripts
│   ├── preprocessing.py         # Image preprocessing functions
│   ├── ocr.py                   # OCR functions
│   ├── data_preparation.py      # Functions to prepare data for BERT
│   ├── model_training.py        # Model training functions
│   └── validation.py            # Model validation functions
├── /config
│   └── config.yaml              # Configuration file for parameters and settings
├── /results
│   └── (output from model training and evaluation, like saved models and logs)
├── /logs
│   └── (log files from model training and evaluation)
├── requirements.txt             # List of required Python packages
├── README.md                    # Project overview and instructions
└── main.py                      # Entry point for running the entire pipeline
