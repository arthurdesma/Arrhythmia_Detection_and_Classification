# Cardiac Arrhythmia Detection and Classification from ECG Data

This project aims to identify irregularities in electrocardiogram (ECG) data using operational artificial intelligence models. It was developed by Arthur DESMAZURES, Paul GILQUIN, Pauline GOUILLART, Candice MARCHAND, and Mathilde SCHLIENGER as part of the E4 Project at ESIEE Paris.

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
  - [Cardiac Function](#cardiac-function)
  - [ECG](#ecg)
  - [Arrhythmias](#arrhythmias)
- [Dataset](#dataset) 
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Models](#models)
    - [Decision Tree](#decision-tree)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)
    - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
    - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
    - [CNN Time Series](#cnn-time-series)
- [Results](#results)
  - [Model Selection](#model-selection)
  - [User Interface](#user-interface)
    - [Design](#design)
    - [Development](#development)
    - [AI Integration](#ai-integration)
    - [Result Interpretation](#result-interpretation)
- [Ethics](#ethics)
- [Conclusion](#conclusion)
  - [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Introduction
This project focuses on developing predictive models to identify cardiac arrhythmias from ECG data using the MIT-BIH Arrhythmia Database from PhysioNet. It involved understanding arrhythmias and their ECG representation, selecting and testing various machine learning models, and integrating the best models into a user-friendly interface for healthcare professionals.

## Background

### Cardiac Function
The heart is a complex organ that pumps blood through the circulatory system. It consists of four chambers (right and left atria, right and left ventricles) that work together to ensure continuous blood flow. The cardiac cycle is a series of mechanical and electrical events that occur in less than a second, regulated by electrical signals originating from the sinoatrial node.

### ECG
An electrocardiogram (ECG) is a non-invasive diagnostic tool that captures the heart's electrical activity over a period of time through electrodes placed on the patient's skin. It is essential for evaluating heart rhythm, detecting arrhythmias, and diagnosing various cardiac diseases. The standard 12-lead ECG provides a comprehensive view of cardiac activity from different angles.

### Arrhythmias
Arrhythmias are abnormal heart rhythms that can be identified by ECG. Some common types include ventricular fibrillation, bradycardia, tachycardia, and extrasystoles. Causes can vary from physical abnormalities to lifestyle factors, and symptoms range from mild to severe. Early detection and treatment are crucial for patient outcomes.

## Dataset
The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects between 1975 and 1979. The recordings were digitized at 360 samples per second and annotated by at least two cardiologists. This dataset provides a diverse representation of ECG morphologies and artifacts encountered in clinical practice.

## Methodology

### Data Preprocessing
- Grouped arrhythmias into 5 categories for model training
- Applied MinMaxScaler to normalize data between 0 and 1
- Denoised signal using PyWavelets library

### Models
Several classification models were tested:

#### Decision Tree
Decision trees provide simple visual interpretations of decisions and are effective for handling heterogeneous, non-linear data.

#### Support Vector Machine (SVM)
SVMs are efficient in finding maximum margin separating hyperplanes between classes, making them robust, especially in high-dimensional spaces.

#### Long Short-Term Memory (LSTM) 
LSTMs are designed to recognize patterns in data sequences, making them ideal for applications like speech recognition and time series prediction.

#### Convolutional Neural Network (CNN)
CNNs excel at analyzing images by exploiting the spatial structure of data, making them perfect for computer vision and image analysis tasks.

#### CNN Time Series
A CNN time series model was developed to classify specific anomalies in the abnormal ECGs.

## Results

### Model Selection
The CNN model demonstrated the best performance in categorizing healthy vs abnormal ECGs based on precision, recall, F1 score, ROC AUC, and log loss metrics. A subsequent CNN time series model was developed to classify specific anomalies in the abnormal ECGs.

### User Interface

#### Design
The user interface was designed using Figma and converted to a Tkinter GUI using Tkinter-Designer. It features several pages for file selection, ECG visualization, AI prediction review, anomaly summary, and detailed AI results.

#### Development
The interface was developed in Python using Tkinter. Pages were connected and button functionalities were implemented to enable smooth user interaction and data flow.

#### AI Integration
The trained AI models were integrated into the user interface to assist physicians in their diagnosis. The models highlight anomalies in red and healthy waves in blue on the ECG plot.

#### Result Interpretation
The interface provides a summary of detected anomalies and a breakdown of specific anomaly types (Q, S, F, V) predicted by the AI model. These results are intended to support, not replace, the physician's judgment.

## Ethics
The project discusses ethical considerations around the use of AI in healthcare, including risks, limitations, and the need for human oversight by medical professionals. Relevant regulations like GDPR and the proposed AI Act are also highlighted. Key ethical points include:
- Defining validation thresholds for AI results
- Ensuring physician verification of AI classifications
- Addressing potential biases and errors in AI models
- Clarifying liability in case of AI-related errors
- Adhering to data protection and privacy regulations

## Conclusion
This project successfully developed an AI-assisted ECG analysis tool for detecting and classifying cardiac arrhythmias. The combination of a high-performing CNN model and a user-friendly interface has the potential to support healthcare professionals in their diagnostic process.

### Future Work
Potential future improvements include:
- Enhancing model accuracy with more data and expert input
- Further developing the user interface with more robust tools
- Tailoring the application to specific healthcare roles
- Conducting user studies to evaluate the tool's effectiveness and usability

## Acknowledgments
The team extends their gratitude to Prof. Adrien UGON for his invaluable support and guidance throughout the project, and to ESIEE Paris for facilitating this meaningful learning experience.
