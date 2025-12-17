# Time_Series_Anomaly_Detection
The objective of this project is to build an end-to-end machine learning solution that detects anomalies in time-series sensor data collected from industrial equipment. The focus is on identifying unusual sensor patterns that deviate from normal operating behavior without relying on labeled anomaly data.
# Time Series Anomaly Detection for IoT Sensors

## Overview
This project implements an end-to-end anomaly detection system for time-series IoT sensor data to identify abnormal equipment behavior that may indicate early failure or maintenance needs. The solution reflects real-world industrial conditions where labeled anomalies are not available.

---

## Dataset

This project uses the NASA Bearing Dataset.

Due to file size limits, the raw dataset is not included in this repository.

Download the dataset from:
https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset

After downloading, place it in the following structure:

data/
└── 1st_test/



---

## Methodology
1. Raw sensor files were loaded and ordered chronologically  
2. Statistical features (mean, standard deviation, RMS, max) were extracted per time window  
3. Features were standardized for model compatibility  
4. Anomalies were detected using two unsupervised approaches:
   - Isolation Forest
   - Autoencoder (deep learning)
5. Results were validated using domain reasoning and visual inspection  

---

## Models Used
- **Isolation Forest:** Detects rare and abnormal patterns efficiently  
- **Autoencoder:** Learns normal behavior and flags deviations using reconstruction error  

---

## Evaluation Strategy
Labeled anomalies were not available; therefore, traditional supervised metrics such as accuracy and precision were not used. Model performance was validated through:
- Visualization of anomalies in time-series context  
- Expected bearing degradation behavior  
- Agreement between both models  

---

## Key Findings
- Sensor variability and vibration energy increase prior to bearing failure  
- Anomalies are rare during healthy operation and increase near failure  
- The approach supports predictive maintenance use cases  

---

## How to Run
1. Place the NASA Bearing Dataset inside the `data/` directory  
2. Open the Jupyter notebook:
3. Run all cells sequentially  

---

## Requirements
- Python 3.x  
- pandas, numpy, matplotlib  
- scikit-learn  
- tensorflow / keras  

Install dependencies:
