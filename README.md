## Problem Statement Discussion

The challenge at hand involves the classification of electrocardiogram (ECG) signals into distinct diagnostic groups, with the ultimate goal of predicting blood pressure (BP) levels in patients. Analyzing ECG data can unveil specific patterns and anomalies, offering crucial insights into an individual's cardiovascular health. Integrating these ECG findings with BP prediction enhances our capability to detect and manage hypertension at an early stage, facilitating proactive healthcare interventions.

## Process and Approach

In embarking on my inaugural deep learning project, I began with a MATLAB code sample designed for classifying ECG data from the PhysioNet 2017 challenge. This dataset comprised single short ECG lead recordings categorized into two classes: Normal and Atrial fibrillation.

Translating the MATLAB example into Python and acquainting myself with implementing Deep Neural Networks, I progressed to seeking multi-class ECG datasets. The PTB-XL dataset from Physionet, consisting of 21,799 clinical 12-lead ECG signals, caught my attention. It was categorized into five diagnostic classes: ‘Normal ECG’, ‘Myocardial Infarction’, ‘ST/T Change’, ‘Conduction Disturbance’, and ‘Hypertrophy’.

Encountering an imbalance with approximately 10,000 signals classified as Normal, I addressed this issue by augmenting signals from minority classes. This involved repeating available signals and introducing noise to each one, achieving a dual objective of balancing class signals and increasing the overall dataset size.

Initially, I experimented with simple deep CNN models featuring 2-3 convolutional layers and fully connected layers. This approach yielded a test accuracy of 83%, but further improvement proved challenging.

Subsequently, I explored the Residual Network model. Adapting the original ResNet architecture, I implemented several modifications to enhance accuracy.

The outcome was a training accuracy of 97% and a validation accuracy of 93%.

This journey represents a systematic approach to leveraging deep learning for ECG classification, showcasing the evolution from initial exploration to the adoption of advanced models for enhanced accuracy.
