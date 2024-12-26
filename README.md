# Ransomware Detection through Processor and Disk Usage Analysis

## Overview
This project focuses on detecting ransomware attacks by analyzing processor and disk usage data. By monitoring anomalies in resource usage patterns, the system can identify potential ransomware activities and alert users before significant damage occurs.

## Key Features
- **Real-time Monitoring**: Continuously tracks processor and disk activity to detect abnormal patterns.
- **Anomaly Detection**: Uses machine learning models to identify deviations in system performance.
- **Alert System**: Sends alerts when potential ransomware behavior is detected.


## Project Workflow
1. **Data Collection**: Gather data from processor and disk usage logs.
2. **Feature Extraction**: Extract relevant features such as read/write speeds, CPU utilization, and I/O operations.
3. **Data Splitting**: Split data into training and testing sets.
4. **Model Training**: Train machine learning models on normal and malicious usage patterns.
5. **Anomaly Detection**: Apply trained models to detect unusual activities.


## Technology Stack
- **Programming Language**: Python
- **Libraries**:
  - Pandas (data manipulation)
  - Scikit-learn (machine learning)
  - Matplotlib/Seaborn (visualization)
  - NumPy (numerical computations)
- **Tools**:
  - Jupyter Notebooks
  
## Future Enhancements
- **Integration with Firewalls**
- **Multi-System Monitoring**
- **Automated Threat Mitigation**




