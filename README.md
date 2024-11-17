# AI-Powered-Predictive-Maintenance-System-for-Vehicles-with-Real-Time-Data-Visualization-and-Analysis
This project aims to implement an AI-driven predictive maintenance system for engines, focusing on using Gradient Boosting Models to predict engine conditions based on sensor data. The solution includes:

- Data analysis and feature engineering.
- A trained model saved as a .pkl file.
- A user-friendly interface using Streamlit to predict engine health.


# Introduction
Predictive maintenance powered by AI-driven analytics is transforming industries by shifting from reactive to proactive maintenance strategies, particularly in the automotive sector. Volkswagen is at the forefront of leveraging this technology to enhance the reliability and performance of their vehicles. Through an intelligent maintenance ecosystem, Volkswagen integrates real-time sensor data, historical maintenance records, and external factors like weather and driving patterns to predict potential engine failures with high accuracy.

This system uses advanced machine learning models, such as Gradient Boosting, to process large datasets and provide actionable insights. These predictions enable preemptive alerts for maintenance, optimize spare parts inventory, and enhance customer satisfaction by minimizing vehicle downtime and ensuring better safety and reliability. The system's continuous learning ability allows for improved predictions over time.

The implementation is divided into several phases: data integration, AI model development, testing with select OEMs, and full-scale rollout across networks. By adopting this AI-driven predictive maintenance model, Volkswagen aims to revolutionize vehicle management, providing enhanced maintenance solutions for better vehicle performance and customer experience.


# Features
- Collects real-time sensor data from vehicles.
- Preprocesses sensor data for reliability.
- Utilizes GBM machine learning model for predictive maintenance.
- Integrates a web application interface using Streamlit for real-time data visualization and predictions.
- Predicts maintenance probability based on model output.
- Estimates maintenance date 2-3 weeks in advance.
- Provides probability percentage for potential part failure.
- Enhances predictive accuracy and allows proactive intervention.
- Facilitates timely maintenance scheduling to minimize downtime.


# Advantages
- Regular maintenance guarantees the best possible performance from your car.
- Prior to starting lengthy trips, resources are optimised and emergency repairs are avoided.
- Increases the safety of both passengers and drivers.
- Reduces unplanned breakdowns, increasing supply chain efficiency.
- Ensures dependable transport services, which raises customer satisfaction.


# Repository Contents

- **Data/**: Contains the dataset used for training the model.
  - **engine.csv**: Contains engine performance data.
- **models/**: Includes the trained model (model.pkl) used for inference
- **app/**: Streamlit app for user interaction and predictions.
- **notebooks/**: A notebook with exploratory data analysis (EDA) and model training processes.
- **images/**: Stores visual outputs like architecture diagrams and ROC.
- **requirements.txt**: Lists Python libraries required for running the project.


# Dependencies
The project requires the following Python libraries:
- Pandas


![arch1](https://github.com/user-attachments/assets/79b49b70-5563-4553-ad84-8b3928a5c19d)

- Numpy
- scikit-learn
- Matplotlib
- seaborn
- streamlit
- pickle-mixin












