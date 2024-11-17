# AI-Powered-Predictive-Maintenance-System-for-Vehicles-with-Real-Time-Data-Visualization-and-Analysis
This project aims to implement an AI-driven predictive maintenance system for engines, focusing on using Gradient Boosting Models to predict engine conditions based on sensor data. The solution includes:

- Data analysis and feature engineering.
- A trained model saved as a .pkl file.
- A user-friendly interface using Streamlit to predict engine health.


# 1. Introduction
Predictive maintenance powered by AI-driven analytics is transforming industries by shifting from reactive to proactive maintenance strategies, particularly in the automotive sector. Volkswagen is at the forefront of leveraging this technology to enhance the reliability and performance of their vehicles. Through an intelligent maintenance ecosystem, Volkswagen integrates real-time sensor data, historical maintenance records, and external factors like weather and driving patterns to predict potential engine failures with high accuracy.

This system uses advanced machine learning models, such as Gradient Boosting, to process large datasets and provide actionable insights. These predictions enable preemptive alerts for maintenance, optimize spare parts inventory, and enhance customer satisfaction by minimizing vehicle downtime and ensuring better safety and reliability. The system's continuous learning ability allows for improved predictions over time.

The implementation is divided into several phases: data integration, AI model development, testing with select OEMs, and full-scale rollout across networks. By adopting this AI-driven predictive maintenance model, Volkswagen aims to revolutionize vehicle management, providing enhanced maintenance solutions for better vehicle performance and customer experience.

# 2. Advantages
- Regular maintenance guarantees the best possible performance from your car.
- Prior to starting lengthy trips, resources are optimised and emergency repairs are avoided.
- Increases the safety of both passengers and drivers.
- Reduces unplanned breakdowns, increasing supply chain efficiency.
- Ensures dependable transport services, which raises customer satisfaction.

# 3. Features
- Collects real-time sensor data from vehicles.
- Preprocesses sensor data for reliability.
- Utilizes GBM machine learning model for predictive maintenance.
- Integrates a web application interface using Streamlit for real-time data visualization and predictions.
- Predicts maintenance probability based on model output.
- Estimates maintenance date 2-3 weeks in advance.
- Provides probability percentage for potential part failure.
- Enhances predictive accuracy and allows proactive intervention.
- Facilitates timely maintenance scheduling to minimize downtime.

# 4. Dataset
File: engine_data.csv
- The dataset contains sensor readings like engine RPM, lubricant oil pressure, fuel pressure, and more, along with labels indicating engine condition.

# 5. Architecture


![arch1](https://github.com/user-attachments/assets/796ffabf-3b81-4a60-a413-27de3f451a47)

# 6. Usage
- Access the web interface via localhost:8501.
- Adjust the sliders to simulate sensor values and get predictions.
- View confidence levels for predictions and suggestions for maintenance.

![Values1](https://github.com/user-attachments/assets/39e3a577-b8c3-4587-b4ed-c699955f1361)




# 5. Repository Contents

- **Data/**: Contains the dataset used for training the model.
  - **engine.csv**: Contains engine performance data.
- **models/**: Includes the trained model (model.pkl) used for inference
- **app/**: Streamlit app for user interaction and predictions.
- **notebooks/**: A notebook with exploratory data analysis (EDA) and model training processes.
- **images/**: Stores visual outputs like architecture diagrams and ROC.
- **requirements.txt**: Lists Python libraries required for running the project.


# 6. Dependencies
The project requires the following Python libraries:
- Pandas
![download](https://github.com/user-attachments/assets/bc870cbc-1a76-40f0-ab50-114cfe183937)
- Numpy
![download](https://github.com/user-attachments/assets/3fcfdadf-239d-4bb8-a84b-2e879ce639d7)
- scikit-learn
![download](https://github.com/user-attachments/assets/2607b27e-26a3-42d9-80fa-037ae9247a41)
- Matplotlib
![download](https://github.com/user-attachments/assets/471432df-7693-4dab-8fbe-28db6abc5bd5)


- seaborn
![download](https://github.com/user-attachments/assets/f47a7af8-418f-4bce-9272-5544f0bee576)

- streamlit
![download](https://github.com/user-attachments/assets/583ea90c-fa2d-4102-a9f5-c5842f4c94b4)


# 7. Results

ROC Curve: Displays the true positive rate vs. false positive rate for the model (image from images/roc_curve.png)
![download](https://github.com/user-attachments/assets/3c2f0c4e-29bd-423e-a66f-cd385ded5775)

# 8. Future Work

- Integration with IoT devices for real-time data collection.
- Expand model capabilities to detect additional engine faults.
- Build an alert system for predictive maintenance notifications.










