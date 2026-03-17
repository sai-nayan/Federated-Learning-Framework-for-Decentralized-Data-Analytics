# Federated Learning Framework for Decentralized Data Analytics

This repository contains a comprehensive Streamlit-based dashboard for orchestrating and evaluating Federated Learning (FL) models across multiple industries. It leverages the [Flower](https://flower.ai/) framework alongside popular machine learning backends (PyTorch, Scikit-Learn, and XGBoost) to simulate distributed training processes natively on your machine!

![Healthcare Domain Training Result](https://github.com/user-attachments/assets/your-image-url-here) <!-- Replace with actual image url or local path if needed -->

## Features
- **Healthcare (Neural Networks)**: Simulates deep learning image classification across medical institutions using a custom PyTorch MedicalResNet model.
- **Finance (XGBoost)**: Demonstrates federated aggregation of tree-based models across banks using XGBoost for credit risk assessment. 
- **Education (Random Forest)**: Simulates privacy-preserving student performance prediction across discrete schools using Scikit-Learn Random Forests.
- **Live Streamlit Dashboard**: Offers a beautiful multi-domain interactive visualizer with domain-specific theming, pulsing client node animations, and real-time federated training telemetry mappings.
- **Deterministic Simulation Control**: Provides robust fallback mechanics and deterministic centralized metric mapping to gracefully emulate complex FL topologies.

## Tech Stack
*   **Frontend**: Streamlit, HTML/CSS
*   **Federation Engine**: Flower (`flwr`), Ray
*   **Machine Learning**: PyTorch, Scikit-Learn, XGBoost, Pandas, Numpy 
*   **Data Visualization**: Altair

## Setup & Installation

**Prerequisites:** Python 3.9+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sai-nayan/Federated-Learning-Framework-for-Decentralized-Data-Analytics.git
   cd Federated-Learning-Framework-for-Decentralized-Data-Analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   Navigate to `http://localhost:8503` (or the port specified by Streamlit) in your browser.

## Project Structure
- `app.py`: The main Streamlit dashboard application.
- `federated_nn.py`: The PyTorch Neural Network federation script for Healthcare.
- `federated_finance.py`: The XGBoost federation script for Finance.
- `federated_student.py`: The Random Forest federation script for Education.
- `data/`: Contains the `.csv` demographic datasets for the various domains.

## License
MIT License
