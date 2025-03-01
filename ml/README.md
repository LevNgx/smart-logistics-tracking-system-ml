# 🚀 Smart Logistics - Machine Learning Module

## 📌 Overview
This module contains all the **Machine Learning** logic for predicting **shipment delays**, optimizing routes, and detecting anomalies. It provides a **FastAPI service** to expose ML predictions to the backend.
## 🛠️ **Setup Instructions**
### **1️⃣ Install Python & Virtual Environment**
Ensure you have **Python 3.8+** installed. Then, create a virtual environment:

```bash
cd ml
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

Install all required Python packages using:
pip install -r requirements.txt

To train the model, run:
python src/train_model.py
✅ This will save the trained model inside the models/ directory.

To expose ML predictions via FastAPI, run:
uvicorn src.api:app --reload

📌 **Explanation of Dependencies:**
- `pandas`: Data manipulation.
- `numpy`: Numerical computations.
- `scikit-learn`: Machine learning models.
- `tensorflow`: LSTM time-series model.
- `matplotlib`: Data visualization.
- `seaborn`: Statistical plots.
- `fastapi`: API for ML predictions.
- `uvicorn`: Runs FastAPI server.
- `joblib`: Saves and loads ML models.
- `networkx`: For route optimization.
