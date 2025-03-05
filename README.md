**Smart Logistics & Delivery Tracking System** 🚀

**Overview**
The Smart Logistics & Delivery Tracking System is a full-stack application that provides real-time shipment tracking, AI-powered delay predictions, and route optimization to enhance logistics efficiency. This project integrates machine learning, cloud computing, and modern DevOps practices, making it a cutting-edge solution for supply chain management.

**Features**
✅ Real-Time Shipment Tracking – Monitor deliveries and status updates via an interactive dashboard.
✅ AI-Powered Delay Prediction – Uses machine learning to predict delays based on historical and live data.
✅ Route Optimization – Suggests optimal delivery routes using AI.
✅ Microservices Architecture – Backend services built using Spring Boot & Java for scalability.
✅ SQL Database Integration – Stores shipment and tracking details securely.
✅ Cloud Deployment – Hosted on AWS (EC2, RDS, S3) for high availability.
✅ CI/CD Pipeline – Automates builds and deployments using Jenkins, Docker, and Kubernetes.

**Tech Stack 💻**

1. React	Frontend (User Interface & Dashboard)
2. Spring Boot	Backend (Microservices & REST APIs)
3. MySQL Database (Stores shipment & tracking data)
4. Python (Flask/FastAPI, Scikit-Learn)	Machine Learning (Delay Prediction & Route Optimization)
5. aws (EC2, RDS, S3)	Cloud Deployment
6. Docker & Kubernetes	Containerization & Orchestration
7. Jenkins	CI/CD Automation
8. GitHub	Version Control



This project predicts **shipment delays** using a **hybrid machine learning model** with:
- ✅ **XGBoost** (structured tabular data)
- ✅ **LSTM Neural Networks** (time-series learning)
- ✅ **TensorBoard** for real-time training visualization

📌 Running XGBoost on Different Systems
XGBoost requires OpenMP for optimized parallel processing.
Depending on your operating system, you may need extra steps to install OpenMP.
✅ MacOS (Apple Silicon & Intel)
Run the following command:
brew install libomp

📌 If XGBoost still fails with an OpenMP error, try setting the environment variable manually:
export PATH="/opt/homebrew/opt/libomp/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

📌 Then, reinstall XGBoost:
pip uninstall xgboost -y
pip install xgboost

