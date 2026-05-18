# AgriDetect

## AI-Powered Smart Crop Monitoring and Disease Detection System

AgriDetect is an intelligent crop monitoring and disease detection platform that leverages Artificial Intelligence, Computer Vision, and Machine Learning to identify crop diseases from leaf images and provide smart agricultural insights.

The system is designed to support farmers and agricultural monitoring by enabling fast disease analysis, confidence-based predictions, and pesticide recommendations using deep learning techniques.

---

# Live Deployment

The project is deployed and accessible online:

https://agridetect1.netlify.app/

Note: The model is currently under active development and may occasionally produce inaccurate predictions for unsupported crops or unclear images.

---

# Features

- AI-based crop disease detection
- Deep Learning image classification model
- Leaf image analysis using Computer Vision
- FastAPI-powered backend API
- Interactive frontend interface
- Confidence scoring for predictions
- Smart pesticide recommendations
- Healthy leaf detection
- Multi-crop support
- Real-time image upload and analysis
- Live deployed web application

---

# Supported Crops

The current model is trained specifically on:

- Rice
- Cotton
- Wheat
- Tomato
- Potato
- Pepper
- Maize

The system may not provide accurate predictions for unsupported crops, pests, or unrelated plant images.

---

# Recommended Usage

For best results:

- Upload clear images of a single leaf
- Use good lighting conditions
- Ensure the affected area is visible
- Test using supported crop images only
- Avoid blurry or unrelated plant images

Sample images from the supported dataset can be added in the `sample_images/` folder for testing purposes.

---

# Technologies Used

## Programming and Development
- Python
- HTML5
- CSS3
- JavaScript

## AI and Machine Learning
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas

## Backend and APIs
- FastAPI
- REST APIs
- Uvicorn

## Database and Cloud
- Firebase
- SQLite

## Hardware and IoT (Planned Integration)
- Arduino
- Raspberry Pi
- IoT Sensors

---

# Project Structure

```text
AgriDetect/
│
├── backend/
│   ├── main.py
│   ├── agridetect_model.h5
│   ├── class_indices.json
│
├── frontend/
│   ├── index.html
│
├── sample_images/
│
├── images/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# System Workflow

```text
Leaf Image Upload
        ↓
Frontend Interface
        ↓
FastAPI Backend API
        ↓
TensorFlow Deep Learning Model
        ↓
Disease Prediction
        ↓
Confidence Score Generation
        ↓
Pesticide Recommendation
```

---

# AI Model Details

| Property | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Model Type | CNN-based Image Classifier |
| Image Input Size | 224 × 224 |
| Disease Classes | 32+ |
| Image Processing | OpenCV |

---

# Installation and Setup

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AgriDetect.git
cd AgriDetect
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Start Backend Server

```bash
cd backend
uvicorn main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

# API Endpoint

## Analyze Crop

```http
POST /api/analyze-crop
```

### Input
- Leaf image file

### Output

```json
{
  "pest_name": "Rice Brown Spot",
  "confidence": 94.2,
  "pesticide_recommendation": "Official remedy recommendation..."
}
```

---

# Key Functionalities

## Disease Detection
Identifies crop diseases from uploaded leaf images using deep learning models.

## Confidence-Based Predictions
Displays AI confidence scores to improve reliability and reduce incorrect diagnosis.

## Smart Recommendations
Provides pesticide and remedy suggestions for detected crop diseases.

## Healthy Leaf Classification
Detects healthy leaves to avoid unnecessary pesticide usage.

---

# Future Improvements

- Mobile application integration
- Multilingual support
- Cloud deployment improvements
- Real-time camera scanning
- Weather-aware disease analysis
- Fertilizer recommendation system
- IoT sensor integration
- Drone-assisted crop monitoring

---

# Author

## Rushil Gupta

Computer Science Undergraduate  
AI, Robotics, IoT, and Automation Enthusiast

---

# Project Vision

AgriDetect aims to combine Artificial Intelligence and smart agricultural technologies to support efficient farming practices, improve disease detection speed, and reduce crop losses through technology-driven solutions.

---

# Disclaimer

This project is intended for educational and research purposes.  
Agricultural recommendations should be verified with agricultural experts before real-world implementation.

---

# Support

If you found this project useful:

- Star the repository
- Fork the project
- Share feedback
- Contribute improvements

