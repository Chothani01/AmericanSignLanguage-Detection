# ✋ American Sign Language (ASL) Detection

A real-time American Sign Language (ASL) hand sign detection system built using **Python**, **OpenCV**, and **MediaPipe**.  

This project allows you to detect static ASL letters and numbers from webcam input and classify them into specific gestures using a trained Random Forest model.

---


## ⚡ Features

- ✅ Real-time hand tracking and detection
- ✅ Supports static ASL letters
- ✅ Uses MediaPipe for robust 21-point hand landmark detection
- ✅ Feature normalization for higher accuracy
- ✅ Modular code (easy to extend to more gestures or dynamic signs)
- ✅ Works on standard webcams (no extra hardware needed)

---

## 🏗️ Project Structure

AmericanSignLanguage-Detection/
├── data/ # Dataset images (folders per class)
├── HandTrackingModule.py # Hand tracking and landmark extraction module
├── main.py # Real-time detection script
├── train.py # Training script
├── model.pkl # Trained Random Forest model
├── README.md # Project documentation
