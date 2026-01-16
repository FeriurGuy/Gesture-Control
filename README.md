# Smart Hand Gesture Recognition System

A computer vision application that translates real-time hand gestures into spoken audio commands. This system leverages **MediaPipe** for hand landmark detection and **Scikit-Learn** for gesture classification, featuring a virtual touchless interface and low-latency audio feedback.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Python](https://img.shields.io/badge/Mediapipe-green)
![License](https://img.shields.io/badge/license-MIT-grey)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [Project Structure](#project-structure)
- [License](#license)

## Overview
This project facilitates Human-Computer Interaction (HCI) by converting static hand gestures into audible speech. It provides an end-to-end pipeline allowing users to record custom datasets, train a machine learning model, and execute real-time inference using a standard webcam.

## Key Features
* **Real-time Detection:** High-performance hand tracking using MediaPipe (21 landmarks).
* **Virtual Interface:** "Touchless" virtual buttons controlled by fingertip coordinates.
* **Custom Data Collection:** Integrated tools to record and label new gesture datasets easily.
* **Audio Caching Engine:** Implements MD5 hashing to cache generated audio, eliminating network latency for repeated phrases.
* **Robust Classification:** Uses a Random Forest Classifier to ensure high accuracy with minimal computational load.

## System Architecture
1.  **Input Layer:** Captures video feed and extracts hand landmarks (x, y, z coordinates).
2.  **Processing Layer:** Normalizes data and predicts the gesture class using the pre-trained model.
3.  **Interaction Layer:** Checks for virtual button collisions (e.g., Toggle Voice) and manages audio output queues.

## Installation

### Prerequisites
* Python 3.8 or higher
* Webcam

### Steps
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/FeriurGuy/Gesture-Control.git
    cd Gesture-Control
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage Workflow

This system is designed in three modular stages. Follow this order to get started:

### 1. Data Collection (`collect_data.py`)
Use this script to create a dataset for a new gesture.
```bash
python collect_data.py
