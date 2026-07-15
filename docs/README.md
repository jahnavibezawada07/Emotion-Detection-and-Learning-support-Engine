# Emotion Detection & Learning Support Engine

An AI-powered application that detects students' emotions from text using deep learning models and provides personalized learning support through Google Gemini AI.

## Project Overview

The Emotion Detection & Learning Support Engine combines BiLSTM and BERT-based emotion classification to analyze students' learning-related text. The system identifies emotional states, detects mixed emotions, generates AI-assisted learning guidance, and presents interactive analytics through a Streamlit dashboard.

## Features

* BiLSTM and BERT emotion classification
* Mixed emotion detection
* AI-generated learning support using Google Gemini
* Interactive Plotly analytics dashboard
* CSV-based interaction history
* Cached model loading for improved performance
* User-friendly Streamlit interface

## Project Structure

```text
├── app.py
├── src/
├── models/
├── data/
├── docs/
├── requirements.txt
└── README.md
```

## Documentation

The **docs** folder contains the complete project documentation, including project reports, design documents, user manuals, and other supporting materials related to the development and implementation of the system.

## Technologies Used

* Python
* TensorFlow
* Hugging Face Transformers
* Streamlit
* Plotly
* Pandas
* Scikit-learn
* Google Gemini API

## Getting Started

1. Clone the repository.
2. Install the required dependencies.
3. Configure the `.env` file with your Gemini API key.
4. Run the application using:

```bash
streamlit run app.py
```

## Author

**Jahnavi Bezawada**

Bachelor of Technology – Computer Science & Engineering (Data Science)

