Voice Authenticity Detection API

A lightweight FastAPI-based service that detects whether a voice recording is human or AI-generated using Wav2Vec2 embeddings and a machine learning classifier.

Features

Accepts Base64-encoded audio

Detects AI-generated vs human voice

Uses Wav2Vec2 deep audio embeddings

Scikit-learn classifier with PCA

API key authentication

Rate limiting

Response caching

Project Structure
project-root/
│
├── app/
│   ├── models/                # Saved ML artifacts
│   │   ├── scaler.joblib
│   │   ├── pca.joblib
│   │   └── classifier.joblib
│   │
│   ├── feature_extractor.py  # Wav2Vec2 feature extraction
│   ├── inference.py          # Prediction pipeline
│   ├── main.py               # FastAPI application
│   ├── rate_limiter.py       # API rate limiting
│   ├── schemas.py            # Request/response models
│   ├── security.py           # API key validation
│   └── utils.py              # Audio preprocessing
│
├── Dataset/                  # Training dataset
│
├── train_model.py            # Model training script
├── requirements.txt
├── .env
└── README.md

Requirements

Python 3.9+

pip

virtualenv (recommended)

Setup Instructions
1. Clone the repository
git clone <repo-url>
cd <project-folder>

2. Create virtual environment

Mac/Linux

python3 -m venv venv
source venv/bin/activate


Windows

python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables

Create a .env file in the root directory:

API_KEY=your_secret_api_key

5. Ensure model files exist

Inside:

app/models/


You must have:

scaler.joblib
pca.joblib
classifier.joblib


If not, train the model.

Training the Model
Dataset Structure
Dataset/
├── Human/
│   ├── English/
│   ├── Hindi/
│   └── ...
│
└── AI/
    ├── English/
    ├── Hindi/
    └── ...


Each language folder contains .wav or .mp3 files.

Run training
python train_model.py


After training, the following files will be generated:

scaler.joblib
pca.joblib
classifier.joblib


Move them to:

app/models/

Running the API

Start the FastAPI server:

uvicorn app.main:app --reload


Server will run at:

http://127.0.0.1:8000


Interactive docs:

http://127.0.0.1:8000/docs

API Endpoint
POST /api/voice-detection

Detect whether audio is AI-generated or human.

Headers
x-api-key: your_secret_api_key
Content-Type: application/json

Request Body
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_AUDIO_STRING"
}

Success Response
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.92
}

Error Response
{
  "status": "error",
  "message": "Invalid API key"
}

Rate Limiting

15 requests per minute per API key

If exceeded:

{
  "status": "error",
  "message": "Too many requests. Please try again later."
}

Caching

Predictions are cached for 5 minutes

Prevents recomputation for identical audio

Model Pipeline

Audio preprocessing

Silence trimming and normalization

Wav2Vec2 embedding extraction

Feature scaling

PCA dimensionality reduction

Classifier prediction

Tech Stack

FastAPI

PyTorch

HuggingFace Transformers

Scikit-learn

LightGBM

Librosa

Example cURL Request
curl -X POST "http://127.0.0.1:8000/api/voice-detection" \
-H "x-api-key: your_secret_api_key" \
-H "Content-Type: application/json" \
-d '{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_STRING"
}'

Notes

Maximum audio length: 12 seconds

Maximum audio size: 5 MB

Supported formats: MP3

Future Improvements

Redis-based rate limiting

GPU batch inference

Torch-based classifier head

Multi-language calibration

Real-time streaming support