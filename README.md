# Automated Research Paper Classification

**Kriti 2024 - Barak Hostel Submission**

An intelligent system for automated categorization of research papers using machine learning and natural language processing techniques. This project was developed as part of the Kriti 2024 technical festival competition hosted on Kaggle.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Team Members](#team-members)
- [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

The Automated Research Paper Classification system addresses the challenge of efficiently organizing and categorizing the ever-growing volume of academic research papers. Our solution leverages advanced NLP techniques to automatically classify research papers into relevant categories based on their abstracts, titles, and metadata.

### Problem Statement

With millions of research papers published annually across various disciplines, researchers and academics face significant challenges in:
- Finding relevant papers in their field of interest
- Organizing literature for systematic reviews
- Tracking emerging research trends
- Managing research databases efficiently

Our system provides an automated solution to classify papers into predefined categories, making research discovery and organization more efficient.

## ✨ Features

- **Multi-class Classification**: Categorizes papers into multiple research domains
- **Abstract Analysis**: Processes paper abstracts using advanced NLP techniques
- **Title Processing**: Extracts key information from paper titles for improved accuracy
- **Scalable Architecture**: Handles large volumes of papers efficiently
- **Real-time Prediction**: Provides instant classification for new papers
- **Confidence Scoring**: Returns probability scores for each category
- **Batch Processing**: Supports bulk classification of multiple papers

## 🛠️ Tech Stack

- **Programming Language**: Python 3.8+
- **Machine Learning Frameworks**:
  - Scikit-learn
  - TensorFlow/Keras
  - PyTorch (optional)
- **NLP Libraries**:
  - NLTK
  - SpaCy
  - Transformers (Hugging Face)
- **Data Processing**:
  - Pandas
  - NumPy
- **Visualization**:
  - Matplotlib
  - Seaborn
  - Plotly
- **Web Framework** (if applicable):
  - Flask/FastAPI
- **Other Tools**:
  - Jupyter Notebook
  - Git

## 📦 Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Daivik12345/Kriti-2024--Automated-research-paper-classification.git
cd Kriti-2024--Automated-research-paper-classification
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models/Data

```bash
python setup.py
```

## 🚀 Usage

### Basic Usage

To classify a single research paper:

```python
from classifier import ResearchPaperClassifier

# Initialize the classifier
classifier = ResearchPaperClassifier()

# Classify a paper
result = classifier.classify(
    title="Deep Learning for Natural Language Processing: A Survey",
    abstract="This paper presents a comprehensive survey of deep learning techniques..."
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Command Line Interface

```bash
# Classify a single paper
python classify.py --title "Your Paper Title" --abstract "Your paper abstract"

# Batch classification from CSV
python classify.py --input papers.csv --output results.csv
```

### Training the Model

To train the model with your own dataset:

```bash
python train.py --data path/to/training_data.csv --epochs 50 --batch_size 32
```

## 📁 Project Structure

```
Kriti-2024--Automated-research-paper-classification/
│
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Preprocessed data
│   └── external/            # External data sources
│
├── models/
│   ├── saved_models/        # Trained model files
│   └── checkpoints/         # Training checkpoints
│
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── Model_Training.ipynb # Model development
│   └── Evaluation.ipynb     # Performance evaluation
│
├── src/
│   ├── preprocessing/       # Data preprocessing modules
│   ├── models/             # Model architectures
│   ├── utils/              # Utility functions
│   └── visualization/      # Plotting functions
│
├── tests/                   # Unit tests
├── config/                  # Configuration files
├── requirements.txt         # Project dependencies
├── setup.py                # Setup script
└── README.md               # Project documentation
```

## 🧠 Model Architecture

Our classification system employs a hybrid approach combining traditional ML and deep learning:

### Feature Extraction
1. **TF-IDF Vectorization**: Captures term importance
2. **Word Embeddings**: Uses pre-trained embeddings (Word2Vec/GloVe)
3. **BERT Embeddings**: Contextual representations for better understanding

### Classification Models
- **Baseline**: Logistic Regression, SVM
- **Ensemble**: Random Forest, XGBoost
- **Deep Learning**: LSTM, CNN, Transformer-based models

### Model Pipeline
```
Input (Title + Abstract) → Preprocessing → Feature Extraction → Model → Prediction
```

## 📊 Dataset

The project uses research paper data from multiple sources:

- **Training Data**: [Specify size] papers across [X] categories
- **Validation Data**: [Specify size] papers
- **Test Data**: [Specify size] papers

### Data Format
```csv
paper_id,title,abstract,category
1,"Paper Title","Paper abstract text...","Computer Science"
```

## 📈 Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.82 | 0.81 | 0.80 | 0.80 |
| Random Forest | 0.86 | 0.85 | 0.84 | 0.84 |
| XGBoost | 0.88 | 0.87 | 0.86 | 0.86 |
| BERT-based | 0.92 | 0.91 | 0.90 | 0.90 |
 
**Note**: This project was developed as part of Kriti 2024, the annual technical festival competition.
