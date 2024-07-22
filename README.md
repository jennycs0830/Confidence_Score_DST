# Confidence Score Estimation for Dialogue State Tracking (DST)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
Estimating a model's confidence in its outputs is critical for conversational AI systems based on LLMs, particularly for reducing hallucinations and preventing over-reliance. In this project, we explore various methods to quantify and leverage model uncertainty, focusing on DST in TODS. Our approach aims to provide well-calibrated confidence scores, improving the overall performance of the dialogue systems.

## Dataset
Our experiments use the MultiWOZ 2.2 corpus, a multi-domain task-oriented dialogue dataset. MultiWOZ is a human-human written dialogue dataset with 8K/1K/1K samples for training/validation/testing. We focus on five domains: Restaurant, Hotel, Train, Attraction, and Taxi. Each domain includes turn-level annotations and descriptions of slot labels.

## Methods
We evaluate four methods for estimating confidence scores:
1. Softmax
2. Raw token scores
3. Verbalized confidences
4. Combination of the above methods

Additionally, we enhance these methods with a self-probing mechanism. We use the Area Under the Curve (AUC) metric to assess calibration, with higher AUC indicating better calibration. Our experiments demonstrate the effectiveness of fine-tuning open-weight LLMs to achieve better-calibrated confidence scores and improved Joint Goal Accuracy (JGA) by 8.5% in zero-shot scenarios compared to closed-source models.

## Results
Our results show that incorporating confidence scores into the fine-tuning process significantly enhances DST performance. The combined confidence score method generates well-calibrated scores that are moderately correlated with ground truth labels, justifying its superior performance.

## Installation
To install the necessary dependencies for this project, run:
```bash
pip install -r requirements.txt
```

## Usage
To use the code in this repository, follow these steps:
1. Clone the respository
```bash
git clone https://github.com/yourusername/Confidence_Score_DST.git
cd Confidence_Score_DST
```

2. Get __`multiwoz-context-db.vec`__, which is a faiss database.
```bash
python create_faiss_db.py --output_faiss_db multiwoz-context-db.vec
```

3. 

