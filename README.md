# fact-checker
automated fact-checker built based on BERT



# Automated Fact-checker

### Important Files:
* BERT_fact_checker.ipynb : describes the steps of implementation
* src/ bertClassifier.py : contains class and functions for initializing and training the BERT model


### Installed libraries
* transformers 
* datasets 
* sentence_transformers 
* umap-learn


### Important libraries
```bash
import sklearn
from transformers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from src.bertClassifier import *
```

### Implementation steps
1.   Load Data
2.   Process Data
3.   Develop the Model (BERT)
4.   Predict & Evaluate (62% Acc.)
5.   Data Augmentation + Predict & Evaluate (XX% Acc.)
6.   ANNEX - Data visualization
