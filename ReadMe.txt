Predicting Personality Type from Online Forum Text


The project was done in Python 3.6. Make sure to also install all the imported packages. 

The dataset used can be found here: https://www.kaggle.com/datasnaek/mbti-type

The main scripts are:
config.py - configure the path of the source dataset
lstm.py - uses a Long Short Term Memory neural network for classification
svm.py - uses a Support Vector Machine for classification
gridsearch.py, gridsearch_full_label.py - use gridsearch to find the best parameters
ensemble.py - uses an ensemble classifier
heatmap.py - generates the heatmap of the LIWC features 
