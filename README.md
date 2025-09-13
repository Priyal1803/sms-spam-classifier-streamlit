# SMS Spam Classifier

This project is a *Machine Learning-based SMS Spam Classifier* that identifies whether an SMS message is Spam or Not Spam.  
It uses TF-IDF Vectorization for feature extraction and multiple classification algorithms like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) for prediction.  
The app is deployed using Streamlit for an interactive web interface.

---

## Project Structure

- *notebooks/* : Jupyter Notebooks for data exploration, feature extraction, and model training.  
- *app.py* :  Main application file (developed in PyCharm), loads the trained model & vectorizer, and classifies SMS messages.  
- *model.pkl* : Serialized trained ML model.  
- *vectorizer.pkl* : Serialized TF-IDF vectorizer.  
- *spam.csv / data/* : Dataset used for training and testing.  
- *requirements.txt* : List of required Python packages.  

---

## Features

- *TF-IDF Vectorization*  
  Converts SMS messages into numerical format suitable for machine learning.  
- *Classification Algorithms Implemented*  
  - Naive Bayes  
  - Logistic Regression  
  - Support Vector Machines (SVM)  
- *Model Serialization*  
  Pre-trained model and vectorizer saved using pickle for reusability and deployment.  
- *Streamlit Web App*  
  User-friendly web interface to input SMS text and check if it’s Spam or Not Spam.  

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier

##  Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install the required dependencie
pip install -r requirements.txt

## Run the app.py file
python app.py


## Dataset
The dataset (spam.csv) contains labeled SMS messages used for training.
It contains labeled SMS messages marked as ham (not spam) or spam.

## 🛠 Technologies Used
-Python

-Scikit-learn

-Pandas, NumPy

-Streamlit

-NLP (TF-IDF Vectorizer)

-Pickle (for model persistence)

# Future Enhancements
-Experiment with deep learning models (LSTMs, Transformers).
-Add real-time SMS/email detection API.
-Expand dataset for better generalization.


## Author 
Priyal Patil
📧 Email: priyalpatil1805@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/priyalpatil03/)  
