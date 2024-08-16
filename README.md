# Salary Prediction Application

This project is a web-based application that predicts mean salary based on median salary, population size, number of jobs, and area size of a London borough. The application is built using Flask for the web interface, Housing in London Boroughs dataset from Rahul Nadar 123 on Kaggle, and a Random Forest model for the prediction.

## Project Structure
SalaryPrediction/
│
├── data/
│   ├── housing_in_london_monthly_variables.csv
│   ├── housing_in_london_yearly_variables.csv
│   ├── london-borough-profiles-2016 Data set.csv
│   └── final_housing_data.csv
│
├── models/
│   ├── housing_price_model.pkl
│   └── housing_scaler.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── data_processing.py
├── model_training.py
├── model_evaluation.py
├── requirements.txt
└── README.md

### Installation
1. Clone the repository:

   git clone https://github.com/oll-iver/SalaryPrediction.git
   cd SalaryPrediction

2. Install requirements:
    
   pip install -r requirements.txt

3. Process and train the data:

   python model_training.py
   python data_processing.py

4. OPTIONAL: Evaluate the data:

    python model_evaluation.py

5. Run the web app:

    python app.py

#### Acknowledgements
The dataset used in this project can be found at www.kaggle.com/datasets/rahulnadar123/housing-in-london-boroughs/code
The ML model is implemented using sklearn.