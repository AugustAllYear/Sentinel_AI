# Sential AI 
Adaptive Fraud Scoring Engine

## Project Overview


## Business Problem

## Project Structure
```
sentinel_ai/
├── .github/
│   └── workflows/
│       └── ci.yml                # CI/CD workflow
├── config/
│   └── config.yaml               # (optional) configuration
├── data/
│   ├── raw/                      # raw input data (ignored by git)
│   └── processed/                # cleaned data
├── models/                       # saved model artifacts (MLflow)
├── src/
│   ├── __init__.py
│   ├── data.py                   # data loading & preprocessing
│   ├── features.py               # feature engineering
│   ├── train.py                  # model training
│   ├── evaluate.py               # evaluation & metrics
│   ├── predict.py                # prediction on new data
│   └── utils.py                  # helpers
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── notebooks/                    # (optional) exploratory notebooks
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py                      # (optional) for packaging  
```

GET SETUP.py!!!!
```
sentinel_ai/
├── .github/workflows/      # CI/CD pipelines
├── config/                 # configuration files
├── data/                   # raw and processed data (gitignored)
├── models/                 # saved models
├── src/                    # source code
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── tests/                  # unit tests
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```
## Data
We used historical campaign data containing:
- Customer demographics: age, income, tenure (months), days since last purchase, average order value
- Campaign attributes: channel (email, social, push), type (promotional, informational, loyalty)
- Engagement flag: whether the customer opened the email (target variable)

The dataset was synthetically generated for demonstration; the methodology was applied to real customer data.

## Methodology
1. **Exploratory Data Analysis**: Visualized feature distributions and relationships with the target. 
2. **Preprocessing**: Scaled numerical features and one‑hot encoded categorical variables.
3. **Modeling**: Trained a Random Forest classifier to predict open probability.
4. **Evaluation**: Used ROC‑AUC and classification report; achieved ROC‑AUC of 0.78.
5. **Simulation**: Compared random targeting with model‑based targeting over six months to quantify business impact.

## Results
- The model identified recency (`last_purchase_days`) as the strongest predictor of opens.
- Targeting the top 30% of customers by predicted probability captures ~68% of all potential opens.
- In a six‑month simulation, switching from random to model‑based targeting increased cumulative opens by **25%**, meeting the business objective.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script `campaign_optimization.py`.
4. (Optional) Replace the synthetic data with your own CSV file, ensuring column names and data types match.

## Continuation and Refinement Suggestions
**Notebook Model_experimentation.ipynb**: this notebook showcases further model exploration and tuning, only on the best performing of the tested models. A GridSearchCV was performed (with cross validation on multiple data splits) to find the best tuning for both models tested, Random Forest and XGBoost.
- - The tuned Random Forest model performed better than the XGBoost and slightly better than the Original Model with a new ROC score of 0.7936
- The model should improve with more data with retraining every 4‑6 months, at which time it is recommended that the experimental models are retrained as well. It is also advisable to spend time engineering new features, ensembling (combining models in one pipeline). List below.

- **A/B Test the Model**: Run a live experiment comparing the model’s top 30% against a random 30% control group to validate the lift.
- **Feature Engineering**: Incorporate additional features such as:
  - Customer lifetime value
  - Previous campaign engagement history (e.g., number of opens in last 3 months)
  - Time‑based features (day of week, season)
  - Average response time
- **Model Improvement**: Experiment with gradient boosting (XGBoost, LightGBM) and tune hyperparameters via cross‑validation.
- **Automate Retraining**: Set up a scheduled pipeline (e.g., monthly) that ingests new campaign data, retrains the model, and updates the scoring system.
- **Deployment**: Package the model as a REST API using Flask or FastAPI, and integrate with the marketing automation platform (e.g., Salesforce Marketing Cloud, Braze).
- **Monitoring**: Track model performance drift over time and set up alerts if ROC‑AUC drops below a threshold.
-- Note:  
Recommended CI/CD platforms: GitHub Actions, GitLab CI, Jenkins, Azure DevOps, CircleCI. For a data science project, GitHub Actions is popular because it integrates with code repositories and is free for public/private repos up to a limit.

If data files being used are below 14GB data can be processed using GitHub actions. Otherwise may need to upgrade to (S3) and trigger jobs.