# EDA Project B — Time-Series Forecasting Starter

Student: Al Azhar Hilal Al Azri  
Student ID: PG12S2540471

This repository contains a starter Streamlit app for Mini Project B. The included dataset slice is stored at:

`data/dataset_sample.csv`

Confirmed time-series setup:
- Timestamp column: `Datetime`
- Target column: `PJME_MW`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository named `EDA-ProjectB-PG12S2540471`.
2. Upload these files exactly:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. On Streamlit Community Cloud, create a new app.
4. Connect the GitHub repository.
5. Use branch `main`.
6. Set the main file path to `app.py`.
7. Deploy.

## OpenRouter API key

The app reads the OpenRouter API key in this order:
1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field inside the app

No API key is hardcoded.

## What to submit

Submit:
- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json`
- Exported `project_card.md`

## Student work required

This starter app intentionally stops before training models. Students must add:
- Time-based split
- Forecasting models
- Metrics table
- Extra dashboard visualizations
- Insights and interpretation
