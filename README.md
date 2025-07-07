# Recon Matcher

## Introduction

###

This is a two layer robust and intelligent model that can give the matching percentage for two transactions by matching their Customer IDs, Policy numbers, Names, Locations, Product, Channel and Amount.

###

The first layer determines the rows in which Customer IDs and Policy Numbers are not matching and they need to be set as very low matching percentage. The rest are sent to the LLMProcessor which determines wheteher the names and locations are matching or not. Then a binary matchers are made for each of the columns (1 for matched, 0 for unmatched and -1 for missing).

###

These binary matchers are then sent to ML model which finally determines the match percentage.

###

**Provided a Summary at the end**

## How to run the code

###

Go to orchestrator.py and run that python file.
**Python version should be 3.10 or above**

## Upload the CSV that need to be matched:

###

Upload the input csv that needs to be matched in the input folder and enter its location in 'DATA_CSV' LLM_matcher/config.py.

## Rulebooks

###

If you want to improve the efficiency of the LLM results, you can edit the 'nameRuleBook' and 'locationRuleBook' CSVs inside the LLM_matcher/rulebooks.

## LLM model

###

Current LLM model is set to mistral. You can change it through LLM_matcher/config.py. Change the LLM_URL and edit the LLM_MODEL as per requirements. If required, make the changes in @retry and \_call_llm function in the class LLMProcessor of LLM_matcher/LLMProcessor.py file (it is just below **init**).

## ML model

###

The default model is model.pkl inside the LLM_matcher folder. It is a supervised learning model based on logistic regression. The labels are matcher (2) for more than 95 match percentage, checker (1) for 80-95 match percenatge and unmatched (0) for less than 80 match percentage.

###

The Customer ID and Policy number are given more weightage. Amount difference, Names and Location are given important weightage as well. Product and Channel are given less weightage. The training dataset (ML_test_models/input/train_v2.csv) has the dataset keeping the weightage in mind, this training dataset is used to train the Logistic Regression model. The training dataset inside ML_test_models/input is designed keeping the weights in mind. You can edit the training dataset or the equations in the logisticalRegression.py file to experiment with different weights.

## Output

###

The final output file is stored in output folder under the name of output.csv.

## Summary

### **AI-assisted reconciliation**

Matches bank-statement rows to policy-record rows using a two-layer pipeline (rule-based + LLM + ML) and outputs a match probability for every pair.

### **Features**

- Fast rule layer to reject obviously wrong pairs (Customer ID / Policy No).
- LLM layer (Mistral by default) for **Name** and **Location** scoring.
- Logistic-Regression model (customisable) blends all signals into a final match-percentage.
- Plug-and-play **rulebooks** so analysts can add domain knowledge without coding.
