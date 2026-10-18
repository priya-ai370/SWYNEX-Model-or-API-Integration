# SWYNEX AI Question Answering Prototype

A small AI/ML prototype based on Task 1: answer user questions by retrieving the closest matching question from a 500-row question-answer dataset.

## Features
- Streamlit web interface
- Includes the Task 1 `task1dataset.csv` (500 question-answer pairs)
- TF-IDF text features and cosine similarity for question matching
- Low-match fallback to reduce unrelated answers
- Shows the matched dataset question and similarity score for transparency
- No external API key required

## Tech stack
Python, Streamlit, Pandas, scikit-learn.

## Run locally (Windows / VS Code)
1. Install Python 3.10 or newer.
2. Extract this ZIP and open the `SWYNEX_AI_QA_Prototype` folder in VS Code.
3. Open **Terminal → New Terminal**, then run:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Example input and output
**Input:** `What does SPIT stand for?`

**Expected answer:** `SPIT stands for Sardar Patel Institute of Technology.`

**Input:** `What is the full name of SPIT?`

**Expected answer:** `The full name is Bharatiya Vidya Bhavan's Sardar Patel Institute of Technology.`

An unrelated question should trigger the fallback when its similarity score is below the configured threshold.

## Evaluation notes
Task 1 sets a target of at least 90% correct answers on test questions and fewer than 10% unrelated answers. These are targets, not measured results. Evaluate the system on questions not used to tune the system; record correct, incorrect, and fallback outcomes before reporting accuracy. TF-IDF similarity is a retrieval baseline, not a generative language model, and a high similarity score does not guarantee correctness.

## Dataset
`data/task1dataset.csv` contains 500 rows with `question` and `answer` columns from the provided Task 1 file.

## Security
No API keys are needed. Do not add passwords, API keys, or confidential personal information to the repository.
