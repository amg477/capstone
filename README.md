Capstone Project: Influencer Identification in Healthcare Policy
=======================================================

This guide provides everything you need to know to run scripts, manage files, and contribute code to the capstone repository.

-----------------------------------------------------
0. Navigate to Project Root
-----------------------------------------------------

Open terminal and navigate to your repo folder:

    cd ~/capstone/capstone

-----------------------------------------------------
1. Set Up the Environment
-----------------------------------------------------

1.1 Install all dependencies:

    pip install -r requirements.txt

1.2 (If using NLTK for text processing):

    python
    >>> import nltk
    >>> nltk.download('punkt')
    >>> nltk.download('stopwords')

-----------------------------------------------------
2. Run Preprocessing Scripts
-----------------------------------------------------

Clean text:

    from text_cleaning import clean_text
    df['clean_text'] = df['text'].apply(clean_text)

Normalize sources:

    from source_normalization import normalize_source
    df['source'] = df['source'].apply(normalize_source)

-----------------------------------------------------
3. Build Narrative Paths
-----------------------------------------------------

Generate influence paths based on events:

    from path_building import build_paths
    paths_df = build_paths(mentions_df, events_df, window_days=7)

-----------------------------------------------------
4. Run Attribution Modeling
-----------------------------------------------------

Run Markov attribution model:

    from attribution_modeling import run_attribution_model
    result = run_attribution_model(paths_df)

-----------------------------------------------------
5. Git Workflow (Use Daily)
-----------------------------------------------------

    git status
    git add .
    git commit -m "Brief description of your change"
    git pull origin main
    git push origin main

-----------------------------------------------------
6. Project Structure
-----------------------------------------------------

capstone/
├── data/              ← raw and processed data files
├── notebooks/         ← EDA and modeling notebooks
├── src/               ← all Python scripts
├── reports/           ← figures and presentation content
├── app/               ← Streamlit app (optional)
├── requirements.txt   ← dependencies
└── README.md

-----------------------------------------------------
7. Project Goals
-----------------------------------------------------

- Identify who originates vs. amplifies healthcare narratives
- Quantify influence based on timing, reach, engagement
- Analyze how narratives propagate over time and across platforms
- (Optional) Deliver interactive dashboard for real-time exploration

