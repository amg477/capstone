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
2. Model Development
-----------------------------------------------------

The Makefile orchestrates the complete data processing pipeline from raw Excel files to cleaned data files ready for analysis and the Streamlit application.

### 2.1 View Available Commands

To see all available Makefile targets:

    make help

### 2.2 Run Complete Pipeline

To run the entire pipeline (all 7 steps):

    make all

This executes:
- Step 1: Combine and sample raw Excel files
- Step 2: Extract person names, process text, detect emotions
- Step 3: Clean person names
- Step 4: Attribution analysis (model development)
- Step 5: PCA clustering and influencer table (model development)
- Step 6: Clean persons_by_row
- Step 7: Clean influencer_table

### 2.3 Run Individual Pipeline Stages

**Data Processing (Steps 1-3):**
Processes raw data and extracts/cleans person names:

    make data-processing

**Model Development (Steps 4-5):**
Runs attribution analysis and PCA clustering (requires step 3 to be completed first):

    make model-dev

**Final Cleaning (Steps 6-7):**
Cleans final output files for Streamlit app (requires steps 4-5 to be completed first):

    make cleaning

### 2.4 Run Individual Steps

You can also run individual steps:

    make step1    # Combine and sample raw Excel files
    make step2    # Extract person names and process text
    make step3    # Clean person names
    make step4    # Attribution analysis
    make step5    # PCA clustering
    make step6    # Clean persons_by_row
    make step7    # Clean influencer_table

### 2.5 Clean Generated Files

To remove all generated parquet files:

    make clean

### 2.6 Output Locations

After running the pipeline, output files are located in:
- Processed data: `data_storage/processed_data/`
- Final data: `data_storage/final_data/`
- Streamlit app data: `data_storage/streamlit_app_data/`

-----------------------------------------------------
3. Streamlit App
-----------------------------------------------------

### 3.1 Prepare Data for Streamlit

Before running the Streamlit app, ensure the data processing pipeline has been completed:

    make all

This generates the required files:
- `final_dataset_with_attribution.parquet`
- `persons_by_row_cleaned.parquet`
- `influencer_table_cleaned.parquet`

### 3.2 Run the Streamlit App

Navigate to the streamlit_app directory and run:

    cd streamlit_app
    streamlit run app.py

Or from the project root:

    streamlit run streamlit_app/app.py

The app will open in your default web browser, typically at `http://localhost:8501`.

-----------------------------------------------------
4. Git Workflow (Use Daily)
-----------------------------------------------------

    git status
    git add .
    git commit -m "Brief description of your change"
    git pull origin main
    git push origin main

-----------------------------------------------------
5. Project Structure
-----------------------------------------------------

capstone/
├── data_storage/      ← raw, processed, and final data files
├── streamlit_app/     ← Streamlit application (app.py, charts.py, etc.)
├── model_development/ ← data processing and model development scripts
├── requirements.txt   ← Python dependencies (single file for entire project)
├── Makefile           ← data processing pipeline automation
└── README.md

-----------------------------------------------------
6. Project Goals
-----------------------------------------------------

- Identify who originates vs. amplifies healthcare narratives
- Quantify influence based on timing, reach, engagement
- Analyze how narratives propagate over time and across platforms
- (Optional) Deliver interactive dashboard for real-time exploration

