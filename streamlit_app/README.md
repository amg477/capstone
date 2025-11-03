# Attribution & PCA Analysis Streamlit App

This Streamlit application visualizes attribution and PCA analysis results to explore how individuals were being talked about in the dataset.

## Features

- **Attribution Analysis**: Visualize how individuals were mentioned with sentiment and visibility metrics
- **PCA Cluster Analysis**: Explore the clustering results from PCA analysis
- **Top Individuals**: View the most mentioned individuals with their metrics
- **Data Explorer**: Interactive table to browse and filter the data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

   **Note:** The emotion analysis feature requires `sentencepiece`. If you encounter tokenizer errors, ensure it's installed:
   ```bash
   pip install sentencepiece
   ```
   
   You may need to restart the app after installing sentencepiece.

2. Ensure the data files are in the correct location:
   - `data_storage/final_data/influencer_table.csv` (or `.parquet`)
   - `data_storage/final_data/final_dataset_with_attribution.parquet`
   - `data_storage/final_data/persons_by_row.csv` (or `.parquet`)

## Running the App

From the project root directory, run:

```bash
streamlit run streamlit_app/app.py
```

Or from within the `streamlit_app` directory:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Filters**: Use the sidebar to filter by individuals or clusters
2. **Tabs**: Navigate between different analysis views:
   - **Attribution Analysis**: Visibility vs sentiment charts
   - **PCA Clusters**: Cluster distribution and metrics comparison
   - **Top Individuals**: Most mentioned individuals
   - **Data Explorer**: Interactive data table with sorting and download

## Data Structure

The app expects the following data structure:

### Influencer Table (`influencer_table.csv`)
- `person_list`: Name of the individual
- `cluster`: Cluster assignment (0 or 1)
- `cluster_label`: Human-readable cluster label
- `mention_count`: Number of mentions
- `vipr_score`: VIPR score (visibility metric)
- `vipr_weight`: VIPR weight
- `sentiment_score`: Average sentiment score
- `circulation_size`: Average circulation size

## Notes

- The app uses caching to improve performance
- Charts are interactive (hover for details, zoom, pan)
- Data can be downloaded as CSV from the Data Explorer tab

