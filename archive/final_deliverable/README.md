# PolicyPath - Healthcare Policy Influence Analysis

A comprehensive Streamlit application for analyzing healthcare policy influence patterns through attribution analysis, dashboard visualizations, and network analysis.

**Built by Georgetown University MSBA Team**: Anna Glass, Jasmin Mendoza, Mohammad Waqas, Mark Saba, Posy Olivetti

## 🚀 How to Run the Application

### Local Development
```bash
cd final_deliverable
python run.py
# or
streamlit run app.py
```

### Streamlit Cloud Deployment
1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io)**
3. **Click "New app"** and connect your repository
4. **Set main file path**: `final_deliverable/app.py`
5. **Add secrets** in the Streamlit Cloud dashboard:
   ```toml
   [data]
   mode = "local"
   data_dir = "data"
   parquet = "final_model_dataset_sample.csv"
   csv = "final_model_dataset_sample.csv"
   attr_csv = "attribution_all_scored_sample.csv"
   logo = "penta_logo.png"
   ```
6. **Deploy!** Your app will be live with sample data

**The application will open in your browser at: http://localhost:8501**

## 📁 Project Structure

```
final_deliverable/
├── app.py                    # Main Streamlit application
├── run.py                    # Application launcher
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .streamlit/
    └── secrets.toml         # Configuration file
```

## 🎯 Features

### 📋 Instructions Tab
- Welcome page with usage instructions
- Data status and tips for analysis

### 🎯 Attribution Tab
- **Item Attribution**: Analyze influence by publication, author, channel, etc.
- **Term Attribution**: Search for specific terms and see their influence scores
- Interactive lookups with detailed explanations

### 📊 Dashboard Tab
- **KPI Metrics**: Key performance indicators at a glance
- **Visualizations**: Bar charts, pie charts, and time series
- **Sankey Diagram**: Flow analysis between different dimensions
- **Sample Data**: View filtered data in tabular format
- **Filters**: Date range, publications, channels, sentiment bands, authors, topics

### 🕸️ Network Tab
- **Network Analysis**: Visualize relationships between entities
- **Edge Analysis**: Understand connection strengths
- **Node Strength**: Identify influential nodes in the network

## 🔧 Configuration

The application uses Streamlit secrets for configuration:

```toml
# .streamlit/secrets.toml
[data]
mode = "local"  # or "azure"
data_dir = "data"
parquet = "final_model_dataset.parquet"
csv = "final_model_dataset.csv"
attr_csv = "attribution_all_scored.csv"
logo = "final_deliverable/penta_logo.png"
```

## 📊 Data Sources

The application supports multiple data sources:

- **Local Files**: CSV and Parquet files in the `data/` directory
- **Azure Blob Storage**: Remote data loading with connection strings
- **Network Data**: `network_edges.csv` for network analysis

## 🎨 Branding

Features Georgetown University MSBA styling with:
- Professional green color scheme (#12715D)
- Custom logo display
- Clean typography and modern layout
- Real-time search suggestions
- Interactive visualizations (Plotly & Altair)

## 📈 Usage Tips

1. **Start with Instructions**: Review the welcome page for guidance
2. **Explore Attribution**: Use item and term lookups with real-time search suggestions
3. **Analyze Dashboard**: Apply filters and explore interactive visualizations
4. **Network Analysis**: Add network_edges.csv for relationship analysis (see TODO notes in code)
5. **Export Results**: Use built-in CSV/JSON export features
6. **Search Suggestions**: Type 2+ characters to see live suggestions
7. **Saved Views**: Save your filter combinations for quick access

## 🛠️ Dependencies

- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `duckdb` - In-memory SQL database
- `altair` - Statistical visualizations
- `plotly` - Interactive charts
- `azure-storage-blob` - Azure integration
- `matplotlib` & `seaborn` - Additional plotting

## 📝 Notes

- The application gracefully handles missing data files
- DuckDB provides fast in-memory SQL queries
- All visualizations are interactive and responsive
- Data is cached for optimal performance