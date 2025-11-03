"""
Data Loading Functions
Handles loading of datasets, models, and cached resources
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

# Global flag for emotion model availability
try:
    import sentencepiece
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False


@st.cache_data(show_spinner=False)
def load_influencer_table():
    """Load influencer table - prefer parquet, fallback to CSV"""
    try:
        # Get project root (assuming app.py is in streamlit_app/)
        project_root = Path(__file__).parent.parent
        
        # Try parquet first (much faster)
        path = project_root / "data_storage" / "final_data" / "influencer_table.parquet"
        if not path.exists():
            alt_path = Path("data_storage") / "final_data" / "influencer_table.parquet"
            if alt_path.exists():
                path = alt_path
            else:
                # Fallback to CSV
                path = project_root / "data_storage" / "final_data" / "influencer_table.csv"
                if not path.exists():
                    alt_path = Path("data_storage") / "final_data" / "influencer_table.csv"
                    if alt_path.exists():
                        path = alt_path
                    else:
                        st.error(f"File not found: {path}")
                        return None
        
        # Load based on file extension
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        
        # Optimize data types to reduce memory
        if 'circulation_size' in df.columns:
            df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce', downcast='float')
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce', downcast='float')
        if 'mention_count' in df.columns:
            df['mention_count'] = pd.to_numeric(df['mention_count'], errors='coerce', downcast='integer')
        
        return df
    except Exception as e:
        st.error(f"Error loading influencer table: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_attribution_dataset():
    """Load attribution dataset"""
    try:
        project_root = Path(__file__).parent.parent
        path = project_root / "data_storage" / "final_data" / "attribution_dataset.parquet"
        
        if not path.exists():
            alt_path = Path("data_storage") / "final_data" / "attribution_dataset.parquet"
            if alt_path.exists():
                path = alt_path
            else:
                st.error(f"File not found: {path}")
                return None
        
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        st.error(f"Error loading attribution dataset: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_final_dataset():
    """Load final dataset with attribution"""
    try:
        project_root = Path(__file__).parent.parent
        path = project_root / "data_storage" / "final_data" / "final_dataset_with_attribution.parquet"
        
        if not path.exists():
            alt_path = Path("data_storage") / "final_data" / "final_dataset_with_attribution.parquet"
            if alt_path.exists():
                path = alt_path
            else:
                return None
        
        # Only load essential columns initially to speed up loading
        # User can specify columns if needed, but for now load all
        df = pd.read_parquet(path)
        
        # Optimize data types
        if 'circulation_size' in df.columns:
            df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce', downcast='float')
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce', downcast='float')
        if 'row_index' in df.columns:
            df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce', downcast='integer')
        
        return df
    except Exception as e:
        st.warning(f"Could not load final dataset: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_persons_by_row():
    """Load persons by row data - prefer parquet, fallback to CSV"""
    try:
        project_root = Path(__file__).parent.parent
        
        # Try parquet first (much faster)
        path = project_root / "data_storage" / "final_data" / "persons_by_row.parquet"
        if not path.exists():
            alt_path = Path("data_storage") / "final_data" / "persons_by_row.parquet"
            if alt_path.exists():
                path = alt_path
            else:
                # Fallback to CSV
                path = project_root / "data_storage" / "final_data" / "persons_by_row.csv"
                if not path.exists():
                    alt_path = Path("data_storage") / "final_data" / "persons_by_row.csv"
                    if alt_path.exists():
                        path = alt_path
                    else:
                        return None
        
        # Load based on file extension
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        
        # Optimize data types
        if 'row_index' in df.columns:
            df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce', downcast='integer')
        
        return df
    except Exception as e:
        st.warning(f"Could not load persons by row: {e}")
        return None


@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model from Hugging Face"""
    if not EMOTION_AVAILABLE:
        return None
    
    model_name = "mrm8488/t5-base-finetuned-emotion"
    
    # Check if sentencepiece is available (required for T5 tokenizers)
    try:
        import sentencepiece
        sentencepiece_available = True
    except ImportError:
        sentencepiece_available = False
        error_msg = """
        ⚠️ **sentencepiece library is required but not found**
        
        Please install it using one of these methods:
        
        1. **Using pip:**
           ```bash
           pip install sentencepiece
           ```
        
        2. **Or install all requirements:**
           ```bash
           pip install -r streamlit_app/requirements.txt
           ```
        
        3. **If using conda:**
           ```bash
           conda install -c conda-forge sentencepiece
           ```
        
        **Note:** You may need to restart your Streamlit app after installation.
        
        The emotion analysis feature will be disabled until sentencepiece is installed.
        """
        st.error(error_msg)
        return None
    
    # Strategy 1: Load tokenizer explicitly with trust_remote_code and use_fast=False
    try:
        # Disable fast tokenizer conversion by explicitly using slow tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            model_name, 
            use_fast=False,
            legacy=False,
            local_files_only=False
        )
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        emotion_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
        return emotion_pipeline
    except Exception as e1:
        # Strategy 2: Try loading from local cache or without legacy mode
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                use_fast=False,
                legacy=True,
                local_files_only=False
            )
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            emotion_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1
            )
            return emotion_pipeline
        except Exception as e2:
            # Strategy 3: Try with AutoTokenizer but explicitly set tokenizer_class
            try:
                from transformers import T5Config
                config = T5Config.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    config=config,
                    tokenizer_type="t5"
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
                emotion_pipeline = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1
                )
                return emotion_pipeline
            except Exception as e3:
                # Strategy 4: Manual tokenizer instantiation with explicit paths
                try:
                    import os
                    from huggingface_hub import snapshot_download
                    
                    # Try to download and use local paths
                    cache_dir = snapshot_download(repo_id=model_name, cache_dir=None)
                    
                    tokenizer = T5Tokenizer.from_pretrained(
                        cache_dir,
                        use_fast=False,
                        local_files_only=True
                    )
                    model = T5ForConditionalGeneration.from_pretrained(
                        cache_dir,
                        local_files_only=True
                    )
                    emotion_pipeline = pipeline(
                        "text2text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=-1
                    )
                    return emotion_pipeline
                except Exception as e4:
                    error_msg = f"Failed to load emotion model after multiple attempts.\n"
                    error_msg += f"Last error: {e4}\n"
                    error_msg += f"Please ensure sentencepiece is installed: pip install sentencepiece"
                    st.error(error_msg)
                    return None

