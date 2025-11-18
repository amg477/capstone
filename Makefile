# Makefile for Data Processing and Model Development Pipeline
# 
# This Makefile orchestrates the complete data processing pipeline from raw Excel files
# to cleaned data files ready for the Streamlit application.
#
# Execution Order:
#   1. data_processing.py - Combine and sample raw Excel files
#   2. names_then_text.py - Extract person names, process text, detect emotions
#   3. clean_people_names.py - Advanced name cleaning and normalization
#   4. attribution.py - Model development: attribution analysis (optional)
#   5. pca.py - Model development: PCA clustering and influencer table (optional)
#   6. clean_persons_by_row.py - Clean persons_by_row file
#   7. clean_influencer_table.py - Clean influencer_table file
#
# Usage:
#   make all              - Run complete pipeline (steps 1-7)
#   make data-processing  - Run steps 1-3 (data processing only)
#   make model-dev        - Run steps 4-5 (model development, requires step 3)
#   make cleaning         - Run steps 6-7 (final cleaning, requires steps 4-5)
#   make clean            - Remove all generated files
#   make help             - Show this help message

.PHONY: all help clean data-processing model-dev cleaning step1 step2 step3 step4 step5 step6 step7

# Directories
DATA_DIR := data_storage
RAW_DIR := $(DATA_DIR)/raw_data
PROCESSED_DIR := $(DATA_DIR)/processed_data
FINAL_DIR := $(DATA_DIR)/final_data
STREAMLIT_DIR := $(DATA_DIR)/streamlit_app_data
MODEL_DEV_DIR := model_development/data_processing

# Python executable (use python3 if available, fallback to python)
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Data Processing and Model Development Pipeline"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make all              - Run complete pipeline (all 7 steps)"
	@echo "  make data-processing  - Run data processing steps (1-3)"
	@echo "  make model-dev        - Run model development steps (4-5, requires step 3)"
	@echo "  make cleaning         - Run final cleaning steps (6-7, requires steps 4-5)"
	@echo ""
	@echo "Individual steps:"
	@echo "  make step1            - Combine and sample raw Excel files"
	@echo "  make step2            - Extract person names and process text"
	@echo "  make step3            - Clean person names"
	@echo "  make step4            - Attribution analysis (model development)"
	@echo "  make step5            - PCA clustering (model development)"
	@echo "  make step6            - Clean persons_by_row"
	@echo "  make step7            - Clean influencer_table"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Remove all generated parquet files"
	@echo "  make help             - Show this help message"
	@echo ""

# Step 1: Combine and sample raw Excel files
step1:
	@echo "=========================================="
	@echo "Step 1: Combining and sampling raw data"
	@echo "=========================================="
	cd $(MODEL_DEV_DIR) && $(PYTHON) data_processing.py

# Step 2: Extract person names, process text, detect emotions
step2:
	@echo "=========================================="
	@echo "Step 2: Extracting person names and processing text"
	@echo "=========================================="
	cd $(MODEL_DEV_DIR) && $(PYTHON) names_then_text.py

# Step 3: Clean person names
step3:
	@echo "=========================================="
	@echo "Step 3: Cleaning person names"
	@echo "=========================================="
	cd $(MODEL_DEV_DIR) && $(PYTHON) clean_people_names.py

# Step 4: Attribution analysis (model development)
step4:
	@echo "=========================================="
	@echo "Step 4: Attribution analysis (model development)"
	@echo "=========================================="
	cd model_development && $(PYTHON) attribution.py

# Step 5: PCA clustering and influencer table (model development)
step5:
	@echo "=========================================="
	@echo "Step 5: PCA clustering (model development)"
	@echo "=========================================="
	cd model_development && $(PYTHON) pca.py

# Step 6: Clean persons_by_row
step6:
	@echo "=========================================="
	@echo "Step 6: Cleaning persons_by_row"
	@echo "=========================================="
	cd $(MODEL_DEV_DIR) && $(PYTHON) clean_persons_by_row.py

# Step 7: Clean influencer_table
step7:
	@echo "=========================================="
	@echo "Step 7: Cleaning influencer_table"
	@echo "=========================================="
	cd $(MODEL_DEV_DIR) && $(PYTHON) clean_influencer_table.py

# Data processing pipeline (steps 1-3)
data-processing:
	@$(MAKE) step1
	@$(MAKE) step2
	@$(MAKE) step3
	@echo ""
	@echo "✓ Data processing complete (steps 1-3)"
	@echo "  Output files in: $(PROCESSED_DIR)/ and $(FINAL_DIR)/"

# Model development pipeline (steps 4-5)
model-dev:
	@$(MAKE) step4
	@$(MAKE) step5
	@echo ""
	@echo "✓ Model development complete (steps 4-5)"
	@echo "  Output files in: $(FINAL_DIR)/"

# Final cleaning pipeline (steps 6-7)
cleaning:
	@$(MAKE) step6
	@$(MAKE) step7
	@echo ""
	@echo "✓ Final cleaning complete (steps 6-7)"
	@echo "  Output files in: $(STREAMLIT_DIR)/"

# Complete pipeline (all steps)
all:
	@$(MAKE) step1
	@$(MAKE) step2
	@$(MAKE) step3
	@$(MAKE) step4
	@$(MAKE) step5
	@$(MAKE) step6
	@$(MAKE) step7
	@echo ""
	@echo "=========================================="
	@echo "✓ Complete pipeline finished successfully!"
	@echo "=========================================="
	@echo ""
	@echo "Output locations:"
	@echo "  Processed data: $(PROCESSED_DIR)/"
	@echo "  Final data:     $(FINAL_DIR)/"
	@echo "  Streamlit app:  $(STREAMLIT_DIR)/"
	@echo ""
	@echo "Streamlit app files ready:"
	@echo "  - final_dataset_with_attribution.parquet"
	@echo "  - persons_by_row_cleaned.parquet"
	@echo "  - influencer_table_cleaned.parquet"

# Clean target - remove generated files
clean:
	@echo "Removing generated parquet files..."
	@find $(PROCESSED_DIR) -name "*.parquet" -type f -delete 2>/dev/null || true
	@find $(FINAL_DIR) -name "*.parquet" -type f -delete 2>/dev/null || true
	@find $(STREAMLIT_DIR) -name "*.parquet" -type f -delete 2>/dev/null || true
	@echo "✓ Clean complete"

# Check if required files exist
check-dependencies:
	@echo "Checking dependencies..."
	@if [ ! -d "$(RAW_DIR)" ]; then \
		echo "✗ ERROR: Raw data directory not found: $(RAW_DIR)"; \
		exit 1; \
	fi
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "✗ ERROR: Python not found. Please install Python 3."; \
		exit 1; \
	fi
	@echo "✓ Dependencies check passed"

