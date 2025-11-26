import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folders
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

# Output dataset path
OUTPUT_PATH = os.path.join(PROCESSED_PATH, "ipl_master_dataset.csv")
OVER_OUTPUT_PATH = os.path.join(PROCESSED_PATH, "ipl_over_by_over.csv")

# Figures and tables (for analyses / dashboard)
FIGURES_PATH = os.path.join("output", "figures")
TABLES_PATH = os.path.join("output", "tables")

# Create directories if they don't exist
for path in [os.path.dirname(OUTPUT_PATH), FIGURES_PATH, TABLES_PATH]:
    os.makedirs(path, exist_ok=True)