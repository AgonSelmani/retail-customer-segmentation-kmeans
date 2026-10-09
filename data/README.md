# Data

This folder holds the input dataset and generated outputs. Raw and processed
CSV files are excluded from Git via `.gitignore`.

## Setup

Place the raw retail transaction file at:

    data/raw/My_retail_data.csv

The file must contain these 14 columns:

    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
    CustomerID, Country, Year, Month, Season, Revenue, Avg_UnitPrice, Customer_seg

## Generated Outputs

After running `python main.py`, these files will be created automatically:

    data/processed/
    ├── customer_level.csv           # aggregated features per customer
    ├── customer_level_scaled.csv    # RobustScaler output
    └── CUSTOMER_SEGMENTS.csv        # final segmented dataset