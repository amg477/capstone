import pandas as pd
import os

def optimized_data_processing():
    """
    Loads, filters to U.S. only, and processes multiple Excel files efficiently.
    Returns: A single, cleaned DataFrame.
    """

    excel_file_paths = [
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_1_df8ce708-7388-44d9-9aeb-a9f1cd72fd6b.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_2_5f3db6ca-8894-45b3-8e08-162e2e88baba.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_3_dd084585-7587-445b-ace2-fdd4eec637f9.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_4_9eb1d7ad-6010-4d1d-a128-b6af3cca5cfa.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_5_53407d72-3f41-4dc2-878e-db6076e73954.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_6_7e7a5c27-9b2a-444c-bec5-92133ae41865.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_7_7e2eaa3d-a08f-4882-a8f9-3a2ad04c91c7.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_8_edeac5df-5b03-424a-9706-8ef3de3827ca.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_d13e454c-6fef-48ac-af60-4406f9204813.xlsx",
    ]

    cols_to_drop = {
        "load_datetime",
        "published_datetime",
        "region",
        "iso_language_code",
        "publisher_site_url",
        "article_url",
    }

    processed_dfs = []

    for path in excel_file_paths:
        df = pd.read_excel(
            path,
            engine="openpyxl",
        )

        # Filter to U.S. only
        df = df[df["country"] == "United States"]

        # Drop unused columns (after filtering to avoid country loss)
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # Drop 'country' column after filtering
        if "country" in df.columns:
            df.drop(columns="country", inplace=True)

        # Convert load_date if present
        if "load_date" in df.columns:
            df["load_date"] = pd.to_datetime(df["load_date"], errors="coerce")

        processed_dfs.append(df)

    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Drop columns with >90% missing values
    missing_mask = (final_df.isna() | (final_df == "")).mean() > 0.9
    final_df.drop(columns=final_df.columns[missing_mask], inplace=True)

    return final_df

if __name__ == "__main__":
    print("Starting data processing")

    final_df = optimized_data_processing()

    output_path = "data/processed/processed_data.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Data processing complete. Saved to: {output_path}")