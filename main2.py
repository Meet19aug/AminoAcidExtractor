import pdfplumber
import pandas as pd
import os

# Input and output paths
pdf_path = "sample.pdf"
output_dir = "extracted_tables"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_tables_from_pdf(pdf_path, output_dir):
    """
    Extracts tables from the PDF and saves them as CSV files.
    """
    with pdfplumber.open(pdf_path) as pdf:
        table_count = 0

        for page_number, page in enumerate(pdf.pages):
            print(f"Processing Page {page_number + 1}")

            # Extract tables on the page
            tables = page.extract_tables()

            for table_index, table in enumerate(tables):
                # Convert to pandas DataFrame
                df = pd.DataFrame(table[1:], columns=table[0])  # Use first row as headers

                # Save each table as a CSV file
                csv_file = os.path.join(output_dir, f"table_page{page_number + 1}_table{table_index + 1}.csv")
                df.to_csv(csv_file, index=False)

                table_count += 1
                print(f"Table extracted: {csv_file}")

        if table_count == 0:
            print("No tables found in the PDF.")
        else:
            print(f"Total {table_count} tables extracted and saved in '{output_dir}'.")

# Run the extraction
extract_tables_from_pdf(pdf_path, output_dir)
