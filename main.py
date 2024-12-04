from flask import Flask, request, jsonify, send_file, make_response, render_template
import camelot
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import io
import re
def extract_table_metadata(pdf_path, page_number):
    """
    Extract table number and description from the PDF page
    Returns tuple of (table_number, table_description)
    """
    # Read the page text
    tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='stream')
    if not tables:
        return None, None

    # Get the text around the table
    text = tables[0].df.to_string()

    # Try to find table number using regex
    table_num_match = re.search(r'Table\s+(\d+)', text, re.IGNORECASE)
    table_number = table_num_match.group(1) if table_num_match else None

    # Try to find table description (usually follows table number)
    table_desc_match = re.search(r'Table\s+\d+[.:]\s*([^\n]+)', text, re.IGNORECASE)
    table_description = table_desc_match.group(1).strip() if table_desc_match else None

    return table_number, table_description

def is_category_row(row):
    """
    Check if a row represents a category (first cell has value, rest are empty)
    """
    if pd.isna(row.iloc[0]) or str(row.iloc[0]).strip() == '':
        return False

    # Check if all other cells are empty
    return row.iloc[1:].isna().all() or (row.iloc[1:].astype(str).str.strip() == '').all()

def is_potential_table(df):
    """
    Enhanced check to determine if a DataFrame is likely to be a genuine table
    """
    if df is None or df.empty:
        return False

    # Minimum dimensions for a valid table
    if df.shape[1] < 2 or df.shape[0] < 3:
        return False

    # Check for structured data patterns
    def has_structured_pattern(df):
        # Check if there's consistent data in columns
        non_empty_cells = df.notna().sum()
        row_consistency = non_empty_cells / len(df) > 0.3
        return row_consistency.mean() > 0.5

    # Check for numeric content
    def has_numeric_content(df):
        numeric_ratio = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().mean().mean()
        return numeric_ratio > 0.3

    # Combined checks
    return (has_structured_pattern(df) and
            has_numeric_content(df) and
            not df.iloc[:, 1:].isna().all().all())



app = Flask(__name__)

def clean_special_characters(text):
    """
    Clean and standardize special characters in text
    """
    if pd.isna(text):
        return ''

    text = str(text).strip()
    # Replace various forms of plus-minus symbol
    text = re.sub(r'[±\+\-]+', '±', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def make_unique_headers(headers):
    """
    Ensure headers are unique by adding numbers to duplicates
    """
    seen = {}
    unique_headers = []

    for header in headers:
        base_header = str(header).strip()
        if not base_header:
            base_header = 'Column'

        if base_header in seen:
            seen[base_header] += 1
            unique_headers.append(f"{base_header}_{seen[base_header]}")
        else:
            seen[base_header] = 0
            unique_headers.append(base_header)

    return unique_headers

def identify_column_headers(df):
    """
    Enhanced function to identify column headers from complex tables
    """
    header_candidates = []
    header_idx = -1

    # First pass: look for multiple header rows that might need to be combined
    for idx, row in df.iterrows():
        if row.isna().all():
            continue

        values = row.astype(str).apply(clean_special_characters)

        # Skip rows that are mostly numbers or symbols
        if values.str.match(r'^[\d\s\±\-\+\.]+$').mean() > 0.7:
            continue

        non_empty_ratio = values.str.len().gt(0).mean()
        non_numeric_ratio = (~values.str.match(r'^\d*\.?\d+$')).mean()
        avg_length = values.str.len().mean()

        if (non_empty_ratio > 0.3 and
                non_numeric_ratio > 0.5 and
                avg_length < 50):
            header_candidates.append((idx, values))

    if header_candidates:
        # If we have multiple header rows, try to combine them
        if len(header_candidates) > 1:
            # Combine consecutive header rows
            combined_headers = []
            prev_idx = None
            for idx, values in header_candidates:
                if prev_idx is not None and idx - prev_idx > 1:
                    break
                if not combined_headers:
                    combined_headers = values.tolist()
                else:
                    for i, val in enumerate(values):
                        if val.strip() and combined_headers[i].strip():
                            combined_headers[i] = f"{combined_headers[i]} {val}".strip()
                        elif val.strip():
                            combined_headers[i] = val.strip()
                prev_idx = idx
            header_idx = header_candidates[-1][0]
            headers = combined_headers
        else:
            header_idx = header_candidates[0][0]
            headers = header_candidates[0][1].tolist()

        # Clean and make headers unique
        headers = [clean_special_characters(h) for h in headers]
        headers = make_unique_headers(headers)

        return headers, header_idx

    # If no clear headers found, generate default ones
    return make_unique_headers([f'Column_{i+1}' for i in range(df.shape[1])]), -1

def validate_and_clean_table_data(df):
    """
    Enhanced validation and cleaning for complex table data
    """
    if df is None or df.empty:
        return None

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Clean all cell values
    for col in df.columns:
        df[col] = df[col].apply(clean_special_characters)

    # Remove completely empty rows and columns
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    if df.shape[0] < 3 or df.shape[1] < 2:
        return None

    # Identify and set headers
    headers, header_idx = identify_column_headers(df)

    if header_idx >= 0:
        data_df = df.iloc[header_idx + 1:].copy()
    else:
        data_df = df.copy()

    # Reset index and set unique column names
    data_df.reset_index(drop=True, inplace=True)
    data_df.columns = headers

    # Process categories
    try:
        data_df = process_categories(data_df)
    except Exception as e:
        print(f"Warning: Error in category processing - {str(e)}")
        # Continue without category processing if it fails
        pass

    if data_df.empty or data_df.shape[1] < 2:
        return None

    return data_df

def process_categories(df):
    """
    Enhanced category processing with better error handling
    """
    try:
        processed_rows = []
        current_category = None

        for idx, row in df.iterrows():
            row_values = row.astype(str).apply(clean_special_characters)

            if is_category_row(row_values):
                current_category = row_values.iloc[0].strip()
                continue

            if current_category and not row_values.isna().all():
                new_row = row.copy()
                first_val = clean_special_characters(new_row.iloc[0])
                if first_val:
                    new_row.iloc[0] = f"{current_category} - {first_val}"
                processed_rows.append(new_row)
            elif not row_values.isna().all():
                processed_rows.append(row)

        if processed_rows:
            result_df = pd.DataFrame(processed_rows)
            result_df.columns = df.columns
            return result_df
        return df

    except Exception as e:
        print(f"Warning: Error in category processing - {str(e)}")
        return df

@app.route('/extract-tables', methods=['POST'])
def extract_tables():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "File format not supported, please upload a PDF"}), 400

    try:
        temp_pdf_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_pdf_path)

        # Use lattice flavor for complex tables with lines
        tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='stream')
        if not tables:
            print("Using lattice")
            # Try stream flavor if lattice doesn't find tables
            tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')

        if not tables:
            return jsonify({"message": "No tables found in the PDF"}), 200

        temp_zip_stream = io.BytesIO()

        print(len(tables))

        tables.export("foo.csv", f='csv')
        with zipfile.ZipFile(temp_zip_stream, 'w') as temp_zip:
            for i, table in enumerate(tables):
                try:
                    df = table.df

                    if is_potential_table(df):
                        table_number, table_description = extract_table_metadata(
                            temp_pdf_path,
                            table.parsing_report['page']
                        )

                        validated_df = validate_and_clean_table_data(df)

                        if validated_df is not None:
                            # Create metadata with matching columns
                            num_cols = len(validated_df.columns)
                            # Combine metadata and data
                            final_df = pd.concat([validated_df], ignore_index=True)

                            # Write to CSV
                            csv_stream = io.StringIO()
                            final_df.to_csv(csv_stream, index=False)
                            csv_stream.seek(0)
                            csv_filename = f"Table_{table_number or (i+1)}_processed.csv"
                            temp_zip.writestr(csv_filename, csv_stream.read())

                except Exception as e:
                    print(f"Error processing Table {i+1}: {str(e)}")
                    continue

        temp_zip_stream.seek(0)

        response = make_response(send_file(
            temp_zip_stream,
            mimetype='application/zip',
            as_attachment=True,
            download_name='processed_tables.zip'
        ))
        response.headers['Content-Disposition'] = 'attachment; filename=processed_tables.zip'
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)