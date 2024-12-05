from dotenv import load_dotenv
import pprint
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
import os
import re
import csv
import tabula
import pandas as pd
import string
import json

load_dotenv()

def process_table_content(table_text):
    """
    Process the raw table text into structured rows with proper headers.

    Args:
        table_text (str): Raw table text from PDF

    Returns:
        list: Processed rows including headers
    """
    lines = table_text.strip().split('\n')

    # Find the header line by looking for a line with mostly column headers
    header_line_idx = None
    max_header_score = 0

    for idx, line in enumerate(lines):
        if idx < len(lines) - 1:  # Avoid last line
            words = line.split()
            if not words:
                continue

            # Calculate a "header score" based on characteristics of typical headers
            non_numeric = sum(1 for w in words if not any(c.isdigit() for c in w))
            header_score = non_numeric / len(words) if words else 0

            # Additional header indicators
            if any(word.upper() == word for word in words):  # Contains uppercase words
                header_score += 0.2

            if header_score > max_header_score:
                max_header_score = header_score
                header_line_idx = idx

    if header_line_idx is None:
        return []

    # Process headers
    header_cells = ['Sample']  # First column will always be Sample
    header_cells.extend(lines[header_line_idx].split())

    # Process data rows
    data_rows = []
    current_row = []

    for line in lines[header_line_idx + 1:]:
        cells = line.strip().split()
        if not cells:
            continue

        # Check if this is a new row or continuation of previous row
        if len(current_row) == 0 or (
                not cells[0].replace('.', '').replace('-', '').isdigit() and  # First value is not a number
                not cells[0].startswith('±') and  # Not a standard deviation row
                not cells[0].startswith('+')):    # Not a continuation marker
            if current_row:
                # Pad or trim row to match header length
                while len(current_row) < len(header_cells):
                    current_row.append('')
                data_rows.append(current_row[:len(header_cells)])
            current_row = cells
        else:
            # Continue previous row with additional values
            current_row.extend(cells)

    # Add the last row if it exists
    if current_row:
        while len(current_row) < len(header_cells):
            current_row.append('')
        data_rows.append(current_row[:len(header_cells)])

    # Combine all rows with header
    all_rows = [header_cells]
    all_rows.extend(data_rows)

    return all_rows

def extract_doi(text):
    """
    Extract DOI from the document text.

    Args:
        text (str): Full document text

    Returns:
        str: Extracted DOI or None if not found
    """
    # Pattern to match DOI formats
    doi_match = re.search(r'http://dx\.doi\.org/[^\s]+', text)
    if not doi_match:
        # Backup patterns
        doi_match = re.search(r'(?:doi:|DOI:|https?://doi\.org/)[^\s]+', text)

    doi = doi_match.group(0) if doi_match else None
    return doi

def extract_table_metadata(text, table_start_index):
    """
    Extract metadata associated with a table, including title and additional notes.

    Args:
        text (str): Full document text
        table_start_index (int): Starting index of the table in the text

    Returns:
        dict: Metadata associated with the table
    """
    metadata = {
        'title': None,
        'note': None,
        'abbreviations': None
    }

    # Search for potential title before the table (look at preceding lines)
    title_search_range = text[:table_start_index].split('\n')[-3:]  # Look at previous 3 lines
    for line in reversed(title_search_range):
        # Check for title-like lines (start with a number or have descriptive words)
        if re.match(r'^\d+\.', line) or any(word in line.lower() for word in ['table', 'of', 'scores', 'composition']):
            metadata['title'] = line.strip()
            break

    # Search for notes or explanations after the table
    note_pattern = r'(?:Note:|^a\s|The\s)'
    note_match = re.search(f'{note_pattern}.*', text[table_start_index:], re.MULTILINE)
    if note_match:
        metadata['note'] = note_match.group(0).strip()

    # Look for abbreviations section
    abbr_match = re.search(r'a?\s*[Aa]bbreviations?:?\s*(.*)', text[table_start_index:], re.MULTILINE)
    if abbr_match:
        metadata['abbreviations'] = abbr_match.group(1).strip()

    return metadata

def extract_and_save_tables(documents, doi, output_dir='output', ):
    """
    Extract tables from documents with enhanced metadata and save them as CSV files

    Args:
        documents (list): List of document objects
        output_dir (str): Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("DOI of extract", doi)

    # Define a pattern to detect Markdown-like table structures
    table_pattern = re.compile(
        r"(?<=\n)\|(?:[^|\n]+\|)+\n\|(?:[-:]+\|)+\n(?:\|(?:[^|\n]+\|)+\n)+",
        re.MULTILINE
    )

    for doc_index, doc in enumerate(documents):
        text = doc.text
        tables = list(table_pattern.finditer(text))

        if tables:
            print(f"Document {doc_index+1} contains {len(tables)} table(s)")

            # Use alphabetic suffixes for multiple tables on the same page
            table_suffixes = string.ascii_lowercase

            for table_index, table_match in enumerate(tables):
                # Get the full table match
                table = table_match.group(0)

                # Extract metadata
                metadata = extract_table_metadata(text, table_match.start())

                # Prepare filename
                suffix = table_suffixes[table_index] if len(tables) > 1 else ''
                base_filename = f"{doc_index+1}{suffix}"

                rows = process_table_content(table)

                # Process table into CSV
                rows = [row.strip().split('|')[1:-1] for row in table.strip().split('\n')]

                # Remove alignment rows
                rows = [row for row in rows if not all(set(cell.strip()) <= {'-', ':'} for cell in row)]

                # Ensure first column header is "Sample"
                if rows:
                    headers = rows[0]
                    headers[0] = "Sample"
                    # Add Title and DOI columns
                    headers.extend(["Title", "DOI"])
                    rows[0] = headers

                    # Add metadata to first data row
                    if len(rows) > 1:
                        rows[1].extend([metadata['title'] if metadata['title'] else '', doi if doi else ''])

                    # Add empty rows for spacing
                    rows.append([''] * len(headers))  # Empty row for spacing

                    # Add abbreviations if available
                    if metadata['abbreviations']:
                        rows.append(['Abbreviations:', metadata['abbreviations']] + [''] * (len(headers) - 2))

                    # Add notes if available
                    if metadata['note']:
                        rows.append(['Note:', metadata['note']] + [''] * (len(headers) - 2))

                # Write table to CSV
                csv_filepath = os.path.join(output_dir, f"{base_filename}.csv")
                with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerows(rows)

                # Write metadata to JSON
                metadata_filepath = os.path.join(output_dir, f"{base_filename}_metadata.json")
                with open(metadata_filepath, 'w', encoding='utf-8') as jsonfile:
                    metadata['doi'] = doi  # Add DOI to metadata
                    json.dump(metadata, jsonfile, indent=2)

                print(f"Saved table to {base_filename}.csv")
                print(f"Saved metadata to {base_filename}_metadata.json")
        else:
            print(f"Document {doc_index+1} does not contain any tables")

if __name__ == '__main__':
    parser = LlamaParse(
        result_type="markdown"  # "markdown" and "text" are available
    )

    file_extractor = {".pdf": parser}

    nest_asyncio.apply()
    documents = SimpleDirectoryReader(input_files=['sample1.pdf'], file_extractor=file_extractor).load_data()

    pprint.pprint(documents)
    extract_and_save_tables(documents)
