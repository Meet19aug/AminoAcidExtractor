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
    # This assumes notes typically start with "Note:", "a", or have additional information
    note_pattern = r'(?:Note:|^a\s|The\s)'
    note_match = re.search(f'{note_pattern}.*', text[table_start_index:], re.MULTILINE)
    if note_match:
        metadata['note'] = note_match.group(0).strip()

    # Look for abbreviations section
    abbr_match = re.search(r'a?\s*[Aa]bbreviations?:?\s*(.*)', text[table_start_index:], re.MULTILINE)
    if abbr_match:
        metadata['abbreviations'] = abbr_match.group(1).strip()

    return metadata

def extract_and_save_tables(documents, output_dir='output'):
    """
    Extract tables from documents with enhanced metadata and save them as CSV files
    
    Args:
        documents (list): List of document objects
        output_dir (str): Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

                # Process table into CSV
                rows = [row.strip().split('|')[1:-1] for row in table.strip().split('\n')]

                # Remove alignment rows
                rows = [row for row in rows if not all(set(cell.strip()) <= {'-', ':'} for cell in row)]

                # Write table to CSV
                csv_filepath = os.path.join(output_dir, f"{base_filename}.csv")
                with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)

                    # Clean and write rows
                    cleaned_rows = []
                    for row in rows:
                        cleaned_row = [cell.strip() for cell in row]
                        cleaned_rows.append(cleaned_row)

                    csv_writer.writerows(cleaned_rows)

                # Write metadata to JSON
                metadata_filepath = os.path.join(output_dir, f"{base_filename}_metadata.json")
                with open(metadata_filepath, 'w', encoding='utf-8') as jsonfile:
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
    documents = SimpleDirectoryReader(input_files=['sample.pdf'], file_extractor=file_extractor).load_data()

    print(documents)

    pprint.pprint(documents)
    # Assuming 'documents' is already defined from the previous context
    extract_and_save_tables(documents)

