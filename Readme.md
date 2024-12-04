
# Extract Amino Acid Data from PDFs Using Advanced AI Technology

This project automates the extraction of amino acid data and metadata from PDF documents, leveraging **Camelot** for tabular data extraction and **LlamaIndex** for enhanced performance and accuracy. The solution includes a **Flask-powered frontend** for user interaction.

## Features

- **Table Extraction**: Extract structured tables from PDFs using Camelot and LlamaIndex.
- **Metadata Enrichment**: Automatically extract and save table metadata, including titles, notes, and abbreviations.
- **Advanced Validation**: Validate and clean tables for consistency and usability.
- **DOI and Title Detection**: Identify and attach document DOI and title to the extracted tables.
- **Flexible Output**: Save extracted tables as CSV files and metadata as JSON.
- **Web Interface**: User-friendly web application for uploading PDFs and downloading extracted data.

## Technologies Used

- **Python Libraries**: 
  - `Camelot` for PDF table extraction
  - `LlamaIndex` for structured document analysis
  - `pdfplumber` for metadata extraction
  - `Flask` for the web interface
  - `Pandas` and `NumPy` for data manipulation
- **Environment Management**: `dotenv` for configuration
- **Frontend Integration**: Flask-based API for interaction

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/extract-amino-acids.git
   cd extract-amino-acids
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4: **Configure Your API Key**:

To use Llama Cloud, you need to set your API key in the environment variables. Follow these steps:

1. Get your API key from the [LlamaIndex Cloud Documentation](https://docs.cloud.llamaindex.ai/llamaparse/getting_started/get_an_api_key).
2. Add the following line to your `.env` file or your terminal environment:

   ```bash
   LLAMA_CLOUD_API_KEY=llx-your-api-key
   ```

Replace `llx-your-api-key` with the API key you generated in step 1.

5. **Run the Flask app**:
   ```bash
   python main.py
   ```

6. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`.

## Usage

### Flask Web Interface

1. Upload a PDF file using the web interface.
2. The application extracts tables and metadata from the PDF.
3. Download the processed data as a ZIP file containing:
   - CSV files for tables
   - JSON files for metadata

### Direct Script Execution

To run the extraction process directly:
1. Place your PDF files in the `input_files` directory.
2. Modify and execute `extract_complex_pdf.py`:
   ```bash
   python extract_complex_pdf.py
   ```

The extracted tables will be saved in the `output` directory.

## File Descriptions

- **`app.py`**: Flask application for handling PDF uploads and table extraction.
- **`extract_complex_pdf.py`**: Core logic for table extraction and metadata processing.
- **`requirements.txt`**: List of required Python packages.

## Examples

### Extracted Output
- **Table CSV**: Contains structured table data with enriched column headers.
- **Metadata JSON**: Contains metadata such as table titles, notes, abbreviations, DOI, and document title.

Example Metadata:
```json
{
  "title": "Table 1. Composition of amino acids",
  "note": "Data represent mean values ± standard deviations.",
  "abbreviations": "AA: Amino Acids; SD: Standard Deviation"
}
```

## Dependencies

- Python 3.12
- Required Python libraries (see `requirements.txt`)

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature/your-feature`).
3. Commit your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the contributors of Camelot, LlamaIndex, and Flask for enabling the core functionalities of this project.
