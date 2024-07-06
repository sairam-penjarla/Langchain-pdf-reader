## Langchain-pdf-reader

This repository provides a Python script for using Langchain to answer questions based on a PDF document. 

### Functionality

1. **Process a PDF:** The script can read text content from a local PDF file.
2. **Text Splitting:** It splits the extracted text into smaller chunks for efficient processing.
3. **OpenAI Embeddings:** The script utilizes OpenAI to generate embeddings for the text chunks.
4. **Document Search Index:** A FAISS-based index is created for fast retrieval of relevant text sections based on user queries.
5. **Question Answering with LLM:** The script leverages Langchain's Question Answering chain with OpenAI to answer user questions using the retrieved document sections.

**Note:** This script provides a basic example. You can extend it to support functionalities like:

* Handling multiple PDF documents.
* Creating a web interface for user interaction.
* Integrating with different LLM providers or question answering models.


### Installation

1. Clone this repository:

```bash
git clone https://github.com/sairam-penjarla/Langchain-pdf-reader.git
```

2. Install the required libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```

**Note:** You'll need to set your OpenAI API key in the environment variable `OPENAI_API_KEY`.

### Usage

1. Replace `'budget_speech.pdf'` in the script with the path to your desired PDF file.
2. Define your questions in the `query` variables.
3. Run the script:

```bash
python langchain_pdf_reader.py
```

The script will output the answers to your questions based on the processed PDF content.

### Contributing

We welcome contributions to improve this project. Feel free to fork the repository, make changes, and submit a pull request.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
