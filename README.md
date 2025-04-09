
# Exploring Table Representations for RAG on Large Documents

[![Download Thesis WIP](https://img.shields.io/badge/Download--PDF-Thesis--WIP-orange)](https://www.overleaf.com/read/mqphwrjjhytz#5746a7)

| ![Tab Tree Overview](data/TabTreeOverview.png) |
| :----------------------------------------------: |
|            Figure 1: Tab Tree Overview            |

### Introduction / Motivation

This repository explores the capabilities of LLMs for Table QA in documents containing both text and tables. While LLMs have revolutionized tasks involving unstructured text, their architecture — rooted in the transformer model — is primarily designed for sequential data, making it less suited to the inherently two-dimensional structure of tabular data. Existing research highlights the limitations of LLMs in handling tabular heterogeneity, data sparsity, and correlations, leaving room for improvement in tasks like Table QA, particularly when tables are embedded in complex documents.    

This project aims to bridge these gaps by introducing a framework for Table QA on large documents (incorporating tables and text), with the following core components:  
- **TabTree Serialization Model**: A new table serialization method that models tables as directed tree structures.
- **RAG Framework**: Retrieval Augmented Generation framework tailored for the specific Table QA task on large documents, incorporating multiple table-aware mechanisms.
- **Evaluation Dataset for Table QA on Large Documents**: A manually labeled evaluation dataset to benchmark the effectiveness of table representations and retrieval strategies.  

### Research Questions  

The thesis addresses the following core research questions:  
- How well do LLMs perform in Table QA on large documents with integrated text and tables?  
- Which table representation yields the best quality in Table QA?  
  - How does table representation affect retrieval effectiveness in a RAG pipeline?  
  - How does it influence the generative QA process of LLMs?  
- Which components of a RAG pipeline are most critical for generating accurate responses with respect to tabular data? 

### Getting Started

#### Prerequisites

Before starting, ensure you have the following installed:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Python 3.12](https://www.python.org/)

#### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate rag-project
   ```

   You might change the name of the environment in the `environment.yml` file if needed.

5. **Set up environment variables**:  
   Create a `.env` file in the root directory by copying the contents of `.env.example`. The `.env` file contains sensitive information like API keys and environment-specific configurations. Ensure that your environment variables are correctly set before running the project. 
   
---

#### Running the Code

Before running any code, you need to set up Ollama and Qdrant as services using Docker Compose.

1. **Set up Qdrant**:  
   The `compose.yml` file sets up Qdrant (for vector storage). Make sure these services are running before executing code related to indexing and retrieval tasks. You can start the services with the following command:

   ```bash
   docker-compose -f compose.yml up -d
   ```

2. **Run Code**: 
   We provide different configuration files for the different evaluated tasks of the thesis, i.e. Context / Header Detection, Document / Dataset Analysis, Vanilla Table QA, Chunk Retrieval, and Table QA on Full Documents. For detailed information refer to the documentation within the Thesis.

   To simply index the two evaluation documents of the SEC Filing data, i.e., `data/sec_filings/awk-20231231.htm` and `data/sec_filings/uber-20231231.htm` and run a chat interface using HTML table serialization, you can use the default configuration and run the following command:

   ```bash
   python -m src
   ```

   **Note:** Each paramater in the configuration files can be overwritten by passing it as a command line argument.

   To run the code for the different evaluated tasks, you can run one of the predefined configuration within the `config` folder. The following tasks are available:

   - **Context / Header Detection**:  
     ```bash
     python -m src --config-path ./configs/context_detection/context_detection.toml
     ```

   - **Document / Dataset Analysis**: 
      - **SEC-Filing Dataset**:  
        ```bash
        python -m src --config-path ./configs/document_analysis/sec_filings.toml
        ``` 
      - **WikiTableQuestions Dataset**:  
        ```bash
        python -m src --config-path ./configs/document_analysis/wikitablequestions.toml
        ```

   - **Vanilla Table QA (using WikiTableQuestions Dataset)**:
      - **Baselines (HTML, CSV, JSON, Markdown)**
       ```bash
       python -m src --config-path ./configs/qa_only/baselines.toml
       ```
      - **TabTree Primary Subtree Selection**:  
        ```bash
        python -m src --config-path ./configs/qa_only/tabtree_primary_subtree.toml
        ```
      - **TabTree**:  
        ```bash
        python -m src --config-path ./configs/qa_only/tabtree.toml
        ```
   - **Chunk Retrieval**:
      - **Baselines (HTML, CSV, JSON, Markdown)**:  
        ```bash
        python -m src --config-path ./configs/retrieval/baselines.toml
        ```
      - **TabTree Primary Subtree Selection**:  
        ```bash
        python -m src --config-path ./configs/retrieval/tabtree_primary_subtree.toml
        ```
      - **TabTree**:  
        ```bash
        python -m src --config-path ./configs/retrieval/tabtree.toml
        ```
      - **Table Summaries**:  
        ```bash
        python -m src --config-path ./configs/retrieval/table_summary.toml
        ```
   - **Table QA on Full Documents**:
      - **Baselines (HTML, CSV, JSON, Markdown)**:  
        ```bash
        python -m src --config-path ./configs/full_qa/baselines.toml
        ```
      - **TabTree**:  
        ```bash
        python -m src --config-path ./configs/full_qa/tabtree.toml
        ```
      - **Table Summaries**:  
        ```bash
        python -m src --config-path ./configs/full_qa/table_summary.toml
        ```


3. **Run Tests for the TabTree Model**
   If you want to run the tests for the TabTree model, you can do so by running the following command:

   ```bash
   python -m unittest discover -s src/tests -p tabtree_model_test.py
   ```


