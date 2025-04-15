# NLP 2 Project

## Overview
This project focuses on Natural Language Processing (NLP) techniques and applications. It includes various modules and tools to process, analyze, and extract insights from textual data.

## Features
- Text preprocessing (tokenization, stemming, lemmatization, etc.)
- Sentiment analysis
- Named Entity Recognition (NER)
- Topic modeling
- Custom NLP pipelines
- Usage of Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

## Installation

### Using Poetry
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/nlp_2.git
    ```
2. Navigate to the project directory:
    ```bash
    cd nlp_2
    ```
3. Install dependencies using Poetry:
    ```bash
    poetry install
    ```
4. Activate the Poetry environment:
    ```bash
    poetry shell
    ```

## Usage
1. Run the main script:
    ```bash
    python application.py
    ```
2. Define the necessary environment variables in a `.env` file. This file should be created in the root directory of the project (`nlp_2/`). The required variables include:
    - `PINECONE_API_KEY`: Pinecone database API KEY.
    - `PINECONE_INDEX_NAME`: Name of the Pinecone's index.
    - `PINECONE_SPACE_NAME`: Name of the Pinecone's space.
    - `GROQ_API_KEY`: Groq's API KEY.
    - `FLASK_SECRET_KEY`:  Key of your Flask application.

## Project Structure
```
nlp_2/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Source code
│   ├── preprocessing/  # Text preprocessing scripts
│   ├── models/         # Machine learning models
│   └── utils/          # Utility functions
├── tests/              # Unit tests
├── config/             # Configuration files
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add feature-name"
    ```
4. Push to your branch:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or suggestions, please contact [alexbarria](mailto:alexbarria14@gmail.com).


## Demo
![Chatbot Demo](./img/Chatbot%20Record%20GIF.gif)
