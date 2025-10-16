# Customer Support RAG Application

An AI-powered customer support chatbot for a digital contract platform using Retrieval Augmented Generation (RAG).

## Features

- Chat-based interface for customer support
- Answers questions based on support documentation
- Automatically escalates complex issues to WhatsApp support
- Uses Mistral AI for natural language understanding

## Setup

1. Install required packages:
```
   pip3 install --break-system-packages langchain-text-splitters langchain-community langgraph langchain-mistralai openpyxl pandas
```

2. Update the WhatsApp support link in `customer_support_agent.py`

3. Add your support documentation to the `SupportDocuments/` folder as a .csv or .xlsx file

4. Run the application:
```
   python3 customer_support_agent.py
```

## Documentation Format

Your spreadsheet should have these columns:
- **Question** (or **Topic**): The question or topic
- **Answer** (or **Content**): The detailed answer
- **Category** (optional): Category for organization

## Requirements

- Python 3.8+
- Mistral AI API key