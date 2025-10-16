# Customer Support RAG Application

An AI-powered customer support chatbot for a digital contract platform using Retrieval Augmented Generation (RAG).

## Features

- Chat-based interface for customer support
- Answers questions based on support documentation
- Automatically escalates complex issues to WhatsApp support
- Uses Mistral AI for natural language understanding
- **NEW:** Persistent ticket storage with Supabase (optional)
- **NEW:** Admin dashboard API endpoints for ticket management
- **NEW:** Conversation history tracking

## Setup

1. Install required packages:
```
   pip3 install --break-system-packages langchain-text-splitters langchain-community langgraph langchain-mistralai openpyxl pandas flask flask-cors supabase
```

2. Update the WhatsApp support link in `customer_support_agent.py`

3. Add your support documentation to the `SupportDocuments/` folder as a .csv or .xlsx file

4. (Optional) Set up Supabase for persistent storage:
   - See [SUPABASE_SETUP.md](SUPABASE_SETUP.md) for detailed instructions
   - Set environment variables: `SUPABASE_URL` and `SUPABASE_KEY`
   - Without Supabase, the app runs in stateless mode

5. Run the application:
```
   python3 customer_support_agent.py
```

## Documentation Format

Your spreadsheet should have these columns:
- **Question** (or **Topic**): The question or topic
- **Answer** (or **Content**): The detailed answer
- **Category** (optional): Category for organization

## API Endpoints

### Chat Endpoint
- `POST /chat` - Send customer questions and receive AI responses
  - Request: `{ "question": "...", "customer_info": {...}, "ticket_id": "..." }`
  - Response includes `ticket_id` for tracking

### Admin Dashboard Endpoints (requires Supabase)
- `GET /api/tickets` - List all tickets (supports `?status=` filter)
- `GET /api/tickets/:id` - Get ticket details with conversation history
- `PATCH /api/tickets/:id` - Update ticket (e.g., mark as resolved)
- `GET /api/tickets/summary` - Get ticket counts by status

### Health Check
- `GET /api/health` - Check API status and Supabase connection

## Requirements

- Python 3.8+
- Mistral AI API key
- (Optional) Supabase account for persistence