# Nakamoto Blockchain Simulation

A simulation platform for analyzing blockchain behavior and hashrate control mechanisms.

## Project Structure

```
Nakamoto/
├── src/
│   ├── models/          # Core blockchain models
│   ├── api/            # Flask API endpoints
│   ├── data/           # Data processing utilities
│   └── simulation/     # Simulation engine
├── tests/              # Test files
├── data/               # CSV and JSON data files
├── static/             # Images and static assets
├── pages/             # Streamlit pages
├── app.py             # Main Streamlit application
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask API:
```bash
python -m src.api.routes
```

2. Start the Streamlit interface:
```bash
streamlit run app.py
```

## Development

- The main simulation logic is in `src/simulation/engine.py`
- Blockchain models are defined in `src/models/blockchain.py`
- API endpoints are in `src/api/routes.py`
- The Streamlit interface is in `app.py`

## Testing

Run tests using:
```bash
python -m pytest tests/
```
