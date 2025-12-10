# Grocery Store Product Recognition System

A FastAPI-based web application that uses computer vision and deep learning to recognize grocery store products.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Getting Started

### Download and Extract the Project

1. **Download the project archive**
   - Download `Final.tar.gz` from the project repository

2. **Extract the archive**

   **Linux/macOS:**
   ```bash
   tar -xzf Final.tar.gz
   cd Final
   ```

   **Windows (using PowerShell or Command Prompt):**
   ```cmd
   tar -xzf Final.tar.gz
   cd Final
   ```

   **Windows (using 7-Zip or WinRAR):**
   - Right-click `Final.tar.gz` → Extract Here
   - Navigate to the extracted `Final` folder

## Project Structure

```
Final/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── model.py                # ProductClassifier stub
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # FastAPI routes and endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models and ProductClassifier
│   └── services/
│       ├── __init__.py
│       ├── database.py         # SQLite database service
│       └── vision.py           # Image processing and classification
├── data/
│   ├── products.csv            # Product catalog data
│   └── products.db             # SQLite database (auto-generated)
├── models/                      # Directory for trained model weights
├── scripts/
│   └── train_stub.py           # Training script placeholder
├── requirements.txt             # Python dependencies
└── ReadMe.txt                   # This file
```

## Installation

### Step 1: Create a Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Update pip

```bash
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The `requirements.txt` includes all necessary packages including PyTorch. If you encounter issues with PyTorch installation, you may need to install it separately:

**For CPU-only (Linux/macOS/Windows):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU support (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
    

## Running the Application

### Start the server

**WSL/Ubuntu:**
bash
uvicorn app.main:app --reload


**Windows Command Prompt:**
cmd
uvicorn app.main:app --reload


### Access the application

- **Main Application**: http://127.0.0.1:8000
- **API Documentation (Swagger UI)**: http://127.0.0.1:8000/docs
- **Alternative API Docs (ReDoc)**: http://127.0.0.1:8000/redoc

### Stop the server

Press Ctrl + C in the terminal






GET /

Returns basic API information and status.

### Product Recognition

POST /predict

Upload an image to identify the grocery product.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
json
{
  "product": "Product Name",
  "confidence": 0.95,
  "location": "Aisle 3, Shelf 2"
}


### Additional Endpoints
[Add your other endpoints here]

## Troubleshooting

### PyTorch Import Error

If you see:

OSError: libtorch_global_deps.so: cannot open shared object file


**Solution:**
bash
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir


### Permission Denied (OneDrive)

If you get permission errors when creating virtual environment in OneDrive:

**Solution:** Move your project out of OneDrive folder or pause OneDrive sync temporarily.

### "Got unexpected extra argument (reload)"

Make sure you're using `--reload` (double dash) not `-- reload` (space between dashes).

**Correct:**
bash
uvicorn app.main:app --reload


### Module Not Found Errors

Make sure your virtual environment is activated:

**WSL/Ubuntu:**
bash
source .venv/bin/activate


**Windows:**
cmd
.venv\Scripts\activate


Then reinstall dependencies:
bash
pip install -r requirements.txt


### Port Already in Use

If port 8000 is already in use, specify a different port:
bash
uvicorn app.main:app --reload --port 8001


## Development

### Creating requirements.txt

To generate a requirements file from your current environment:
bash
pip freeze > requirements.txt


## Contributors

- Gabriel Rivera
- Will Andrews
- Gabe Valladares

