# Grocery Store Product Recognition System

A FastAPI-based web application that uses computer vision and deep learning to recognize grocery store products.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)


## Installation

### Windows (WSL/Ubuntu)

1. **Navigate to project directory**
   bash
   cd /mnt/c/Users/gabri/Desktop/Final/Final
   

2. **Create a virtual environment**
   bash
   python3 -m venv .venv
   

3. **Activate the virtual environment**
   bash
   source .venv/bin/activate
   

4. **Update pip**
   bash
   pip install --upgrade pip
  

5. **Install system dependencies (if needed)**
   bash
   sudo apt-get update
   sudo apt-get install -y libgomp1 libopenblas-dev
   

6. **Install PyTorch (CPU version)**
   bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   

7. **Install other project dependencies**
   bash
   pip install -r requirements.txt
   
   
   If you don't have a `requirements.txt`, install manually:
  
   pip install fastapi uvicorn pydantic pillow numpy python-multipart
   

Windows (Command Prompt)

 **Note**: It's recommended to move your project out of OneDrive to avoid permission issues.

1. **Move project out of OneDrive (Optional but Recommended)**
   cmd
   cd C:\Users\gabri\Desktop
   mkdir Projects
   xcopy /E /I C:\Users\gabri\OneDrive\Desktop\Final\Final Projects\Final
   cd Projects\Final
   

2. **Create a virtual environment**
   cmd
   python -m venv .venv
   

3. **Activate the virtual environment**
   cmd
   .venv\Scripts\activate
   

4. **Update pip**
   cmd
   python -m pip install --upgrade pip
   

5. **Install PyTorch (CPU version)**
    cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   

6. **Install other project dependencies**
   cmd
   pip install -r requirements.txt
    

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

