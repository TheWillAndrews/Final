from __future__ import annotations

from app.services.vision import (
    process_bytes_for_single_product,
    fake_multi_detections,
    draw_detections,
)

from datetime import datetime
from pathlib import Path
import io
import base64
from PIL import Image

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from app.services.database import get_db         # new DB accessor

from app.models.schemas import (
    ProductClassifier,
    ProductLocation,
    DetectedProduct,
    LocateResponse,
)

router = APIRouter()

# Global singletons for demo
classifier = ProductClassifier()
db = get_db()
log_path = Path("data/requests.log")


def _log_request(result: dict) -> None:
    """Append a simple log line for each prediction."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            ts = datetime.now().isoformat(timespec="seconds")
            f.write(
                f"{ts} | label={result.get('predicted_label')} | "
                f"conf={result.get('confidence'):.3f} | "
                f"found={result.get('found_in_database')} | "
                f"aisle={result.get('location', {}).get('aisle')} | "
                f"section={result.get('location', {}).get('section')}\n"
            )
    except Exception:
        # Logging should never break the app; ignore any file errors
        pass

@router.post("/products/basic", response_model=LocateResponse)
async def classify_basic_product(file: UploadFile = File(...)) -> LocateResponse:
    """
    MVP endpoint:
    - Takes an image
    - Uses ProductClassifier (banana / apple / neither)
    - Does NOT touch the database
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    file_bytes = await file.read()

    # Open with PIL
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    # New ProductClassifier.predict returns (label, conf, topk)
    label, confidence, _ = classifier.predict(image)

    detected = DetectedProduct(name=label, confidence=confidence)

    if label == "neither":
        msg = "No banana or apple detected."
    else:
        msg = f"Detected {label}."

    # location=None because we’re ignoring DB for this MVP
    return LocateResponse(
        detected_product=detected,
        location=None,
        message=msg,
    )

@router.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Upload UI for single-product prediction with drag & drop and nav."""
    return """
    <html>
      <head>
        <title>Grocery Aisle Product Finder</title>
        <style>
          body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f4f7;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 60px;
          }
          .container {
            background: #ffffff;
            padding: 24px 28px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
            width: 420px;
          }
          .nav {
            display: flex;
            gap: 12px;
            font-size: 0.8rem;
            margin-bottom: 12px;
          }
          .nav a {
            color: #2563eb;
            text-decoration: none;
          }
          .nav a:hover {
            text-decoration: underline;
          }
          h1 {
            margin-top: 0;
            font-size: 1.5rem;
          }
          p {
            color: #4b5563;
            font-size: 0.9rem;
          }
          form {
            margin-top: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;   
          }
          .dropzone {
            border: 2px dashed #cbd5f5;
            border-radius: 10px;
            padding: 18px;
            text-align: center;
            font-size: 0.9rem;
            color: #6b7280;
            cursor: pointer;
            background: #f9fafb;
          }
          .dropzone.dragover {
            background: #eff6ff;
            border-color: #2563eb;
            color: #1d4ed8;
          }
          .file-status {
            margin-top: 6px;
            font-size: 0.8rem;
            color: #6b7280;
          }
          button {
            border: none;
            padding: 10px 14px;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-weight: 600;
            cursor: pointer;
          }
          button:hover {
            background: #1d4ed8;
          }
          .note {
            margin-top: 12px;
            font-size: 0.8rem;
            color: #6b7280;
          }
          .api-note {
            margin-top: 16px;
            font-size: 0.8rem;
            color: #9ca3af;    
          }
          code {
            background: #f3f4f6;
            padding: 2px 4px; 
            border-radius: 4px;
            font-size: 0.8rem;
          }
          a {
            color: #2563eb;
            text-decoration: none;
          }
          a:hover {  
            text-decoration: underline;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav">
            <a href="/">Home</a>
            <a href="/products">Products</a>
            <a href="/detect">Detect</a>
            <a href="/docs">API Docs</a>
          </div>
          
          <h1>Grocery Aisle Product Finder 🛒</h1>
          <p>Upload a product image and see which aisle &amp; section it belongs to.</p>
        
          <form id="upload-form" action="/predict" enctype="multipart/form-data" method="post">
            <input id="file-input" name="file" type="file" accept="image/*" required hidden>
            <div id="dropzone" class="dropzone">
              Drag &amp; drop an image here, or click to browse.
            </div>
            <div id="file-status" class="file-status"></div>
            <button type="submit">Find product location</button>
          </form>
           
          <div class="note">
            Try a photo of a <strong>banana</strong>, <strong>apple</strong>,
            <strong>cereal</strong>, or <strong>milk</strong> for a nicer-looking result.
          </div>
    
          <div class="api-note">
            API users: send a POST with <code>multipart/form-data</code> to
            <code>/api/predict</code> with field <code>file</code> to get raw JSON.
          </div>
                 
          <div class="api-note">
            Browse the <a href="/products">product catalog</a> or try the
            <a href="/detect">multi-object detection demo</a>.
          </div>
        </div>
        <script>
          (function () {
            const dropzone = document.getElementById('dropzone');
            const fileInput = document.getElementById('file-input');
            const status = document.getElementById('file-status');
        
            if (!dropzone || !fileInput) return;

            function updateStatus() {
              if (fileInput.files && fileInput.files.length > 0) {
                const name = fileInput.files[0].name;
                status.textContent = `Selected: ${name}`;
              } else {
                status.textContent = '';
              }
            }
    
            dropzone.addEventListener('click', () => fileInput.click());
           
            dropzone.addEventListener('dragover', (e) => {
              e.preventDefault();
              dropzone.classList.add('dragover');
            });
    
            ['dragleave', 'dragend'].forEach((type) => {
              dropzone.addEventListener(type, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
              });
            });

            dropzone.addEventListener('drop', (e) => {
              e.preventDefault();
              dropzone.classList.remove('dragover');
              if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                updateStatus();
              }
            });

            fileInput.addEventListener('change', updateStatus);
          })();
        </script>
      </body>
    </html>
    """

@router.get("/products", response_class=HTMLResponse)
async def list_products() -> str:
    """Simple page to list all products from the database."""
    products = db.all_products()

    rows_html = ""
    for p in products:
        stock_text = "In stock" if p.in_stock else "Out of stock"
        rows_html += f"""
        <tr>
          <td>{p.product_name}</td>
          <td>{p.category}</td>
          <td>{p.aisle}</td>
          <td>{p.section}</td>
          <td>${p.price:.2f}</td>
          <td>{stock_text}</td>
        </tr>
        """

    return f"""
    <html>
      <head>
        <title>Product Catalog</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f4f7;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 40px;
          }}
          .container {{
            background: #ffffff;
            padding: 24px 28px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
            width: 800px;
          }}
          h1 {{
            margin-top: 0;
            font-size: 1.4rem;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.9rem;
          }}
          th, td {{
            border-bottom: 1px solid #e5e7eb;
            padding: 6px 4px;
            text-align: left;
          }}
          th {{
            font-weight: 600;
            color: #4b5563;
          }}
          .links {{
            margin-top: 14px;
            font-size: 0.8rem;
          }}
          .links a {{
            color: #2563eb;
            text-decoration: none;
            margin-right: 12px;
          }}
          .links a:hover {{
            text-decoration: underline;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Product Catalog</h1>
          <table>
            <thead>
              <tr>
                <th>Product</th>
                <th>Category</th>
                <th>Aisle</th>
                <th>Section</th>
                <th>Price</th>
                <th>Stock</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>

          <div class="links">
            <a href="/">← Back to single-product finder</a>
            <a href="/detect">Multi-object detection demo</a>
          </div>
        </div>
      </body>
    </html>
    """

@router.post("/predict", response_class=HTMLResponse)
async def predict_html(file: UploadFile = File(...)) -> str:
    """HTML version – shows a pretty result card plus image preview + confidence bar."""
    file_bytes = await file.read()

    # --- CV: use vision service for image -> prediction ---
    try:
        image, label, confidence_value = process_bytes_for_single_product(
            file_bytes, file.content_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- DB lookup ---
    product_info: ProductLocation | None = db.lookup(label)

    # Build result dict in the same shape as before
    if product_info is None:
        result = {
            "predicted_label": label,
            "confidence": confidence_value,
            "found_in_database": False,
            "message": f"No location info found for '{label}'.",
            "location": None,
            "meta": None,
        }
    else:
        result = {
            "predicted_label": label,
            "confidence": confidence_value,
            "found_in_database": True,
            "location": {
                "aisle": product_info.aisle,
                "section": product_info.section,
            },
            "meta": {
                "category": product_info.category,
                "price": product_info.price,
                "in_stock": product_info.in_stock,
            },
        }

    _log_request(result)

    # --- Build data URL for preview ---
    try:
        b64 = base64.b64encode(file_bytes).decode("ascii")
        mime = file.content_type or "image/jpeg"
        img_data_url = f"data:{mime};base64,{b64}"
    except Exception:
        img_data_url = None

    label = result["predicted_label"]
    confidence_value = result["confidence"]
    confidence_pct = confidence_value * 100.0
    confidence_str = f"{confidence_pct:.1f}%"
    found = result["found_in_database"]

    # Confidence color logic
    if confidence_value >= 0.8:
        bar_color = "#16a34a"  # green
    elif confidence_value >= 0.5:
        bar_color = "#eab308"  # yellow
    else:
        bar_color = "#dc2626"  # red

    if found and result["location"]:
        aisle = result["location"]["aisle"]
        section = result["location"]["section"]
        category = result["meta"]["category"]
        price = result["meta"]["price"]
        in_stock = result["meta"]["in_stock"]
    else:
        aisle = section = category = price = in_stock = None

    status_text = "Found in database ✅" if found else "Not in database ❌"
    stock_text = (
        "In stock"
        if in_stock
        else "Out of stock"
        if in_stock is not None
        else "Unknown"
    )

    preview_block = (
        f'<div class="preview"><div class="preview-label">Uploaded image</div>'
        f'<img src="{img_data_url}" alt="Uploaded image" /></div>'
        if img_data_url
        else ""
    )

    details_block = (
        "".join(
            [
                f'<div class="row"><span class="label">Aisle:</span> <span class="value">{aisle}</span></div>',
                f'<div class="row"><span class="label">Section:</span> <span class="value">{section}</span></div>',
                f'<div class="row"><span class="label">Category:</span> <span class="value">{category}</span></div>',
                f'<div class="row"><span class="label">Price:</span> <span class="value">${price:.2f}</span></div>',
                f'<div class="row"><span class="label">Stock:</span> <span class="value">{stock_text}</span></div>',
            ]
        )
        if found and aisle is not None
        else '<div class="status">No aisle/section data available for this product.</div>'
    )

    return f"""
    <html>
      <head>
        <title>Result – Grocery Aisle Product Finder</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f4f7;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 60px;
          }}
          .container {{
            background: #ffffff;
            padding: 24px 28px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
            width: 460px;
          }}
          h1 {{
            margin-top: 0;
            font-size: 1.4rem;
          }}
          .pill {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 0.75rem;
            background: #eff6ff;
            color: #1d4ed8;
            margin-bottom: 8px;
          }}
          .row {{
            margin: 6px 0;
            font-size: 0.9rem;
          }}
          .label {{
            color: #6b7280;
          }}
          .value {{
            font-weight: 600;
          }}
          .status {{
            margin-top: 12px;
            font-size: 0.9rem;
          }}
          .back {{
            margin-top: 18px;
          }}
          a.button {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 600;
          }}
          a.button:hover {{
            background: #1d4ed8;
          }}
          .preview {{
            margin-top: 12px;
            margin-bottom: 16px;
          }}
          .preview-label {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 4px;
          }}
          .preview img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25);
          }}
          .confidence-container {{
            margin-top: 10px;
            margin-bottom: 12px;
          }}
          .confidence-label {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 4px;
          }}
          .confidence-bar-bg {{
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: #e5e7eb;
            overflow: hidden;
          }}
          .confidence-bar-fill {{
            height: 100%;
            border-radius: 999px;
            transition: width 0.3s ease;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="pill">Prediction: {label}</div>
          <h1>Product location result</h1>

          <div class="row">
            <span class="label">Confidence:</span>
            <span class="value">{confidence_str}</span>
          </div>

          <div class="confidence-container">
            <div class="confidence-label">Confidence level</div>
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill" style="width: {confidence_pct:.1f}%; background: {bar_color};"></div>
            </div>
          </div>

          {preview_block}

          {details_block}

          <div class="status"><strong>{status_text}</strong></div>

          <div class="back">
            <a href="/" class="button">Search another product</a>
          </div>
        </div>
      </body>
    </html>
    """

@router.get("/detect", response_class=HTMLResponse)
async def detect_index() -> str:
    """Upload page for the multi-object detection (simulated) demo with drag & drop + nav."""
    return """
    <html>
      <head>
        <title>Multi-object Detection Demo</title>
        <style>
          body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f4f7;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 60px;
          }
          .container {
            background: #ffffff;
            padding: 24px 28px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
            width: 460px;
          }
          .nav {
            display: flex;
            gap: 12px;
            font-size: 0.8rem;
            margin-bottom: 12px;
          }
          .nav a {
            color: #2563eb;
            text-decoration: none;
          }
          .nav a:hover {
            text-decoration: underline;
          }
          h1 {
            margin-top: 0;
            font-size: 1.4rem;
          }
          p {
            color: #4b5563;
            font-size: 0.9rem;
          }
          form {
            margin-top: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
          }
          .dropzone {
            border: 2px dashed #cbd5f5;
            border-radius: 10px;
            padding: 18px;
            text-align: center;
            font-size: 0.9rem;
            color: #6b7280;
            cursor: pointer;
            background: #f9fafb;
          }
          .dropzone.dragover {
            background: #eff6ff;
            border-color: #2563eb;
            color: #1d4ed8;
          }
          button {
            border: none;
            padding: 10px 14px;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-weight: 600;
            cursor: pointer;
          }
          button:hover {
            background: #1d4ed8;
          }
          .note {
            margin-top: 12px;
            font-size: 0.8rem;
            color: #6b7280;
          }
          a.back-link {
            display: inline-block;
            margin-top: 12px;
            font-size: 0.8rem;
            color: #2563eb;
            text-decoration: none;
          }
          a.back-link:hover {
            text-decoration: underline;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav">
            <a href="/">Home</a>
            <a href="/products">Products</a>
            <a href="/detect">Detect</a>
            <a href="/docs">API Docs</a>
          </div>

          <h1>Multi-object Detection (Simulated) 🧪</h1>
          <p>
            Upload a product image to see a simulated multi-object detection result
            with multiple boxes and labels.
          </p>

          <form id="detect-form" action="/detect" enctype="multipart/form-data" method="post">
            <input id="detect-file-input" name="file" type="file" accept="image/*" required hidden>
            <div id="detect-dropzone" class="dropzone">
              Drag &amp; drop an image here, or click to browse.
            </div>
            <button type="submit">Run detection</button>
          </form>

          <div class="note">
            This is a demo: detections are simulated based on the classifier's prediction.
          </div>

          <a href="/" class="back-link">← Back to single-product finder</a>
        </div>

        <script>
          (function () {
            const dropzone = document.getElementById('detect-dropzone');
            const fileInput = document.getElementById('detect-file-input');

            if (!dropzone || !fileInput) return;

            dropzone.addEventListener('click', () => fileInput.click());

            dropzone.addEventListener('dragover', (e) => {
              e.preventDefault();
              dropzone.classList.add('dragover');
            });

            ['dragleave', 'dragend'].forEach((type) => {
              dropzone.addEventListener(type, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
              });
            });

            dropzone.addEventListener('drop', (e) => {
              e.preventDefault();
              dropzone.classList.remove('dragover');
              if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
              }
            });
          })();
        </script>
      </body>
    </html>
    """

@router.post("/detect", response_class=HTMLResponse)
async def detect_html(file: UploadFile = File(...)) -> str:
    """Simulated multi-object detection with drawn bounding boxes."""
    file_bytes = await file.read()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Load image
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    # Base single-label prediction (reuse classifier)
    base_label, base_conf, _ = classifier.predict(image)

    # Fake multiple detections using vision service helper
    detections = fake_multi_detections(image, base_label)

    # Draw boxes using vision service helper
    annotated = draw_detections(image, detections)

    # Encode original and annotated images for HTML
    def to_data_url(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    original_url = to_data_url(image)
    annotated_url = to_data_url(annotated)

    # Build rows for detections table
    rows_html = ""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rows_html += f"""
        <tr>
          <td>{det['label']}</td>
          <td>{det['confidence']:.2f}</td>
          <td>({x1}, {y1}) – ({x2}, {y2})</td>
        </tr>
        """

    return f"""
    <html>
      <head>
        <title>Multi-object Detection Result</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f4f7;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 40px;
          }}
          .container {{
            background: #ffffff;
            padding: 24px 28px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
            width: 900px;
          }}
          h1 {{
            margin-top: 0;
            font-size: 1.4rem;
          }}
          .layout {{
            display: flex;
            gap: 20px;
            margin-top: 16px;
          }}
          .column {{
            flex: 1;
          }}
          .img-block img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25);
          }}
          .img-label {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 4px;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.85rem;
          }}
          th, td {{
            border-bottom: 1px solid #e5e7eb;
            padding: 6px 4px;
            text-align: left;
          }}
          th {{
            font-weight: 600;
            color: #4b5563;
          }}
          .meta {{
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 8px;
          }}
          a.button {{
            display: inline-block;
            margin-top: 16px;
            padding: 8px 12px;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 600;
          }}
          a.button:hover {{
            background: #1d4ed8;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Multi-object Detection Result (Simulated)</h1>
          <div class="meta">
            Base prediction: <strong>{base_label}</strong> ({base_conf * 100:.1f}% confidence)
          </div>

          <div class="layout">
            <div class="column">
              <div class="img-label">Original uploaded image</div>
              <div class="img-block">
                <img src="{original_url}" alt="Original image" />
              </div>
            </div>
            <div class="column">
              <div class="img-label">Annotated image with simulated detections</div>
              <div class="img-block">
                <img src="{annotated_url}" alt="Annotated detections" />
              </div>
            </div>
          </div>

          <table>
            <thead>
              <tr>
                <th>Label</th>
                <th>Confidence</th>
                <th>Bounding Box (x1, y1 – x2, y2)</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>

          <a href="/detect" class="button">Run another detection</a>
        </div>
      </body>
    </html>
    """

@router.post("/api/predict")
async def predict_api(file: UploadFile = File(...)) -> dict:
    """Raw JSON API endpoint for single-product prediction."""
    file_bytes = await file.read()

    try:
        image, label, confidence_value = process_bytes_for_single_product(
            file_bytes, file.content_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    product_info: ProductLocation | None = db.lookup(label)

    if product_info is None:
        result = {
            "predicted_label": label,
            "confidence": confidence_value,
            "found_in_database": False,
            "message": f"No location info found for '{label}'.",
            "location": None,
            "meta": None,
        }
    else:
        result = {
            "predicted_label": label,
            "confidence": confidence_value,
            "found_in_database": True,
            "location": {
                "aisle": product_info.aisle,
                "section": product_info.section,
            },
            "meta": {
                "category": product_info.category,
                "price": product_info.price,
                "in_stock": product_info.in_stock,
            },
        }

    _log_request(result)
    return result

@router.post("/api/detect")
async def detect_api(file: UploadFile = File(...)) -> dict:
    """Raw JSON API endpoint for simulated multi-object detection."""
    file_bytes = await file.read()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    base_label, base_conf, _ = classifier.predict(image)
    detections = fake_multi_detections(image, base_label)

    return {
        "base_prediction": {
            "label": base_label,
            "confidence": base_conf,
        },
        "detections": detections,
    }

