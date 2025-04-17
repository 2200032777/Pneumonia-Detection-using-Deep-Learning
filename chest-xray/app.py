import os
import torch
import torch.nn as nn
from torchvision.models import densenet121
import torchvision.transforms as transforms
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory # Added send_from_directory
from PIL import Image
import io
from werkzeug.utils import secure_filename # To securely save filenames
import datetime # Make sure datetime is imported
import traceback # For detailed error logging

# --- Configuration ---
MODEL_PATH = "model/best_model.pth" # Ensure this path is correct relative to where app.py is run
NUM_CLASSES = 2
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
UPLOAD_FOLDER = 'static/uploads/' # Save uploads within static folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# IMPORTANT: Use a strong, environment-variable-based secret key in production!
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-default-super-secret-key-for-dev')

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Important!

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = densenet121(weights=None) # Start with no pretrained weights (or specify weights='DEFAULT' if you want ImageNet)
# Modify the classifier structure (must match the structure during training *before* loading state_dict)
try:
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    print("Model classifier structure modified.")
except AttributeError as e:
     print(f"❌ Error accessing model.classifier.in_features: {e}")
     print("Ensure the densenet121 model structure is as expected.")
     exit()


# Load the trained state dictionary
try:
    # Ensure the path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at specified path: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Model state_dict loaded successfully from {MODEL_PATH}")
except FileNotFoundError as e:
    print(f"❌ {e}")
    exit()
except RuntimeError as e:
     print(f"❌ Error loading state_dict: {e}")
     print("This often happens if the model architecture defined here doesn't")
     print("match the architecture used when the model was saved (e.g., classifier mismatch).")
     exit()
except Exception as e:
    print(f"❌ An unexpected error occurred loading the model: {e}")
    exit()

model.to(device)
model.eval() # Set model to evaluation mode (very important!)
print("Model moved to device and set to evaluation mode.")


# --- Image Transformations ---
# Should match the transformations used during validation/testing (excluding augmentations like Flip/Rotation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Ensure Grayscale matches training if needed. If trained on RGB, remove/comment out Grayscale.
    # If the saved model expects 3 channels AFTER grayscale, this is correct.
    # If it was trained on 1 channel grayscale -> 3 channel input layer, adjust accordingly.
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])
print("Image transforms defined.")

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload form."""
    current_year = datetime.date.today().year # Get current year
    # Render the main page, passing the current year for the footer
    return render_template('index.html', current_year=current_year)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload, prediction, and renders results."""
    current_year = datetime.date.today().year # Get current year for footer
    save_path = None # Initialize save_path to None
    prediction_result = None
    confidence_score_str = None
    # image_path_for_template = None # Only needed if displaying server-saved image separately

    # 1. Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part selected.')
        # Redirect back to index, passing the year
        return redirect(url_for('index', current_year=current_year))

    file = request.files['file']

    # 2. Check if the user selected a file
    if file.filename == '':
        flash('No file selected.')
        # Redirect back to index, passing the year
        return redirect(url_for('index', current_year=current_year))

    # 3. Check if the file is allowed and process it
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename) # Sanitize filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file temporarily (optional, only needed if displaying it from server)
            # You might comment this out if only using JS preview
            file.seek(0) # Ensure pointer is at the start before saving
            file.save(save_path)
            print(f"💾 Image temporarily saved to: {save_path}")

            # Prepare image for the model
            file.seek(0) # Reset stream pointer to read for processing
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB") # Ensure 3 channels (RGB)

            # Apply transformations
            image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim and send to device

            # Perform prediction
            with torch.no_grad(): # Disable gradient calculations for inference
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1) # Get probabilities
                confidence, predicted_idx = torch.max(probabilities, 1) # Get highest probability and index
                prediction_result = CLASS_NAMES[predicted_idx.item()] # Get class name
                confidence_score = confidence.item() * 100 # Convert confidence to percentage
                confidence_score_str = f"{confidence_score:.2f}%" # Format for display

            print(f"🧠 Prediction: {prediction_result}, Confidence: {confidence_score_str}")

            # Optional: Construct relative path if displaying server-saved image
            # image_path_for_template = os.path.join('uploads', filename).replace("\\","/")

        except FileNotFoundError as e:
             flash(f"Error: Model file not found. Please check server configuration. Details: {e}")
             print(f"❌ Prediction Error - File Not Found: {e}")
             traceback.print_exc()
              # Redirect back to index, passing the year
             return redirect(url_for('index', current_year=current_year))
        except Exception as e:
            flash(f"An error occurred during processing: {e}")
            print(f"❌ Prediction Error: {e}")
            traceback.print_exc() # Print detailed stack trace to console for debugging

            # Attempt to clean up the saved file if an error occurred after saving
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    print(f"🧹 Cleaned up file due to error: {save_path}")
                except OSError as cleanup_error:
                    print(f"⚠️ Error removing file {save_path} during error cleanup: {cleanup_error}")

            # Redirect back to index, passing the year
            return redirect(url_for('index', current_year=current_year))

        # 4. Render the template WITH results if successful
        return render_template('index.html',
                               prediction=prediction_result,
                               confidence=confidence_score_str,
                               current_year=current_year
                               # Optional: pass image_path if needed by template
                               # image_path=image_path_for_template
                              )

    # 5. Handle invalid file type
    else:
        flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}')
        # Redirect back to index, passing the year
        return redirect(url_for('index', current_year=current_year))


# Optional: Route to serve uploaded files directly - useful for debugging
# Flask automatically serves files from the 'static' folder anyway if path is correct in HTML
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serves uploaded files from the UPLOAD_FOLDER."""
    print(f"Serving file: {filename} from {app.config['UPLOAD_FOLDER']}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask application...")
    # Set host='0.0.0.0' to make the app accessible on your local network
    # debug=True is useful for development but should be False in production
    app.run(debug=True, host='0.0.0.0', port=5000)