from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- CROP MAPPING FOR AMBIGUOUS FOLDERS ---
# This ensures that folders like "Brown Spot" are displayed as "Rice Brown Spot"
FOLDER_TO_CROP = {
    "Bacterial Leaf Blight": "Rice",
    "Brown Spot": "Rice",
    "Leaf Blast": "Rice",
    "Leaf scald": "Rice",
    "Sheath Blight": "Rice",
    "Healthy Rice Leaf": "Rice",
    "Blight": "Maize",
    "Common_Rust": "Maize",
    "Gray_Leaf_Spot": "Maize",
    "Bacterial Blight": "Cotton",
    "Curl Virus": "Cotton",
    "Herbicide Growth Damage": "Cotton",
    "Leaf Hopper Jassids": "Cotton",
    "Leaf Redding": "Cotton",
    "Leaf Variegation": "Cotton"
}

# =====================================================================
# MASTER CIBRC KNOWLEDGE BASE (Mapped to your EXACT 32 Folders)
# =====================================================================
MASTER_REMEDY_LIST = {
    # --- RICE ---
    "Bacterial Leaf Blight": {"chemical": "Streptocycline + Copper Oxychloride", "dosage": "100g + 2.5kg", "phi": "Check Label", "application": "Apply at boot leaf stage or as soon as streaks appear."},
    "Brown Spot": {"chemical": "Propiconazole 25% EC", "dosage": "500 ml", "phi": "30 Days", "application": "Balanced fertilizer application helps reduce severity."},
    "Leaf Blast": {"chemical": "Tricyclazole 75% WP", "dosage": "300 - 400 g", "phi": "30 Days", "application": "Apply at neck-break stage for maximum protection."},
    "Leaf scald": {"chemical": "Carbendazim 50% WP", "dosage": "1.0 kg", "phi": "15 Days", "application": "Apply at early appearance of leaf scald symptoms."},
    "Sheath Blight": {"chemical": "Hexaconazole 5% SC", "dosage": "1000 ml", "phi": "30 Days", "application": "Direct spray towards the base of the plant."},

    # --- MAIZE ---
    "Blight": {"chemical": "Mancozeb 75% WP", "dosage": "1.5 - 2.0 kg", "phi": "25 Days", "application": "Apply when first symptoms appear on lower leaves."},
    "Common_Rust": {"chemical": "Azoxystrobin 18.2% + Cyproconazole 7.3% SC", "dosage": "500 ml", "phi": "21 Days", "application": "Early stage application is critical for rust control."},
    "Gray_Leaf_Spot": {"chemical": "Pyraclostrobin 20% WG", "dosage": "500 g", "phi": "20 Days", "application": "Monitor lower leaves during high humidity periods."},

    # --- COTTON ---
    "Bacterial Blight": {"chemical": "Streptomycin Sulphate + Tetracycline Hydrochloride", "dosage": "100 g", "phi": "Check Label", "application": "Apply as soon as water-soaked spots appear."},
    "Curl Virus": {"chemical": "Afidopyropen 50 g/L DC", "dosage": "1000 ml", "phi": "15 Days", "application": "Targets whitefly vectors that transmit the virus."},
    "Herbicide Growth Damage": {"chemical": "Activated Charcoal / Flush with Water", "dosage": "N/A", "phi": "N/A", "application": "Foliar damage. Flush with clean water."},
    "Leaf Hopper Jassids": {"chemical": "Flonicamid 50% WG", "dosage": "150 - 200 g", "phi": "15 Days", "application": "Spray when hopper population reaches economic threshold."},
    "Leaf Redding": {"chemical": "Magnesium Sulphate (1%) + Urea (1%)", "dosage": "5 kg + 5 kg", "phi": "N/A", "application": "Corrective foliar spray for physiological deficiency."},
    "Leaf Variegation": {"chemical": "None Required (Genetic/Environmental)", "dosage": "N/A", "phi": "N/A", "application": "Non-pathogenic; no chemical spray needed."},

    # --- PEPPER ---
    "Pepper_bell__Bacterial_spot": {"chemical": "Copper Oxychloride 50% WP", "dosage": "2.5 kg", "phi": "Check Label", "application": "Spray early in the morning or late evening."},

    # --- POTATO ---
    "Potato__Early_blight": {"chemical": "Mancozeb 75% WP", "dosage": "1.5 - 2.0 kg", "phi": "26 Days", "application": "Repeat spray at 10-14 day intervals."},
    "Potato__Late_blight": {"chemical": "Cymoxanil 8% + Mancozeb 64% WP", "dosage": "1.5 kg", "phi": "10 Days", "application": "Crucial during cool, foggy, and moist weather."},

    # --- TOMATO ---
    "Tomato__Target_Spot": {"chemical": "Boscalid 25.2% + Pyraclostrobin 12.8% WG", "dosage": "500 - 600 g", "phi": "3 Days", "application": "Provide maximum coverage of the inner canopy."},
    "Tomato__Tomato_mosaic_virus": {"chemical": "Imidacloprid 17.8% SL", "dosage": "250 ml", "phi": "15 Days", "application": "Control aphids to prevent secondary viral spread."},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"chemical": "Thiamethoxam 25% WG", "dosage": "200 g", "phi": "20 Days", "application": "Manage whitefly to prevent virus transmission."},
    "Tomato_Bacterial_spot": {"chemical": "Copper Oxychloride 50% WP", "dosage": "2.5 kg", "phi": "Check Label", "application": "Avoid overhead irrigation during infestation."},
    "Tomato_Early_blight": {"chemical": "Mancozeb 75% WP", "dosage": "1.5 - 2.0 kg", "phi": "26 Days", "application": "Start spray as soon as brown spots appear."},
    "Tomato_Late_blight": {"chemical": "Cymoxanil 8% + Mancozeb 64% WP", "dosage": "1.5 kg", "phi": "10 Days", "application": "Highly destructive in high humidity."},
    "Tomato_Leaf_Mold": {"chemical": "Difenoconazole 25% EC", "dosage": "500 ml", "phi": "5 Days", "application": "Focus spray on the undersides of leaves."},
    "Tomato_Septoria_leaf_spot": {"chemical": "Chlorothalonil 75% WP", "dosage": "1.0 - 1.25 kg", "phi": "14 Days", "application": "Protect young plants from ground-splashed spores."},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"chemical": "Abamectin 1.9% EC", "dosage": "750 - 1000 ml", "phi": "3 Days", "application": "Ensure thorough wetting of leaves."}
}

# 1. LOAD MODEL
try:
    model = tf.keras.models.load_model("agridetect_model.h5")
    with open("class_indices.json", "r") as f:
        labels = json.load(f)
    print("✅ Model & Exact 32-Class Mapping Loaded Successfully!")
except Exception as e:
    print("⏳ Model setup error:", e)
    model, labels = None, {}

@app.post("/api/analyze-crop")
async def analyze_crop(file: UploadFile = File(...)):
    if not model: return {"error": "Model not loaded."}
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        folder_name = labels.get(str(class_idx), "Unknown")

        # --- SMART NAME FORMATTING ---
        # 1. Determine the display name (Always include Crop + Disease)
        display_name = folder_name.replace("__", " ").replace("_", " ")
        
        # If the folder doesn't already have the crop name, prepend it from our dictionary
        if folder_name in FOLDER_TO_CROP and FOLDER_TO_CROP[folder_name] not in display_name:
            display_name = f"{FOLDER_TO_CROP[folder_name]} {display_name}"
            
        display_name = display_name.title()

        # 2. Confidence Guard (UPGRADED TO 70% FOR BETTER ACCURACY)
        if confidence < 0.70:
            return {
                "pest_name": "Unclear Image",
                "confidence": round(confidence * 100, 1),
                "pesticide_recommendation": "The AI is not highly confident (Below 70%). To ensure a safe and accurate pesticide recommendation, please upload a clearer, well-lit photo of a single leaf."
            }

        # 3. Healthy Classes
        if "healthy" in folder_name.lower() or folder_name == "Healthy":
            return {
                "pest_name": display_name,
                "confidence": round(confidence * 100, 1),
                "pesticide_recommendation": "Plant looks healthy! No chemical treatment required. Maintain standard nutrition."
            }

        # 4. Direct Knowledge Base Lookup
        remedy_data = MASTER_REMEDY_LIST.get(folder_name)

        if remedy_data:
            remedy_text = (
                f"Official CIBRC Remedy: {remedy_data['chemical']}\n"
                f"APPLICATION: Mix {remedy_data['dosage']} in 500-750 Liters of water per hectare.\n"
                f"NOTES: {remedy_data['application']}\n"
                f"SAFE HARVEST (PHI): {remedy_data['phi']}."
            )
        else:
            remedy_text = f"Condition identified as {display_name}. Please consult local CIBRC registry for formulation details."

        return {
            "pest_name": display_name,
            "confidence": round(confidence * 100, 1),
            "pesticide_recommendation": remedy_text
        }

    except Exception as e:
        return {"error": str(e)}