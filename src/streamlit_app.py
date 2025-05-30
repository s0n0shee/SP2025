import streamlit as st
from PIL import Image
import numpy as np
import io
import tempfile
import os
import matplotlib.pyplot as plt
from ensemble_backend import ensemble_predict_with_densenet_gradcam, IMG_SIZE, CLASS_NAMES, MODEL_CONFIGS
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="Explainable Multi-classification of Retinal Diseases Using Ensembled Transfer Learning Models and Grad-CAM",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🩺"
)

# --- Class name formatting and descriptions ---
CLASS_DISPLAY_NAMES = {
    "age_degeneration": "Age-Related Macular Degeneration",
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
    "others": "Others"
}

# For table display: abbreviate AMD and DR
CLASS_TABLE_DISPLAY_NAMES = {
    "age_degeneration": "AMD",
    "cataract": "Cataract",
    "diabetic_retinopathy": "DR",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
    "others": "Others"
}

# --- Citations and mapping for clickable references ---
CITATION_LIST = [
    ("NHS, 'Diabetic retinopathy', nhs.uk, 2023, December 4. [Online]. Available: https://www.nhs.uk/conditions/diabetic-retinopathy/", "https://www.nhs.uk/conditions/diabetic-retinopathy/"),
    ("M. D. Abràmoff, M. K. Garvin, and M. Sonka, 'Retinal imaging and image analysis', IEEE Reviews in Biomedical Engineering, vol. 3, pp. 169–208, 2010.", None),
    ("National Eye Institute, 'Age-Related Macular Degeneration (AMD) | National Eye Institute', 2024, October 2. [Online]. Available: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration", "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration"),
    ("A. A. E. F. Elsharif and S. S. Abu-Naser, 'Retina diseases diagnosis using deep learning', 2022.", None),
    ("A. Jackson, 'Glaucoma Facts and Stats - Glaucoma Research Foundation', Glaucoma Research Foundation, 2024, February 20. [Online]. Available: https://glaucoma.org/articles/glaucoma-facts-and-stats", "https://glaucoma.org/articles/glaucoma-facts-and-stats"),
    ("National Eye Institute, 'Glaucoma | National Eye Institute', 2024a, September 6. [Online]. Available: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/glaucoma", "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/glaucoma"),
    ("A. K. Schuster, C. Erb, E. M. Hoffmann, T. Dietlein, and N. Pfeiffer, 'The diagnosis and treatment of glaucoma', Deutsches Ärzteblatt International, vol. 117, no. 13, p. 225, 2020.", None),
    ("National Eye Institute, 'Cataracts | National Eye Institute', 2024b, September 18. [Online]. Available: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/cataracts", "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/cataracts"),
    ("L. M. Kankanala, G. Jayashree, R. Balakrishnan, and A. Bhargava, 'Automated cataract grading using slit-lamp images with machine learning', J. Ophthalmol, 2021.", None)
]

# Map class to citation indices (1-based)
CLASS_CITATION_INDICES = {
    "age_degeneration": [3, 4],
    "cataract": [8, 9],
    "diabetic_retinopathy": [1, 2],
    "glaucoma": [5, 6, 7],
    "normal": [],
    "others": []
}

CLASS_DESCRIPTIONS = {
    "age_degeneration": "A progressive eye disease that damages the macula, the part of the retina responsible for central vision. Caused by aging, it appears in two forms: dry (with drusen and pigment changes) and wet (with abnormal blood vessel growth and hemorrhages). Leads to central vision loss.",
    "cataract": "A condition where the eye’s lens becomes cloudy, resulting in blurry or dim vision. Common with aging, but also linked to UV exposure and smoking. In fundus images, it appears as a hazy or unclear view of the retina.",
    "diabetic_retinopathy": "A diabetes-related disease that damages retinal blood vessels due to high blood sugar. It causes hemorrhages, microaneurysms, exudates, and fluid leakage. May result in permanent vision loss if untreated.",
    "glaucoma": "An optic nerve disease caused by increased intraocular pressure or fluid buildup. Leads to optic nerve damage, cupping of the optic disc, and peripheral vision loss. Fundus images often show an enlarged cup-to-disc ratio and nerve fiber thinning.",
    "normal": "Healthy eyes with no retinal or optic nerve abnormalities. The optic disc, blood vessels, and macula appear clear and normal in fundus images.",
    "others": "Miscellaneous ocular diseases not categorized under major classes. May include rare or less common conditions affecting the retina or optic nerve."
}

CITATION_TEXT = '''
[18] NHS, "Diabetic retinopathy," nhs.uk, 2023, December 4. [Online]. Available: https://www.nhs.uk/conditions/diabetic-retinopathy/
[19] M. D. Abràmoff, M. K. Garvin, and M. Sonka, "Retinal imaging and image analysis," IEEE Reviews in Biomedical Engineering, vol. 3, pp. 169–208, 2010.
[20] National Eye Institute, "Age-Related Macular Degeneration (AMD) | National Eye Institute," 2024, October 2. [Online]. Available: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration
[21] A. A. E. F. Elsharif and S. S. Abu-Naser, "Retina diseases diagnosis using deep learning," 2022.
[22] A. Jackson, "Glaucoma Facts and Stats - Glaucoma Research Foundation," Glaucoma Research Foundation, 2024, February 20. [Online]. Available: https://glaucoma.org/articles/glaucoma-facts-and-stats
[23] National Eye Institute, "Glaucoma | National Eye Institute," 2024a, September 6. [Online]. Available: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/glaucoma
[24] A. K. Schuster, C. Erb, E. M. Hoffmann, T. Dietlein, and N. Pfeiffer, "The diagnosis and treatment of glaucoma," Deutsches Ärzteblatt International, vol. 117, no. 13, p. 225, 2020.
[25] National Eye Institute, "Cataracts | National Eye Institute," 2024b, September 18. [Online]. Available: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/cataracts
[26] L. M. Kankanala, G. Jayashree, R. Balakrishnan, and A. Bhargava, "Automated cataract grading using slit-lamp images with machine learning," J. Ophthalmol, 2021.
[27] Kaggle, "Ocular disease recognition," 2020, September 24. [Online]. Available: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
'''

# --- PDF Generation Function ---
def generate_report_pdf(image_path, label, confidence, avg_probs, heatmap_path, model_preds):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    Story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    Story.append(Paragraph("Retinal Disease Classification Report", title_style))
    Story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    Story.append(Spacer(1, 20))
    
    # Images
    img = Image.open(image_path)
    img_path = "temp_original.png"
    img.save(img_path)
    img_original = RLImage(img_path, width=3*inch, height=3*inch)
    Story.append(Paragraph("Original Image:", styles["Heading2"]))
    Story.append(img_original)
    Story.append(Spacer(1, 20))
    
    # Add heatmap
    img_heatmap = RLImage(heatmap_path, width=3*inch, height=3*inch)
    Story.append(Paragraph("Grad-CAM Heatmap:", styles["Heading2"]))
    Story.append(img_heatmap)
    Story.append(Spacer(1, 20))
    
    # Classification Results
    Story.append(Paragraph("Classification Results", styles["Heading2"]))
    Story.append(Paragraph(f"<b>Predicted Class:</b> {CLASS_DISPLAY_NAMES[label.lower()]}", styles["Normal"]))
    Story.append(Paragraph(f"<b>Confidence Score:</b> {confidence:.3f}", styles["Normal"]))
    Story.append(Spacer(1, 20))
    
    # Description
    Story.append(Paragraph("Disease Description", styles["Heading2"]))
    Story.append(Paragraph(CLASS_DESCRIPTIONS[label.lower()], styles["Normal"]))
    Story.append(Spacer(1, 20))
    
    # Probability Table
    Story.append(Paragraph("Model & Ensemble Class Probabilities", styles["Heading2"]))
    # Use abbreviations for table
    class_names = [CLASS_TABLE_DISPLAY_NAMES[c] for c in CLASS_NAMES]
    table_data = [["Model"] + class_names]
    
    # Add individual model predictions
    for i, pred in enumerate(model_preds):
        row = [MODEL_CONFIGS[i]['name']] + [f"{pred[0][j]:.4f}" for j in range(len(CLASS_NAMES))]
        table_data.append(row)
    
    # Add ensemble average
    avg_row = ["Ensemble Average"] + [f"{avg_probs[j]:.4f}" for j in range(len(CLASS_NAMES))]
    table_data.append(avg_row)    # Create table with custom column widths
    # Calculate column widths as percentage of page width
    page_width = letter[0] - doc.leftMargin - doc.rightMargin
    # Make all columns equal width except the model name column
    col_width = page_width * 0.13  # Same width for all disease columns
    col_widths = [page_width * 0.22] + [col_width] * len(class_names)
    
    # Helper function to wrap text
    def wrap_text(text, width=15):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        return '\n'.join(lines)
    
    # Wrap headers and all cell contents
    for row_idx in range(len(table_data)):
        for col_idx in range(len(table_data[row_idx])):
            cell_content = str(table_data[row_idx][col_idx])
            if row_idx == 0:  # Headers
                table_data[row_idx][col_idx] = wrap_text(cell_content, width=20)
            else:  # Data cells - format numbers
                if col_idx > 0:  # Skip model name column
                    table_data[row_idx][col_idx] = f"{float(cell_content):.4f}"
                else:
                    table_data[row_idx][col_idx] = wrap_text(cell_content, width=20)
    
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),  # Smaller font for entire table
        ('LEADING', (0, 0), (-1, -1), 10),  # Line spacing for the entire table
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Vertical alignment# Slightly smaller font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),  # Reduce cell padding
        ('RIGHTPADDING', (0, 0), (-1, -1), 4)
    ])
    Story.append(table)
    
    # Build PDF
    doc.build(Story)
    
    # Clean up temporary files
    os.remove(img_path)
    
    buffer.seek(0)
    return buffer

# --- Tabs ---
tabs = st.tabs(["Main", "About"])

with tabs[0]:
    st.markdown("""    <style>
    /* Global styles */
    .stApp { 
        background-color: #f8f9fa !important; 
        color: #1a1a1a !important;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* Layout and spacing */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
        gap: 1.5rem !important;
    }
    
    [data-testid="stHorizontalBlock"] > div {
        flex: 1;
        min-width: 0;
    }

    /* Header and title */
    .stMarkdown h1 {
        font-size: 1.05rem !important;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1.2rem;
        line-height: 1.15 !important;
    }

    /* Navigation Tabs */
    .stTabs {
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        padding: 0;
        display: flex;
        justify-content: center;
        gap: 2rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        min-width: 100px;
        text-align: center;
        font-size: 1rem;
        color: #666666;
        font-weight: 500;
        padding: 0.75rem 0;
        transition: all 0.2s ease;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1976d2 !important;
        font-weight: 500;
        border-bottom: 2px solid #1976d2;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        border-bottom: 2px solid #e0e0e0;
    }

    /* Main container and layout */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
        margin-bottom: 0rem;
    }

    /* File uploader */
    .stUploadedFile {
        background: white;
        border-radius: 8px;
        padding: 1.25rem;
        border: 2px dashed #e0e0e0;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }
    .stUploadedFile:hover {
        border-color: #1976D2;
    }

    /* Image containers */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .image-container img {
        width: 100%;
        height: 300px;
        object-fit: contain;
        border-radius: 4px;
    }
    .image-container .caption {
        margin-top: 0.75rem;
        color: #666;
        font-size: 0.9rem;
        text-align: center;
    }

    /* Disease card */
    .disease-card {
        background: white;
        border-radius: 8px;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
    }
    .disease-card .disease-title {
        color: #1976D2;
        font-size: 1.5rem !important;
        font-weight: 600;
        line-height: 1.2;
    }
    .disease-card h4 {
        color: #666666;
        font-weight: 500;
        font-size: 1rem;
        margin-bottom: -0.3rem;
    }

    /* Confidence score */
    .confidence-score {
        font-size: 1.1rem;
        color: #1a1a1a;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Description section */
    .description-section {
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #333;
    }

    /* Probability Table */
    .table-section {
        margin-top: 1.5rem;
    }
    .table-section h4 {
        color: #1a1a1a;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .stDataFrame {
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .dataframe {
        color: #1a1a1a;
        font-size: 0.9rem;
        width: 100%;
        margin: 0 !important;
    }
    .dataframe thead th {
        background-color: #f5f7fa !important;
        color: #1a1a1a !important;
        font-weight: 600;
        padding: 0.75rem 1rem !important;
        border-bottom: 2px solid #e0e0e0;
        text-align: left;
    }
    .dataframe tbody td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid #e0e0e0;
    }
    .dataframe tr:last-child td {
        background-color: #f5f7fa;
        font-weight: 600;
        color: #1976D2;
    }

    /* Citations and references */
    .desc-cite {
        font-size: 0.9rem;
        color: #1976D2;
        margin-left: 0.3rem;
        font-weight: 500;
    }
    .desc-cite a {
        color: #1976D2 !important;
        text-decoration: none;
        border-bottom: 1px dotted #1976D2;
        transition: all 0.2s ease;
    }
    .desc-cite a:hover {
        background-color: rgba(25,118,210,0.1);
    }
    .references {
        font-size: 0.9rem;
        color: #666666;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
        line-height: 1.6;
    }
    .references a {
        color: #1976D2 !important;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .references a:hover {
        text-decoration: underline;
    }

    /* Buttons */
    .stButton button {
        background-color: #1976D2;
        color: white !important;
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        width: auto;
        margin: 0 auto;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stButton button:hover {
        background-color: #1565C0;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .stMarkdown h1 {
        font-size: 1.05rem !important;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1.2rem;
        line-height: 1.15 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Explainable Multi-classification of Retinal Diseases Using Ensembled Transfer Learning Models and Grad-CAM")
    st.markdown("Welcome! This app predicts the classification of **retinal fundus images** to six disease categories, namely: Age-Related Macular Degeneration, Cataracts, Diabetic Retinopathy, Glaucoma, Others (Unique), and Normal. To start, **upload an image below.**")   
    uploaded_file = st.file_uploader(
        "Choose an image file" if "file_uploader_label" not in st.session_state else st.session_state.file_uploader_label,
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    # Apply black color to file uploader label
    st.markdown("""
        <style>
        .stMarkdown + div [data-baseweb="file-uploader"] > div > div {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            with st.spinner("Classifying and generating Grad-CAM heatmap..."):
                label, confidence, avg_probs, overlay, model_preds = ensemble_predict_with_densenet_gradcam(temp_path)

            # --- Display layout ---
            col1, col2, col3 = st.columns([1, 1, 1.1], gap="large")
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)          
            with col3:
                desc = CLASS_DESCRIPTIONS[label.lower()]
                cite_indices = CLASS_CITATION_INDICES[label.lower()]
                st.markdown(f"""
                    <div class='disease-card'>
                        <div class='disease-title'>{CLASS_DISPLAY_NAMES[label.lower()]}</div>
                        <h4>Class Prediction</h4>
                        <div class='confidence-score'>
                            Confidence: <b>{confidence:.3f}</b>
                        </div>                    <div class='description-section'>
                            <b>Description</b><br>
                            <p style="text-align: justify; margin: 0.5rem 0;">{desc}{' '.join([f"<a href='#ref{idx}' class='desc-cite' id='desc-cite-{idx}'>[{idx}]</a>" for idx in cite_indices]) if cite_indices else ''}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            # --- Probability Table ---
            st.markdown("<div class='table-section'>", unsafe_allow_html=True)
            st.markdown("<h4>Model & Ensemble Class Probabilities</h4>", unsafe_allow_html=True)
            class_names = [CLASS_TABLE_DISPLAY_NAMES[c] for c in CLASS_NAMES]
            table_data = []
            for i, pred in enumerate(model_preds):
                row = [float(f"{pred[0][j]:.4f}") for j in range(len(CLASS_NAMES))]
                table_data.append(row)
            avg_row = [float(f"{avg_probs[j]:.4f}") for j in range(len(CLASS_NAMES))]
            model_names = [m['name'] for m in MODEL_CONFIGS]
            df = pd.DataFrame(table_data, columns=class_names, index=model_names)
            df.loc['Ensemble Avg'] = avg_row
            df = df.rename_axis('Model').reset_index()
            def highlight_ensemble(s):
                return ['font-weight: bold' if v == 'Ensemble Avg' else '' for v in s]        # Add CSS for table column width and text wrapping
            st.markdown("""            <style>            .stDataFrame [data-testid=\"stDataFrameDataCell\"] div,
            .stDataFrame [data-testid=\"stDataFrameHeaderCell\"] div {
                text-align: center !important;
                justify-content: center !important;
                width: 100px !important;
                min-width: 100px !important;
                max-width: 100px !important;
                white-space: pre-line !important;
                overflow-wrap: break-word !important;
                word-break: break-word !important;
                padding: 8px 8px !important;
                font-size: 0.9rem !important;
                line-height: 1.3 !important;
                height: auto !important;
                display: flex !important;
                align-items: center !important;
            }
            .stDataFrame [data-testid=\"stDataFrameDataCell\"]:first-child div,
            .stDataFrame [data-testid=\"stDataFrameHeaderCell\"]:first-child div {
                width: 140px !important;
                min-width: 140px !important;
                max-width: 140px !important;
                text-align: left !important;
                justify-content: flex-start !important;
                white-space: pre-line !important;
                word-break: break-word !important;
            }            .stDataFrame thead tr th {
                background-color: #f5f7fa !important;
                font-weight: 600 !important;
            }
            /* Style for the ensemble average row */
            .stDataFrame tbody tr:last-child {
                background-color: #f5f7fa !important;
                font-weight: bold !important;
            }
            .stDataFrame tbody tr:last-child td div {
                color: #1976D2 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] button {
            background-color: #2196F3;
            color: white;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: -40px;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #1976D2;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Save heatmap for PDF
        heatmap_path = "temp_heatmap.png"
        overlay.save(heatmap_path)
          # Generate PDF and add download button
        pdf_buffer = generate_report_pdf(temp_path, label, confidence, avg_probs, heatmap_path, model_preds)
        st.markdown("<div style='text-align: center; margin: 0;'>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Download Classification Report (PDF)",
            data=pdf_buffer,
            file_name=f"retinal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="pdf_download"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

        # --- Citations at the bottom ---
        st.markdown("<div class='references'><b>Description References</b><br>" + ''.join([
            f"<div id='ref{idx+1}'><b>[{idx+1}]</b> " +
            (f"<a href='{url}' target='_blank'>{text}</a>" if url else text) + "</div>"
            for idx, (text, url) in enumerate(CITATION_LIST)
        ]) + "</div>", unsafe_allow_html=True)
        
        # Clean up temporary files
        os.remove(temp_path)
        os.remove(heatmap_path)

with tabs[1]:
    st.markdown("""
    <div style='max-width: 900px; margin: 0 auto;'>
    <div class='disease-card' style='background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2rem 2.5rem 2rem 2.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
    <h2 style='color: #1976D2; font-size: 1.15rem; font-weight: 600; line-height: 1.25;'>Project Description</h2>
    <div class='description-section' style='font-size: 1.05rem; text-align: justify; line-height: 1.7;'>
    This project was developed by Roshan Bacud, a Computer Science student at the University of the Philippines Manila as part of her undergraduate thesis titled: <b>Explainable Multi-classification of Retinal Diseases Using Ensembled Transfer Learning Models and Grad-CAM</b>.<br><br>
    In the study, eight deep learning models were evaluated, from which the three performing were ensembled to build a Clinical Decision Support System (CDSS) for retinal disease diagnosis. These models are: EfficientNetB5, EfficientNetB5 (Optimized), and DenseNet201 (Optimized).<br><br>
    This CDSS predicts user-submitted retinal fundus images to the following categories: Age-Related Macular Degeneration, Cataracts, Diabetic Retinopathy, Glaucoma, Others (Unique), and Normal. This then visualizes model attention using Grad-CAM heatmaps to explain the classification, along with probability scores.<br><br>
    <b>Disclaimer: This tool is not a substitute for professional medical diagnosis.</b><br></br>
    </div>
    <h2 style='color: #1976D2; font-size: 1.15rem; font-weight: 600; line-height: 1.25;'>Disease Categories</h2>
    <div class='description-section' style='font-size: 1.05rem; text-align: justify; line-height: 1.7;'>
    <b>Diabetic retinopathy (DR)</b> is a complication of diabetes where high blood sugar levels damage the retina <a href='#references' class='desc-cite'>[1]</a>. It encompasses complications in the retinal vessel walls derived from blood vessel and nerve cell damage from hyperglycemia, where blood glucose is elevated. Specifically, two conditions emerge from such damage: ischemia and diabetic macular edema. Ischemia refers to the emergence of new blood vessels that are likely to rupture due to weakness, causing damage to vision or even permanent loss of sight, leading to proliferative diabetic retinopathy <a href='#references' class='desc-cite'>[2]</a>. Diabetic macular edema, on the other hand, causes the blood-retinal barrier to break down, causing fluid leakage into the macula and possibly affecting central vision. Signs of this complication include hemorrhages, exudates, and microaneurysms. In fundus photography, DR is commonly detected through these features—which manifest as red or pink spots, white flaky areas (cotton wool spots), and narrow crooked blood vessels. Lack of treatment may lead to visual impairment or even blindness.<br><br>
    <b>Age-related Macular Degeneration (AMD)</b> is a disease attributed to aging, primarily affecting central vision. This is where the macula, a part of the retina that controls straight-ahead and sharp vision, is damaged <a href='#references' class='desc-cite'>[3]</a>. In AMD, there is an excessive growth of abnormal blood vessels in the eye that eventually leads to loss of vision due to detachment of the retinal pigmented epithelium. Its major categories are dry (non-neovascular, exhibiting drusen) and wet (choroidal neovascularization, exhibiting hemorrhages) AMD, where vision loss is usually gradual with the former and more sight-threatening with the latter <a href='#references' class='desc-cite'>[4]</a>. Additionally, AMD progression starts with the dry form and may advance to the wet form at later stages <a href='#references' class='desc-cite'>[4]</a>. AMD is commonly detected in fundus photography through drusen, geographic atrophy, hemorrhages, and pigment clumping—manifesting as yellow patches, pigment changes, dark red deposits, and small pale spots.<br><br>
    <b>Glaucoma</b> is a disease primarily classified as a neuropathy, not a retinopathy, caused by the destruction of the optic nerve. When the eye fluid (aqueous humor) does not circulate properly in the front part of the eye, it damages the axons and ganglion cells of the retina <a href='#references' class='desc-cite'>[5]</a>. It starts asymptomatically and progresses as impairment in peripheral vision. According to the Glaucoma Research Foundation, it is the second leading cause of blindness, and all age ranges are at risk <a href='#references' class='desc-cite'>[6]</a>. Without treatment, it can eventually cause blindness; however, with early treatment, the damage can be halted <a href='#references' class='desc-cite'>[7]</a>. Optic disc cupping, or the enlargement or deepening of the cup (the central depression in the optic disc), is a hallmark of this condition. This cup and disc are visible in fundus photography. Images with this disease primarily show a larger cup-to-disc ratio, signifying an enlarged cup. Other notable features that indicate glaucoma include degeneration of the optic nerve head <a href='#references' class='desc-cite'>[8]</a>, bleeding, thinning of nerve fibers, and blocked veins.<br><br>
    <b>Cataract</b> is characterized by clouding of the lens in the eye and is more prevalent with aging. It makes vision blurry, hazy, and less colorful, and over time may lead to vision loss <a href='#references' class='desc-cite'>[9]</a>. It is currently the leading cause of visual impairment and blindness worldwide. Other factors that contribute to cataracts are smoking and exposure to ultraviolet light <a href='#references' class='desc-cite'>[10]</a>. Typically, cataracts are clinically diagnosed using tonometry, visual acuity testing, and a dilated eye exam, and are classified based on the location and appearance of clouding within the lens <a href='#references' class='desc-cite'>[5]</a>. However, in fundus photography, it is observed as a cloudy or hazy capture with reduced clarity.
    </div>
    </div>
    <div id='references' class='references'><b>References</b><br>
    <div id='ref1'><b>[1]</b> <a href='https://www.nhs.uk/conditions/diabetic-retinopathy/' target='_blank'>NHS, “Diabetic retinopathy,” nhs.uk, Dec. 4, 2023.</a></div>
    <div id='ref2'><b>[2]</b> M. D. Abràmoff, M. K. Garvin, and M. Sonka, “Retinal imaging and image analysis,” <i>IEEE Reviews in Biomedical Engineering</i>, vol. 3, pp. 169–208, 2010.</div>
    <div id='ref3'><b>[3]</b> <a href='https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration' target='_blank'>National Eye Institute, “Age-Related Macular Degeneration (AMD) | National Eye Institute,” Oct. 2, 2024.</a></div>
    <div id='ref4'><b>[4]</b> A. A. E. F. Elsharif and S. S. Abu-Naser, “Retina diseases diagnosis using deep learning,” 2022.</div>
    <div id='ref5'><b>[5]</b> <a href='https://doi.org/10.3390/jimaging9040084' target='_blank'>S. Muchuchuti and S. Viriri, “Retinal disease detection using deep learning techniques: A comprehensive review,” <i>Journal of Imaging</i>, vol. 9, no. 4, p. 84, 2023.</a></div>
    <div id='ref6'><b>[6]</b> <a href='https://glaucoma.org/articles/glaucoma-facts-and-stats' target='_blank'>A. Jackson, “Glaucoma facts and stats - Glaucoma Research Foundation,” Glaucoma Research Foundation, Feb. 20, 2024.</a></div>
    <div id='ref7'><b>[7]</b> <a href='https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/glaucoma' target='_blank'>National Eye Institute, “Glaucoma | National Eye Institute,” Sep. 6, 2024.</a></div>
    <div id='ref8'><b>[8]</b> A. K. Schuster, C. Erb, E. M. Hoffmann, T. Dietlein, and N. Pfeiffer, “The diagnosis and treatment of glaucoma,” <i>Deutsches Ärzteblatt International</i>, vol. 117, no. 13, pp. 225–234, 2020.</div>
    <div id='ref9'><b>[9]</b> <a href='https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/cataracts' target='_blank'>National Eye Institute, “Cataracts | National Eye Institute,” Sep. 18, 2024.</a></div>
    <div id='ref10'><b>[10]</b> L. M. Kankanala, G. Jayashree, R. Balakrishnan, and A. Bhargava, “Automated cataract grading using slit-lamp images with machine learning,” <i>Journal of Ophthalmology</i>, vol. 2021, Article ID 6698029, 2021.</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
