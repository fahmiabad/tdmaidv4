# Import necessary libraries
import streamlit as st
from datetime import datetime, timedelta, time, date
import math
import pandas as pd
import os
import json
import fitz  # PyMuPDF for PDF processing
import numpy as np
import openai
from pathlib import Path
import faiss  # For vector search
import re
import tempfile

# --- 1. SET UP OPENAI API ---
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai.api_key:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.")

# --- 2. SET PAGE CONFIG ---
st.set_page_config(
    page_title="TDM-AID (Vancomycin) with RAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. MINIMAL CUSTOM CSS (Optional - for fine-tuning) ---
st.markdown("""
<style>
    /* Add slight padding to containers for visual separation */
    .stContainer {
        padding: 10px 5px; /* Adjust as needed */
    }
    /* Style metric labels for consistency */
    .stMetric > label {
        font-weight: bold;
        color: #555; /* Slightly darker label */
    }
    /* Ensure buttons have consistent width in columns */
    .stButton>button {
        width: 100%;
    }
    /* Add a subtle border to containers used as cards */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    /* Style for the main title */
    h1 {
        color: #4F6BF2; /* Primary color from original CSS */
        padding-top: 10px; /* Add padding to align title with logo */
    }
    /* Style for the subtitle */
    .subtitle {
        color: #666;
        font-size: 0.9em;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    /* Custom styling for recommendation dose */
    .recommendation-dose {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2A3C5D; /* Dark color from original CSS */
        margin-bottom: 5px;
    }
    .recommendation-description {
        font-size: 0.9rem;
        color: #666;
    }
    /* Align logo column vertically */
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"]:first-child {
       display: flex;
       align-items: center; /* Vertically center content in the first column */
    }
    /* Style for the citation references */
    .citation {
        background-color: #f0f7ff;
        border-left: 3px solid #4F6BF2;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    /* Style for the LLM reasoning section */
    .reasoning {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-top: 15px;
    }
    .reasoning-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #2A3C5D;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. RAG SYSTEM SETUP ---

# Class to handle document processing
class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.chunks = []
        self.chunk_size = 1000
        self.overlap = 200
        
    def extract_text_from_pdf(self):
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return None
    
    def chunk_text(self, text):
        if not text:
            return []
        
        # Split text into sections based on headers
        section_pattern = r'\n(?=[A-Z][A-Z\s]+(?:\n|:))'
        sections = re.split(section_pattern, text)
        
        chunks = []
        for section in sections:
            if not section.strip():
                continue
                
            # If section is already small enough, keep it as is
            if len(section) <= self.chunk_size:
                chunks.append(section.strip())
            else:
                # Split large sections into smaller chunks with overlap
                words = section.split()
                current_chunk = []
                current_size = 0
                
                for word in words:
                    current_chunk.append(word)
                    current_size += len(word) + 1  # +1 for space
                    
                    if current_size >= self.chunk_size:
                        chunks.append(" ".join(current_chunk).strip())
                        # Keep last few words for overlap
                        overlap_size = 0
                        overlap_chunk = []
                        for w in reversed(current_chunk):
                            if overlap_size + len(w) + 1 <= self.overlap:
                                overlap_chunk.insert(0, w)
                                overlap_size += len(w) + 1
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_size = overlap_size
                
                # Add the last chunk if not empty
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
        
        self.chunks = chunks
        return chunks

# Class to handle RAG operations
class RAGSystem:
    def __init__(self):
        self.embeddings = None
        self.chunks = None
        self.index = None
        self.is_initialized = False
        self.temp_dir = None
        
    def embed_chunks(self, chunks):
        if not chunks:
            return None
        
        embeddings = []
        try:
            for i in range(0, len(chunks), 20):  # Process in batches of 20 to avoid API limits
                batch = chunks[i:i+20]
                
                # Get embeddings for the batch using OpenAI API
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                # Extract the embedding data from the response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            self.embeddings = np.array(embeddings, dtype=np.float32)
            return self.embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None
    
    def build_index(self):
        if self.embeddings is None:
            return False
        
        try:
            # Initialize FAISS index
            dimension = self.embeddings.shape[1]  # Get dimension from first embedding
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            return True
        except Exception as e:
            st.error(f"Error building search index: {e}")
            return False
    
    def search(self, query, k=3):
        if not self.is_initialized or self.index is None or self.chunks is None:
            return []
        
        try:
            # Get embedding for the query
            query_response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = np.array([query_response.data[0].embedding], dtype=np.float32)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Return the relevant chunks
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.chunks):  # Ensure index is valid
                    results.append({
                        "text": self.chunks[idx],
                        "score": float(distances[0][i])
                    })
            
            return results
        except Exception as e:
            st.error(f"Error during search: {e}")
            return []
    
    def initialize_from_pdf(self, pdf_path):
        try:
            # Create temporary directory if it doesn't exist
            if not self.temp_dir:
                self.temp_dir = tempfile.TemporaryDirectory()
            
            # Process the PDF
            processor = DocumentProcessor(pdf_path)
            text = processor.extract_text_from_pdf()
            if text:
                self.chunks = processor.chunk_text(text)
                self.embeddings = self.embed_chunks(self.chunks)
                success = self.build_index()
                self.is_initialized = success
                return success
            return False
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            return False
    
    def initialize_from_saved(self, chunks_path, embeddings_path):
        try:
            # Load chunks and embeddings from saved files
            with open(chunks_path, 'r') as f:
                self.chunks = json.load(f)
            
            self.embeddings = np.load(embeddings_path)
            
            # Build the index
            success = self.build_index()
            self.is_initialized = success
            return success
        except Exception as e:
            st.error(f"Error loading saved RAG data: {e}")
            return False
    
    def save_data(self, chunks_path, embeddings_path):
        try:
            # Save chunks as JSON
            with open(chunks_path, 'w') as f:
                json.dump(self.chunks, f)
            
            # Save embeddings as numpy array
            np.save(embeddings_path, self.embeddings)
            
            return True
        except Exception as e:
            st.error(f"Error saving RAG data: {e}")
            return False
    
    def cleanup(self):
        if self.temp_dir:
            self.temp_dir.cleanup()

# LLM Reasoning component
class LLMReasoner:
    def __init__(self):
        self.model = "gpt-4o"  # Use GPT-4o for best reasoning capabilities
    
    def generate_reasoning(self, patient_data, calculation_results, relevant_guidelines):
        try:
            # Combine relevant guidelines into context
            guideline_context = "\n\n".join([item["text"] for item in relevant_guidelines])
            
            # Format patient data for prompt
            patient_info = f"""
Patient Information:
- Age: {patient_data.get('age', 'N/A')} years
- Weight: {patient_data.get('weight', 'N/A')} kg
- Sex: {patient_data.get('sex', 'N/A')}
- SCr: {patient_data.get('scr', 'N/A')} µmol/L
- CrCl: {patient_data.get('crcl', 'N/A')} mL/min
- Clinical notes: {patient_data.get('clinical_notes', 'N/A')}
            """
            
            # Format calculation results for prompt
            calc_info = ""
            if calculation_results:
                calc_info = "Calculation Results:\n"
                for key, value in calculation_results.items():
                    calc_info += f"- {key}: {value}\n"
            
            # Create prompt for the LLM
            prompt = f"""You are a clinical pharmacist specialist in antimicrobial stewardship and pharmacokinetics. 
You are analyzing a vancomycin dosing regimen for a patient. Please provide your expert reasoning on the proposed dosing regimen.

{patient_info}

{calc_info}

Based on the relevant clinical guidelines, provide your reasoning:

{guideline_context}

Please provide:
1. An assessment of the current dosing and levels
2. Your reasoning on whether the calculated dose is appropriate
3. Any factors that might necessitate dose adjustment beyond the basic calculations
4. Specific rationale tied to the guidelines
5. Any monitoring recommendations

Format your response as a concise clinical consultation note. Include specific citations to the guidelines when making recommendations.
"""
            
            # Call OpenAI API for reasoning
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a clinical pharmacist specialist providing expert consultation on vancomycin dosing."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            st.error(f"Error generating LLM reasoning: {e}")
            return f"Unable to generate reasoning due to an error: {str(e)}"

# --- 5. INITIALIZE THE RAG SYSTEM ---
@st.cache_resource
def get_rag_system():
    rag = RAGSystem()
    
    # Check if we have saved data
    base_path = Path(".")
    chunks_path = base_path / "vanco_chunks.json"
    embeddings_path = base_path / "vanco_embeddings.npy"
    
    if chunks_path.exists() and embeddings_path.exists():
        if rag.initialize_from_saved(chunks_path, embeddings_path):
            return rag
    
    # If no saved data or loading failed, process the PDF
    pdf_path = base_path / "Vanco.pdf"
    if pdf_path.exists():
        if rag.initialize_from_pdf(pdf_path):
            # Save the data for future use
            rag.save_data(chunks_path, embeddings_path)
            return rag
    
    # If PDF doesn't exist, show error
    st.error("Vancomycin guidelines PDF not found. Please make sure 'Vanco.pdf' is in the app directory.")
    return rag

# --- 6. HELPER FUNCTIONS ---

# Calculate the time difference in hours
def hours_diff(start, end):
    """Calculates the difference between two time objects in hours."""
    today = date.today() # Use date.today() instead of datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)
    # Handle overnight intervals or when end time is on the next day
    if dt_end <= dt_start:
        dt_end += timedelta(days=1)
    return (dt_end - dt_start).total_seconds() / 3600

# Format time difference for display
def format_hours_minutes(decimal_hours):
    """Formats decimal hours into a 'X hours Y minutes' string."""
    if decimal_hours is None or decimal_hours < 0:
        return "Invalid time"
    total_minutes = int(round(decimal_hours * 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours > 0 and minutes > 0:
        return f"{hours} hours {minutes} minutes"
    elif hours > 0:
        return f"{hours} hours"
    elif minutes >= 0: # Show 0 minutes if exactly 0 hours
        return f"{minutes} minutes"
    else:
        return "Invalid time"

# Calculate CrCl using Cockcroft-Gault
def calculate_crcl(age, wt, scr_umol, is_female):
    """Calculates Creatinine Clearance (CrCl) in mL/min."""
    if not all([age, wt, scr_umol]): # Basic validation
        return None
    try:
        scr_mgdl = scr_umol / 88.4
        if scr_mgdl <= 0: return None # Avoid division by zero or negative SCr

        crcl = ((140 - age) * wt) / (72 * scr_mgdl)
        if is_female:
            crcl *= 0.85
        return max(0, crcl) # Ensure CrCl is not negative
    except (ValueError, TypeError, ZeroDivisionError):
        return None

# Determine status vs target range
def check_target_status(value, target_range):
    """Checks if a value is within, below, or above the target range."""
    if value is None or target_range is None:
        return "N/A"
    lower, upper = target_range
    try:
        value = float(value)
        if lower is not None and value < lower:
            return "BELOW TARGET"
        # Handle open-ended upper range (e.g., >600)
        elif upper is not None and value > upper:
            return "ABOVE TARGET"
        elif lower is not None and upper is None and value < lower: # For targets like >600
             return "BELOW TARGET"
        elif lower is not None: # Covers >= lower when upper is None, or lower <= value <= upper
             return "WITHIN TARGET"
        else: # Should not happen with valid inputs
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"

# Display status with appropriate color/icon
def display_status(status_text):
    """Displays status text with Streamlit formatting."""
    if status_text == "WITHIN TARGET":
        st.success(f"✓ {status_text}")
    elif status_text == "BELOW TARGET":
        st.warning(f"⚠️ {status_text}")
    elif status_text == "ABOVE TARGET":
        st.error(f"⚠️ {status_text}") # Use error for above target
    else:
        st.info("Status N/A")

# Display simple progress bar like indicator
def display_level_indicator(label, value, target_range, unit):
    """Displays a value relative to its target range using st.metric and status text."""
    if value is None or target_range is None:
        st.metric(label=f"{label} ({unit})", value="N/A")
        return

    lower, upper = target_range
    status = check_target_status(value, target_range)
    value_str = f"{value:.1f}"

    delta_str = None
    delta_color = "off" # 'normal', 'inverse', 'off'

    if status == "WITHIN TARGET":
        delta_str = "Target Met"
        # No delta color change needed for within target
    elif status == "BELOW TARGET":
        delta_str = f"↓ {lower}"
        delta_color = "inverse" # Red for below target
    elif status == "ABOVE TARGET" and upper is not None:
        delta_str = f"↑ {upper}"
        delta_color = "inverse" # Red for above target
    elif status == "ABOVE TARGET" and upper is None: # Handle > target case
        delta_str = f"↑ {lower}" # Indicate it's above the minimum threshold
        delta_color = "normal"

    st.metric(label=f"{label} ({unit})", value=value_str, delta=delta_str, delta_color=delta_color)

# Function to formulate RAG query based on patient data and calculation results
def formulate_rag_query(patient_data, calculation_results, target_desc):
    """Creates a query for the RAG system based on patient parameters and calculation results."""
    query_parts = []
    
    # Add patient-specific information to query
    if patient_data.get('crcl'):
        if patient_data['crcl'] < 30:
            query_parts.append("vancomycin dosing severe renal impairment")
        elif patient_data['crcl'] < 60:
            query_parts.append("vancomycin dosing moderate renal impairment")
    
    # Add calculation-specific information
    if calculation_results:
        if 'AUC Status' in calculation_results:
            if calculation_results['AUC Status'] == "BELOW TARGET":
                query_parts.append("vancomycin AUC below target recommendations")
            elif calculation_results['AUC Status'] == "ABOVE TARGET":
                query_parts.append("vancomycin AUC above target toxicity")
            else:
                query_parts.append("vancomycin AUC within target monitoring")
        
        if 'Trough Status' in calculation_results:
            if calculation_results['Trough Status'] == "BELOW TARGET":
                query_parts.append("vancomycin trough below target recommendations")
            elif calculation_results['Trough Status'] == "ABOVE TARGET":
                query_parts.append("vancomycin trough above target toxicity")
            else:
                query_parts.append("vancomycin trough within target monitoring")
    
    # Add target-related information
    if "Empirical" in target_desc:
        query_parts.append("vancomycin empirical dosing guidelines AUC 400-600")
    elif "Definitive/Severe" in target_desc:
        query_parts.append("vancomycin severe infection dosing guidelines AUC greater than 600")
    
    # Add clinical context if provided
    if patient_data.get('clinical_notes') and len(patient_data['clinical_notes']) > 10:
        # Extract keywords from clinical notes
        notes = patient_data['clinical_notes'].lower()
        if "meningitis" in notes:
            query_parts.append("vancomycin CNS infection meningitis")
        if "endocarditis" in notes:
            query_parts.append("vancomycin endocarditis dosing")
        if "pneumonia" in notes:
            query_parts.append("vancomycin pneumonia dosing")
        if "dialysis" in notes or "hd" in notes or "crrt" in notes:
            query_parts.append("vancomycin renal replacement therapy dialysis")
        if "icu" in notes or "sepsis" in notes or "septic" in notes:
            query_parts.append("vancomycin critical care sepsis")
        if "obesity" in notes or "obese" in notes:
            query_parts.append("vancomycin dosing obesity")
    
    # Combine query parts, ensuring we're not repeating terms
    unique_parts = list(set(query_parts))
    
    # If no specific parts, use a general query
    if not unique_parts:
        unique_parts = ["vancomycin dosing guidelines AUC monitoring recommendations"]
    
    # Combine parts into a single query string
    return " ".join(unique_parts)

# Display RAG results in the UI
def display_rag_results(results):
    """Displays the retrieved guideline sections."""
    if not results:
        st.info("No relevant guideline sections found.")
        return
    
    st.subheader("Relevant Guideline Sections")
    
    for i, result in enumerate(results):
        with st.expander(f"Guideline Section {i+1}"):
            st.markdown(f'<div class="citation">{result["text"]}</div>', unsafe_allow_html=True)

# Render LLM reasoning in the UI
def display_llm_reasoning(reasoning_text):
    """Displays the LLM reasoning with appropriate formatting."""
    if not reasoning_text:
        st.info("No expert reasoning available.")
        return
    
    st.subheader("Expert Pharmacokinetic Analysis")
    st.markdown(f'<div class="reasoning"><p class="reasoning-title">Clinical Pharmacist Assessment</p>{reasoning_text}</div>', unsafe_allow_html=True)

# Render interpretation using Streamlit components
def render_interpretation_st(trough_status, trough_measured, auc_status, auc24, thalf, interval_h, new_dose, target_desc, pk_method):
    """Renders interpretation using st.expander and st.markdown."""

    if not all([trough_status, trough_measured is not None, auc_status, auc24 is not None, thalf is not None, interval_h, new_dose is not None, target_desc]):
         st.warning("Interpretation cannot be generated due to missing calculation results.")
         return

    # Determine recommendation text based on status
    rec_action = "adjust" # Default
    if "BELOW" in trough_status or "BELOW" in auc_status:
        rec_action = "increase"
    elif "ABOVE" in trough_status or "ABOVE" in auc_status:
        rec_action = "decrease"
    elif "WITHIN" in trough_status and "WITHIN" in auc_status:
        rec_action = "maintain" # Or suggest confirmation

    # Get target values from description for display
    target_trough_str = "N/A"
    target_auc_str = "N/A"
    if "Empirical" in target_desc:
        target_trough_str = "10-15 mg/L"
        target_auc_str = "400-600 mg·h/L"
    elif "Definitive/Severe" in target_desc:
        target_trough_str = "15-20 mg/L"
        target_auc_str = ">600 mg·h/L" # Displaying lower bound

    # Build the interpretation sections
    with st.expander("View Detailed Interpretation", expanded=True):
        st.subheader("Assessment")
        assessment_text = f"""
        - **Method:** {pk_method}
        - **Measured Trough:** {trough_measured:.1f} mg/L ({trough_status.lower()})
        - **Calculated AUC₂₄:** {auc24:.1f} mg·h/L ({auc_status.lower()})
        - **Calculated Half-life:** {f'{thalf:.1f}' if thalf is not None else 'N/A'} h
        - **Current Interval:** q{interval_h}h (Interval is likely {'appropriate' if thalf is not None and interval_h >= thalf * 1.2 else 'potentially too long/short relative to half-life'})
        """
        st.markdown(assessment_text)

        st.subheader("Recommendation")
        recommendation_text = f"""
        - **Action:** Consider **{rec_action}ing** the dose.
        - **Suggested Regimen:** **{new_dose} mg q{interval_h}h**
        - **Goal:** To achieve target AUC ({target_auc_str}) and target trough ({target_trough_str}).
        """
        st.markdown(recommendation_text)

        st.subheader("Rationale")
        rationale_text = f"""
        The recommendation aims to align the patient's exposure (AUC₂₄) and trough levels with the selected therapeutic targets ({target_desc}).
        The adjustment is based on the calculated pharmacokinetic parameters derived from the {pk_method.lower()} analysis, which provides {'an individualized' if 'Peak & Trough' in pk_method else 'an estimated'} assessment compared to population averages.
        """
        st.markdown(rationale_text)

        st.subheader("Follow-up")
        followup_text = """
        - Draw the next trough level before the 3rd or 4th dose of the *new* regimen to confirm target attainment.
        - Continue to monitor renal function (e.g., SCr), clinical signs of infection, and potential vancomycin toxicity.
        - Adjust therapy based on clinical response and subsequent levels.
        """
        st.markdown(followup_text)

# --- 7. MAIN APP STRUCTURE ---
def main():
    # Initialize RAG system
    rag_system = get_rag_system()
    llm_reasoner = LLMReasoner()
    
    # --- HEADER ---
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        logo_url = "logo.png"
        try:
            st.image(logo_url, width=150)
        except Exception as e:
            st.markdown('<div style="font-size: 40px; margin-top: 10px;">💊</div>', unsafe_allow_html=True)

    with col_title:
        st.title("TDM-AID by HTAR")
        st.markdown('<p class="subtitle">VANCOMYCIN MODULE WITH RAG + LLM REASONING</p>', unsafe_allow_html=True)

    # Check if RAG system is initialized
    if not rag_system.is_initialized:
        st.warning("RAG system not initialized. Some advanced features may not be available.")

    # --- SIDEBAR CONTENT ---
    with st.sidebar:
        st.header("⚙️ Patient Information")

        with st.container(): # Group patient details
             # Use columns for better layout
            col_pid, col_ward = st.columns(2)
            with col_pid:
                pid = st.text_input("Patient ID", placeholder="MRN12345")
            with col_ward:
                ward = st.text_input("Ward/Unit", placeholder="ICU")

        with st.container(): # Group demographics
            col_age, col_wt = st.columns(2)
            with col_age:
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=65, step=1)
            with col_wt:
                wt = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")

            col_scr, col_fem = st.columns(2)
            with col_scr:
                scr_umol = st.number_input("SCr (µmol/L)", min_value=10.0, max_value=2000.0, value=88.0, step=1.0, format="%.0f", help="Serum Creatinine")
            with col_fem:
                 # Add vertical space to align checkbox
                st.markdown('<div style="height: 29px;"></div>', unsafe_allow_html=True)
                fem = st.checkbox("Female", value=False)

        with st.container(): # Group target selection
            st.subheader("Therapeutic Target")
            target_level_desc = st.selectbox(
                "Select target level:",
                options=[
                    "Empirical (Target AUC₂₄ 400-600 mg·h/L; Trough ~10-15 mg/L)",
                    "Definitive/Severe (Target AUC₂₄ >600 mg·h/L; Trough ~15-20 mg/L)"
                ],
                index=0,
                label_visibility="collapsed",
                help="Select the desired therapeutic range based on indication and severity."
            )
            # Define target ranges based on selection
            if "Empirical" in target_level_desc:
                target_trough_range = (10.0, 15.0)
                target_auc_range = (400.0, 600.0)
            else: # Definitive/Severe
                target_trough_range = (15.0, 20.0)
                target_auc_range = (600.0, None) # Use None for open-ended upper bound

        with st.container(): # Group clinical notes
            st.subheader("Clinical Context")
            clinical_notes = st.text_area(
                "Clinical Notes",
                placeholder="Enter relevant clinical context (e.g., infection type, organ function, source control status...)",
                height=100,
                label_visibility="collapsed"
            )

        # Calculate CrCl in sidebar for use in all tabs
        crcl = calculate_crcl(age, wt, scr_umol, fem)
        if crcl is not None:
            st.metric(label="Estimated CrCl (mL/min)", value=f"{crcl:.1f}")
        else:
            st.warning("Cannot calculate CrCl. Check Age, Weight, SCr.")
            
        # Toggle for LLM features
        st.divider()
        st.subheader("Advanced Features")
        use_rag = st.toggle("Use RAG Guidelines", value=True, help="Retrieve relevant guideline sections for this case")
        use_llm = st.toggle("Use LLM Reasoning", value=True, help="Generate expert reasoning using GPT-4o")
        
    # --- MAIN AREA TABS ---
    tab1, tab2, tab3 = st.tabs(["Initial Dose", "Trough-Only Analysis", "Peak & Trough Analysis"])
    
    # --- INITIAL DOSE TAB ---
    with tab1:
        st.header("Initial Loading Dose Calculator")
        st.caption("Calculate an appropriate one-time loading dose based on patient weight.")

        col1, col2 = st.columns([3,1])

        with col1:
            st.info("""
            **Calculation Method:**
            - Uses weight-based dosing (typically 20-35 mg/kg, commonly 25 mg/kg).
            - Aims to rapidly achieve therapeutic concentrations.
            - Dose is rounded to the nearest 250mg increment.
            - **Note:** Renal function (CrCl) primarily guides the *maintenance* dose interval, not typically the loading dose amount. However, extreme renal impairment might warrant caution or adjustment.
            """)

        with col2:
             # Add vertical space to align button better
            st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
            calc_load_button = st.button("Calculate Loading Dose", key="calc_load")

        if calc_load_button:
            if wt is None or wt <= 0:
                st.error("Please enter a valid patient weight in the sidebar.")
            else:
                # Standard loading dose calculation (e.g., 25 mg/kg)
                loading_dose_raw = wt * 25
                # Round to nearest 250 mg
                loading_dose_rounded = round(loading_dose_raw / 250) * 250
                # Max dose consideration (optional, based on local guidelines)
                max_loading_dose = 3000 # Example cap
                loading_dose_final = min(loading_dose_rounded, max_loading_dose)

                st.success(f"**Recommended Initial Loading Dose: {loading_dose_final} mg** (one-time)")
                st.caption(f"Calculated based on 25 mg/kg for {wt} kg weight, rounded to nearest 250 mg.")
                if loading_dose_final >= max_loading_dose:
                     st.warning(f"Loading dose capped at {max_loading_dose} mg.")
                
                # Collect data for RAG and LLM
                patient_data = {
                    'age': age,
                    'weight': wt,
                    'sex': 'Female' if fem else 'Male',
                    'scr': scr_umol,
                    'crcl': crcl,
                    'clinical_notes': clinical_notes
                }
                
                calculation_results = {
                    'Calculation Method': 'Initial Loading Dose',
                    'Calculated Dose (raw)': f"{loading_dose_raw:.1f} mg",
                    'Rounded Dose': f"{loading_dose_rounded} mg",
                    'Final Recommended Loading Dose': f"{loading_dose_final} mg (one-time)"
                }
                
                # Use RAG system to retrieve relevant guidelines if enabled
                if use_rag and rag_system.is_initialized:
                    query = f"vancomycin loading dose guidelines {wt}kg patient CrCl {crcl:.1f} {clinical_notes}"
                    results = rag_system.search(query, k=3)
                    if results:
                        display_rag_results(results)
                    
                    # Use LLM reasoning if enabled
                    if use_llm:
                        reasoning_text = llm_reasoner.generate_reasoning(patient_data, calculation_results, results)
                        display_llm_reasoning(reasoning_text)

                # Prepare data for download
                report_data = f"""Vancomycin Initial Loading Dose Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Patient Information:
- ID: {pid if pid else 'N/A'}
- Ward: {ward if ward else 'N/A'}
- Age: {age} years
- Weight: {wt} kg
- Sex: {'Female' if fem else 'Male'}
- SCr: {scr_umol} µmol/L
- Estimated CrCl: {f'{crcl:.1f}' if crcl is not None else 'N/A'} mL/min (for maintenance dose reference)

Loading Dose Calculation:
- Basis: 25 mg/kg
- Calculated Dose (raw): {loading_dose_raw:.1f} mg
- Rounded Dose: {loading_dose_rounded} mg
- Final Recommended Loading Dose: {loading_dose_final} mg (one-time)

Clinical Notes:
{clinical_notes if clinical_notes else 'N/A'}

Disclaimer: For educational purposes only. Verify with clinical guidelines.
"""
                st.download_button(
                    label="📄 Download Loading Dose Report",
                    data=report_data,
                    file_name=f"vanco_loading_dose_{pid if pid else 'report'}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
                
    # --- TROUGH-ONLY TAB ---
    with tab2:
        st.header("Trough-Only Analysis")
        st.caption("Estimate PK parameters and suggest dose adjustments based on a single steady-state trough level.")

        # Input Form
        with st.form(key="trough_only_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Current Regimen")
                dose_current_to = st.number_input("Current Dose (mg)", min_value=250, step=250, value=1000, key="to_dose")
                interval_current_to = st.selectbox(
                    "Dosing Interval (hours)",
                    options=[6, 8, 12, 18, 24, 36, 48],
                    index=2, # Default to q12h
                    format_func=lambda x: f"q{x}h",
                    key="to_interval"
                )
            with col2:
                st.subheader("Trough Level & Timing")
                trough_measured_to = st.number_input("Measured Trough (mg/L)", min_value=0.1, max_value=100.0, value=12.5, step=0.1, format="%.1f", key="to_trough")
                dose_time_to = st.time_input("Last Dose Given At", value=time(8, 0), step=timedelta(minutes=15), key="to_dose_time")
                sample_time_to = st.time_input("Trough Level Drawn At", value=time(19, 30), step=timedelta(minutes=15), key="to_sample_time") # Assumes draw before next dose

            submitted_to = st.form_submit_button("Run Trough Analysis")

        # --- TROUGH-ONLY RESULTS ---
        if submitted_to:
            # Input validation
            if not all([dose_current_to, interval_current_to, trough_measured_to, dose_time_to, sample_time_to]):
                 st.error("Please ensure all inputs are provided.")
            elif crcl is None:
                 st.error("Cannot perform analysis: CrCl could not be calculated. Check patient details in the sidebar.")
            else:
                time_since_last_dose_h = hours_diff(dose_time_to, sample_time_to)

                timing_valid_to = True
                if time_since_last_dose_h <= 0:
                    st.error("Timing Error: Trough sample time must be after the last dose time.")
                    timing_valid_to = False
                elif time_since_last_dose_h >= interval_current_to:
                     st.warning(f"Timing Warning: Trough drawn {format_hours_minutes(time_since_last_dose_h)} after the dose, which is longer than the interval (q{interval_current_to}h). Ensure this timing is correct and represents a true trough.")
                     # Proceed with calculation but warn user

                # Perform calculations only if timing is valid (or warning accepted)
                if timing_valid_to:
                    with st.spinner("Analyzing trough level..."):
                        try:
                            # --- Trough-Only PK Calculations (Simplified Population Estimates) ---
                            # Use population Vd estimate (e.g., 0.7 L/kg)
                            vd_est = 0.7 * wt
                            # Estimate Ke based on CrCl (example formula, use locally validated one if available)
                            # Example: Ke = 0.00083 * CrCl + 0.0044 (adjust based on literature/guidelines)
                            ke_est = 0.00083 * crcl + 0.0044
                            if ke_est <= 0: ke_est = 0.001 # Avoid zero or negative Ke

                            # Estimate Cmax based on measured trough and estimated Ke
                            # C_trough = Cmax_ss * exp(-Ke * (tau - T_inf)) -> Approximation if T_inf is short
                            # More accurately: C_trough = [(Dose/Vd) / (1-exp(-Ke*tau))] * exp(-Ke*(tau - T_inf))
                            # Simpler approximation for trough-only: C_trough = C_peak_est * exp(-Ke * time_since_peak)
                            # This method is less accurate as it relies heavily on population Vd and Ke.
                            # A common simplified approach estimates Cpeak from trough:
                            # Cpeak_est = trough_measured_to / math.exp(-ke_est * (interval_current_to - 1)) # Assuming 1hr infusion approx.
                            # This is highly approximate. A better trough-only method uses Bayesian estimation or assumes steady state.

                            # Let's use a steady-state assumption approach:
                            # C_trough_ss = (Dose / (Vd * Ke * tau)) * (Ke * tau * exp(-Ke * (tau - T_inf))) / (1 - exp(-Ke * tau)) -- Complex
                            # Alternative: Use the measured trough to back-calculate a Ke if Vd is assumed, or vice-versa.
                            # Let's stick to population Ke/Vd for simplicity here, acknowledging limitations.

                            cl_est = ke_est * vd_est
                            thalf_est = math.log(2) / ke_est if ke_est > 0 else float('inf')
                            auc_interval_est = dose_current_to / cl_est if cl_est > 0 else float('inf')
                            auc24_est = auc_interval_est * (24 / interval_current_to) if interval_current_to > 0 else float('inf')

                            # --- Dose Recommendation ---
                            # Target the midpoint of the AUC range for calculation
                            target_auc_mid = (target_auc_range[0] + target_auc_range[1]) / 2 if target_auc_range[1] is not None else target_auc_range[0] + 100 # Aim slightly above min if open-ended
                            target_auc_interval = target_auc_mid * (interval_current_to / 24)
                            new_dose_raw = target_auc_interval * cl_est
                            new_dose_rounded = round(new_dose_raw / 250) * 250 if new_dose_raw > 0 else 0

                            # --- Prepare Results ---
                            trough_status = check_target_status(trough_measured_to, target_trough_range)
                            auc_status = check_target_status(auc24_est, target_auc_range)

                            pk_results = {
                                'Calculation Method': 'Trough-Only (Population Estimate)',
                                'Est. CrCl (mL/min)': f"{crcl:.1f}",
                                'Est. Vd (L)': f"{vd_est:.1f}",
                                'Est. Ke (h⁻¹)': f"{ke_est:.4f}",
                                'Est. CL (L/h)': f"{cl_est:.2f}",
                                'Est. t½ (h)': f"{thalf_est:.1f}",
                                'Est. AUC₂₄ (mg·h/L)': f"{auc24_est:.1f}",
                                'Measured Trough (mg/L)': f"{trough_measured_to:.1f}",
                                'Trough Status': trough_status,
                                'AUC Status': auc_status,
                                'Suggested New Dose': f"{new_dose_rounded} mg q{interval_current_to}h"
                            }

                            # --- Display Results ---
                            st.subheader("Analysis Results")
                            st.info(f"Trough drawn **{format_hours_minutes(time_since_last_dose_h)}** after last dose (Interval: q{interval_current_to}h). Calculations use population estimates based on patient demographics.")

                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.subheader("Target Status")
                                display_level_indicator("Measured Trough", trough_measured_to, target_trough_range, "mg/L")
                                display_level_indicator("Estimated AUC₂₄", auc24_est, target_auc_range, "mg·h/L")

                            with res_col2:
                                st.subheader("Recommendation")
                                st.markdown(f'<p class="recommendation-dose">{new_dose_rounded} mg q{interval_current_to}h</p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="recommendation-description">Suggested dose to achieve target {target_level_desc.split("(")[1].split(";")[0].strip()}.</p>', unsafe_allow_html=True)


                            st.subheader("Estimated PK Parameters")
                            pk_col1, pk_col2, pk_col3 = st.columns(3)
                            with pk_col1:
                                st.metric("Est. Vd (L)", f"{vd_est:.1f}")
                                st.metric("Est. CL (L/h)", f"{cl_est:.2f}")
                            with pk_col2:
                                st.metric("Est. Ke (h⁻¹)", f"{ke_est:.4f}")
                                st.metric("Est. t½ (h)", f"{thalf_est:.1f}")
                            with pk_col3:
                                st.metric("Est. AUC₂₄ (mg·h/L)", f"{auc24_est:.1f}")
                                st.metric("CrCl (mL/min)", f"{crcl:.1f}")

                            # Render Interpretation
                            render_interpretation_st(
                                trough_status=trough_status,
                                trough_measured=trough_measured_to,
                                auc_status=auc_status,
                                auc24=auc24_est,
                                thalf=thalf_est,
                                interval_h=interval_current_to,
                                new_dose=new_dose_rounded,
                                target_desc=target_level_desc,
                                pk_method="Trough-Only (Population Estimate)"
                            )
                            
                            # Collect patient data for RAG and LLM
                            patient_data = {
                                'age': age,
                                'weight': wt,
                                'sex': 'Female' if fem else 'Male',
                                'scr': scr_umol,
                                'crcl': crcl,
                                'clinical_notes': clinical_notes
                            }
                            
                            # Use RAG system to retrieve relevant guidelines if enabled
                            if use_rag and rag_system.is_initialized:
                                query = formulate_rag_query(patient_data, pk_results, target_level_desc)
                                results = rag_system.search(query, k=3)
                                if results:
                                    display_rag_results(results)
                                
                                # Use LLM reasoning if enabled
                                if use_llm:
                                    reasoning_text = llm_reasoner.generate_reasoning(patient_data, pk_results, results)
                                    display_llm_reasoning(reasoning_text)

                            # Download Report
                            report_data = f"""Vancomycin TDM Report (Trough-Only)
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Patient Information:
- ID: {pid if pid else 'N/A'}
- Ward: {ward if ward else 'N/A'}
- Age: {age} years
- Weight: {wt} kg
- Sex: {'Female' if fem else 'Male'}
- SCr: {scr_umol} µmol/L
- Estimated CrCl: {f'{crcl:.1f}' if crcl is not None else 'N/A'} mL/min

Current Regimen & Level:
- Dose: {dose_current_to} mg q{interval_current_to}h
- Last Dose Time: {dose_time_to.strftime('%H:%M')}
- Trough Sample Time: {sample_time_to.strftime('%H:%M')} ({format_hours_minutes(time_since_last_dose_h)} after dose)
- Measured Trough: {trough_measured_to:.1f} mg/L

Target: {target_level_desc}
- Target Trough: {target_trough_range[0]}-{target_trough_range[1]} mg/L
- Target AUC₂₄: {target_auc_range[0]}-{target_auc_range[1] if target_auc_range[1] else '>'} mg·h/L

Analysis Results (Population Estimates):
{pd.Series(pk_results).to_string()}

Clinical Notes:
{clinical_notes if clinical_notes else 'N/A'}

Disclaimer: Trough-only analysis uses population estimates and has limitations. Individual PK may vary. Clinical correlation required.
"""
                            st.download_button(
                                label="📄 Download Trough Analysis Report",
                                data=report_data,
                                file_name=f"vanco_trough_analysis_{pid if pid else 'report'}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )

                        except Exception as e:
                            st.error(f"An error occurred during calculation: {e}")
                            st.exception(e) # Show traceback for debugging
                            
    # --- PEAK & TROUGH TAB ---
    with tab3:
        st.header("Peak & Trough Analysis")
        st.caption("Calculate individual PK parameters using paired peak and trough levels (Sawchuk-Zaske method).")

        # Input Form
        with st.form(key="peak_trough_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Regimen & Levels")
                dose_levels_pt = st.number_input("Dose Administered (mg)", min_value=250, step=250, value=1000, key="pt_dose")
                interval_pt = st.selectbox(
                    "Dosing Interval (hours)",
                    options=[6, 8, 12, 18, 24, 36, 48],
                    index=2, # Default q12h
                    format_func=lambda x: f"q{x}h",
                    key="pt_interval"
                )
                peak_measured_pt = st.number_input("Measured Peak (mg/L)", min_value=0.1, max_value=200.0, value=30.0, step=0.1, format="%.1f", key="pt_peak")
                trough_measured_pt = st.number_input("Measured Trough (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f", key="pt_trough")

            with col2:
                st.subheader("Timing Information")
                infusion_start_time_pt = st.time_input("Infusion Start Time", value=time(8, 0), step=timedelta(minutes=15), key="pt_inf_start")
                infusion_end_time_pt = st.time_input("Infusion End Time", value=time(9, 0), step=timedelta(minutes=15), key="pt_inf_end")
                peak_sample_time_pt = st.time_input("Peak Sample Time", value=time(10, 0), step=timedelta(minutes=15), key="pt_peak_time", help="Typically 1-2h post-infusion end")
                trough_sample_time_pt = st.time_input("Trough Sample Time", value=time(19, 30), step=timedelta(minutes=15), key="pt_trough_time", help="Immediately before next dose")

            submitted_pt = st.form_submit_button("Run Peak & Trough Analysis")

        # --- PEAK & TROUGH RESULTS ---
        if submitted_pt:
            # Input validation
            if not all([dose_levels_pt, interval_pt, peak_measured_pt, trough_measured_pt, infusion_start_time_pt, infusion_end_time_pt, peak_sample_time_pt, trough_sample_time_pt]):
                 st.error("Please ensure all inputs are provided.")
            elif crcl is None:
                 st.error("Cannot perform analysis: CrCl could not be calculated. Check patient details in the sidebar.")
            elif peak_measured_pt <= trough_measured_pt:
                 st.error("Input Error: Peak level must be higher than trough level.")
            else:
                # Calculate timing differences
                infusion_duration_h = hours_diff(infusion_start_time_pt, infusion_end_time_pt)
                time_from_inf_end_to_peak = hours_diff(infusion_end_time_pt, peak_sample_time_pt)
                time_from_peak_to_trough = hours_diff(peak_sample_time_pt, trough_sample_time_pt)
                time_from_inf_end_to_trough = hours_diff(infusion_end_time_pt, trough_sample_time_pt) # T'

                # Validate timings
                timing_valid_pt = True
                if infusion_duration_h <= 0:
                    st.error("Timing Error: Infusion end time must be after start time.")
                    timing_valid_pt = False
                if time_from_inf_end_to_peak < 0:
                    st.error("Timing Error: Peak sample time must be after infusion end time.")
                    timing_valid_pt = False
                if time_from_peak_to_trough <= 0:
                    st.error("Timing Error: Trough sample time must be after peak sample time.")
                    timing_valid_pt = False
                # Check if trough is drawn reasonably before the next dose (e.g. within interval duration)
                time_from_inf_start_to_trough = hours_diff(infusion_start_time_pt, trough_sample_time_pt)
                if time_from_inf_start_to_trough >= interval_pt:
                     st.warning(f"Timing Warning: Trough drawn {format_hours_minutes(time_from_inf_start_to_trough)} after infusion start, which is >= interval (q{interval_pt}h). Ensure this is a true trough before the *next* dose.")
                     # Allow calculation but warn

                if timing_valid_pt:
                    with st.spinner("Analyzing peak and trough levels..."):
                        try:
                            # --- Sawchuk-Zaske PK Calculations ---
                            # Calculate Ke
                            if time_from_peak_to_trough == 0: raise ValueError("Time between peak and trough samples cannot be zero.")
                            ke_ind = math.log(peak_measured_pt / trough_measured_pt) / time_from_peak_to_trough
                            if ke_ind <= 0: raise ValueError("Calculated Ke is non-positive. Check levels and times.")

                            thalf_ind = math.log(2) / ke_ind

                            # Extrapolate Cmax (at end of infusion) and Cmin (true trough)
                            cmax_extrap = peak_measured_pt * math.exp(ke_ind * time_from_inf_end_to_peak)
                            # Cmin_extrap = trough_measured_pt * math.exp(ke_ind * time_trough_to_next_dose_start) # Needs time to next dose
                            # Alternative Cmin calculation using Cmax_extrap and interval:
                            # Ensure interval > infusion duration for Cmin calculation
                            if interval_pt <= infusion_duration_h: raise ValueError("Dosing interval must be longer than infusion duration.")
                            cmin_extrap = cmax_extrap * math.exp(-ke_ind * (interval_pt - infusion_duration_h))

                            # Calculate Vd
                            # Ensure infusion duration is positive before using in calculation
                            if infusion_duration_h <= 0: raise ValueError("Infusion duration must be positive for Vd calculation.")
                            term1 = dose_levels_pt / (ke_ind * infusion_duration_h)
                            term2 = (1 - math.exp(-ke_ind * infusion_duration_h))
                            # Denominator term: (Cmax_extrap - Cmin_extrap * exp(-Ke * T_inf))
                            denominator_vd = cmax_extrap - (cmin_extrap * math.exp(-ke_ind * infusion_duration_h))
                            if denominator_vd == 0: raise ValueError("Calculation error: Vd denominator is zero.")
                            vd_ind = term1 * (term2 / denominator_vd) # More robust formula

                            if vd_ind <= 0: raise ValueError("Calculated Vd is non-positive. Check inputs.")

                            # Calculate CL
                            cl_ind = ke_ind * vd_ind

                            # Calculate AUC
                            auc_interval_ind = dose_levels_pt / cl_ind
                            auc24_ind = auc_interval_ind * (24 / interval_pt)

                            # --- Dose Recommendation ---
                            target_auc_mid = (target_auc_range[0] + target_auc_range[1]) / 2 if target_auc_range[1] is not None else target_auc_range[0] + 100
                            target_auc_interval = target_auc_mid * (interval_pt / 24)
                            new_dose_raw = target_auc_interval * cl_ind
                            new_dose_rounded = round(new_dose_raw / 250) * 250

                            # --- Prepare Results ---
                            trough_status = check_target_status(trough_measured_pt, target_trough_range) # Use measured trough for status check
                            auc_status = check_target_status(auc24_ind, target_auc_range)

                            pk_results = {
                                'Calculation Method': 'Peak & Trough (Individualized)',
                                'Individual Vd (L)': f"{vd_ind:.1f}",
                                'Individual Ke (h⁻¹)': f"{ke_ind:.4f}",
                                'Individual CL (L/h)': f"{cl_ind:.2f}",
                                'Individual t½ (h)': f"{thalf_ind:.1f}",
                                'Individual AUC₂₄ (mg·h/L)': f"{auc24_ind:.1f}",
                                'Measured Peak (mg/L)': f"{peak_measured_pt:.1f}",
                                'Measured Trough (mg/L)': f"{trough_measured_pt:.1f}",
                                'Extrapolated Cmax (mg/L)': f"{cmax_extrap:.1f}",
                                'Extrapolated Cmin (mg/L)': f"{cmin_extrap:.1f}",
                                'Trough Status': trough_status,
                                'AUC Status': auc_status,
                                'Suggested New Dose': f"{new_dose_rounded} mg q{interval_pt}h"
                            }

                            # --- Display Results ---
                            st.subheader("Analysis Results")
                            st.info(f"Infusion Duration: **{format_hours_minutes(infusion_duration_h)}**, Time Infusion End to Peak: **{format_hours_minutes(time_from_inf_end_to_peak)}**, Time Peak to Trough: **{format_hours_minutes(time_from_peak_to_trough)}**")

                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.subheader("Target Status")
                                display_level_indicator("Measured Trough", trough_measured_pt, target_trough_range, "mg/L")
                                display_level_indicator("Calculated AUC₂₄", auc24_ind, target_auc_range, "mg·h/L")

                            with res_col2:
                                st.subheader("Recommendation")
                                st.markdown(f'<p class="recommendation-dose">{new_dose_rounded} mg q{interval_pt}h</p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="recommendation-description">Suggested dose based on individual PK to achieve target {target_level_desc.split("(")[1].split(";")[0].strip()}.</p>', unsafe_allow_html=True)

                            st.subheader("Individual PK Parameters")
                            pk_col1, pk_col2, pk_col3 = st.columns(3)
                            with pk_col1:
                                st.metric("Individual Vd (L)", f"{vd_ind:.1f}")
                                st.metric("Individual CL (L/h)", f"{cl_ind:.2f}")
                            with pk_col2:
                                st.metric("Individual Ke (h⁻¹)", f"{ke_ind:.4f}")
                                st.metric("Individual t½ (h)", f"{thalf_ind:.1f}")
                            with pk_col3:
                                st.metric("Individual AUC₂₄ (mg·h/L)", f"{auc24_ind:.1f}")
                                st.metric("Extrap. Cmax (mg/L)", f"{cmax_extrap:.1f}")
                                # st.metric("Extrap. Cmin (mg/L)", f"{cmin_extrap:.1f}") # Cmin often less critical to display than AUC/Trough

                            # Render Interpretation
                            render_interpretation_st(
                                trough_status=trough_status,
                                trough_measured=trough_measured_pt, # Use measured trough for interpretation context
                                auc_status=auc_status,
                                auc24=auc24_ind,
                                thalf=thalf_ind,
                                interval_h=interval_pt,
                                new_dose=new_dose_rounded,
                                target_desc=target_level_desc,
                                pk_method="Peak & Trough (Individualized)"
                            )
                            
                            # Collect patient data for RAG and LLM
                            patient_data = {
                                'age': age,
                                'weight': wt,
                                'sex': 'Female' if fem else 'Male',
                                'scr': scr_umol,
                                'crcl': crcl,
                                'clinical_notes': clinical_notes
                            }
                            
                            # Use RAG system to retrieve relevant guidelines if enabled
                            if use_rag and rag_system.is_initialized:
                                query = formulate_rag_query(patient_data, pk_results, target_level_desc)
                                results = rag_system.search(query, k=3)
                                if results:
                                    display_rag_results(results)
                                
                                # Use LLM reasoning if enabled
                                if use_llm:
                                    reasoning_text = llm_reasoner.generate_reasoning(patient_data, pk_results, results)
                                    display_llm_reasoning(reasoning_text)

                            # Download Report
                            report_data = f"""Vancomycin TDM Report (Peak & Trough)
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Patient Information:
- ID: {pid if pid else 'N/A'}
- Ward: {ward if ward else 'N/A'}
- Age: {age} years
- Weight: {wt} kg
- Sex: {'Female' if fem else 'Male'}
- SCr: {scr_umol} µmol/L
- Estimated CrCl: {f'{crcl:.1f}' if crcl is not None else 'N/A'} mL/min

Regimen, Levels & Timing:
- Dose: {dose_levels_pt} mg q{interval_pt}h
- Infusion Start: {infusion_start_time_pt.strftime('%H:%M')}
- Infusion End: {infusion_end_time_pt.strftime('%H:%M')} (Duration: {format_hours_minutes(infusion_duration_h)})
- Peak Sample Time: {peak_sample_time_pt.strftime('%H:%M')} ({format_hours_minutes(time_from_inf_end_to_peak)} post-infusion)
- Trough Sample Time: {trough_sample_time_pt.strftime('%H:%M')}
- Measured Peak: {peak_measured_pt:.1f} mg/L
- Measured Trough: {trough_measured_pt:.1f} mg/L

Target: {target_level_desc}
- Target Trough: {target_trough_range[0]}-{target_trough_range[1]} mg/L
- Target AUC₂₄: {target_auc_range[0]}-{target_auc_range[1] if target_auc_range[1] else '>'} mg·h/L

Analysis Results (Individualized PK):
{pd.Series(pk_results).to_string()}

Clinical Notes:
{clinical_notes if clinical_notes else 'N/A'}

Disclaimer: For educational purposes only. Verify calculations and clinical correlation.
"""
                            st.download_button(
                                label="📄 Download Peak & Trough Analysis Report",
                                data=report_data,
                                file_name=f"vanco_peak_trough_analysis_{pid if pid else 'report'}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )

                        except ValueError as ve: # Catch specific calculation errors
                            st.error(f"Calculation Error: {ve}. Please check input values and timings.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during calculation: {e}")
                            st.exception(e) # Show traceback for debugging
                            
    # --- FOOTER ---
    st.markdown("---")
    st.caption("""
        **Disclaimer:** This tool is intended for educational and informational purposes only and does not constitute medical advice.
        Calculations are based on standard pharmacokinetic principles but may need adjustment based on individual patient factors and clinical context.
        Always consult with qualified healthcare professionals and local guidelines for clinical decision-making.
        *Reference: Basic Clinical Pharmacokinetics (6th Ed.), Clinical Pharmacokinetics Pharmacy Handbook (2nd Ed.), ASHP/IDSA Vancomycin Guidelines.*

        Developed by Dr. Fahmi Hassan (fahmibinabad@gmail.com), Enhanced with RAG and LLM reasoning.
    """)
    
    # Cleanup RAG system when the app is closed
    try:
        rag_system.cleanup()
    except:
        pass

if __name__ == "__main__":
    main()
