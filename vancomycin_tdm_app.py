# vancomycin_tdm_app.py
"""
Single-file Streamlit app for Vancomycin TDM with RAG-guided LLM interpretation,
time-based inputs, target-level selection, PDF chunking via pypdf,
current dosing interval input, SCr input in µmol/L, and clinical notes.
Uses Streamlit secrets for OpenAI API key.
Peak/Trough Vd and Expected Levels updated based on provided formulas.
Increased LLM max_tokens and refined LLM prompt for better recommendations.
V2: Refined LLM prompt in interpret() to explicitly compare levels/AUC to targets.
V3: Added code-based target status checks (Trough/AUC vs. Target) displayed before LLM interpretation.
"""
import os
import math
import re # Import regex for parsing target strings
import streamlit as st
from datetime import datetime, timedelta, time
import logging # Added for better error logging

# --- 1. SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Vancomycin TDM App", layout="wide")

# --- 2. CONFIGURE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
SCR_CONVERSION_FACTOR = 88.4 # Conversion factor from mg/dL to µmol/L

# --- 3. LOAD SECRET ---
# Use Streamlit secrets to get the OpenAI API key
try:
    # Check if running on Streamlit Cloud and secrets are available
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        logging.info("OpenAI API key loaded from Streamlit secrets.")
    else:
        # Fallback for local development (optional, but can be useful)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Defer st.error call until the main app body
            logging.error("OpenAI API key not found in Streamlit secrets or environment variables.")
        else:
            logging.info("OpenAI API key loaded from environment variable for local testing.")
            os.environ["OPENAI_API_KEY"] = openai_api_key

except Exception as e:
    # Defer st.error call until the main app body
    logging.error(f"Error loading OpenAI API key: {e}")

# Check if API key was loaded successfully before proceeding with imports/RAG
api_key_loaded = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]

# --- 4. IMPORTS (Place imports after key check to avoid errors if key fails) ---
# Only import these if the API key is likely available
libraries_loaded = False # Initialize
if api_key_loaded:
    try:
        from pypdf import PdfReader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        # Updated imports for newer Langchain structure:
        from langchain_openai import OpenAIEmbeddings # Moved to langchain-openai
        from langchain_community.vectorstores import FAISS # Moved to langchain-community
        from langchain.chains import RetrievalQA # Still in langchain core (usually)
        from langchain_openai import OpenAI # LLM moved to langchain-openai

        logging.info("Required libraries imported successfully.")
        libraries_loaded = True
    except ImportError as e:
        # Defer st.error call
        logging.error(f"Failed to import required libraries: {e}")
        libraries_loaded = False
else:
    # Log error if key wasn't loaded (will be shown in UI later)
    logging.error("API key not loaded, cannot import dependent libraries.")


# --- 5. LOAD & INDEX GUIDELINE PDF USING pypdf ---
@st.cache_resource # Caches the resource across reruns
def load_rag_chain(pdf_path: str) -> RetrievalQA | None:
    """Loads the PDF, chunks text, creates embeddings, and builds a RAG chain."""
    # Ensure API key and libraries are loaded before trying to use them
    if not api_key_loaded or not libraries_loaded:
         logging.error("Cannot load RAG chain because API key or libraries failed to load.")
         return None
    try:
        # Check if the PDF file exists
        if not os.path.exists(pdf_path):
            logging.error(f"Guideline PDF not found at path: {pdf_path}")
            return None

        # Extract full text
        logging.info(f"Loading PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        full_text = ""
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            except Exception as page_err:
                logging.warning(f"Could not extract text from page {i+1} in {pdf_path}: {page_err}")
        logging.info(f"Extracted text length: {len(full_text)} characters.")
        if not full_text:
            logging.error(f"No text extracted from PDF: {pdf_path}")
            return None

        # Chunk text
        logging.info("Chunking text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        logging.info(f"Created {len(chunks)} text chunks.")
        if not chunks:
            logging.error("Text splitting resulted in zero chunks.")
            return None

        # Embed and index
        logging.info("Creating embeddings and FAISS index...")
        embeddings = OpenAIEmbeddings()
        index = FAISS.from_texts(chunks, embeddings)
        logging.info("FAISS index created successfully.")

        # Build QA chain
        logging.info("Building RetrievalQA chain...")
        # --- Increased max_tokens for potentially longer interpretations ---
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, max_tokens=512), # Increased token limit
            chain_type="stuff",
            retriever=index.as_retriever()
        )
        logging.info("RetrievalQA chain built successfully.")
        return qa

    except Exception as e:
        logging.exception("Error during RAG chain loading:")
        return None

# Attempt to load the RAG chain
PDF_FILENAME = 'clinical-pharmacokinetics-pharmacy-handbook-ccph-2nd-edition-rev-2.0_0-2.pdf' # Make sure this file exists
qa_chain = load_rag_chain(PDF_FILENAME) # This might return None if setup failed

# --- 6. TIME DIFFERENCE HELPER ---
def hours_diff(start: time, end: time) -> float:
    """Calculates the difference between two times in hours, handling overnight intervals."""
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)
    if dt_end < dt_start:
        dt_end += timedelta(days=1)
    return (dt_end - dt_start).total_seconds() / 3600

# --- 7. PK CALC FUNCTIONS ---

def convert_scr_to_mgdl(scr_umol: float) -> float:
    """Converts SCr from µmol/L to mg/dL."""
    if scr_umol <= 0:
        return 0.0
    return scr_umol / SCR_CONVERSION_FACTOR

def calculate_crcl(age: int, weight: float, scr_mgdl: float, female: bool = False) -> float:
    """
    Calculates Creatinine Clearance (CrCl) using Cockcroft-Gault.
    Requires SCr in mg/dL.
    """
    if scr_mgdl <= 0:
        logging.warning(f"Invalid SCr (mg/dL) for CrCl calculation: {scr_mgdl}. Returning 0.")
        return 0.0
    base = ((140 - age) * weight) / (72 * scr_mgdl)
    crcl = base * 0.85 if female else base
    return max(0, crcl)

# Renamed from calculate_vd
def calculate_population_vd(weight: float, age: int | None = None) -> float:
    """Calculates population Volume of Distribution (Vd). Uses standard 0.7 L/kg."""
    if weight <= 0:
        logging.warning(f"Invalid weight ({weight}) for Vd calculation. Returning 0.")
        return 0.0
    vd = 0.7 * weight
    logging.info(f"Calculated population Vd using 0.7 L/kg (Weight={weight}kg): {vd:.1f} L")
    return vd

def round_dose(dose: float) -> int:
    """Rounds the dose to the nearest 250mg increment."""
    if dose < 0:
        return 0
    rounded = int(round(dose / 250.0) * 250)
    return max(250, rounded) if dose > 0 else 0

def calculate_initial_dose(age: int, weight: float, scr_mgdl: float, female: bool) -> int:
    """Calculates an initial loading dose based on CrCl (requires SCr in mg/dL)."""
    crcl = calculate_crcl(age, weight, scr_mgdl, female)
    mg_kg = 25
    loading_dose = weight * mg_kg
    logging.info(f"Calculated initial dose: CrCl={crcl:.1f}, mg/kg={mg_kg}, Raw Dose={loading_dose:.1f}")
    return round_dose(loading_dose)

# --- New function for Individual Vd based on Image Formula (e) ---
def calculate_individual_vd(dose: float, weight: float, cmax: float, ke: float, interval_h: float) -> float:
    """
    Calculates individual Volume of Distribution (Vd) using the formula:
    Vd(L) = Dose(mg) / [ Cmax(mg/L) * (1 - exp(-Ke*T)) ]
    (Derived from image formula assuming it calculates absolute Vd)
    """
    if dose <= 0 or cmax <= 0 or ke <= 0 or interval_h <= 0:
        logging.warning(f"Invalid inputs for individual Vd calculation: dose={dose}, cmax={cmax}, ke={ke}, interval={interval_h}. Returning 0.")
        return 0.0

    try:
        term_exp = 1 - math.exp(-ke * interval_h)
        if term_exp <= 0:
            logging.warning(f"Exponential term invalid ({term_exp}) in individual Vd calculation. Returning 0.")
            return 0.0

        vd_absolute = dose / (cmax * term_exp)

        logging.info(f"Calculated individual Vd (absolute): {vd_absolute:.1f} L")
        return vd_absolute
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating individual Vd: {e}. Inputs: dose={dose}, cmax={cmax}, ke={ke}, interval={interval_h}")
        return 0.0

def calculate_ke_trough(dose_int: float, vd: float, trough: float, time_since_last_dose_h: float) -> float:
    """
    Calculates the elimination rate constant (Ke) from trough level,
    using the ACTUAL time elapsed since the last dose was given.
    """
    if trough <= 0 or vd <= 0 or time_since_last_dose_h <= 0:
        logging.warning(f"Invalid input for Ke calculation: trough={trough}, vd={vd}, time_since_dose={time_since_last_dose_h}. Returning Ke=0.")
        return 0
    try:
        # Estimate Cmax using rise from trough: Cmax_est = trough + (dose / Vd)
        cmax_est = trough + (dose_int / vd)
        if cmax_est <= trough:
             logging.warning(f"Calculated Cmax_est ({cmax_est:.2f}) not greater than trough ({trough:.2f}). Check inputs (Vd might be too large or dose too small). Returning Ke=0.")
             return 0
        # ke = ln(cmax_est / trough) / time_since_last_dose_h
        ke = math.log(cmax_est / trough) / time_since_last_dose_h
        logging.info(f"Calculated Ke (trough-only): Cmax_est={cmax_est:.2f}, Trough={trough:.2f}, TimeSinceDose={time_since_last_dose_h:.2f}h => Ke={ke:.4f} h⁻¹")
        return max(0, ke)
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating Ke (trough-only): {e}. Inputs: dose={dose_int}, vd={vd}, trough={trough}, time_since_dose={time_since_last_dose_h}")
        return 0

# --- New function for Expected Cmax based on Image Formula (f) ---
def calculate_expected_cmax(dose: float, vd: float, interval_h: float, ke: float) -> float:
    """
    Calculates the expected Cmax using the formula:
    Expected Cmax = Dose (mg) / [Vd(L) × (1−e−KeT)]
    """
    if dose <= 0 or vd <= 0 or ke <= 0 or interval_h <= 0:
        logging.warning(f"Invalid inputs for expected Cmax calculation: dose={dose}, vd={vd}, ke={ke}, interval={interval_h}. Returning 0.")
        return 0.0

    try:
        term_exp = 1 - math.exp(-ke * interval_h)
        if term_exp <= 0:
            logging.warning(f"Exponential term invalid ({term_exp}) in expected Cmax calculation. Returning 0.")
            return 0.0

        # Uses absolute Vd (L)
        expected_cmax = dose / (vd * term_exp)
        logging.info(f"Calculated expected Cmax: {expected_cmax:.2f} mg/L")
        return expected_cmax
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating expected Cmax: {e}. Inputs: dose={dose}, vd={vd}, ke={ke}, interval={interval_h}")
        return 0.0

# --- New function for Expected Cmin based on Image Formula (g) ---
def calculate_expected_cmin(expected_cmax: float, ke: float, interval_h: float) -> float:
    """
    Calculates the expected Cmin using the formula:
    Expected Cmin = Expected Cmax × e−KeT
    """
    if expected_cmax <= 0 or ke <= 0 or interval_h <= 0:
        logging.warning(f"Invalid inputs for expected Cmin calculation: expected_cmax={expected_cmax}, ke={ke}, interval={interval_h}. Returning 0.")
        return 0.0

    try:
        expected_cmin = expected_cmax * math.exp(-ke * interval_h)
        logging.info(f"Calculated expected Cmin: {expected_cmin:.2f} mg/L")
        return expected_cmin
    except (ValueError, OverflowError) as e:
        logging.error(f"Math error calculating expected Cmin: {e}. Inputs: expected_cmax={expected_cmax}, ke={ke}, interval={interval_h}")
        return 0.0

def calculate_new_dose_trough(ke: float, vd: float, current_interval_h: float, target_trough: float = 15.0) -> int:
    """
    Calculates a new dose to reach a target trough level, using the calculated Ke
    and the CURRENT dosing interval (as the default interval for the new dose).
    Uses the formula: Dose = Cmin_target * Vd * (1 - exp(-Ke*T)) / exp(-Ke*T)
    """
    if ke <= 0 or vd <= 0 or current_interval_h <= 0 or target_trough <= 0:
        logging.warning(f"Cannot calculate new dose due to invalid Ke ({ke:.4f}) or other inputs (Vd={vd:.1f}, CurrentInterval={current_interval_h:.1f}, Target={target_trough:.1f}). Returning 0.")
        return 0
    try:
        term_exp = math.exp(-ke * current_interval_h)
        if term_exp == 1: # Avoid division by zero if Ke or T is effectively 0
            logging.warning("Exponential term denominator is 1 (exp(-KeT)=1) in new dose calculation (Ke or interval near zero). Cannot calculate dose.")
            return 0
        if term_exp <= 1e-9: # Consider denominator effectively zero if exp(-KeT) is very small
             logging.warning(f"Denominator exp(-KeT) ({term_exp:.2e}) near zero in new dose calculation. Check Ke and interval. Returning 0.")
             return 0 # Return 0 as dose would approach infinity

        numerator = target_trough * vd * (1 - term_exp)
        denominator = term_exp
        new_dose = numerator / denominator
        logging.info(f"Calculated new dose (trough-target): Target={target_trough:.1f}, Vd={vd:.1f}, Ke={ke:.4f}, CurrentInterval={current_interval_h:.1f}h => Raw New Dose={new_dose:.1f}")
        return round_dose(new_dose)
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        # Catch ZeroDivisionError explicitly if the check above fails somehow
        logging.error(f"Math error calculating new dose (trough-target): {e}. Inputs: target={target_trough}, vd={vd}, ke={ke}, current_interval={current_interval_h}")
        return 0

def calculate_auc24_trough(dose_int: float, ke: float, vd: float, current_interval_h: float) -> float:
    """
    Estimates AUC24 based on the current dose, calculated Ke, Vd,
    and the CURRENT dosing interval. Uses AUC = Dose_interval / CL_interval = Dose_interval / (Ke * Vd)
    Then AUC24 = AUC_interval * (24 / current_interval_h)
    """
    if ke <= 0 or vd <= 0 or current_interval_h <= 0:
        logging.warning(f"Cannot calculate AUC24 due to invalid Ke ({ke:.4f}) or other inputs (Vd={vd:.1f}, CurrentInterval={current_interval_h:.1f}). Returning AUC24=0.")
        return 0.0
    cl = ke * vd
    if cl <= 0:
        logging.warning(f"Calculated CL is zero or negative (Ke={ke:.4f}, Vd={vd:.1f}). Cannot calculate AUC24. Returning 0.")
        return 0.0
    try:
        # Calculate AUC over one dosing interval
        auc_interval = dose_int / cl
        # Extrapolate to 24 hours
        auc24 = auc_interval * (24 / current_interval_h)
        logging.info(f"Calculated AUC24 (trough-derived): Dose={dose_int}, CL={cl:.2f}, CurrentInterval={current_interval_h:.1f}h => AUC_interval={auc_interval:.1f}, AUC24={auc24:.1f}")
        return max(0, auc24)
    except ZeroDivisionError:
        logging.error(f"Division by zero calculating AUC24 (current_interval={current_interval_h}). Returning 0.")
        return 0.0

# --- Updated Peak/Trough Calculation Function ---
def calculate_pk_params_peak_trough(
    c_trough: float, c_peak_measured: float, infusion_duration_h: float,
    time_from_infusion_end_to_peak_draw_h: float, current_interval_h: float, # Use CURRENT interval here
    dose: float, weight: float # Added parameters for Vd calculation
) -> dict | None:
    """
    Calculates PK parameters using peak and trough levels (Sawchuk-Zaske method).
    Uses CURRENT dosing interval. Calculates individual Vd using image formula (e).
    Calculates Expected Cmax/Cmin using image formulas (f, g).
    """
    # Time between the peak draw time and the trough draw time (assuming trough is pre-next dose)
    time_between_peak_draw_and_trough_draw_h = current_interval_h - infusion_duration_h - time_from_infusion_end_to_peak_draw_h

    if time_between_peak_draw_and_trough_draw_h <= 0 or c_peak_measured <= 0 or c_trough <= 0 or c_peak_measured <= c_trough:
        logging.warning(f"Invalid inputs for peak/trough calculation: TimeBetweenSamples={time_between_peak_draw_and_trough_draw_h:.2f}, Cpeak={c_peak_measured:.2f}, Ctrough={c_trough:.2f}.")
        return None
    try:
        # Calculate Ke using the two measured concentrations
        ke = math.log(c_peak_measured / c_trough) / time_between_peak_draw_and_trough_draw_h
        if ke <= 0:
             logging.warning(f"Calculated Ke is zero or negative ({ke:.4f}) in peak/trough method.")
             return None
        # Calculate t1/2
        half_life_h = math.log(2) / ke
        # Calculate Cmax extrapolated to end of infusion
        c_max_extrapolated = c_peak_measured * math.exp(ke * time_from_infusion_end_to_peak_draw_h)
        # Use measured trough as actual Cmin (pre-dose)
        c_min_actual = c_trough

        # Calculate individual Vd using the simplified absolute Vd formula
        # Vd(L) = Dose(mg) / [ Cmax_extrapolated(mg/L) * (1 - exp(-Ke*T)) ]
        term_exp_vd = 1 - math.exp(-ke * current_interval_h)
        if term_exp_vd <= 0 or c_max_extrapolated <= 0:
             logging.warning(f"Invalid terms for individual Vd calculation: Cmax_extrap={c_max_extrapolated:.2f}, term_exp={term_exp_vd:.4f}")
             return None
        vd_ind = dose / (c_max_extrapolated * term_exp_vd)
        if vd_ind <= 0:
            logging.warning("Individual Vd calculation failed or resulted in non-positive value. Cannot proceed.")
            return None

        # AUC calculation using Dose/CL method
        cl_ind = ke * vd_ind
        if cl_ind <= 0:
             logging.warning(f"Calculated individual CL is zero or negative (Ke={ke:.4f}, Vd_ind={vd_ind:.1f}). Cannot calculate AUC.")
             return None
        auc_interval_total = dose / cl_ind
        auc24 = auc_interval_total * (24 / current_interval_h)

        # Calculate expected levels for the *current* dose using calculated individual Vd
        expected_cmax = calculate_expected_cmax(dose, vd_ind, current_interval_h, ke)
        expected_cmin = calculate_expected_cmin(expected_cmax, ke, current_interval_h)

        logging.info(f"Calculated PK Params (peak/trough): Ke={ke:.4f}, t1/2={half_life_h:.1f}, Cmax_extrap={c_max_extrapolated:.1f}, Cmin_actual={c_min_actual:.1f}, Vd_ind={vd_ind:.1f} L, CL_ind={cl_ind:.2f} L/h, AUC_interval={auc_interval_total:.1f}, AUC24={auc24:.1f}, ExpCmax={expected_cmax:.1f}, ExpCmin={expected_cmin:.1f}")
        return {
            'ke': ke,
            't_half': half_life_h,
            'Cmax_extrapolated': c_max_extrapolated, # Cmax at end of infusion
            'Cmin_actual': c_min_actual, # Measured trough (assumed pre-dose)
            'Vd_individual': vd_ind, # Vd calculated from formula (e) - absolute L
            'CL_individual': cl_ind, # Individual Clearance
            'AUC_interval': auc_interval_total, # AUC over the dosing interval
            'AUC24': auc24, # AUC extrapolated to 24h
            'Expected_Cmax': expected_cmax, # Calculated from formula (f)
            'Expected_Cmin': expected_cmin  # Calculated from formula (g)
        }
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating PK parameters (peak/trough): {e}.")
        return None

# --- 8. LLM INTERPRETATION (Refined Prompt V2 - Keep as is for now) ---
def interpret(crcl: float, pk_results: dict, target_level_desc: str, clinical_notes: str) -> str:
    """Generates interpretation and recommendations using the RAG chain, including clinical notes."""
    if qa_chain is None:
        logging.warning("QA chain not loaded. Skipping interpretation.")
        return "Interpretation unavailable: RAG system failed to load. Check API key, PDF file, and library installations."

    # Extract relevant values safely
    current_interval_info = pk_results.get('Current Dosing Interval', 'N/A')
    time_since_dose_info = pk_results.get('Time Since Last Dose (at Trough Draw)', 'N/A') # Only for trough mode
    measured_trough = pk_results.get('Measured Trough', pk_results.get('Cmin_actual', 'N/A')) # Get trough from either mode
    calculated_auc = pk_results.get('Estimated AUC24', pk_results.get('AUC24', 'N/A')) # Get AUC from either mode
    calculated_thalf = pk_results.get('Estimated t½', pk_results.get('t_half', 'N/A')) # Get t_half from either mode
    expected_cmax = pk_results.get('Expected_Cmax', 'N/A')
    expected_cmin = pk_results.get('Expected_Cmin', 'N/A')
    trough_status = pk_results.get('Trough Status vs Target', 'N/A') # Get code-checked status
    auc_status = pk_results.get('AUC Status vs Target', 'N/A') # Get code-checked status


    # Format values for prompt, handling N/A
    measured_trough_str = f"{measured_trough:.1f} mg/L" if isinstance(measured_trough, (float, int)) else str(measured_trough)
    calculated_auc_str = f"{calculated_auc:.1f} mg·h/L" if isinstance(calculated_auc, (float, int)) else str(calculated_auc)
    calculated_thalf_str = f"{calculated_thalf:.1f} h" if isinstance(calculated_thalf, (float, int)) else str(calculated_thalf)
    expected_cmax_str = f"{expected_cmax:.1f} mg/L" if isinstance(expected_cmax, (float, int)) else str(expected_cmax)
    expected_cmin_str = f"{expected_cmin:.1f} mg/L" if isinstance(expected_cmin, (float, int)) else str(expected_cmin)

    time_context = f"Consider the actual time the trough was drawn ({time_since_dose_info})." if time_since_dose_info != 'N/A' else ""
    notes_context = f"\n\nClinical Notes Provided:\n{clinical_notes}" if clinical_notes else ""
    half_life_info = f"Calculated half-life is {calculated_thalf_str}." if calculated_thalf_str != 'N/A' else ""

    # Add the objective status check result to the context for the LLM
    objective_status_context = f"\nObjective Status Check:\n- Trough Status: {trough_status}\n- AUC Status: {auc_status}\n"

    prompt = f"""
Context: You are a clinical pharmacokinetics expert interpreting vancomycin TDM results based on the Clinical Pharmacokinetics Pharmacy Handbook, 2nd edition.

Patient Information Summary:
- Estimated Creatinine Clearance (CrCl): {crcl:.1f} mL/min
- Selected Therapeutic Target: {target_level_desc}
{notes_context}

Current Regimen & Monitoring Results:
{pk_results}
{objective_status_context}
Task: Provide a concise, structured interpretation and recommendation, considering all provided information. Structure your response clearly under the following headings: Assessment, Recommendation, Rationale, Follow-up. **Ensure your assessment statements are strictly consistent with the numerical data AND the Objective Status Check provided above.**

1.  **Assessment:**
    * **Trough Level:** State the Measured Trough ({measured_trough_str}), the target trough range from '{target_level_desc}', and the objective status ({trough_status}).
    * **AUC Level:** State the Calculated AUC ({calculated_auc_str}), the target AUC range from '{target_level_desc}', and the objective status ({auc_status}).
    * **Overall Goal:** Based *only* on the objective status check, is the overall therapeutic goal currently being met?
    * **Clearance:** Is the patient clearing the drug as expected based on CrCl? {time_context}
    * **Interval Appropriateness:** Is the current dosing interval ({current_interval_info}) appropriate given the calculated half-life ({half_life_info})?
    * **Clinical Alignment:** Does the clinical picture (from notes, if provided) align with the levels? (e.g., therapeutic levels but ongoing fever?)
    * **Expected vs. Measured:** If available, compare measured levels (Trough: {measured_trough_str}) to the 'Expected_Cmin' ({expected_cmin_str}) calculated for the current regimen. Are they similar or significantly different? (Also consider Cmax if applicable: Measured Peak/Extrapolated vs Expected Cmax {expected_cmax_str}).

2.  **Recommendation:**
    * Clearly state whether a dose/interval change is needed based on the Assessment (especially the objective status).
    * If adjustment is needed:
        * Suggest a specific dose (rounded to nearest 250mg) AND a standard clinical interval (e.g., q8h, q12h, q24h). Reference the 'Suggested New Dose' field from the results if available and appropriate.
        * Prioritize maintaining the current interval unless the half-life or clinical context strongly suggests a change. If changing the interval, explain why.
    * If no change needed, state the current regimen is appropriate.
    * **Prioritization:** If clinical notes suggest poor response despite 'therapeutic' levels, prioritize recommending a dose increase or broader clinical review over simply stating levels are adequate. Conversely, if notes indicate clinical improvement despite slightly low levels, consider if maintaining the current dose is acceptable.

3.  **Rationale:**
    * Explain *why* the recommendation is being made.
    * Link the decision directly to the specific findings in the Assessment (objective status, levels vs. target, AUC vs. target, Ke, Vd, CrCl, half-life vs. interval, clinical notes). Be specific about which parameters justify the recommendation.

4.  **Follow-up:**
    * Suggest *specific* monitoring. When should the next level be drawn (e.g., trough before 3rd/4th dose of new regimen)?
    * Should renal function (SCr) be monitored more frequently?
    * Mention any other relevant clinical monitoring based on the notes.

Use the provided guideline knowledge. Be specific and clinically oriented. Ensure the response is complete and directly supported by the provided data and objective status check.
"""
    logging.info(f"Generating interpretation with prompt:\n{prompt}")
    try:
        # Use invoke for newer Langchain versions if available
        if hasattr(qa_chain, 'invoke'):
            response = qa_chain.invoke({"query": prompt})
            # Handle both dict and string responses from invoke/run
            if isinstance(response, dict) and 'result' in response:
                response_text = response['result']
            elif isinstance(response, str):
                 response_text = response
            else:
                 response_text = str(response) # Fallback
                 logging.warning(f"Unexpected response type from qa_chain.invoke: {type(response)}. Converted to string.")
        # Fallback for older Langchain versions or different chain types
        elif hasattr(qa_chain, 'run'):
             response_text = qa_chain.run(prompt) # type: ignore
        else:
             logging.error("QA chain object does not have a recognized execution method ('invoke' or 'run').")
             return "Interpretation failed: Could not execute the RAG chain."

        logging.info(f"LLM Interpretation received (length: {len(response_text)}).")
        return response_text
    except Exception as e:
        logging.error(f"Error running RAG QA chain for interpretation: {e}")
        logging.exception("RAG Chain execution error:")
        return f"Interpretation failed: Error communicating with the RAG system ({e}). Check API key and model availability."

# --- 9. HELPER FUNCTION for Target Status Check ---
def get_target_ranges(target_desc: str) -> tuple[tuple[float|None, float|None], tuple[float|None, float|None]]:
    """Parses the target description string to get numerical trough and AUC ranges."""
    trough_range = (None, None)
    auc_range = (None, None)

    # Extract Trough range (e.g., ~10-15 mg/L or ~15-20 mg/L)
    trough_match = re.search(r'Trough\s*~\s*([\d.]+)\s*-\s*([\d.]+)', target_desc)
    if trough_match:
        try:
            trough_range = (float(trough_match.group(1)), float(trough_match.group(2)))
        except ValueError:
            logging.warning(f"Could not parse trough range from: {trough_match.group(0)}")

    # Extract AUC range (e.g., 400-600 mg·h/L or >600 mg·h/L)
    auc_match_range = re.search(r'AUC24\s*([\d.]+)\s*-\s*([\d.]+)', target_desc)
    auc_match_greater = re.search(r'AUC24\s*>\s*([\d.]+)', target_desc)

    if auc_match_range:
        try:
            auc_range = (float(auc_match_range.group(1)), float(auc_match_range.group(2)))
        except ValueError:
            logging.warning(f"Could not parse AUC range from: {auc_match_range.group(0)}")
    elif auc_match_greater:
        try:
            # Represent ">X" as (X, infinity) - use None for infinity
            auc_range = (float(auc_match_greater.group(1)), None)
        except ValueError:
            logging.warning(f"Could not parse AUC lower bound from: {auc_match_greater.group(0)}")

    return trough_range, auc_range

def check_target_status(value: float | None, target_range: tuple[float|None, float|None]) -> str:
    """Compares a value against a target range tuple (min, max). Handles None for infinity."""
    if value is None or value <= 0:
        return "N/A (Invalid Value)"
    if target_range == (None, None):
        return "N/A (Invalid Target)"

    lower_bound, upper_bound = target_range

    if lower_bound is not None and upper_bound is not None: # Range like 10-15 or 400-600
        if value < lower_bound:
            return f"BELOW ({lower_bound}-{upper_bound})"
        elif value > upper_bound:
            return f"ABOVE ({lower_bound}-{upper_bound})"
        else:
            return f"WITHIN ({lower_bound}-{upper_bound})"
    elif lower_bound is not None and upper_bound is None: # Range like >600
        # Add a small tolerance for floating point comparisons, e.g., 600.01 is considered WITHIN
        tolerance = 1e-6
        if value <= lower_bound - tolerance:
             return f"BELOW (>{lower_bound})"
        else: # value > lower_bound
             return f"WITHIN (>{lower_bound})"
    # Add cases for <X or other range types if needed
    else:
        return "N/A (Unhandled Target Format)"

# --- 10. STREAMLIT UI ---

# Display initial errors if setup failed
if not api_key_loaded:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable (OPENAI_API_KEY) for local testing. App functionality is limited.")
    st.stop()
if not libraries_loaded:
    st.error(f"Failed to import required libraries. Please ensure all dependencies are installed (streamlit, pypdf, langchain, langchain-community, langchain-openai, faiss-cpu, tiktoken). App functionality is limited.")
    st.stop()
if qa_chain is None:
     st.error("Critical Error: The RAG guideline interpretation system could not be loaded (PDF not found or API/library issue). Interpretation features will be disabled.")
     # Allow app to run for calculations, but interpretation won't work.

# Main App Title
st.title("🧪 TDM-AID by HTAR (Vancomycin module)")
st.markdown("Calculates PK parameters and provides interpretation based on Clinical Pharmacokinetics Pharmacy Handbook (2nd edition)")

# Sidebar for inputs
st.sidebar.header("⚙️ Mode & Target")
mode = st.sidebar.radio("Select Calculation Mode:", ["Initial Dose", "Trough-Only", "Peak & Trough"])

target_level_desc = st.sidebar.selectbox(
    "Select Therapeutic Target:",
    options=[
        "Empirical (Target AUC24 400-600 mg·h/L; Trough ~10-15 mg/L)",
        "Definitive/Severe (Target AUC24 >600 mg·h/L; Trough ~15-20 mg/L)"
    ],
    index=0, # Default to Empirical
    help="Select the desired therapeutic goal based on infection type and severity."
)

# Determine numerical targets for calculations based on selection
target_trough_range_calc, target_auc_range_calc = get_target_ranges(target_level_desc)

# Define the single value to *target* for dose calculation (e.g., midpoint or lower bound)
target_trough_for_dose_calc = 0.0
target_auc_for_dose_calc = 0.0

if "Empirical" in target_level_desc:
    target_trough_for_dose_calc = 12.5 # Midpoint of 10-15
    target_auc_for_dose_calc = 500 # Midpoint target for dose calc
elif "Definitive/Severe" in target_level_desc:
    target_trough_for_dose_calc = 17.5 # Midpoint of 15-20
    target_auc_for_dose_calc = 600 # Lower end target for dose calc


st.sidebar.header("👤 Patient Information")
col_pid, col_ward = st.sidebar.columns(2)
with col_pid:
    pid = st.text_input("Patient ID", placeholder="e.g., MRN12345")
with col_ward:
    ward = st.text_input("Ward/Unit", placeholder="e.g., ICU, 5W")

col_age, col_wt = st.sidebar.columns(2)
with col_age:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=65, step=1)
with col_wt:
    wt = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")

col_scr, col_fem = st.sidebar.columns(2)
with col_scr:
    scr_umol = st.number_input(
        "SCr (µmol/L)",
        min_value=10.0, max_value=2000.0, value=88.0, step=1.0, format="%.0f",
        help="Serum Creatinine in µmol/L. Will be converted to mg/dL for CrCl calculation."
    )
with col_fem:
    fem = st.checkbox("Female", value=False)

clinical_notes = st.sidebar.text_area(
    "Clinical Notes (Optional)",
    placeholder="Enter relevant clinical context, e.g., 'Persistent fever >38.5C', 'Worsening renal function', 'Source control achieved', 'Concurrent nephrotoxins'...",
    height=100,
    help="Provide brief clinical context to aid interpretation."
)

# --- Helper function to build a downloadable report ---
def build_report(lines: list[str], scr_umol_report: float, clinical_notes_report: str) -> str:
    """Formats patient info and results into a text report."""
    scr_mgdl_report = convert_scr_to_mgdl(scr_umol_report)
    # Calculate population Vd for the header
    vd_pop_report = calculate_population_vd(wt, age) if wt > 0 else 0.0
    crcl_report = calculate_crcl(age, wt, scr_mgdl_report, fem) if age > 0 and wt > 0 and scr_mgdl_report > 0 else 0.0

    hdr = [
        "--- Vancomycin TDM Report ---",
        f"Patient ID: {pid if pid else 'N/A'}",
        f"Ward/Unit: {ward if ward else 'N/A'}",
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Patient Info: Age={age} yrs, Weight={wt} kg, Sex={'Female' if fem else 'Male'}, SCr={scr_umol_report:.0f} µmol/L",
        f"Estimated CrCl (using SCr ~{scr_mgdl_report:.2f} mg/dL): {crcl_report:.1f} mL/min" if crcl_report is not None else "N/A (Invalid Input)",
        # Report population Vd estimate in header
        f"Estimated Population Vd (0.7 L/kg): {vd_pop_report:.1f} L" if vd_pop_report is not None else "N/A (Invalid Input)",
        f"Selected Target: {target_level_desc}",
    ]
    if clinical_notes_report:
        hdr.extend([
            "--- Clinical Notes Provided ---",
            clinical_notes_report.replace('\n', '\n  ') # Indent notes slightly
        ])
    hdr.append("--- Results & Interpretation ---")
    # Use a more robust way to format lines from dict for report
    report_body = []
    for k, v in lines.items(): # Assuming lines is now the pk_results dict
         report_body.append(f"{k}: {v}")

    # Find interpretation if it exists and add it formatted
    interpretation_key = "RAG Interpretation"
    interpretation_content = lines.get(interpretation_key, None)
    if interpretation_content:
         report_body.append(f"--- {interpretation_key} ---")
         report_body.append(interpretation_content.replace('\n', '\n  '))
         # Remove it from the main list if it was added separately
         if interpretation_key in lines:
             # This part might need adjustment depending on how interpretation is added
             pass # Logic to avoid duplication if needed

    return "\n".join(hdr + report_body)

# --- Main Area Logic ---
results_container = st.container()

# --- SCr Conversion (Do it once here after input) ---
scr_mgdl = convert_scr_to_mgdl(scr_umol)

if mode == "Initial Dose":
    with results_container:
        st.subheader("🚀 Initial Loading Dose Calculation")
        st.markdown("Calculates a one-time loading dose based on patient weight and estimated renal function (CrCl). Uses ~25 mg/kg, rounded.")
        calc_button = st.button("Calculate Loading Dose", key="calc_initial")

        if calc_button:
            if wt <= 0 or age <= 0 or scr_mgdl <= 0:
                st.warning("Please enter valid patient information (Age > 0, Weight > 0, SCr > 0).")
            else:
                with st.spinner("Calculating..."):
                    initial_dose_mg = calculate_initial_dose(age, wt, scr_mgdl, fem)
                    crcl_calc = calculate_crcl(age, wt, scr_mgdl, fem)
                    st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                    st.success(f"Recommended Initial Loading Dose: **{initial_dose_mg} mg** (Administer once)")

                    report_dict = { # Use dict for report
                        "Mode": "Initial Loading Dose",
                        "Calculated Loading Dose": f"{initial_dose_mg} mg (one time)"
                    }
                    report_data = build_report(report_dict, scr_umol, clinical_notes)
                    st.download_button(
                        label="📥 Download Report (.txt)",
                        data=report_data,
                        file_name=f"{pid or 'patient'}_vanco_initial_dose_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

elif mode == "Trough-Only":
    st.sidebar.header("💉 Trough-Only Monitoring")
    st.sidebar.markdown("Enter details about the *current* regimen, the **current dosing interval**, and the measured trough level.")
    dose_int_current = st.sidebar.number_input("Current Dose per Interval (mg)", min_value=250, step=250, value=1000)
    current_interval_h = st.sidebar.selectbox(
        "Current Dosing Interval (hours)",
        options=[6, 8, 12, 18, 24, 36, 48], index=2, format_func=lambda x: f"q{x}h",
        help="Select the patient's current dosing frequency (e.g., q12h)."
    )
    dose_time = st.sidebar.time_input("Time of Last Dose Administered", value=time(8, 0), step=timedelta(minutes=15))
    sample_time = st.sidebar.time_input("Time Trough Level Drawn", value=time(19, 30), step=timedelta(minutes=15), help="Actual time the level was drawn.")
    trough_measured = st.sidebar.number_input("Measured Trough Level (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f")

    with results_container:
        st.subheader("📉 Trough-Only Analysis")
        st.markdown("Estimates PK parameters and suggests dose adjustments based on a single trough level and the **current** dosing interval.")

        time_since_last_dose_h = hours_diff(dose_time, sample_time)
        if time_since_last_dose_h > 0:
            st.info(f"Trough drawn **{time_since_last_dose_h:.2f} hours** after the last dose. Current interval: **q{current_interval_h}h**.")
        else:
            st.warning("Sample time must be after the last dose time.")

        calc_button = st.button("Run Trough-Only Analysis", key="run_trough")

        if calc_button:
             if dose_int_current <= 0 or trough_measured <= 0 or time_since_last_dose_h <= 0 or wt <= 0 or age <= 0 or scr_mgdl <= 0 or current_interval_h <= 0:
                 st.warning("Please ensure Dose (>0), Measured Trough (>0), Weight (>0), Age (>0), SCr (>0), Current Interval (>0), and a valid time difference (>0h) are entered.")
             else:
                 with st.spinner("Analyzing Trough Level..."):
                     crcl_calc = calculate_crcl(age, wt, scr_mgdl, fem)
                     # Use population Vd for trough-only estimate
                     vd_calc = calculate_population_vd(wt, age) # Use population Vd
                     ke_calc = calculate_ke_trough(dose_int_current, vd_calc, trough_measured, time_since_last_dose_h)

                     interpretation_text = "N/A (Calculation Error or RAG disabled)"
                     pk_results = {"Error": "Initial calculation failed or Ke invalid."}
                     new_dose_calc = 0 # Initialize
                     auc24_calc = 0.0 # Initialize
                     t_half_calc = 0.0 # Initialize
                     trough_status = "N/A" # Initialize
                     auc_status = "N/A" # Initialize


                     if ke_calc > 0 and vd_calc > 0:
                         t_half_calc = math.log(2) / ke_calc
                         auc24_calc = calculate_auc24_trough(dose_int_current, ke_calc, vd_calc, current_interval_h)
                         # Calculate new dose targeting the midpoint trough for the selected range
                         new_dose_calc = calculate_new_dose_trough(ke_calc, vd_calc, current_interval_h, target_trough=target_trough_for_dose_calc)

                         # --- Perform Status Check ---
                         trough_status = check_target_status(trough_measured, target_trough_range_calc)
                         auc_status = check_target_status(auc24_calc, target_auc_range_calc)
                         # --- End Status Check ---

                         st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                         col1, col2, col3 = st.columns(3)
                         # Display population Vd used
                         col1.metric(label="Est. Population Vd", value=f"{vd_calc:.1f} L")
                         col2.metric(label="Estimated Ke", value=f"{ke_calc:.4f} h⁻¹")
                         col3.metric(label="Estimated t½", value=f"{t_half_calc:.1f} h")
                         st.metric(label=f"Estimated AUC₂₄ (based on q{current_interval_h}h)", value=f"{auc24_calc:.1f} mg·h/L", help=f"Target: {target_level_desc.split(';')[0]}") # Show AUC target part
                         st.metric(label=f"Suggested New Dose (for target ~{target_trough_for_dose_calc} mg/L)", value=f"{new_dose_calc} mg q{current_interval_h}h" if new_dose_calc > 0 else "N/A", help=f"Rounded dose for the current q{current_interval_h}h interval to achieve target trough.")

                         pk_results = {
                             'Calculation Mode': 'Trough-Only',
                             'Current Dose': f"{dose_int_current} mg",
                             'Current Dosing Interval': f"q{current_interval_h}h",
                             'Time Since Last Dose (at Trough Draw)': f"{time_since_last_dose_h:.2f} h",
                             'Measured Trough': f"{trough_measured:.1f} mg/L",
                             'Trough Status vs Target': trough_status, # Add status
                             'Estimated Population Vd': f"{vd_calc:.1f} L",
                             'Estimated Ke': f"{ke_calc:.4f} h⁻¹",
                             'Estimated t½': f"{t_half_calc:.1f} h",
                             'Estimated AUC24 (for current interval)': f"{auc24_calc:.1f} mg·h/L",
                             'AUC Status vs Target': auc_status, # Add status
                             'Suggested New Dose (for target trough)': f"{new_dose_calc} mg q{current_interval_h}h" if new_dose_calc > 0 else "N/A"
                         }
                         # Only run interpretation if RAG chain loaded
                         if qa_chain:
                              interpretation_text = interpret(crcl_calc, pk_results, target_level_desc, clinical_notes)
                         else:
                              interpretation_text = "Interpretation disabled: RAG system not loaded."

                     else:
                         st.error("Could not calculate valid Ke and/or Vd. Cannot proceed with AUC/New Dose calculation or interpretation.")
                         pk_results = { # Update results to show failure
                             'Calculation Mode': 'Trough-Only',
                             'Current Dose': f"{dose_int_current} mg",
                             'Current Dosing Interval': f"q{current_interval_h}h",
                             'Time Since Last Dose (at Trough Draw)': f"{time_since_last_dose_h:.2f} h",
                             'Measured Trough': f"{trough_measured:.1f} mg/L",
                             'Trough Status vs Target': check_target_status(trough_measured, target_trough_range_calc), # Still check trough
                             'Error': 'Failed to calculate valid Ke/Vd from trough level.',
                             'Estimated Population Vd': f"{vd_calc:.1f} L" if vd_calc > 0 else "N/A",
                             'Estimated Ke': "Invalid",
                             'Estimated t½': "N/A",
                             'Estimated AUC24': "N/A",
                             'AUC Status vs Target': "N/A (Calculation Failed)",
                             'Suggested New Dose': "N/A"
                         }
                         interpretation_text = "Interpretation unavailable due to calculation error."


                     # --- Display Status Check before Interpretation ---
                     st.divider() # Add visual separation
                     st.subheader("🎯 Target Status Check (Code-Based)")
                     if 'Error' not in pk_results:
                         st.markdown(f" - **Measured Trough:** {trough_measured:.1f} mg/L -> **{trough_status}**")
                         st.markdown(f" - **Estimated AUC₂₄:** {auc24_calc:.1f} mg·h/L -> **{auc_status}**")
                     else:
                         # Show status even if calc failed partially
                         st.markdown(f" - **Measured Trough:** {trough_measured:.1f} mg/L -> **{pk_results.get('Trough Status vs Target','N/A')}**")
                         st.markdown(f" - **Estimated AUC₂₄:** N/A -> **{pk_results.get('AUC Status vs Target','N/A')}**")
                         st.warning("Status based on available data; AUC calculation failed.")
                     st.divider()
                     # --- End Display Status Check ---

                     st.subheader("💬 Interpretation & Recommendation (AI-Generated)")
                     st.markdown(interpretation_text) # Display interpretation or error message

                     # Prepare dict for report
                     report_dict = pk_results.copy()
                     if interpretation_text not in ["N/A (Calculation Error or RAG disabled)", "Interpretation disabled: RAG system not loaded.", "Interpretation unavailable due to calculation error."]:
                         report_dict["RAG Interpretation"] = interpretation_text

                     report_data = build_report(report_dict, scr_umol, clinical_notes)
                     st.download_button(
                         label="📥 Download Report (.txt)",
                         data=report_data,
                         file_name=f"{pid or 'patient'}_vanco_trough_{datetime.now().strftime('%Y%m%d')}.txt",
                         mime="text/plain"
                     )

elif mode == "Peak & Trough":
    st.sidebar.header("📈 Peak & Trough Monitoring")
    st.sidebar.markdown("Enter the **dose administered**, the **current dosing interval**, infusion times, and both peak and trough levels.")
    dose_for_levels = st.sidebar.number_input("Dose Administered (mg)", min_value=250, step=250, value=1000, help="Dose given before levels drawn.")
    current_interval_h_pt = st.sidebar.selectbox(
        "Current Dosing Interval (hours)",
        options=[6, 8, 12, 18, 24, 36, 48], index=2, format_func=lambda x: f"q{x}h", key="interval_pt",
        help="Select the patient's current dosing frequency."
    )
    infusion_start_time = st.sidebar.time_input("Infusion Start Time", value=time(8, 0), step=timedelta(minutes=15))
    infusion_end_time = st.sidebar.time_input("Infusion End Time", value=time(9, 0), step=timedelta(minutes=15), help="End time of infusion.")
    peak_sample_time = st.sidebar.time_input("Peak Sample Time", value=time(10, 0), step=timedelta(minutes=15), help="Time peak level drawn.")
    trough_sample_time = st.sidebar.time_input("Trough Sample Time", value=time(19, 30), step=timedelta(minutes=15), help="Time trough level drawn (usually immediately before next dose).")
    c_trough_measured = st.sidebar.number_input("Measured Trough (Cmin) (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f")
    c_peak_measured = st.sidebar.number_input("Measured Peak (Cpeak) (mg/L)", min_value=0.1, max_value=200.0, value=30.0, step=0.1, format="%.1f", help="Level measured at 'Peak Sample Time'.")

    with results_container:
        st.subheader("📊 Peak & Trough Analysis (Sawchuk-Zaske)")
        st.markdown("Calculates individual PK parameters using measured levels and the **current** dosing interval.")

        infusion_duration_h = hours_diff(infusion_start_time, infusion_end_time)
        time_from_infusion_end_to_peak_draw_h = hours_diff(infusion_end_time, peak_sample_time)

        valid_times = True
        if infusion_duration_h <= 0:
             st.warning("Infusion End Time must be after Infusion Start Time.")
             valid_times = False
        if time_from_infusion_end_to_peak_draw_h < 0:
             st.warning("Peak Sample Time must be after Infusion End Time.")
             valid_times = False

        if valid_times:
             st.info(f"Calculated Durations: Infusion={infusion_duration_h:.2f}h | Delay to Peak Draw={time_from_infusion_end_to_peak_draw_h:.2f}h. Current Interval: **q{current_interval_h_pt}h**.")
             time_between_peak_draw_and_trough_draw_h_check = current_interval_h_pt - infusion_duration_h - time_from_infusion_end_to_peak_draw_h
             if time_between_peak_draw_and_trough_draw_h_check <= 0:
                  st.warning("Timing Error: The interval is too short for the specified infusion and peak draw times. Ke calculation will likely fail.")

        calc_button = st.button("Run Peak & Trough Analysis", key="run_peak_trough")

        if calc_button:
            if not valid_times:
                st.error("Please correct the infusion/peak time errors before running analysis.")
            elif c_trough_measured <= 0 or c_peak_measured <= 0:
                st.warning("Measured Peak and Trough levels must be > 0.")
            elif c_peak_measured <= c_trough_measured:
                st.warning("Measured Peak level must be > Measured Trough level.")
            elif dose_for_levels <= 0:
                 st.warning("Please enter the dose administered (> 0 mg).")
            elif wt <= 0 or age <= 0 or scr_mgdl <= 0:
                 st.warning("Please enter valid patient information (Age > 0, Weight > 0, SCr > 0).")
            else:
                with st.spinner("Analyzing Peak & Trough Levels..."):
                    crcl_calc = calculate_crcl(age, wt, scr_mgdl, fem)
                    # Pass dose and weight for individual Vd calculation
                    pk_params = calculate_pk_params_peak_trough(
                        c_trough=c_trough_measured, c_peak_measured=c_peak_measured,
                        infusion_duration_h=infusion_duration_h,
                        time_from_infusion_end_to_peak_draw_h=time_from_infusion_end_to_peak_draw_h,
                        current_interval_h=current_interval_h_pt,
                        dose=dose_for_levels, # Pass dose
                        weight=wt # Pass weight
                    )

                    interpretation_text = "N/A (Calculation Error or RAG disabled)"
                    pk_results = {"Error": "Failed to calculate PK parameters from Peak & Trough data."}
                    new_dose_suggestion = "N/A" # Initialize
                    trough_status = "N/A" # Initialize
                    auc_status = "N/A" # Initialize

                    if pk_params:
                        ke_ind = pk_params['ke']
                        thalf_ind = pk_params['t_half']
                        cmax_ind = pk_params['Cmax_extrapolated']
                        cmin_ind = pk_params['Cmin_actual'] # This is the measured trough
                        auc24_ind = pk_params['AUC24']
                        vd_ind = pk_params['Vd_individual']
                        cl_ind = pk_params['CL_individual']
                        exp_cmax_ind = pk_params['Expected_Cmax']
                        exp_cmin_ind = pk_params['Expected_Cmin']

                        # --- Perform Status Check ---
                        trough_status = check_target_status(cmin_ind, target_trough_range_calc)
                        auc_status = check_target_status(auc24_ind, target_auc_range_calc)
                        # --- End Status Check ---

                        st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Individual Vd", value=f"{vd_ind:.1f} L" if vd_ind > 0 else "N/A", help="Calculated using Dose / [Cmax_extrap × (1−e−KeT)]")
                        col2.metric(label="Individual Ke", value=f"{ke_ind:.4f} h⁻¹")
                        col3.metric(label="Individual t½", value=f"{thalf_ind:.1f} h")

                        col4, col5 = st.columns(2)
                        col4.metric(label="Est. Cmax (End of Infusion)", value=f"{cmax_ind:.1f} mg/L")
                        col5.metric(label="Measured Cmin (Trough)", value=f"{cmin_ind:.1f} mg/L")

                        col6, col7 = st.columns(2)
                        col6.metric(label="Expected Cmax (for current dose)", value=f"{exp_cmax_ind:.1f} mg/L" if exp_cmax_ind > 0 else "N/A", help="Predicted Cmax based on individual Vd/Ke")
                        col7.metric(label="Expected Cmin (for current dose)", value=f"{exp_cmin_ind:.1f} mg/L" if exp_cmin_ind > 0 else "N/A", help="Predicted Cmin based on individual Vd/Ke")

                        st.metric(label=f"Individual AUC₂₄ (based on q{current_interval_h_pt}h)", value=f"{auc24_ind:.1f} mg·h/L", help=f"Target: {target_level_desc.split(';')[0]}") # Show AUC target part

                        # --- Suggested Dose Calculation using Individual Parameters ---
                        if ke_ind > 0 and vd_ind > 0 and current_interval_h_pt > 0 and cl_ind > 0 and target_auc_for_dose_calc > 0:
                             try:
                                 # Target Dose = Target AUC_interval * CL_ind
                                 target_auc_interval = target_auc_for_dose_calc * (current_interval_h_pt / 24.0)
                                 new_dose_raw = target_auc_interval * cl_ind
                                 new_dose_rounded = round_dose(new_dose_raw)
                                 new_dose_suggestion = f"{new_dose_rounded} mg q{current_interval_h_pt}h"
                                 st.metric(label=f"Suggested Dose (for Target AUC ~{target_auc_for_dose_calc})", value=new_dose_suggestion, help=f"Calculated to achieve target AUC ({target_auc_for_dose_calc}) using individual CL and current interval.")
                             except Exception as dose_calc_err:
                                 st.warning(f"Could not calculate suggested dose: {dose_calc_err}")
                                 new_dose_suggestion = "N/A (Calculation Error)"
                        else:
                             st.warning("Cannot suggest new dose without valid Individual Ke, Vd, Interval, and Target AUC.")
                             new_dose_suggestion = "N/A (Missing Parameters)"


                        pk_results = {
                            'Calculation Mode': 'Peak & Trough',
                            'Dose Administered': f"{dose_for_levels} mg",
                            'Current Dosing Interval': f"q{current_interval_h_pt}h",
                            'Infusion Duration': f"{infusion_duration_h:.2f} h",
                            'Time to Peak Draw (post-infusion)': f"{time_from_infusion_end_to_peak_draw_h:.2f} h",
                            'Measured Peak (at draw time)': f"{c_peak_measured:.1f} mg/L",
                            'Measured Trough (Cmin)': f"{c_trough_measured:.1f} mg/L",
                            'Trough Status vs Target': trough_status, # Add status
                            'Individual Vd (Formula e)': f"{vd_ind:.1f} L" if vd_ind > 0 else "N/A",
                            'Individual Ke': f"{ke_ind:.4f} h⁻¹",
                            'Individual t½': f"{thalf_ind:.1f} h",
                            'Individual CL': f"{cl_ind:.2f} L/h",
                            'Est. Cmax (End of Infusion)': f"{cmax_ind:.1f} mg/L",
                            'Expected Cmax (Formula f)': f"{exp_cmax_ind:.1f} mg/L" if exp_cmax_ind > 0 else "N/A",
                            'Expected Cmin (Formula g)': f"{exp_cmin_ind:.1f} mg/L" if exp_cmin_ind > 0 else "N/A",
                            'AUC_interval': f"{pk_params['AUC_interval']:.1f} mg·h/L",
                            'Individual AUC24 (for current interval)': f"{auc24_ind:.1f} mg·h/L",
                            'AUC Status vs Target': auc_status, # Add status
                            'Suggested New Dose (for target AUC)': new_dose_suggestion
                        }
                        # Only run interpretation if RAG chain loaded
                        if qa_chain:
                             interpretation_text = interpret(crcl_calc, pk_results, target_level_desc, clinical_notes)
                        else:
                             interpretation_text = "Interpretation disabled: RAG system not loaded."

                    else:
                        st.error("Failed to calculate PK parameters from Peak & Trough data. Check input values and timings.")
                        # Keep pk_results as the initial error dictionary
                        pk_results['Dose Administered'] = f"{dose_for_levels} mg" # Add context
                        pk_results['Current Dosing Interval'] = f"q{current_interval_h_pt}h"
                        pk_results['Measured Peak'] = f"{c_peak_measured:.1f} mg/L"
                        pk_results['Measured Trough'] = f"{c_trough_measured:.1f} mg/L"
                        pk_results['Trough Status vs Target'] = check_target_status(c_trough_measured, target_trough_range_calc) # Still check trough
                        pk_results['AUC Status vs Target'] = "N/A (Calculation Failed)"
                        interpretation_text = "Interpretation unavailable due to calculation error."


                    # --- Display Status Check before Interpretation ---
                    st.divider() # Add visual separation
                    st.subheader("🎯 Target Status Check (Code-Based)")
                    if 'Error' not in pk_results:
                        st.markdown(f" - **Measured Trough:** {cmin_ind:.1f} mg/L -> **{trough_status}**")
                        st.markdown(f" - **Calculated AUC₂₄:** {auc24_ind:.1f} mg·h/L -> **{auc_status}**")
                    else:
                        # Show status even if calc failed partially
                        st.markdown(f" - **Measured Trough:** {c_trough_measured:.1f} mg/L -> **{pk_results.get('Trough Status vs Target','N/A')}**")
                        st.markdown(f" - **Calculated AUC₂₄:** N/A -> **{pk_results.get('AUC Status vs Target','N/A')}**")
                        st.warning("Status based on available data; PK parameter calculation failed.")
                    st.divider()
                    # --- End Display Status Check ---

                    st.subheader("💬 Interpretation & Recommendation (AI-Generated)")
                    st.markdown(interpretation_text) # Display interpretation or error message

                    # Prepare dict for report
                    report_dict = pk_results.copy()
                    if interpretation_text not in ["N/A (Calculation Error or RAG disabled)", "Interpretation disabled: RAG system not loaded.", "Interpretation unavailable due to calculation error."]:
                        report_dict["RAG Interpretation"] = interpretation_text

                    report_data = build_report(report_dict, scr_umol, clinical_notes)
                    st.download_button(
                        label="📥 Download Report (.txt)",
                        data=report_data,
                        file_name=f"{pid or 'patient'}_vanco_peak_trough_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )


# --- Footer ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational and informational purposes only. Consult official guidelines and clinical judgment for patient care decisions.")
st.caption(f"Guideline source: Clinical Pharmacokinetics Pharmacy Handbook (2nd ed.) | App version: V3 | Last updated: {datetime.now().strftime('%Y-%m-%d')}")

