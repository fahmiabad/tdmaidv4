# vancomycin_tdm_app.py
"""
Single-file Streamlit app for Vancomycin TDM with RAG-guided LLM interpretation,
time-based inputs, target-level selection, PDF chunking via pypdf,
and explicit intended dosing interval input.
Uses Streamlit secrets for OpenAI API key.
"""
import os
import math
import streamlit as st
from datetime import datetime, timedelta, time
import logging # Added for better error logging

# --- 1. SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Vancomycin TDM App", layout="wide")

# --- 2. CONFIGURE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 3. LOAD SECRET ---
# Use Streamlit secrets to get the OpenAI API key
try:
    # Check if running on Streamlit Cloud and secrets are available
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        logging.info("OpenAI API key loaded from Streamlit secrets.")
        # Avoid calling st.info here, do it later in the main UI flow if needed
    else:
        # Fallback for local development (optional, but can be useful)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Defer st.error call until the main app body
            logging.error("OpenAI API key not found in Streamlit secrets or environment variables.")
            # We can't call st.stop() here either before the main app runs
            # We'll handle the missing key later in the main UI flow
        else:
            logging.info("OpenAI API key loaded from environment variable for local testing.")
            # Set it for Langchain if loaded locally
            os.environ["OPENAI_API_KEY"] = openai_api_key

except Exception as e:
    # Defer st.error call until the main app body
    logging.error(f"Error loading OpenAI API key: {e}")
    # Cannot call st.stop() here

# Check if API key was loaded successfully before proceeding with imports/RAG
api_key_loaded = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]

# --- 4. IMPORTS (Place imports after key check to avoid errors if key fails) ---
# Only import these if the API key is likely available
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
    libraries_loaded = False # Can't load libraries if API key failed

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
            # Defer st.error
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
            # Defer st.error
            logging.error(f"No text extracted from PDF: {pdf_path}")
            return None

        # Chunk text
        logging.info("Chunking text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        logging.info(f"Created {len(chunks)} text chunks.")
        if not chunks:
            # Defer st.error
            logging.error("Text splitting resulted in zero chunks.")
            return None

        # Embed and index
        logging.info("Creating embeddings and FAISS index...")
        # Ensure API key is implicitly used by Langchain components
        embeddings = OpenAIEmbeddings()
        index = FAISS.from_texts(chunks, embeddings)
        logging.info("FAISS index created successfully.")

        # Build QA chain
        logging.info("Building RetrievalQA chain...")
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0), # Use the OpenAI LLM
            chain_type="stuff",        # Chain type for combining documents
            retriever=index.as_retriever() # Use the FAISS index as the retriever
        )
        logging.info("RetrievalQA chain built successfully.")
        return qa

    except Exception as e:
        # Defer st.error
        logging.exception("Error during RAG chain loading:") # Log full traceback
        return None

# Attempt to load the RAG chain
# IMPORTANT: Make sure this PDF file is in the same directory as your script
# or provide the full path. When deploying to Streamlit Cloud, ensure this
# file is included in your repository.
PDF_FILENAME = 'clinical-pharmacokinetics-pharmacy-handbook-ccph-2nd-edition-rev-2.0_0-2.pdf'
qa_chain = load_rag_chain(PDF_FILENAME) # This might return None if setup failed

# --- 6. TIME DIFFERENCE HELPER ---
def hours_diff(start: time, end: time) -> float:
    """Calculates the difference between two times in hours, handling overnight intervals."""
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)

    # If end time is earlier than start time, assume it's the next day
    if dt_end < dt_start:
        dt_end += timedelta(days=1)

    return (dt_end - dt_start).total_seconds() / 3600

# --- 7. PK CALC FUNCTIONS ---
# (Using standard pharmacokinetic formulas)

def calculate_crcl(age: int, weight: float, scr: float, female: bool = False) -> float:
    """Calculates Creatinine Clearance (CrCl) using Cockcroft-Gault."""
    if age <= 0 or weight <=0 or scr <= 0: # Add age/weight check
        logging.warning(f"Invalid input for CrCl calculation: age={age}, weight={weight}, scr={scr}. Returning 0.")
        return 0.0
    base = ((140 - age) * weight) / (72 * scr)
    crcl = base * 0.85 if female else base
    return max(0, crcl) # CrCl cannot be negative

def calculate_vd(weight: float, age: int | None = None) -> float:
    """Calculates Volume of Distribution (Vd). Uses standard 0.7 L/kg."""
    if weight <= 0:
        logging.warning(f"Invalid weight ({weight}) for Vd calculation. Returning 0.")
        return 0.0
    vd = 0.7 * weight
    logging.info(f"Calculated Vd using 0.7 L/kg (Weight={weight}kg): {vd:.1f} L")
    return vd


def round_dose(dose: float) -> int:
    """Rounds the dose to the nearest 250mg increment."""
    if dose < 0:
        return 0
    # Round to nearest 250, ensuring minimum is 250 if dose > 0
    rounded = int(round(dose / 250.0) * 250)
    return max(250, rounded) if dose > 0 else 0


def calculate_initial_dose(age: int, weight: float, scr: float, female: bool) -> int:
    """Calculates an initial loading dose based on CrCl."""
    crcl = calculate_crcl(age, weight, scr, female)
    mg_kg = 25 # Consider adjusting based on CrCl or clinical context if needed
    loading_dose = weight * mg_kg
    logging.info(f"Calculated initial dose: CrCl={crcl:.1f}, mg/kg={mg_kg}, Raw Dose={loading_dose:.1f}")
    return round_dose(loading_dose)

def calculate_ke_trough(dose_int: float, vd: float, trough: float, time_since_last_dose_h: float) -> float:
    """
    Calculates the elimination rate constant (Ke) from trough level,
    using the ACTUAL time elapsed since the last dose was given.
    """
    if trough <= 0 or vd <= 0 or time_since_last_dose_h <= 0:
        logging.warning(f"Invalid input for Ke calculation: trough={trough}, vd={vd}, time_since_dose={time_since_last_dose_h}. Returning Ke=0.")
        return 0 # Avoid math errors with invalid inputs

    try:
        cmax_est = trough + (dose_int / vd)
        if cmax_est <= trough: # Should not happen if dose > 0
             logging.warning(f"Calculated Cmax_est ({cmax_est:.2f}) not greater than trough ({trough:.2f}). Check inputs. Returning Ke=0.")
             return 0
        # Ke = ln(Cmax / Cmin) / time_since_last_dose
        ke = math.log(cmax_est / trough) / time_since_last_dose_h
        logging.info(f"Calculated Ke (trough-only): Cmax_est={cmax_est:.2f}, Trough={trough:.2f}, TimeSinceDose={time_since_last_dose_h:.2f}h => Ke={ke:.4f} h⁻¹")
        return max(0, ke) # Ke cannot be negative
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating Ke (trough-only): {e}. Inputs: dose={dose_int}, vd={vd}, trough={trough}, time_since_dose={time_since_last_dose_h}")
        return 0


def calculate_new_dose_trough(ke: float, vd: float, intended_interval_h: float, target_trough: float = 15.0) -> int:
    """
    Calculates a new dose to reach a target trough level, using the calculated Ke
    and the INTENDED dosing interval.
    """
    if ke <= 0 or vd <= 0 or intended_interval_h <= 0 or target_trough <= 0:
        logging.warning(f"Cannot calculate new dose due to invalid Ke ({ke:.4f}) or other inputs (Vd={vd:.1f}, IntendedInterval={intended_interval_h:.1f}, Target={target_trough:.1f}). Returning 0.")
        return 0 # Return 0 or handle appropriately

    try:
        term_exp = math.exp(-ke * intended_interval_h)
        if term_exp == 1: # Avoid division by zero if Ke or interval is effectively zero
            logging.warning("Exponential term is 1 in new dose calculation (Ke or interval near zero).")
            return 0 # Or handle differently, implies no elimination
        if term_exp == 0: # Avoid division by zero if interval or Ke is huge
             logging.warning("Denominator zero in new dose calculation (exp term is zero). Check Ke and interval.")
             return 0

        numerator = target_trough * vd * (1 - term_exp)
        denominator = term_exp
        new_dose = numerator / denominator

        logging.info(f"Calculated new dose (trough-only): Target={target_trough:.1f}, Vd={vd:.1f}, Ke={ke:.4f}, IntendedInterval={intended_interval_h:.1f}h => Raw New Dose={new_dose:.1f}")
        return round_dose(new_dose)
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating new dose (trough-only): {e}. Inputs: target={target_trough}, vd={vd}, ke={ke}, intended_interval={intended_interval_h}")
        return 0 # Return 0 or original dose on error


def calculate_auc24_trough(dose_int: float, ke: float, vd: float, intended_interval_h: float) -> float:
    """
    Estimates AUC24 based on the current dose, calculated Ke, Vd,
    and the INTENDED dosing interval.
    """
    if ke <= 0 or vd <= 0 or intended_interval_h <= 0:
        logging.warning(f"Cannot calculate AUC24 due to invalid Ke ({ke:.4f}) or other inputs (Vd={vd:.1f}, IntendedInterval={intended_interval_h:.1f}). Returning AUC24=0.")
        return 0.0

    cl = ke * vd
    if cl <= 0:
        logging.warning(f"Calculated CL is zero or negative (Ke={ke:.4f}, Vd={vd:.1f}). Cannot calculate AUC24. Returning 0.")
        return 0.0

    try:
        daily_dose = dose_int * (24 / intended_interval_h)
        auc24 = daily_dose / cl
        logging.info(f"Calculated AUC24 (trough-only): Dose={dose_int}, Daily Dose={daily_dose:.1f}, CL={cl:.2f}, IntendedInterval={intended_interval_h:.1f}h => AUC24={auc24:.1f}")
        return max(0, auc24) # AUC cannot be negative
    except ZeroDivisionError:
        logging.error(f"Division by zero calculating AUC24 (intended_interval={intended_interval_h}). Returning 0.")
        return 0.0


def calculate_pk_params_peak_trough(
    c_trough: float, c_peak_measured: float, infusion_duration_h: float,
    time_from_infusion_end_to_peak_draw_h: float, intended_interval_h: float # Use INTENDED interval here
) -> dict | None:
    """
    Calculates PK parameters (Ke, t1/2, Cmax, Cmin_actual, AUC24)
    using peak and trough levels (Sawchuk-Zaske method).
    Uses the INTENDED dosing interval for calculations involving interval length.
    """
    time_between_samples_h = intended_interval_h - infusion_duration_h - time_from_infusion_end_to_peak_draw_h

    if time_between_samples_h <= 0 or c_peak_measured <= 0 or c_trough <= 0 or c_peak_measured <= c_trough:
        logging.warning(f"Invalid inputs for peak/trough calculation: "
                        f"TimeBetweenSamples={time_between_samples_h:.2f} (based on intended interval {intended_interval_h:.1f}h), "
                        f"Cpeak={c_peak_measured:.2f}, Ctrough={c_trough:.2f}. Cannot calculate PK parameters.")
        return None

    try:
        ke = math.log(c_peak_measured / c_trough) / time_between_samples_h
        if ke <= 0:
             logging.warning(f"Calculated Ke is zero or negative ({ke:.4f}) in peak/trough method. Check levels and times.")
             return None

        half_life_h = math.log(2) / ke
        c_max_extrapolated = c_peak_measured * math.exp(ke * time_from_infusion_end_to_peak_draw_h)
        c_min_actual = c_trough

        # Calculate AUC for one INTENDED dosing interval using trapezoidal rule approximation
        auc_interval_infusion_part = infusion_duration_h * (c_min_actual + c_max_extrapolated) / 2
        auc_interval_elimination_part = (c_max_extrapolated - c_min_actual) / ke
        auc_interval_total = auc_interval_infusion_part + auc_interval_elimination_part

        # Scale interval AUC to 24 hours based on the INTENDED interval
        auc24 = auc_interval_total * (24 / intended_interval_h)

        logging.info(f"Calculated PK Params (peak/trough): Ke={ke:.4f}, t1/2={half_life_h:.1f}, "
                     f"Cmax_extrap={c_max_extrapolated:.1f}, Cmin_actual={c_min_actual:.1f}, AUC24={auc24:.1f} (based on intended interval {intended_interval_h:.1f}h)")

        return {
            'ke': ke,
            't_half': half_life_h,
            'Cmax_extrapolated': c_max_extrapolated,
            'Cmin_actual': c_min_actual,
            'AUC24': auc24
        }
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating PK parameters (peak/trough): {e}. Inputs: "
                      f"Ctrough={c_trough}, Cpeak={c_peak_measured}, Infusion={infusion_duration_h}, "
                      f"PeakDelay={time_from_infusion_end_to_peak_draw_h}, IntendedInterval={intended_interval_h}")
        return None


# --- 8. LLM INTERPRETATION ---
def interpret(crcl: float, pk_results: dict, target_level_desc: str) -> str:
    """Generates interpretation and recommendations using the RAG chain."""
    # Check if RAG chain is available (it might be None if loading failed)
    if qa_chain is None:
        logging.warning("QA chain not loaded. Skipping interpretation.")
        return "Interpretation unavailable: RAG system failed to load. Check API key, PDF file, and library installations."

    intended_interval_info = pk_results.get('Intended Dosing Interval', 'N/A')
    # Include Time Since Last Dose if available (Trough-Only mode)
    time_since_dose_info = pk_results.get('Time Since Last Dose (at Trough Draw)', 'N/A')
    time_context = f"Consider the actual time the trough was drawn ({time_since_dose_info})." if time_since_dose_info != 'N/A' else ""

    prompt = f"""
Context: You are a clinical pharmacokinetics expert interpreting vancomycin therapeutic drug monitoring (TDM) results based on established guidelines (provided via internal knowledge source).

Patient Information Summary:
- Estimated Creatinine Clearance (CrCl): {crcl:.1f} mL/min
- Selected Therapeutic Target: {target_level_desc}

Current Regimen & Monitoring Results:
{pk_results}

Task: Provide a concise interpretation of these results relative to the target. Include:
1.  Assessment: Are the current levels/AUC within the target range for the intended interval ({intended_interval_info})? Is the patient clearing the drug as expected based on CrCl? {time_context}
2.  Recommendation: Suggest specific adjustments to the dose and/or interval, if needed, to achieve the target. Use standard clinical intervals (e.g., q8h, q12h, q24h). If a new dose is suggested, provide it rounded to the nearest 250mg for the *intended* interval. If no change is needed, state that clearly.
3.  Rationale: Briefly explain why the recommendation is being made, linking it to the PK results (Ke, AUC, levels), target, CrCl, and intended interval.
4.  Follow-up: Suggest when to re-check levels, if appropriate (e.g., next trough before 3rd/4th dose of new regimen).

Use the provided guideline knowledge to inform your response. Be specific and clinically oriented.
"""
    logging.info(f"Generating interpretation with prompt:\n{prompt}")
    try:
        # Use .invoke() for newer Langchain versions if .run() is deprecated
        if hasattr(qa_chain, 'invoke'):
            response = qa_chain.invoke({"query": prompt})
            # Handle potential dictionary response from invoke
            if isinstance(response, dict) and 'result' in response:
                response_text = response['result']
            elif isinstance(response, str):
                 response_text = response
            else:
                # Attempt to extract text if it's another object type
                response_text = str(response)
                logging.warning(f"Unexpected response type from qa_chain.invoke: {type(response)}. Converted to string.")

        else:
             # Fallback for older versions
             response_text = qa_chain.run(prompt) # type: ignore

        logging.info(f"LLM Interpretation received: {response_text}")
        return response_text
    except Exception as e:
        logging.error(f"Error running RAG QA chain for interpretation: {e}")
        logging.exception("RAG Chain execution error:") # Log traceback
        return f"Interpretation failed: Error communicating with the RAG system ({e}). Check API key and model availability."

# --- 9. STREAMLIT UI ---

# Display initial errors if setup failed
if not api_key_loaded:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable (OPENAI_API_KEY) for local testing. App functionality is limited.")
    st.stop() # Stop execution if key is missing

if not libraries_loaded:
    st.error(f"Failed to import required libraries. Please ensure all dependencies are installed (streamlit, pypdf, langchain, langchain-community, langchain-openai, faiss-cpu, tiktoken). You might need to run: pip install -U langchain langchain-community langchain-openai faiss-cpu tiktoken. App functionality is limited.")
    st.stop() # Stop execution if libraries are missing

# Main App Title
st.title("🧪 Vancomycin TDM with RAG Interpretation")
st.markdown("Calculates PK parameters and provides guideline-based interpretation using RAG.")

# Check if RAG system loaded correctly AFTER attempting to load it
if qa_chain is None:
    st.error("Critical Error: The RAG guideline interpretation system could not be loaded. Potential issues: PDF file path incorrect, PDF content unreadable, or problem during embedding/indexing. Interpretation features will be disabled.")
    # Allow calculations to proceed, but interpretation will be unavailable.

# Sidebar for inputs
st.sidebar.header("⚙️ Mode & Target")
mode = st.sidebar.radio("Select Calculation Mode:", ["Initial Dose", "Trough-Only", "Peak & Trough"])

target_level_desc = st.sidebar.selectbox(
    "Select Therapeutic Target:",
    options=[
        "Empirical (Target AUC24 400-600 mg·h/L; Trough ~10-15 mg/L)",
        "Definitive/Severe (Target AUC24 >600 mg·h/L; Trough ~15-20 mg/L)"
    ],
    index=0,
    help="Select the desired therapeutic goal based on infection type and severity. This guides interpretation."
)

# Set numerical target based on selection for calculation purposes
if "Empirical" in target_level_desc:
    target_trough_for_calc = 12.5
    target_auc_range = "400-600"
else:
    target_trough_for_calc = 17.5
    target_auc_range = ">600"


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
    scr = st.number_input("SCr (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1, format="%.1f", help="Serum Creatinine")
with col_fem:
    fem = st.checkbox("Female", value=False)


# Helper function to build a downloadable report
def build_report(lines: list[str]) -> str:
    """Formats patient info and results into a text report."""
    crcl_report = calculate_crcl(age, wt, scr, fem) if age > 0 and wt > 0 and scr > 0 else 0.0
    vd_report = calculate_vd(wt, age) if wt > 0 else 0.0
    hdr = [
        "--- Vancomycin TDM Report ---",
        f"Patient ID: {pid if pid else 'N/A'}",
        f"Ward/Unit: {ward if ward else 'N/A'}",
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Patient Info: Age={age} yrs, Weight={wt} kg, Sex={'Female' if fem else 'Male'}, SCr={scr} mg/dL",
        f"Estimated CrCl: {crcl_report:.1f} mL/min" if crcl_report is not None else "N/A (Invalid Input)",
        f"Estimated Vd (0.7 L/kg): {vd_report:.1f} L" if vd_report is not None else "N/A (Invalid Input)",
        f"Selected Target: {target_level_desc}",
        "--- Results & Interpretation ---"
    ]
    return "\n".join(hdr + lines)

# --- Main Area Logic ---
results_container = st.container()

if mode == "Initial Dose":
    with results_container:
        st.subheader("🚀 Initial Loading Dose Calculation")
        st.markdown("Calculates a one-time loading dose based on patient weight and estimated renal function (CrCl). Uses ~25 mg/kg, rounded.")
        calc_button = st.button("Calculate Loading Dose", key="calc_initial")

        if calc_button:
            if wt <= 0 or age <= 0 or scr <= 0:
                st.warning("Please enter valid patient information (Age > 0, Weight > 0, SCr > 0).")
            else:
                with st.spinner("Calculating..."):
                    initial_dose_mg = calculate_initial_dose(age, wt, scr, fem)
                    crcl_calc = calculate_crcl(age, wt, scr, fem)
                    st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                    st.success(f"Recommended Initial Loading Dose: **{initial_dose_mg} mg** (Administer once)")

                    report_lines = [
                        f"Mode: Initial Loading Dose",
                        f"Calculated Loading Dose: {initial_dose_mg} mg (one time)"
                    ]
                    report_data = build_report(report_lines)
                    st.download_button(
                        label="📥 Download Report (.txt)",
                        data=report_data,
                        file_name=f"{pid or 'patient'}_vanco_initial_dose_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

elif mode == "Trough-Only":
    st.sidebar.header("💉 Trough-Only Monitoring")
    st.sidebar.markdown("Enter details about the *current* regimen, the **intended dosing interval**, and the measured trough level.")
    dose_int_current = st.sidebar.number_input("Current Dose per Interval (mg)", min_value=250, step=250, value=1000) # Min dose 250
    intended_interval_h = st.sidebar.selectbox(
        "Intended Dosing Interval (hours)",
        options=[6, 8, 12, 18, 24, 36, 48],
        index=2,
        format_func=lambda x: f"q{x}h",
        help="Select the standard dosing frequency for this patient (e.g., q12h)."
    )
    dose_time = st.sidebar.time_input("Time of Last Dose Administered", value=time(8, 0), step=timedelta(minutes=15))
    sample_time = st.sidebar.time_input("Time Trough Level Drawn", value=time(19, 30), step=timedelta(minutes=15), help="Actual time the level was drawn.")
    trough_measured = st.sidebar.number_input("Measured Trough Level (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f") # Min trough 0.1

    with results_container:
        st.subheader("📉 Trough-Only Analysis")
        st.markdown("Estimates PK parameters (Ke, t½, AUC24) and suggests dose adjustments based on a single trough level and the **intended** dosing interval.")

        time_since_last_dose_h = hours_diff(dose_time, sample_time)
        # Display calculated time difference, check if positive
        if time_since_last_dose_h > 0:
            st.info(f"Trough drawn **{time_since_last_dose_h:.2f} hours** after the last dose. Intended interval: **q{intended_interval_h}h**.")
        else:
            st.warning("Sample time must be after the last dose time (potentially on the next day).")

        calc_button = st.button("Run Trough-Only Analysis", key="run_trough")

        if calc_button:
             # Check all inputs including positive time difference
             if dose_int_current <= 0 or trough_measured <= 0 or time_since_last_dose_h <= 0 or wt <= 0 or age <= 0 or scr <= 0 or intended_interval_h <= 0:
                 st.warning("Please ensure Dose (>0), Measured Trough (>0), Weight (>0), Age (>0), SCr (>0), Intended Interval (>0), and a valid time difference (>0h) are entered.")
             else:
                with st.spinner("Analyzing Trough Level..."):
                    crcl_calc = calculate_crcl(age, wt, scr, fem)
                    vd_calc = calculate_vd(wt, age)

                    ke_calc = calculate_ke_trough(dose_int_current, vd_calc, trough_measured, time_since_last_dose_h)

                    if ke_calc > 0 and vd_calc > 0:
                        t_half_calc = math.log(2) / ke_calc
                        auc24_calc = calculate_auc24_trough(dose_int_current, ke_calc, vd_calc, intended_interval_h)
                        new_dose_calc = calculate_new_dose_trough(ke_calc, vd_calc, intended_interval_h, target_trough=target_trough_for_calc)

                        # Display results
                        st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Estimated Vd", value=f"{vd_calc:.1f} L")
                        col2.metric(label="Estimated Ke", value=f"{ke_calc:.4f} h⁻¹")
                        col3.metric(label="Estimated t½", value=f"{t_half_calc:.1f} h")

                        st.metric(label=f"Estimated AUC₂₄ (based on q{intended_interval_h}h)", value=f"{auc24_calc:.1f} mg·h/L",
                                  help=f"Target range: {target_auc_range} mg·h/L")
                        st.metric(label=f"Suggested New Dose (for target ~{target_trough_for_calc} mg/L)",
                                  value=f"{new_dose_calc} mg q{intended_interval_h}h" if new_dose_calc > 0 else "Calculation Error",
                                  help=f"Dose rounded to nearest 250mg for the selected q{intended_interval_h}h interval.")

                        pk_results = {
                            'Calculation Mode': 'Trough-Only',
                            'Current Dose': f"{dose_int_current} mg",
                            'Intended Dosing Interval': f"q{intended_interval_h}h",
                            'Time Since Last Dose (at Trough Draw)': f"{time_since_last_dose_h:.2f} h",
                            'Measured Trough': f"{trough_measured:.1f} mg/L",
                            'Estimated Vd': f"{vd_calc:.1f} L",
                            'Estimated Ke': f"{ke_calc:.4f} h⁻¹",
                            'Estimated t½': f"{t_half_calc:.1f} h",
                            'Estimated AUC24 (for intended interval)': f"{auc24_calc:.1f} mg·h/L",
                            'Suggested New Dose (for target trough)': f"{new_dose_calc} mg q{intended_interval_h}h" if new_dose_calc > 0 else "N/A"
                        }
                        interpretation_text = interpret(crcl_calc, pk_results, target_level_desc) # Call interpret even if RAG failed (will return error msg)

                    else:
                        st.error("Could not calculate Ke (check inputs: trough > 0, vd > 0, time since dose > 0). Cannot proceed with AUC/New Dose calculation.")
                        pk_results = {"Error": "Ke calculation failed."}
                        interpretation_text = "N/A (Calculation Error)"


                    # Display Interpretation (will show error if RAG failed)
                    st.subheader("💬 RAG Interpretation & Recommendation")
                    st.markdown(interpretation_text)

                    # Prepare report (only if calc succeeded)
                    if ke_calc > 0 and vd_calc > 0:
                        report_lines = [
                            f"Mode: Trough-Only Analysis",
                            f"Current Dose: {dose_int_current} mg",
                            f"Intended Interval: q{intended_interval_h}h",
                            f"Last Dose Time: {dose_time.strftime('%H:%M')}",
                            f"Trough Sample Time: {sample_time.strftime('%H:%M')}",
                            f"Time Since Last Dose: {time_since_last_dose_h:.2f} h",
                            f"Measured Trough: {trough_measured:.1f} mg/L",
                            f"--- Calculated Parameters ---",
                            f"Est. Vd: {vd_calc:.1f} L",
                            f"Est. Ke: {ke_calc:.4f} h⁻¹",
                            f"Est. t½: {t_half_calc:.1f} h",
                            f"Est. AUC24 (for q{intended_interval_h}h): {auc24_calc:.1f} mg·h/L",
                            f"--- Recommendation ---",
                            f"Suggested New Dose (Target ~{target_trough_for_calc} mg/L): {new_dose_calc} mg q{intended_interval_h}h" if new_dose_calc > 0 else "N/A",
                            f"--- RAG Interpretation ---",
                            interpretation_text.replace('\n', '\n  ')
                        ]
                        report_data = build_report(report_lines)
                        st.download_button(
                            label="📥 Download Report (.txt)",
                            data=report_data,
                            file_name=f"{pid or 'patient'}_vanco_trough_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )

elif mode == "Peak & Trough":
    st.sidebar.header("📈 Peak & Trough Monitoring")
    st.sidebar.markdown("Enter infusion times, the **intended dosing interval**, and both peak and trough levels.")
    intended_interval_h_pt = st.sidebar.selectbox(
        "Intended Dosing Interval (hours)",
        options=[6, 8, 12, 18, 24, 36, 48],
        index=2,
        format_func=lambda x: f"q{x}h",
        key="interval_pt",
        help="Select the standard dosing frequency for this patient (e.g., q12h)."
    )
    infusion_start_time = st.sidebar.time_input("Infusion Start Time", value=time(8, 0), step=timedelta(minutes=15))
    infusion_end_time = st.sidebar.time_input("Infusion End Time", value=time(9, 0), step=timedelta(minutes=15), help="End time of the vancomycin infusion.")
    peak_sample_time = st.sidebar.time_input("Peak Sample Time", value=time(10, 0), step=timedelta(minutes=15), help="Time peak level was drawn (typically 1-2h after infusion ends).")
    trough_sample_time = st.sidebar.time_input("Trough Sample Time", value=time(19, 30), step=timedelta(minutes=15), help="Time trough level was drawn (just before next dose).")
    dose_for_levels = st.sidebar.number_input("Dose Administered (mg)", min_value=250, step=250, value=1000, help="The dose given *before* these peak/trough levels were drawn.") # Min dose 250
    c_trough_measured = st.sidebar.number_input("Measured Trough (Cmin) (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f") # Min trough 0.1
    c_peak_measured = st.sidebar.number_input("Measured Peak (Cpeak) (mg/L)", min_value=0.1, max_value=200.0, value=30.0, step=0.1, format="%.1f", help="This is the level measured at 'Peak Sample Time', *not* the true Cmax.") # Min peak 0.1


    with results_container:
        st.subheader("📊 Peak & Trough Analysis (Sawchuk-Zaske)")
        st.markdown("Calculates individual PK parameters (Ke, t½, Vd, AUC24) using measured peak and trough levels and the **intended** dosing interval.")

        infusion_duration_h = hours_diff(infusion_start_time, infusion_end_time)
        time_from_infusion_end_to_peak_draw_h = hours_diff(infusion_end_time, peak_sample_time)

        # Display calculated durations and check validity
        valid_times = True
        if infusion_duration_h > 0 and time_from_infusion_end_to_peak_draw_h >= 0:
             st.info(f"Calculated Durations: Infusion={infusion_duration_h:.2f}h | Delay to Peak Draw={time_from_infusion_end_to_peak_draw_h:.2f}h. Intended Interval: **q{intended_interval_h_pt}h**.")
             # Check if intended interval makes sense with infusion/peak times
             if intended_interval_h_pt <= infusion_duration_h + time_from_infusion_end_to_peak_draw_h:
                  st.warning("Intended interval is very short based on infusion duration and peak draw time. Check inputs.")
                  # Don't necessarily block calculation, but warn user.
        else:
             st.warning("Infusion/Peak times are invalid. End time must be after start time, Peak time must be at or after end time.")
             valid_times = False

        calc_button = st.button("Run Peak & Trough Analysis", key="run_peak_trough")

        if calc_button:
            # Validate inputs
            if not valid_times:
                 st.error("Please correct the infusion/peak time errors before running.")
            elif c_trough_measured <= 0 or c_peak_measured <= 0:
                st.warning("Measured Peak and Trough levels must be greater than 0.")
            elif c_peak_measured <= c_trough_measured:
                st.warning("Measured Peak level must be greater than Measured Trough level.")
            elif dose_for_levels <= 0:
                 st.warning("Please enter the dose administered (> 0 mg).")
            elif wt <= 0 or age <= 0 or scr <= 0:
                 st.warning("Please enter valid patient information (Age > 0, Weight > 0, SCr > 0).")
            else: # Inputs seem valid, proceed with calculation
                with st.spinner("Analyzing Peak & Trough Levels..."):
                    crcl_calc = calculate_crcl(age, wt, scr, fem)

                    pk_params = calculate_pk_params_peak_trough(
                        c_trough=c_trough_measured,
                        c_peak_measured=c_peak_measured,
                        infusion_duration_h=infusion_duration_h,
                        time_from_infusion_end_to_peak_draw_h=time_from_infusion_end_to_peak_draw_h,
                        intended_interval_h=intended_interval_h_pt
                    )

                    interpretation_text = "N/A (Calculation Error)" # Default
                    pk_results = {"Error": "Failed to calculate PK parameters from Peak & Trough data."} # Default

                    if pk_params:
                        ke_ind = pk_params['ke']
                        thalf_ind = pk_params['t_half']
                        cmax_ind = pk_params['Cmax_extrapolated']
                        cmin_ind = pk_params['Cmin_actual']
                        auc24_ind = pk_params['AUC24']

                        vd_ind = 0.0 # Initialize Vd
                        try:
                            auc_interval_ind = auc24_ind / (24 / intended_interval_h_pt) if intended_interval_h_pt > 0 else 0
                            if ke_ind > 0 and auc_interval_ind > 0:
                                vd_ind = dose_for_levels / (ke_ind * auc_interval_ind)
                            else:
                                st.warning("Could not calculate Vd due to zero Ke or AUC_interval.")
                        except ZeroDivisionError:
                             st.warning("Could not calculate Vd due to division by zero (check intended interval).")


                        # Display results
                        st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Individual Vd", value=f"{vd_ind:.1f} L" if vd_ind > 0 else "N/A")
                        col2.metric(label="Individual Ke", value=f"{ke_ind:.4f} h⁻¹")
                        col3.metric(label="Individual t½", value=f"{thalf_ind:.1f} h")

                        col4, col5 = st.columns(2)
                        col4.metric(label="Est. Cmax (End of Infusion)", value=f"{cmax_ind:.1f} mg/L")
                        col5.metric(label="Measured Cmin (Trough)", value=f"{cmin_ind:.1f} mg/L")

                        st.metric(label=f"Individual AUC₂₄ (based on q{intended_interval_h_pt}h)", value=f"{auc24_ind:.1f} mg·h/L",
                                  help=f"Target range: {target_auc_range} mg·h/L")

                        # Suggest new dose based on target AUC and individual CL
                        new_dose_suggestion = "N/A"
                        if ke_ind > 0 and vd_ind > 0 and intended_interval_h_pt > 0:
                             cl_ind = ke_ind * vd_ind
                             target_auc_numeric = float(target_auc_range.split('-')[0]) if '-' in target_auc_range else float(target_auc_range.replace('>', ''))
                             if target_auc_numeric > 0:
                                 try:
                                     target_auc_interval = target_auc_numeric * (intended_interval_h_pt / 24.0)
                                     new_dose_raw = target_auc_interval * cl_ind
                                     new_dose_rounded = round_dose(new_dose_raw)
                                     new_dose_suggestion = f"{new_dose_rounded} mg q{intended_interval_h_pt}h"
                                     st.metric(label=f"Suggested Dose (for Target AUC ~{target_auc_numeric})", value=new_dose_suggestion)
                                 except Exception as dose_calc_err:
                                     st.warning(f"Could not calculate suggested dose: {dose_calc_err}")
                        else:
                             st.warning("Cannot suggest new dose without valid Ke, Vd, and Intended Interval.")


                        pk_results = {
                            'Calculation Mode': 'Peak & Trough',
                            'Dose Administered': f"{dose_for_levels} mg",
                            'Intended Dosing Interval': f"q{intended_interval_h_pt}h",
                            'Infusion Duration': f"{infusion_duration_h:.2f} h",
                            'Time to Peak Draw (post-infusion)': f"{time_from_infusion_end_to_peak_draw_h:.2f} h",
                            'Measured Peak (at draw time)': f"{c_peak_measured:.1f} mg/L",
                            'Measured Trough (Cmin)': f"{c_trough_measured:.1f} mg/L",
                            'Individual Vd': f"{vd_ind:.1f} L" if vd_ind > 0 else "N/A",
                            'Individual Ke': f"{ke_ind:.4f} h⁻¹",
                            'Individual t½': f"{thalf_ind:.1f} h",
                            'Est. Cmax (End of Infusion)': f"{cmax_ind:.1f} mg/L",
                            'Individual AUC24 (for intended interval)': f"{auc24_ind:.1f} mg·h/L",
                            'Suggested New Dose (for target AUC)': new_dose_suggestion
                        }
                        interpretation_text = interpret(crcl_calc, pk_results, target_level_desc) # Call interpret

                    else:
                        st.error("Failed to calculate PK parameters from Peak & Trough data. Please check the input values (levels, times, interval) for consistency.")
                        # pk_results and interpretation_text retain default error values

                    # Display Interpretation (will show error if RAG failed or calc failed)
                    st.subheader("💬 RAG Interpretation & Recommendation")
                    st.markdown(interpretation_text)

                    # Prepare report (only if calc succeeded)
                    if pk_params:
                        report_lines = [
                            f"Mode: Peak & Trough Analysis",
                            f"Dose Administered: {dose_for_levels} mg",
                            f"Intended Interval: q{intended_interval_h_pt}h",
                            f"Infusion Start: {infusion_start_time.strftime('%H:%M')}, End: {infusion_end_time.strftime('%H:%M')} (Duration: {infusion_duration_h:.2f} h)",
                            f"Peak Sample Time: {peak_sample_time.strftime('%H:%M')} ({time_from_infusion_end_to_peak_draw_h:.2f} h post-infusion)",
                            f"Trough Sample Time: {trough_sample_time.strftime('%H:%M')}",
                            f"Measured Peak (at draw): {c_peak_measured:.1f} mg/L",
                            f"Measured Trough (Cmin): {c_trough_measured:.1f} mg/L",
                            f"--- Calculated Individual Parameters (based on q{intended_interval_h_pt}h) ---",
                            f"Ind. Vd: {vd_ind:.1f} L" if vd_ind > 0 else "N/A",
                            f"Ind. Ke: {ke_ind:.4f} h⁻¹",
                            f"Ind. t½: {thalf_ind:.1f} h",
                            f"Est. Cmax (End of Infusion): {cmax_ind:.1f} mg/L",
                            f"Ind. AUC24: {auc24_ind:.1f} mg·h/L",
                            f"--- Recommendation ---",
                            f"Suggested New Dose (Target AUC ~{target_auc_numeric if 'target_auc_numeric' in locals() and vd_ind > 0 else 'N/A'}): {new_dose_suggestion}",
                            f"--- RAG Interpretation ---",
                            interpretation_text.replace('\n', '\n  ')
                        ]
                        report_data = build_report(report_lines)
                        st.download_button(
                            label="📥 Download Report (.txt)",
                            data=report_data,
                            file_name=f"{pid or 'patient'}_vanco_peak_trough_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )


# --- Footer ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational and informational purposes only. Consult official guidelines and clinical judgment for patient care decisions.")
st.caption(f"Guideline source loaded: `{PDF_FILENAME}` | App last updated: {datetime.now().strftime('%Y-%m-%d')}")
