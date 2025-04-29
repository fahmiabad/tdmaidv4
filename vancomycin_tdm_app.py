# vancomycin_tdm_app.py
"""
Single-file Streamlit app for Vancomycin TDM with RAG-guided LLM interpretation,
time-based inputs, target-level selection, and PDF chunking via pypdf.
Uses Streamlit secrets for OpenAI API key.
"""
import os
import math
import streamlit as st
from datetime import datetime, timedelta, time
import logging # Added for better error logging

# --- 1. CONFIGURE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. LOAD SECRET ---
# Use Streamlit secrets to get the OpenAI API key
try:
    # Check if running on Streamlit Cloud and secrets are available
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        logging.info("OpenAI API key loaded from Streamlit secrets.")
    else:
        # Fallback for local development (optional, but can be useful)
        # You might still want to use environment variables locally
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable (OPENAI_API_KEY) for local testing.")
            logging.error("OpenAI API key not found in Streamlit secrets or environment variables.")
            st.stop() # Stop the app if the key isn't found
        else:
            logging.info("OpenAI API key loaded from environment variable for local testing.")
            # Set it for Langchain if loaded locally
            os.environ["OPENAI_API_KEY"] = openai_api_key

except Exception as e:
    st.error(f"Error loading OpenAI API key: {e}")
    logging.error(f"Error loading OpenAI API key: {e}")
    st.stop()

# --- 3. IMPORTS (Place imports after key check to avoid errors if key fails) ---
try:
    from pypdf import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    logging.info("Required libraries imported successfully.")
except ImportError as e:
    st.error(f"Failed to import required libraries: {e}. Please ensure all dependencies are installed (streamlit, pypdf, langchain, openai, faiss-cpu, tiktoken).")
    logging.error(f"Failed to import required libraries: {e}")
    st.stop()


# --- 4. LOAD & INDEX GUIDELINE PDF USING pypdf ---
@st.cache_resource # Caches the resource across reruns
def load_rag_chain(pdf_path: str) -> RetrievalQA | None:
    """Loads the PDF, chunks text, creates embeddings, and builds a RAG chain."""
    try:
        # Check if the PDF file exists
        if not os.path.exists(pdf_path):
            st.error(f"Guideline PDF not found at path: {pdf_path}")
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
            st.error(f"Could not extract any text from the PDF: {pdf_path}. The PDF might be image-based or corrupted.")
            logging.error(f"No text extracted from PDF: {pdf_path}")
            return None

        # Chunk text
        logging.info("Chunking text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        logging.info(f"Created {len(chunks)} text chunks.")
        if not chunks:
            st.error("Text splitting resulted in zero chunks. Check PDF content and splitter settings.")
            logging.error("Text splitting resulted in zero chunks.")
            return None

        # Embed and index
        logging.info("Creating embeddings and FAISS index...")
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
        st.error(f"Failed to load or process the PDF for RAG: {e}")
        logging.exception("Error during RAG chain loading:") # Log full traceback
        return None

# Attempt to load the RAG chain
# IMPORTANT: Make sure this PDF file is in the same directory as your script
# or provide the full path. When deploying to Streamlit Cloud, ensure this
# file is included in your repository.
PDF_FILENAME = 'clinical-pharmacokinetics-pharmacy-handbook-ccph-2nd-edition-rev-2.0_0-2.pdf'
qa_chain = load_rag_chain(PDF_FILENAME)

# --- 5. TIME DIFFERENCE HELPER ---
def hours_diff(start: time, end: time) -> float:
    """Calculates the difference between two times in hours, handling overnight intervals."""
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)

    # If end time is earlier than start time, assume it's the next day
    if dt_end < dt_start:
        dt_end += timedelta(days=1)

    return (dt_end - dt_start).total_seconds() / 3600

# --- 6. PK CALC FUNCTIONS ---
# (Using standard pharmacokinetic formulas)

def calculate_crcl(age: int, weight: float, scr: float, female: bool = False) -> float:
    """Calculates Creatinine Clearance (CrCl) using Cockcroft-Gault."""
    if scr <= 0: # Avoid division by zero
        return float('inf') # Or handle as appropriate, maybe return 0 or raise error
    base = ((140 - age) * weight) / (72 * scr)
    crcl = base * 0.85 if female else base
    return max(0, crcl) # CrCl cannot be negative

def calculate_vd(weight: float, age: int | None = None) -> float:
    """Calculates Volume of Distribution (Vd). Uses age-based formula if age is provided."""
    # Using the formula provided, default to 0.7 L/kg if age is not given
    # Source for age-based formula needs verification for general applicability.
    # Commonly used Vd for vancomycin is ~0.7 L/kg.
    if age is not None and age >= 0:
         # Ensure this formula is appropriate for your context/guidelines
         # vd = (0.17 * age + 0.22 * weight + 15)
         # Using a more standard 0.7 L/kg as a robust default/alternative:
         vd = 0.7 * weight
         logging.info(f"Calculated Vd using 0.7 L/kg: {vd:.1f} L")
         return vd
    else:
        vd = 0.7 * weight
        logging.info(f"Calculated Vd using 0.7 L/kg (age not provided): {vd:.1f} L")
        return vd


def round_dose(dose: float) -> int:
    """Rounds the dose to the nearest 250mg increment."""
    if dose < 0:
        return 0
    return int(round(dose / 250) * 250)

def calculate_initial_dose(age: int, weight: float, scr: float, female: bool) -> int:
    """Calculates an initial loading dose based on CrCl."""
    crcl = calculate_crcl(age, weight, scr, female)
    # Dosing strategy based on CrCl threshold (verify this against guidelines)
    mg_kg = 20 if crcl <= 30 else 25 # Example: 20 mg/kg if CrCl <= 30, else 25 mg/kg
    loading_dose = weight * mg_kg
    logging.info(f"Calculated initial dose: CrCl={crcl:.1f}, mg/kg={mg_kg}, Raw Dose={loading_dose:.1f}")
    return round_dose(loading_dose)

def calculate_ke_trough(dose_int: float, vd: float, trough: float, interval_h: float) -> float:
    """Calculates the elimination rate constant (Ke) from trough level."""
    if trough <= 0 or vd <= 0 or interval_h <= 0:
        logging.warning(f"Invalid input for Ke calculation: trough={trough}, vd={vd}, interval={interval_h}. Returning Ke=0.")
        return 0 # Avoid math errors with invalid inputs

    # Estimate Cmax based on trough and dose addition
    # Assumes instantaneous distribution, which is an approximation
    try:
        cmax_est = trough + (dose_int / vd)
        if cmax_est <= trough: # Should not happen if dose > 0
             logging.warning(f"Calculated Cmax_est ({cmax_est:.2f}) not greater than trough ({trough:.2f}). Check inputs. Returning Ke=0.")
             return 0
        # Ke = ln(Cmax / Cmin) / interval
        ke = math.log(cmax_est / trough) / interval_h
        logging.info(f"Calculated Ke (trough-only): Cmax_est={cmax_est:.2f}, Trough={trough:.2f}, Interval={interval_h:.2f}h => Ke={ke:.4f} h⁻¹")
        return max(0, ke) # Ke cannot be negative
    except (ValueError, OverflowError) as e:
        logging.error(f"Math error calculating Ke (trough-only): {e}. Inputs: dose={dose_int}, vd={vd}, trough={trough}, interval={interval_h}")
        return 0


def calculate_new_dose_trough(dose_int: float, vd: float, trough: float, interval_h: float, target_trough: float = 15.0) -> int:
    """Calculates a new dose to reach a target trough level."""
    ke = calculate_ke_trough(dose_int, vd, trough, interval_h)
    if ke <= 0 or vd <= 0 or interval_h <= 0 or target_trough <= 0:
        logging.warning(f"Cannot calculate new dose due to invalid Ke ({ke:.3f}) or other inputs (Vd={vd:.1f}, Interval={interval_h:.1f}, Target={target_trough:.1f}). Returning original dose.")
        return round_dose(dose_int) # Return original dose if Ke calculation failed

    # Formula: New Dose = Ctarget * Vd * (1 - e^(-Ke*tau)) / e^(-Ke*tau)
    # This formula calculates the dose needed to achieve the target *at the end* of the interval (trough)
    try:
        numerator = target_trough * vd * (1 - math.exp(-ke * interval_h))
        denominator = math.exp(-ke * interval_h)
        if denominator == 0: # Avoid division by zero if interval or Ke is huge
             logging.warning("Denominator zero in new dose calculation. Check Ke and interval.")
             return round_dose(dose_int)

        new_dose = numerator / denominator
        logging.info(f"Calculated new dose (trough-only): Target={target_trough:.1f}, Vd={vd:.1f}, Ke={ke:.4f}, Interval={interval_h:.1f}h => Raw New Dose={new_dose:.1f}")
        return round_dose(new_dose)
    except (ValueError, OverflowError) as e:
        logging.error(f"Math error calculating new dose (trough-only): {e}. Inputs: target={target_trough}, vd={vd}, ke={ke}, interval={interval_h}")
        return round_dose(dose_int) # Return original dose on error


def calculate_auc24_trough(dose_int: float, vd: float, trough: float, interval_h: float) -> float:
    """Estimates AUC24 based on trough level."""
    ke = calculate_ke_trough(dose_int, vd, trough, interval_h)
    if ke <= 0 or vd <= 0 or interval_h <= 0:
        logging.warning(f"Cannot calculate AUC24 due to invalid Ke ({ke:.3f}) or other inputs (Vd={vd:.1f}, Interval={interval_h:.1f}). Returning AUC24=0.")
        return 0.0

    # Clearance (CL) = Ke * Vd
    cl = ke * vd
    if cl <= 0:
        logging.warning(f"Calculated CL is zero or negative (Ke={ke:.4f}, Vd={vd:.1f}). Cannot calculate AUC24. Returning 0.")
        return 0.0

    # Total daily dose = Dose * (24 / interval)
    daily_dose = dose_int * (24 / interval_h)

    # AUC24 = Total Daily Dose / CL
    auc24 = daily_dose / cl
    logging.info(f"Calculated AUC24 (trough-only): Daily Dose={daily_dose:.1f}, CL={cl:.2f} => AUC24={auc24:.1f}")
    return max(0, auc24) # AUC cannot be negative

def calculate_pk_params_peak_trough(
    c_trough: float, c_peak_measured: float, infusion_duration_h: float,
    time_from_infusion_end_to_peak_draw_h: float, dosing_interval_h: float
) -> dict | None:
    """
    Calculates PK parameters (Ke, t1/2, Cmax, Cmin_actual, AUC24)
    using peak and trough levels (Sawchuk-Zaske method).
    """
    # Time between peak draw and trough draw
    time_between_samples_h = dosing_interval_h - infusion_duration_h - time_from_infusion_end_to_peak_draw_h

    if time_between_samples_h <= 0 or c_peak_measured <= 0 or c_trough <= 0 or c_peak_measured <= c_trough:
        logging.warning(f"Invalid inputs for peak/trough calculation: "
                        f"TimeBetweenSamples={time_between_samples_h:.2f}, "
                        f"Cpeak={c_peak_measured:.2f}, Ctrough={c_trough:.2f}. Cannot calculate PK parameters.")
        return None

    try:
        # Calculate Ke = ln(Cpeak / Ctrough) / time_between_samples
        ke = math.log(c_peak_measured / c_trough) / time_between_samples_h
        if ke <= 0:
             logging.warning(f"Calculated Ke is zero or negative ({ke:.4f}) in peak/trough method. Check levels and times.")
             return None

        # Calculate half-life (t1/2) = ln(2) / Ke
        half_life_h = math.log(2) / ke

        # Extrapolate back to find Cmax (concentration at end of infusion)
        # Cmax = Cpeak_measured * e^(Ke * time_from_infusion_end_to_peak_draw)
        c_max_extrapolated = c_peak_measured * math.exp(ke * time_from_infusion_end_to_peak_draw_h)

        # Extrapolate back further to find Cmin (concentration at start of infusion)
        # Cmin = Cmax_extrapolated * e^(-Ke * infusion_duration) --- THIS IS WRONG
        # Correct: Cmin_actual should be the measured trough C_trough
        c_min_actual = c_trough # The measured trough is the actual Cmin for the *end* of the interval

        # Calculate AUC during the elimination phase (from Cmax_extrapolated down to Cmin_actual)
        # This requires integrating from time=0 (end of infusion) to time=tau (end of interval)
        # AUC_elim = (Cmax_extrapolated - Cmin_actual) / Ke  --- THIS IS APPROXIMATE for AUC over interval minus infusion time

        # More accurate AUC calculation for one dosing interval (AUC_tau):
        # AUC_tau = Dose / CL --- We don't have the dose readily here.
        # Alternative: Use trapezoidal rule during infusion + integral during elimination
        # AUC_infusion = (Cmin_actual + Cmax_extrapolated) / 2 * infusion_duration_h  --- This uses extrapolated Cmax, not true peak during infusion
        # AUC_elimination_phase = (Cmax_extrapolated - Cmin_actual) / ke --- This is AUC from end-of-infusion Cmax down to Cmin

        # Let's stick to the standard AUC calculation if we have Ke and Vd (which we need dose for)
        # Or use the simpler AUC = Dose / CL if dose and CL are known.
        # Since this function ONLY uses concentrations and times, we can only estimate AUC based on trapezoids/integration.

        # Let's calculate AUC for the interval using the derived parameters (less accurate without dose/Vd):
        # AUC during infusion (approximate trapezoid)
        # Need C at start of infusion (Cmin_actual) and C at end (Cmax_extrapolated)
        # auc_inf = (c_min_actual + c_max_extrapolated) / 2.0 * infusion_duration_h
        # AUC during elimination (from Cmax_extrapolated down to Cmin_actual)
        # auc_elim = (c_max_extrapolated - c_min_actual) / ke
        # AUC_interval = auc_inf + auc_elim # AUC over one dosing interval

        # The method provided in the original code seems to calculate AUC differently. Let's re-evaluate that:
        # Original: auc_inf = infusion_h * (cmin + cmax) / 2 -> Uses measured trough and extrapolated Cmax
        # Original: auc_elim = (cmax - cmin) / ke -> Uses extrapolated Cmax and measured trough
        # Original: auc24 = (auc_inf + auc_elim) * (24 / interval_h) -> This calculates AUC for one interval and scales to 24h

        # Let's use the original code's AUC logic for consistency, but using our variable names:
        auc_interval_infusion_part = infusion_duration_h * (c_min_actual + c_max_extrapolated) / 2
        auc_interval_elimination_part = (c_max_extrapolated - c_min_actual) / ke
        auc_interval_total = auc_interval_infusion_part + auc_interval_elimination_part
        auc24 = auc_interval_total * (24 / dosing_interval_h)

        logging.info(f"Calculated PK Params (peak/trough): Ke={ke:.4f}, t1/2={half_life_h:.1f}, "
                     f"Cmax_extrap={c_max_extrapolated:.1f}, Cmin_actual={c_min_actual:.1f}, AUC24={auc24:.1f}")

        return {
            'ke': ke,
            't_half': half_life_h,
            'Cmax_extrapolated': c_max_extrapolated, # Concentration at end of infusion
            'Cmin_actual': c_min_actual,           # Measured trough (concentration at start of next infusion)
            'AUC24': auc24
        }
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        logging.error(f"Math error calculating PK parameters (peak/trough): {e}. Inputs: "
                      f"Ctrough={c_trough}, Cpeak={c_peak_measured}, Infusion={infusion_duration_h}, "
                      f"PeakDelay={time_from_infusion_end_to_peak_draw_h}, Interval={dosing_interval_h}")
        return None


# --- 7. LLM INTERPRETATION ---
def interpret(crcl: float, pk_results: dict, target_level_desc: str) -> str:
    """Generates interpretation and recommendations using the RAG chain."""
    if qa_chain is None:
        logging.warning("QA chain not loaded. Skipping interpretation.")
        return "Interpretation unavailable: RAG system failed to load."

    # Construct a clear prompt for the LLM
    prompt = f"""
Context: You are a clinical pharmacokinetics expert interpreting vancomycin therapeutic drug monitoring (TDM) results based on established guidelines (provided via internal knowledge source).

Patient Information Summary:
- Estimated Creatinine Clearance (CrCl): {crcl:.1f} mL/min
- Selected Therapeutic Target: {target_level_desc}

Current Regimen & Monitoring Results:
{pk_results}

Task: Provide a concise interpretation of these results relative to the target. Include:
1.  Assessment: Are the current levels/AUC within the target range? Is the patient clearing the drug as expected based on CrCl?
2.  Recommendation: Suggest specific adjustments to the dose and/or interval, if needed, to achieve the target. If no change is needed, state that. Provide the recommended dose in mg, rounded to the nearest 250mg.
3.  Rationale: Briefly explain why the recommendation is being made, linking it to the PK results, target, and CrCl.
4.  Follow-up: Suggest when to re-check levels, if appropriate.

Use the provided guideline knowledge to inform your response. Be specific and clinically oriented.
"""
    logging.info(f"Generating interpretation with prompt:\n{prompt}")
    try:
        response = qa_chain.run(prompt)
        logging.info(f"LLM Interpretation received: {response}")
        return response
    except Exception as e:
        logging.error(f"Error running RAG QA chain for interpretation: {e}")
        return f"Interpretation failed: Error communicating with the RAG system ({e})."

# --- 8. STREAMLIT UI ---
st.set_page_config(page_title="Vancomycin TDM App", layout="wide")
st.title("🧪 Vancomycin TDM with RAG Interpretation")
st.markdown("Calculates PK parameters and provides guideline-based interpretation using RAG.")

# Check if RAG system loaded correctly
if qa_chain is None:
    st.error("Critical Error: The RAG guideline interpretation system could not be loaded. Please check the PDF file path and API key configuration. Interpretation features will be disabled.")
    # Optionally disable parts of the UI if RAG is essential
    # st.stop() # Or just let the user use the calculation parts

# Sidebar for inputs
st.sidebar.header("⚙️ Mode & Target")
mode = st.sidebar.radio("Select Calculation Mode:", ["Initial Dose", "Trough-Only", "Peak & Trough"])

target_level_desc = st.sidebar.selectbox(
    "Select Therapeutic Target:",
    options=[
        "Empirical (Target AUC24 400-600 mg·h/L; Trough ~10-15 mg/L)",
        "Definitive/Severe (Target AUC24 >600 mg·h/L; Trough ~15-20 mg/L)" # Adjusted range based on common practice
    ],
    index=0,
    help="Select the desired therapeutic goal based on infection type and severity. This guides interpretation."
)

# Set numerical target based on selection for calculation purposes
# Using midpoint of typical trough ranges associated with AUC targets
if "Empirical" in target_level_desc:
    target_trough_for_calc = 12.5
    target_auc_range = "400-600"
else:
    target_trough_for_calc = 17.5
    target_auc_range = ">600" # Or specify e.g., 600-800 if preferred


st.sidebar.header("👤 Patient Information")
# Use columns for better layout in sidebar
col_pid, col_ward = st.sidebar.columns(2)
with col_pid:
    pid = st.text_input("Patient ID", placeholder="e.g., MRN12345")
with col_ward:
    ward = st.text_input("Ward/Unit", placeholder="e.g., ICU, 5W")

col_age, col_wt = st.sidebar.columns(2)
with col_age:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=65, step=1)
with col_wt:
    wt = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")

col_scr, col_fem = st.sidebar.columns(2)
with col_scr:
    scr = st.number_input("SCr (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1, format="%.1f", help="Serum Creatinine")
with col_fem:
    fem = st.checkbox("Female", value=False)


# Helper function to build a downloadable report
def build_report(lines: list[str]) -> str:
    """Formats patient info and results into a text report."""
    crcl_report = calculate_crcl(age, wt, scr, fem)
    vd_report = calculate_vd(wt, age) # Calculate Vd for the report header
    hdr = [
        "--- Vancomycin TDM Report ---",
        f"Patient ID: {pid if pid else 'N/A'}",
        f"Ward/Unit: {ward if ward else 'N/A'}",
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Patient Info: Age={age} yrs, Weight={wt} kg, Sex={'Female' if fem else 'Male'}, SCr={scr} mg/dL",
        f"Estimated CrCl: {crcl_report:.1f} mL/min",
        f"Estimated Vd: {vd_report:.1f} L",
        f"Selected Target: {target_level_desc}",
        "--- Results & Interpretation ---"
    ]
    return "\n".join(hdr + lines)

# --- Main Area Logic ---
results_container = st.container() # Use a container to group outputs

if mode == "Initial Dose":
    with results_container:
        st.subheader("🚀 Initial Loading Dose Calculation")
        st.markdown("Calculates a one-time loading dose based on patient weight and estimated renal function (CrCl).")
        calc_button = st.button("Calculate Loading Dose", key="calc_initial")

        if calc_button:
            if wt <= 0:
                st.warning("Please enter a valid weight (> 0 kg).")
            else:
                with st.spinner("Calculating..."):
                    initial_dose_mg = calculate_initial_dose(age, wt, scr, fem)
                    crcl_calc = calculate_crcl(age, wt, scr, fem)
                    st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                    st.success(f"Recommended Initial Loading Dose: **{initial_dose_mg} mg** (Administer once)")
                    st.caption(f"Calculation based on {20 if crcl_calc <= 30 else 25} mg/kg target, rounded to nearest 250 mg.")

                    # Prepare report
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
    st.sidebar.markdown("Enter details about the *current* regimen and the measured trough level.")
    dose_int_current = st.sidebar.number_input("Current Dose per Interval (mg)", min_value=0, step=250, value=1000)
    dose_time = st.sidebar.time_input("Time of Last Dose Administered", value=time(8, 0), step=timedelta(minutes=15))
    sample_time = st.sidebar.time_input("Time Trough Level Drawn", value=time(19, 30), step=timedelta(minutes=15), help="Ideally drawn just before the next scheduled dose.")
    trough_measured = st.sidebar.number_input("Measured Trough Level (mg/L)", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f")

    with results_container:
        st.subheader("📉 Trough-Only Analysis")
        st.markdown("Estimates PK parameters (Ke, t½, AUC24) and suggests dose adjustments based on a single trough level.")

        # Calculate interval based on times provided
        interval_h = hours_diff(dose_time, sample_time)
        st.info(f"Calculated dosing interval based on times: **{interval_h:.2f} hours** (Trough drawn {interval_h:.2f}h after last dose). Ensure this reflects the actual dosing frequency (e.g., q12h, q24h).")

        calc_button = st.button("Run Trough-Only Analysis", key="run_trough")

        if calc_button:
             if dose_int_current <= 0 or trough_measured <= 0 or interval_h <= 0 or wt <= 0:
                 st.warning("Please ensure Dose, Measured Trough, Weight, and Time Interval are valid (> 0).")
             else:
                with st.spinner("Analyzing Trough Level..."):
                    crcl_calc = calculate_crcl(age, wt, scr, fem)
                    vd_calc = calculate_vd(wt, age) # Use age if available

                    ke_calc = calculate_ke_trough(dose_int_current, vd_calc, trough_measured, interval_h)
                    t_half_calc = math.log(2) / ke_calc if ke_calc > 0 else float('inf')
                    auc24_calc = calculate_auc24_trough(dose_int_current, vd_calc, trough_measured, interval_h)
                    new_dose_calc = calculate_new_dose_trough(dose_int_current, vd_calc, trough_measured, interval_h, target_trough=target_trough_for_calc)

                    # Display results
                    st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Estimated Vd", value=f"{vd_calc:.1f} L")
                    col2.metric(label="Estimated Ke", value=f"{ke_calc:.4f} h⁻¹" if ke_calc > 0 else "N/A")
                    col3.metric(label="Estimated t½", value=f"{t_half_calc:.1f} h" if t_half_calc != float('inf') else "N/A")

                    st.metric(label=f"Estimated AUC₂₄", value=f"{auc24_calc:.1f} mg·h/L",
                              help=f"Target range: {target_auc_range} mg·h/L")
                    st.metric(label=f"Suggested New Dose (for target ~{target_trough_for_calc} mg/L)",
                              value=f"{new_dose_calc} mg q{interval_h:.1f}h" if new_dose_calc > 0 else "Calculation Error",
                              help=f"Dose rounded to nearest 250mg. Interval based on input times.")

                    # Prepare results dict for interpretation
                    pk_results = {
                        'Calculation Mode': 'Trough-Only',
                        'Current Dose': f"{dose_int_current} mg",
                        'Dosing Interval (used)': f"{interval_h:.2f} h",
                        'Measured Trough': f"{trough_measured:.1f} mg/L",
                        'Estimated Vd': f"{vd_calc:.1f} L",
                        'Estimated Ke': f"{ke_calc:.4f} h⁻¹" if ke_calc > 0 else "N/A",
                        'Estimated t½': f"{t_half_calc:.1f} h" if t_half_calc != float('inf') else "N/A",
                        'Estimated AUC24': f"{auc24_calc:.1f} mg·h/L",
                        'Suggested New Dose (for target)': f"{new_dose_calc} mg q{interval_h:.1f}h" if new_dose_calc > 0 else "N/A"
                    }

                    # Get interpretation if RAG chain is loaded
                    if qa_chain:
                        st.subheader("💬 RAG Interpretation & Recommendation")
                        with st.spinner("Generating interpretation using guideline knowledge..."):
                            interpretation_text = interpret(crcl_calc, pk_results, target_level_desc)
                            st.markdown(interpretation_text) # Display interpretation
                    else:
                        st.warning("Interpretation unavailable as RAG system failed to load.")
                        interpretation_text = "N/A (RAG system not loaded)"


                    # Prepare report
                    report_lines = [
                        f"Mode: Trough-Only Analysis",
                        f"Current Dose: {dose_int_current} mg",
                        f"Last Dose Time: {dose_time.strftime('%H:%M')}",
                        f"Trough Sample Time: {sample_time.strftime('%H:%M')}",
                        f"Calculated Interval: {interval_h:.2f} h",
                        f"Measured Trough: {trough_measured:.1f} mg/L",
                        f"--- Calculated Parameters ---",
                        f"Est. Vd: {vd_calc:.1f} L",
                        f"Est. Ke: {ke_calc:.4f} h⁻¹" if ke_calc > 0 else "N/A",
                        f"Est. t½: {t_half_calc:.1f} h" if t_half_calc != float('inf') else "N/A",
                        f"Est. AUC24: {auc24_calc:.1f} mg·h/L",
                        f"--- Recommendation ---",
                        f"Suggested New Dose (Target ~{target_trough_for_calc} mg/L): {new_dose_calc} mg q{interval_h:.1f}h" if new_dose_calc > 0 else "N/A",
                        f"--- RAG Interpretation ---",
                        interpretation_text.replace('\n', '\n  ') # Indent interpretation for report
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
    st.sidebar.markdown("Enter infusion times and both peak and trough levels.")
    infusion_start_time = st.sidebar.time_input("Infusion Start Time", value=time(8, 0), step=timedelta(minutes=15))
    infusion_end_time = st.sidebar.time_input("Infusion End Time", value=time(9, 0), step=timedelta(minutes=15), help="End time of the vancomycin infusion.")
    peak_sample_time = st.sidebar.time_input("Peak Sample Time", value=time(10, 0), step=timedelta(minutes=15), help="Time peak level was drawn (typically 1-2h after infusion ends).")
    trough_sample_time = st.sidebar.time_input("Trough Sample Time", value=time(19, 30), step=timedelta(minutes=15), help="Time trough level was drawn (just before next dose).")

    # Get dose that corresponds to these levels
    dose_for_levels = st.sidebar.number_input("Dose Administered (mg)", min_value=0, step=250, value=1000, help="The dose given *before* these peak/trough levels were drawn.")

    c_trough_measured = st.sidebar.number_input("Measured Trough (Cmin) (mg/L)", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f")
    c_peak_measured = st.sidebar.number_input("Measured Peak (Cpeak) (mg/L)", min_value=0.0, max_value=200.0, value=30.0, step=0.1, format="%.1f", help="This is the level measured at 'Peak Sample Time', *not* the true Cmax.")


    with results_container:
        st.subheader("📊 Peak & Trough Analysis (Sawchuk-Zaske)")
        st.markdown("Calculates individual PK parameters (Ke, t½, Vd, AUC24) using measured peak and trough levels.")

        # Calculate time durations
        infusion_duration_h = hours_diff(infusion_start_time, infusion_end_time)
        time_from_infusion_end_to_peak_draw_h = hours_diff(infusion_end_time, peak_sample_time)
        dosing_interval_h = hours_diff(infusion_start_time, trough_sample_time) # Assumes trough drawn right before next dose starts at same time as previous

        st.info(f"Calculated Durations: Infusion={infusion_duration_h:.2f}h | Delay to Peak Draw={time_from_infusion_end_to_peak_draw_h:.2f}h | Dosing Interval={dosing_interval_h:.2f}h")

        calc_button = st.button("Run Peak & Trough Analysis", key="run_peak_trough")

        if calc_button:
            # Validate inputs
            valid_inputs = True
            if infusion_duration_h <= 0:
                st.warning("Infusion End Time must be after Start Time.")
                valid_inputs = False
            if time_from_infusion_end_to_peak_draw_h < 0: # Can be 0 if drawn exactly at end, but usually delayed
                st.warning("Peak Sample Time should be after Infusion End Time.")
                valid_inputs = False
            if dosing_interval_h <= infusion_duration_h:
                st.warning("Trough Sample Time implies a very short or negative dosing interval.")
                valid_inputs = False
            if c_trough_measured <= 0 or c_peak_measured <= 0:
                st.warning("Measured Peak and Trough levels must be greater than 0.")
                valid_inputs = False
            if c_peak_measured <= c_trough_measured:
                st.warning("Measured Peak level must be greater than Measured Trough level.")
                valid_inputs = False
            if dose_for_levels <= 0:
                 st.warning("Please enter the dose administered (> 0 mg).")
                 valid_inputs = False
            if wt <= 0:
                 st.warning("Please enter patient weight (> 0 kg).")
                 valid_inputs = False

            if valid_inputs:
                with st.spinner("Analyzing Peak & Trough Levels..."):
                    crcl_calc = calculate_crcl(age, wt, scr, fem)

                    # Calculate individual PK parameters using Sawchuk-Zaske
                    pk_params = calculate_pk_params_peak_trough(
                        c_trough=c_trough_measured,
                        c_peak_measured=c_peak_measured,
                        infusion_duration_h=infusion_duration_h,
                        time_from_infusion_end_to_peak_draw_h=time_from_infusion_end_to_peak_draw_h,
                        dosing_interval_h=dosing_interval_h
                    )

                    if pk_params:
                        ke_ind = pk_params['ke']
                        thalf_ind = pk_params['t_half']
                        cmax_ind = pk_params['Cmax_extrapolated']
                        cmin_ind = pk_params['Cmin_actual'] # This is just the measured trough
                        auc24_ind = pk_params['AUC24']

                        # Calculate Vd using Dose, Ke, Cmax, Cmin, Infusion Duration
                        # Formula: Vd = [Dose * (1 - e^(-Ke*Tinf))] / [Ke * Tinf * (Cmax - Cmin * e^(-Ke*Tinf))] -> Complex!
                        # Simpler Vd: Dose = Vd * Ke * AUC_interval -> Vd = Dose / (Ke * AUC_interval)
                        # AUC_interval = AUC24 / (24 / dosing_interval_h)
                        try:
                            auc_interval_ind = auc24_ind / (24 / dosing_interval_h) if dosing_interval_h > 0 else 0
                            if ke_ind > 0 and auc_interval_ind > 0:
                                vd_ind = dose_for_levels / (ke_ind * auc_interval_ind)
                            else:
                                vd_ind = 0 # Cannot calculate
                                st.warning("Could not calculate Vd due to zero Ke or AUC_interval.")
                        except ZeroDivisionError:
                             vd_ind = 0
                             st.warning("Could not calculate Vd due to division by zero (check interval).")


                        # Display results
                        st.metric(label="Estimated CrCl", value=f"{crcl_calc:.1f} mL/min")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Individual Vd", value=f"{vd_ind:.1f} L" if vd_ind > 0 else "N/A")
                        col2.metric(label="Individual Ke", value=f"{ke_ind:.4f} h⁻¹")
                        col3.metric(label="Individual t½", value=f"{thalf_ind:.1f} h")

                        col4, col5 = st.columns(2)
                        col4.metric(label="Est. Cmax (End of Infusion)", value=f"{cmax_ind:.1f} mg/L")
                        col5.metric(label="Measured Cmin (Trough)", value=f"{cmin_ind:.1f} mg/L")

                        st.metric(label=f"Individual AUC₂₄", value=f"{auc24_ind:.1f} mg·h/L",
                                  help=f"Target range: {target_auc_range} mg·h/L")

                        # Suggest new dose (optional, requires target)
                        # New Dose = TargetAUC24 * CL / (24 / Interval)
                        # CL = Ke * Vd
                        new_dose_suggestion = "N/A"
                        if ke_ind > 0 and vd_ind > 0 and dosing_interval_h > 0:
                             cl_ind = ke_ind * vd_ind
                             # Use midpoint of target AUC range for calculation if range given
                             target_auc_numeric = float(target_auc_range.split('-')[0]) if '-' in target_auc_range else float(target_auc_range.replace('>', ''))
                             if target_auc_numeric > 0:
                                 try:
                                     # Dose_per_interval = TargetAUC24 * CL_ind / (24 / interval) -> This is wrong
                                     # Correct: Dose_per_interval = TargetAUC_interval * CL_ind
                                     # TargetAUC_interval = TargetAUC24 / (24 / interval)
                                     target_auc_interval = target_auc_numeric * (dosing_interval_h / 24.0)
                                     new_dose_raw = target_auc_interval * cl_ind
                                     new_dose_rounded = round_dose(new_dose_raw)
                                     new_dose_suggestion = f"{new_dose_rounded} mg q{dosing_interval_h:.1f}h"
                                     st.metric(label=f"Suggested Dose (for Target AUC ~{target_auc_numeric})", value=new_dose_suggestion)
                                 except Exception as dose_calc_err:
                                     st.warning(f"Could not calculate suggested dose: {dose_calc_err}")
                        else:
                             st.warning("Cannot suggest new dose without valid Ke, Vd, and Interval.")


                        # Prepare results dict for interpretation
                        pk_results = {
                            'Calculation Mode': 'Peak & Trough',
                            'Dose Administered': f"{dose_for_levels} mg",
                            'Infusion Duration': f"{infusion_duration_h:.2f} h",
                            'Time to Peak Draw (post-infusion)': f"{time_from_infusion_end_to_peak_draw_h:.2f} h",
                            'Dosing Interval': f"{dosing_interval_h:.2f} h",
                            'Measured Peak (at draw time)': f"{c_peak_measured:.1f} mg/L",
                            'Measured Trough (Cmin)': f"{c_trough_measured:.1f} mg/L",
                            'Individual Vd': f"{vd_ind:.1f} L" if vd_ind > 0 else "N/A",
                            'Individual Ke': f"{ke_ind:.4f} h⁻¹",
                            'Individual t½': f"{thalf_ind:.1f} h",
                            'Est. Cmax (End of Infusion)': f"{cmax_ind:.1f} mg/L",
                            'Individual AUC24': f"{auc24_ind:.1f} mg·h/L",
                            'Suggested New Dose (for target)': new_dose_suggestion
                        }

                        # Get interpretation if RAG chain is loaded
                        if qa_chain:
                            st.subheader("💬 RAG Interpretation & Recommendation")
                            with st.spinner("Generating interpretation using guideline knowledge..."):
                                interpretation_text = interpret(crcl_calc, pk_results, target_level_desc)
                                st.markdown(interpretation_text) # Display interpretation
                        else:
                            st.warning("Interpretation unavailable as RAG system failed to load.")
                            interpretation_text = "N/A (RAG system not loaded)"

                        # Prepare report
                        report_lines = [
                            f"Mode: Peak & Trough Analysis",
                            f"Dose Administered: {dose_for_levels} mg",
                            f"Infusion Start: {infusion_start_time.strftime('%H:%M')}, End: {infusion_end_time.strftime('%H:%M')} (Duration: {infusion_duration_h:.2f} h)",
                            f"Peak Sample Time: {peak_sample_time.strftime('%H:%M')} ({time_from_infusion_end_to_peak_draw_h:.2f} h post-infusion)",
                            f"Trough Sample Time: {trough_sample_time.strftime('%H:%M')}",
                            f"Calculated Interval: {dosing_interval_h:.2f} h",
                            f"Measured Peak (at draw): {c_peak_measured:.1f} mg/L",
                            f"Measured Trough (Cmin): {c_trough_measured:.1f} mg/L",
                            f"--- Calculated Individual Parameters ---",
                            f"Ind. Vd: {vd_ind:.1f} L" if vd_ind > 0 else "N/A",
                            f"Ind. Ke: {ke_ind:.4f} h⁻¹",
                            f"Ind. t½: {thalf_ind:.1f} h",
                            f"Est. Cmax (End of Infusion): {cmax_ind:.1f} mg/L",
                            f"Ind. AUC24: {auc24_ind:.1f} mg·h/L",
                            f"--- Recommendation ---",
                            f"Suggested New Dose (Target AUC ~{target_auc_numeric if 'target_auc_numeric' in locals() else 'N/A'}): {new_dose_suggestion}",
                            f"--- RAG Interpretation ---",
                            interpretation_text.replace('\n', '\n  ') # Indent interpretation
                        ]
                        report_data = build_report(report_lines)
                        st.download_button(
                            label="📥 Download Report (.txt)",
                            data=report_data,
                            file_name=f"{pid or 'patient'}_vanco_peak_trough_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )

                    else:
                        st.error("Failed to calculate PK parameters from Peak & Trough data. Please check the input values (levels, times) for consistency.")
                        pk_results = {"Error": "Failed to calculate PK parameters from Peak & Trough data."}
                        interpretation_text = "N/A (Calculation Error)"


            else:
                 st.error("Please correct the input errors before running the analysis.")
                 pk_results = {"Error": "Invalid inputs provided."}
                 interpretation_text = "N/A (Input Error)"


# --- Footer ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational and informational purposes only. Consult official guidelines and clinical judgment for patient care decisions.")
st.caption(f"Guideline source loaded: `{PDF_FILENAME}` | App last updated: {datetime.now().strftime('%Y-%m-%d')}")

