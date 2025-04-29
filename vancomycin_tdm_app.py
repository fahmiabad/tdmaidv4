# vancomycin_tdm_app.py
"""
Single-file Streamlit app for Vancomycin TDM with RAG-guided LLM interpretation,
time-based inputs, and target-level selection.
"""
import os
import math
import streamlit as st
from datetime import datetime, timedelta, time

# --- 1. SECRET ---
# Set your OpenAI API key for deployment
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 2. IMPORTS FOR RAG ---
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# --- 3. LOAD & INDEX GUIDELINE PDF ---
@st.cache_resource
def load_rag_chain(pdf_path: str) -> RetrievalQA:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(chunks, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=index.as_retriever()
    )
    return qa

qa_chain = load_rag_chain(
    'clinical-pharmacokinetics-pharmacy-handbook-ccph-2nd-edition-rev-2.0_0-2.pdf'
)

# --- 4. TIME DIFFERENCE HELPER ---
def hours_diff(start: time, end: time) -> float:
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)
    if dt_end < dt_start:
        dt_end += timedelta(days=1)
    return (dt_end - dt_start).total_seconds() / 3600

# --- 5. PK CALC FUNCTIONS ---
def calculate_crcl(age, weight, scr, female=False):
    base = ((140 - age) * weight) / (72 * scr)
    return base * 0.85 if female else base

def calculate_vd(weight, age=None):
    return (0.17 * age + 0.22 * weight + 15) if age else 0.7 * weight

def round_dose(dose):
    return int(round(dose / 250) * 250)

def calculate_initial_dose(age, weight, scr, female):
    crcl = calculate_crcl(age, weight, scr, female)
    mg_kg = 25 if crcl > 30 else 20
    return round_dose(weight * mg_kg)

def calculate_ke_trough(dose_int, vd, trough, interval_h):
    if trough <= 0:
        return 0
    cmax = trough + dose_int / vd
    return math.log(cmax / trough) / interval_h

def calculate_new_dose_trough(
    dose_int, vd, trough, interval_h, target=15.0
):
    ke = calculate_ke_trough(dose_int, vd, trough, interval_h)
    if ke <= 0:
        return dose_int
    return target * vd * (1 - math.exp(-ke * interval_h)) / math.exp(-ke * interval_h)

def calculate_auc24_trough(dose_int, vd, trough, interval_h):
    ke = calculate_ke_trough(dose_int, vd, trough, interval_h)
    if ke <= 0:
        return 0
    daily = dose_int * (24 / interval_h)
    cl = ke * vd
    return daily / cl

def calculate_prepost(
    cmin, cpost, infusion_h, peak_delay_h, interval_h
):
    ke = math.log(cmin / cpost) / (
        interval_h - infusion_h - peak_delay_h
    )
    half = math.log(2) / ke
    cmax = cpost * math.exp(ke * peak_delay_h)
    auc_inf = infusion_h * (cmin + cmax) / 2
    auc_elim = (cmax - cmin) / ke
    auc24 = (auc_inf + auc_elim) * (24 / interval_h)
    return {
        'ke': ke,
        't_half': half,
        'Cmax': cmax,
        'Cmin': cmin,
        'AUC24': auc24
    }

# --- 6. LLM INTERPRETATION ---
def interpret(crcl, res, target_level):
    prompt = f"""
You are a clinical pharmacokinetics expert. Using the guideline:
- CrCl: {crcl:.1f} mL/min
- Target level: {target_level}
- PK results: {res}
Provide a concise interpretation, dosing recommendation, and rationale.
"""
    return qa_chain.run(prompt)

# --- 7. STREAMLIT UI ---
st.set_page_config(page_title="Vancomycin TDM App", layout="wide")
st.title("🧪 Vancomycin TDM with Target Selection & RAG")

mode = st.sidebar.radio(
    "Mode:", ["Initial Dose", "Trough-Only", "Peak & Trough"]
)

st.sidebar.header("Therapeutic Target")
target_level = st.sidebar.selectbox(
    "Select target:",
    [
        "Empirical (AUC24 400-600; trough 10-15)",
        "Definitive (AUC24 600-800; trough 15-20)"
    ]
)
if "Empirical" in target_level:
    target_trough = 12.5
else:
    target_trough = 17.5

st.sidebar.header("Patient Info")
pid = st.sidebar.text_input("Patient ID")
ward = st.sidebar.text_input("Ward/Unit")
age = st.sidebar.number_input("Age (yr)", 0, 120, 65)
wt = st.sidebar.number_input("Weight (kg)", 0.0, 300.0, 70.0)
scr = st.sidebar.number_input("Scr (mg/dL)", 0.1, 10.0, 1.0)
fem = st.sidebar.checkbox("Female")

def build_report(lines):
    hdr = [
        f"Patient ID: {pid}",
        f"Ward: {ward}",
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Age: {age}, Wt: {wt} kg, Female: {fem}",
        f"Scr: {scr} mg/dL", f"Target: {target_level}"
    ]
    return "\n".join(hdr + lines)

if mode == "Initial Dose":
    st.header("Initial Loading Dose")
    if st.button("Calculate Loading Dose"):
        dose = calculate_initial_dose(age, wt, scr, fem)
        st.success(f"Loading Dose: {dose} mg (once)")
        rpt = build_report([f"Loading dose: {dose} mg"])
        st.download_button(
            "Download TXT",
            data=rpt,
            file_name=f"{pid}_initial.txt"
        )
elif mode == "Trough-Only":
    st.header("Trough-Only Analysis")
    dose_time = st.sidebar.time_input(
        "Last Dose Time", value=time(8,0), step=900
    )
    sample_time = st.sidebar.time_input(
        "Trough Sample Time", value=time(20,0), step=900
    )
    dose_int = st.sidebar.number_input(
        "Dose per Interval (mg)", 0, step=250, value=1000
    )
    trough = st.sidebar.number_input(
        "Measured Trough (mg/L)", 0.1, 100.0, 10.0
    )
    interval_h = hours_diff(dose_time, sample_time)
    if st.button("Run Trough-Only"):
        crcl = calculate_crcl(age, wt, scr, fem)
        vd = calculate_vd(wt, age)
        ke = calculate_ke_trough(
            dose_int, vd, trough, interval_h
        )
        t_half = math.log(2) / ke if ke > 0 else 0
        new_dose = round_dose(
            calculate_new_dose_trough(
                dose_int, vd, trough, interval_h, target_trough
            )
        )
        auc24 = calculate_auc24_trough(
            dose_int, vd, trough, interval_h
        )
        st.write(
            f"CrCl: {crcl:.1f} mL/min | Vd: {vd:.1f} L"
        )
        st.write(
            f"Interval: {interval_h:.2f} h | Ke: {ke:.3f} h⁻¹ | t½: {t_half:.1f} h"
        )
        st.write(
            f"AUC24: {auc24:.1f} mg·h/L | New Dose: {new_dose} mg q{interval_h:.1f}h"
        )
        res = {
            'mode':'trough',
            'interval':interval_h,
            'ke':ke,
            't_half':t_half,
            'AUC24':auc24,
            'new_dose':new_dose,
            'target':target_level
        }
        interp = interpret(crcl, res, target_level)
        st.subheader("Interpretation")
        st.write(interp)
        rpt = build_report([
            f"Mode: Trough-Only",
            f"Dose Time: {dose_time}",
            f"Sample Time: {sample_time}",
            f"Trough: {trough} mg/L",
            f"New Dose: {new_dose} mg q{interval_h:.1f}h",
            f"AUC24: {auc24:.1f}",
            f"Interp: {interp}"
        ])
        st.download_button(
            "Download TXT",
            data=rpt,
            file_name=f"{pid}_trough.txt"
        )
else:
    st.header("Peak & Trough Analysis")
    start = st.sidebar.time_input(
        "Infusion Start Time", value=time(8,0), step=900
    )
    end = st.sidebar.time_input(
        "Infusion End Time", value=time(9,0), step=900
    )
    peak = st.sidebar.time_input(
        "Peak Sample Time", value=time(10,0), step=900
    )
    trough_time = st.sidebar.time_input(
        "Trough Sample Time", value=time(20,0), step=900
    )
    cmin = st.sidebar.number_input(
        "Cmin (mg/L)", 0.1, 100.0, 10.0
    )
    cpost = st.sidebar.number_input(
        "Cpost (mg/L)", 0.1, 200.0, 30.0
    )
    infusion_h = hours_diff(start, end)
    peak_delay_h = hours_diff(end, peak)
    interval_h = hours_diff(start, trough_time)
    if st.button("Run Peak & Trough"):
        crcl = calculate_crcl(age, wt, scr, fem)
        params = calculate_prepost(
            cmin, cpost, infusion_h, peak_delay_h, interval_h
        )
        st.write(
            f"CrCl: {crcl:.1f} mL/min | Vd: {calculate_vd(wt, age):.1f} L"
        )
        st.write(
            f"Infusion: {infusion_h:.2f} h | Peak Delay: {peak_delay_h:.2f} h | Interval: {interval_h:.2f} h"
        )
        st.write(
            f"Ke: {params['ke']:.3f} h⁻¹ | t½: {params['t_half']:.1f} h"
        )
        st.write(
            f"Cmax: {params['Cmax']:.1f} mg/L | Cmin: {params['Cmin']:.1f} mg/L"
        )
        st.write(f"AUC24: {params['AUC24']:.1f} mg·h/L")
        res = {'mode':'prepost', 'infusion':infusion_h, 'peak_delay':peak_delay_h, 'interval':interval_h, **params, 'target':target_level}
        interp = interpret(crcl, res, target_level)
        st.subheader("Interpretation")
        st.write(interp)
        rpt = build_report([
            f"Mode: Peak & Trough",
            f"Start: {start}", f"End: {end}", f"Peak: {peak}", f"Trough: {trough_time}",
            f"Cmin: {cmin}", f"Cpost: {cpost}",
            f"AUC24: {params['AUC24']:.1f}", f"Interp: {interp}"
        ])
        st.download_button(
            "Download TXT",
            data=rpt,
            file_name=f"{pid}_prepost.txt"
        )
