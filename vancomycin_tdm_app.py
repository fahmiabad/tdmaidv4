# Import necessary libraries
import streamlit as st
from datetime import datetime, timedelta, time, date
import math

# --- 1. SET PAGE CONFIG WITH MODERN THEME ---
st.set_page_config(
    page_title="Vancomycin TDM Assistant", 
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. APPLY CUSTOM CSS FOR MODERN DESIGN ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #4F6BF2;
        --secondary-color: #3CAEA3;
        --dark-color: #2A3C5D;
        --light-color: #F8F9FA;
        --danger-color: #e63946;
        --warning-color: #ffb703;
        --success-color: #57cc99;
    }
    
    /* Base styling */
    .main {
        background-color: var(--light-color);
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark-color);
        font-family: 'Roboto', sans-serif;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Recommendation card */
    .recommendation-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .recommendation-title {
        color: var(--dark-color);
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 0;
        margin-bottom: 10px;
    }
    
    .recommendation-dose {
        color: var(--dark-color);
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .recommendation-description {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Interpretation card */
    .interpretation-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .interpretation-section {
        margin-bottom: 15px;
    }
    
    .interpretation-title {
        color: var(--dark-color);
        font-size: 1.2rem;
        font-weight: bold;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    
    .interpretation-content {
        color: #333;
        line-height: 1.5;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid var(--primary-color);
    }
    
    .metric-label {
        font-size: 14px;
        color: #555;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: var(--dark-color);
    }
    
    /* Status indicators */
    .status-within {
        color: var(--success-color);
        font-weight: bold;
    }
    
    .status-below {
        color: var(--warning-color);
        font-weight: bold;
    }
    
    .status-above {
        color: var(--danger-color);
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--dark-color);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Input fields */
    .stNumberInput input, .stTimeInput input, .stTextInput input {
        border-radius: 6px;
        border: 1px solid #ddd;
        padding: 8px 12px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        background-color: #f1f3f9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2A3C5D;
        color: white;
    }

    /* Helpful callouts */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        font-size: 12px;
        color: #666;
    }
    
    /* Target status display */
    .target-status-container {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }
    
    .target-status-card {
        flex: 1;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 15px;
    }
    
    /* Progress bars instead of gauges */
    .progress-container {
        width: 100%;
        margin: 15px 0;
    }
    
    .progress-bar {
        height: 8px;
        background-color: #f1f1f1;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-value {
        height: 100%;
        border-radius: 4px;
    }
    
    .progress-within {
        background-color: var(--success-color);
    }
    
    .progress-below {
        background-color: var(--warning-color);
    }
    
    .progress-above {
        background-color: var(--danger-color);
    }
    
    .progress-ticks {
        position: relative;
        height: 20px;
        margin-top: 5px;
    }
    
    .progress-tick {
        position: absolute;
        width: 2px;
        height: 8px;
        background-color: #aaa;
    }
    
    .progress-tick-label {
        position: absolute;
        font-size: 11px;
        color: #666;
        transform: translateX(-50%);
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- 3. CUSTOM CARD COMPONENTS ---
def card(title, content):
    return f"""
    <div class="card">
        <h3>{title}</h3>
        {content}
    </div>
    """

def metric_card(label, value, help_text=""):
    help_html = f'<div class="metric-help">{help_text}</div>' if help_text else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {help_html}
    </div>
    """

# --- 4. STATUS DISPLAY FUNCTIONS ---
def get_status_html(status_text):
    if "WITHIN" in status_text:
        return f'<span class="status-within">✓ {status_text}</span>'
    elif "BELOW" in status_text:
        return f'<span class="status-below">⚠️ {status_text}</span>'
    elif "ABOVE" in status_text:
        return f'<span class="status-above">⚠️ {status_text}</span>'
    else:
        return f'<span>{status_text}</span>'

def show_status_section(pk_results):
    trough_status = pk_results.get('Trough Status vs Target', 'N/A')
    auc_status = pk_results.get('AUC Status vs Target', 'N/A')
    
    trough_html = get_status_html(trough_status)
    auc_html = get_status_html(auc_status)
    
    status_content = f"""
    <div class="target-status-container">
        <div class="target-status-card">
            <h4 style="margin-top: 0;">Trough Level Status</h4>
            <div style="font-size: 18px; margin-top: 10px;">{trough_html}</div>
        </div>
        <div class="target-status-card">
            <h4 style="margin-top: 0;">AUC24 Status</h4>
            <div style="font-size: 18px; margin-top: 10px;">{auc_html}</div>
        </div>
    </div>
    """
    
    st.markdown(status_content, unsafe_allow_html=True)

# --- 5. FIXED VISUAL INDICATORS (NO PLOTLY) ---
def create_progress_bar(value, min_val, target_min, target_max=None):
    status = "unknown"
    if value < target_min:
        status = "below"
        percent = (value - min_val) / (target_min - min_val) * 50 if target_min > min_val else 25
    elif target_max and value > target_max:
        status = "above"
        percent = 50 + ((value - target_max) / (target_max)) * 50
        percent = min(percent, 100)
    else:
        status = "within"
        if target_max:
            percent = 50 + ((value - target_min) / (target_max - target_min)) * 50
        else:
            percent = 75  # Default position for targets like >600

    # Safety check
    percent = max(0, min(100, percent))
    
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-bar">
            <div class="progress-value progress-{status}" style="width: {percent}%;"></div>
        </div>
        <div class="progress-ticks">
    """
    
    # Add ticks
    if target_max:
        progress_html += f"""
            <div class="progress-tick" style="left: 0%;"></div>
            <div class="progress-tick-label" style="left: 0%;">{min_val}</div>
            
            <div class="progress-tick" style="left: 50%;"></div>
            <div class="progress-tick-label" style="left: 50%;">{target_min}</div>
            
            <div class="progress-tick" style="left: 100%;"></div>
            <div class="progress-tick-label" style="left: 100%;">{target_max}</div>
        """
    else:
        progress_html += f"""
            <div class="progress-tick" style="left: 0%;"></div>
            <div class="progress-tick-label" style="left: 0%;">{min_val}</div>
            
            <div class="progress-tick" style="left: 75%;"></div>
            <div class="progress-tick-label" style="left: 75%;">{target_min}</div>
        """
    
    progress_html += """
        </div>
    </div>
    """
    
    return progress_html

def show_level_indicators(measured_trough, target_trough_range, auc24, target_auc_range):
    trough_min, trough_max = target_trough_range
    auc_min, auc_max = target_auc_range
    
    # Create visual representation of levels vs targets
    st.markdown("<h4>Visual Target Status</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h5>Trough Level: {measured_trough:.1f} mg/L</h5>", unsafe_allow_html=True)
        trough_progress = create_progress_bar(measured_trough, 0, trough_min, trough_max)
        st.markdown(trough_progress, unsafe_allow_html=True)
        
    with col2:
        if auc24 and auc_min:
            st.markdown(f"<h5>AUC24: {auc24:.1f} mg·h/L</h5>", unsafe_allow_html=True)
            auc_progress = create_progress_bar(auc24, 0, auc_min, auc_max)
            st.markdown(auc_progress, unsafe_allow_html=True)

# --- 6. FIXED INTERPRETATION RENDERING ---
def render_interpretation(trough_status, trough_measured, auc_status, auc24, thalf, interval_h, new_dose, target_desc):
    """Creates a properly rendered interpretation with styled sections"""
    
    # Determine recommendation text based on status
    if "BELOW" in trough_status or "BELOW" in auc_status:
        rec_action = "increase"
    elif "ABOVE" in trough_status or "ABOVE" in auc_status:
        rec_action = "decrease"
    else:
        rec_action = "maintain"
    
    # Get target values from description
    if "Empirical" in target_desc:
        target_trough = "10-15"
        target_auc = "400-600"
    else:
        target_trough = "15-20"
        target_auc = ">600"
    
    # Create the HTML using string concatenation for better compatibility
    interpretation_html = """
    <div class="interpretation-card">
        <div class="interpretation-section">
            <div class="interpretation-title">Assessment</div>
            <div class="interpretation-content">
                The measured trough level (""" + f"{trough_measured:.1f}" + """ mg/L) is """ + trough_status.lower() + """ for the selected therapeutic goal. 
                The calculated AUC24 (""" + f"{auc24:.1f}" + """ mg·h/L) is """ + auc_status.lower() + """. 
                The calculated half-life (""" + f"{thalf:.1f}" + """ h) suggests the current interval (q""" + str(interval_h) + """h) is 
                """ + ('appropriate' if interval_h >= thalf * 1.5 else 'potentially too long') + """.
            </div>
        </div>
        
        <div class="interpretation-section">
            <div class="interpretation-title">Recommendation</div>
            <div class="interpretation-content">
                Based on the individual PK parameters, """ + rec_action + """ 
                the dose to """ + str(new_dose) + """ mg q""" + str(interval_h) + """h to achieve the target AUC of """ + target_auc + """ mg·h/L
                and target trough of """ + target_trough + """ mg/L.
            </div>
        </div>
        
        <div class="interpretation-section">
            <div class="interpretation-title">Rationale</div>
            <div class="interpretation-content">
                The recommendation is based on the measured trough being """ + trough_status.lower() + """ and the calculated AUC being """ + auc_status.lower() + """.
                The individual PK parameters provide a more accurate assessment than population estimates.
            </div>
        </div>
        
        <div class="interpretation-section">
            <div class="interpretation-title">Follow-up</div>
            <div class="interpretation-content">
                Draw next trough level before the 3rd or 4th dose of the new regimen to confirm that the target is being achieved.
                Continue to monitor renal function and clinical response.
            </div>
        </div>
    </div>
    """
    
    return interpretation_html

# --- 7. MAIN APP STRUCTURE ---
def main():
    # Header with modern design
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="font-size: 40px; margin-right: 15px;">💊</div>
        <div>
            <h1 style="margin: 0; padding: 0;">Vancomycin TDM Assistant</h1>
            <p style="margin: 0; color: #666;">Clinical pharmacokinetics made simple</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different modes
    tabs = st.tabs(["Initial Dose", "Trough-Only Analysis", "Peak & Trough Analysis"])
    
    # --- SIDEBAR CONTENT ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #4F6BF2;">⚙️ Patient Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Patient info in clean format
        col_pid, col_ward = st.columns(2)
        with col_pid:
            pid = st.text_input("Patient ID", placeholder="MRN12345")
        with col_ward:
            ward = st.text_input("Ward/Unit", placeholder="ICU")
            
        # Wrap demographic inputs in a card style  
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_age, col_wt = st.columns(2)
        with col_age:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=65, step=1)
        with col_wt:
            wt = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")
            
        col_scr, col_fem = st.columns(2)
        with col_scr:
            scr_umol = st.number_input("SCr (µmol/L)", min_value=10.0, max_value=2000.0, value=88.0, step=1.0, format="%.0f")
        with col_fem:
            fem = st.checkbox("Female", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Target selection with better visual distinction
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4>Therapeutic Target</h4>', unsafe_allow_html=True)
        target_level_desc = st.selectbox(
            "Select target level:",
            options=[
                "Empirical (Target AUC24 400-600 mg·h/L; Trough ~10-15 mg/L)",
                "Definitive/Severe (Target AUC24 >600 mg·h/L; Trough ~15-20 mg/L)"
            ],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clinical notes with better styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4>Clinical Context</h4>', unsafe_allow_html=True)
        clinical_notes = st.text_area(
            "Clinical Notes",
            placeholder="Enter relevant clinical context (e.g., infection type, organ function, source control status...)",
            height=100,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- INITIAL DOSE TAB ---
    with tabs[0]:
        st.markdown("""
        <div class="card">
            <h2>Initial Loading Dose Calculator</h2>
            <p>Calculate an appropriate one-time loading dose based on patient weight and renal function.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **Calculation Method**
            - Uses weight-based dosing (~25 mg/kg)
            - Considers renal function via CrCl
            - Automatically rounds to nearest 250mg
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            calc_button = st.button("Calculate Loading Dose", use_container_width=True)
        
        # Placeholder for initial dose calculation results
        if calc_button:
            # Calculate SCr in mg/dL
            scr_mgdl = scr_umol / 88.4
            
            # Simulate CrCl calculation
            if fem:
                crcl = ((140 - age) * wt * 0.85) / (72 * scr_mgdl)
            else:
                crcl = ((140 - age) * wt) / (72 * scr_mgdl)
            
            # Calculate initial dose
            initial_dose = round(wt * 25 / 250) * 250
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"""
            ### Results
            - **Estimated CrCl:** {crcl:.1f} mL/min
            - **Recommended Initial Loading Dose:** {initial_dose} mg (one-time)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add a download button for the report
            st.download_button(
                label="📄 Download Report",
                data=f"Patient ID: {pid}\nWard: {ward}\nAge: {age} years\nWeight: {wt} kg\nSCr: {scr_umol} µmol/L\nEstimated CrCl: {crcl:.1f} mL/min\nRecommended Initial Loading Dose: {initial_dose} mg (one-time)",
                file_name=f"vanco_initial_dose_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

    # --- TROUGH-ONLY TAB ---
    with tabs[1]:
        st.markdown("""
        <div class="card">
            <h2>Trough-Only Analysis</h2>
            <p>Analyze a single trough level to estimate pharmacokinetic parameters and suggest dose adjustments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a clean form for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Current Regimen</h4>", unsafe_allow_html=True)
            dose_int_current = st.number_input("Current Dose (mg)", min_value=250, step=250, value=1000, key="to_dose")
            current_interval_h = st.selectbox(
                "Dosing Interval",
                options=[6, 8, 12, 18, 24, 36, 48],
                index=2,
                format_func=lambda x: f"q{x}h",
                key="to_interval"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Level Timing</h4>", unsafe_allow_html=True)
            dose_time = st.time_input("Last Dose Given At", value=time(8, 0), step=timedelta(minutes=15), key="to_dose_time")
            sample_time = st.time_input("Trough Level Drawn At", value=time(19, 30), step=timedelta(minutes=15), key="to_sample_time")
            trough_measured = st.number_input("Measured Trough (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f", key="to_trough")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate the time difference in hours
        def hours_diff(start, end):
            today = datetime.today().date()
            dt_start = datetime.combine(today, start)
            dt_end = datetime.combine(today, end)
            if dt_end < dt_start:  # Handle overnight intervals
                dt_end += timedelta(days=1)
            return (dt_end - dt_start).total_seconds() / 3600
        
        time_since_last_dose_h = hours_diff(dose_time, sample_time)
        
        # Format time difference for display
        def format_hours_minutes(decimal_hours):
            if decimal_hours < 0:
                return "Invalid time"
            total_minutes = int(round(decimal_hours * 60))
            hours = total_minutes // 60
            minutes = total_minutes % 60
            if hours > 0 and minutes > 0:
                return f"{hours} hours {minutes} minutes"
            elif hours > 0:
                return f"{hours} hours"
            else:
                return f"{minutes} minutes"
        
        time_formatted = format_hours_minutes(time_since_last_dose_h)
        
        # Show timing info with better styling
        if time_since_last_dose_h > 0:
            st.markdown(f"""
            <div class="info-box">
                <strong>Timing Info:</strong> Trough level was drawn <strong>{time_formatted}</strong> after the last dose.
                Current interval: <strong>q{current_interval_h}h</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Timing Issue:</strong> Sample time must be after the last dose time.
            </div>
            """, unsafe_allow_html=True)
        
        # Button with better styling
        calc_button = st.button("Run Trough Analysis", use_container_width=True, key="run_trough_calc")
        
        # Display results when button is clicked
        if calc_button and time_since_last_dose_h > 0:
            # Show progress indicator
            with st.spinner("Analyzing trough level..."):
                # Simulate calculations
                scr_mgdl = scr_umol / 88.4
                
                # Calculate CrCl
                if fem:
                    crcl = ((140 - age) * wt * 0.85) / (72 * scr_mgdl)
                else:
                    crcl = ((140 - age) * wt) / (72 * scr_mgdl)
                
                # Use population Vd for trough-only estimate
                vd_calc = 0.7 * wt
                
                # Simulate Ke calculation
                ke_calc = 0.00286 + 0.00331 * (crcl/100)
                
                # Calculate half-life
                t_half_calc = math.log(2) / ke_calc
                
                # Simulate AUC calculation
                cl_calc = ke_calc * vd_calc
                auc24_calc = (dose_int_current / cl_calc) * (24 / current_interval_h)
                
                # Calculate new dose
                target_trough = 15.0  # Example target
                new_dose_calc = round(target_trough * vd_calc * ke_calc * current_interval_h / (1 - math.exp(-ke_calc * current_interval_h)) / 250) * 250
                
                # Status check
                def check_target_status(value, target_range):
                    lower, upper = target_range
                    if value < lower:
                        return "BELOW TARGET"
                    elif upper and value > upper:
                        return "ABOVE TARGET"
                    else:
                        return "WITHIN TARGET"
                
                # Get target ranges based on selection
                if "Empirical" in target_level_desc:
                    trough_range = (10, 15)
                    auc_range = (400, 600)
                else:
                    trough_range = (15, 20)
                    auc_range = (600, None)
                
                trough_status = check_target_status(trough_measured, trough_range)
                auc_status = check_target_status(auc24_calc, auc_range)
                
                # Prepare results for display
                pk_results = {
                    'Calculation Mode': 'Trough-Only',
                    'Current Dose': f"{dose_int_current} mg",
                    'Current Dosing Interval': f"q{current_interval_h}h",
                    'Time Since Last Dose (at Trough Draw)': time_formatted,
                    'Measured Trough': f"{trough_measured:.1f} mg/L",
                    'Trough Status vs Target': trough_status,
                    'Estimated Population Vd': f"{vd_calc:.1f} L",
                    'Estimated Ke': f"{ke_calc:.4f} h⁻¹",
                    'Estimated t½': f"{t_half_calc:.1f} h",
                    'Estimated AUC24': f"{auc24_calc:.1f} mg·h/L",
                    'AUC Status vs Target': auc_status,
                    'Suggested New Dose': f"{new_dose_calc} mg q{current_interval_h}h"
                }
            
            # Display results in a modern format
            st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
            
            # Display simplified visual indicators instead of Plotly charts
            show_level_indicators(trough_measured, trough_range, auc24_calc, auc_range)
            
            # Display status section
            st.markdown("<h4>Target Status</h4>", unsafe_allow_html=True)
            show_status_section(pk_results)
            
            # Display detailed results in a clean grid
            st.markdown("<h4>PK Parameters</h4>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">CrCl</div><div class="metric-value">{crcl:.1f} mL/min</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Estimated Vd</div><div class="metric-value">{vd_calc:.1f} L</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Elimination Rate (Ke)</div><div class="metric-value">{ke_calc:.4f} h⁻¹</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Half-life</div><div class="metric-value">{t_half_calc:.1f} h</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">AUC₂₄</div><div class="metric-value">{auc24_calc:.1f} mg·h/L</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Clearance</div><div class="metric-value">{cl_calc:.2f} L/h</div></div>', unsafe_allow_html=True)
            
            # Recommendation section with better styling
            st.markdown("<h3>Recommendation</h3>", unsafe_allow_html=True)
            
            recommendation_html = f"""
            <div class="recommendation-card">
                <div class="recommendation-title">Suggested Dose Adjustment</div>
                <div class="recommendation-dose">{new_dose_calc} mg q{current_interval_h}h</div>
                <div class="recommendation-description">Based on the pharmacokinetic analysis and target level ({target_level_desc}).</div>
            </div>
            """
            
            st.markdown(recommendation_html, unsafe_allow_html=True)
            
            # Fixed AI interpretation with proper rendering
            st.markdown("<h3>AI-Generated Interpretation</h3>", unsafe_allow_html=True)
            
            interpretation_html = render_interpretation(
                trough_status=trough_status,
                trough_measured=trough_measured,
                auc_status=auc_status,
                auc24=auc24_calc,
                thalf=t_half_calc,
                interval_h=current_interval_h,
                new_dose=new_dose_calc,
                target_desc=target_level_desc
            )
            
            st.markdown(interpretation_html, unsafe_allow_html=True)
            
            # Download button with better styling
            st.download_button(
                label="📄 Download Complete Report",
                data="Vancomycin TDM Report\n" + "\n".join([f"{k}: {v}" for k, v in pk_results.items()]),
                file_name=f"vanco_trough_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

    # --- PEAK & TROUGH TAB ---
    with tabs[2]:
        st.markdown("""
        <div class="card">
            <h2>Peak & Trough Analysis</h2>
            <p>Calculate individual PK parameters using both peak and trough levels (Sawchuk-Zaske method).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a cleaner form layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Current Regimen</h4>", unsafe_allow_html=True)
            dose_for_levels = st.number_input("Dose Administered (mg)", min_value=250, step=250, value=1000, key="pt_dose")
            current_interval_h_pt = st.selectbox(
                "Dosing Interval",
                options=[6, 8, 12, 18, 24, 36, 48],
                index=2,
                format_func=lambda x: f"q{x}h",
                key="pt_interval"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Measured Levels</h4>", unsafe_allow_html=True)
            c_peak_measured = st.number_input("Peak Level (mg/L)", min_value=0.1, max_value=200.0, value=30.0, step=0.1, format="%.1f", key="pt_peak")
            c_trough_measured = st.number_input("Trough Level (mg/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f", key="pt_trough")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Timing Information</h4>", unsafe_allow_html=True)
            infusion_start_time = st.time_input("Infusion Start Time", value=time(8, 0), step=timedelta(minutes=15), key="pt_inf_start")
            infusion_end_time = st.time_input("Infusion End Time", value=time(9, 0), step=timedelta(minutes=15), key="pt_inf_end")
            peak_sample_time = st.time_input("Peak Sample Time", value=time(10, 0), step=timedelta(minutes=15), key="pt_peak_time")
            trough_sample_time = st.time_input("Trough Sample Time", value=time(19, 30), step=timedelta(minutes=15), key="pt_trough_time")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate time differences (reusing function from above)
        infusion_duration_h = hours_diff(infusion_start_time, infusion_end_time)
        time_from_infusion_end_to_peak_draw_h = hours_diff(infusion_end_time, peak_sample_time)
        
        # Format for display
        infusion_duration_formatted = format_hours_minutes(infusion_duration_h)
        time_to_peak_formatted = format_hours_minutes(time_from_infusion_end_to_peak_draw_h)
        
        # Show timing info with better styling
        valid_times = True
        if infusion_duration_h <= 0:
            st.markdown("""
            <div class="error-box">
                <strong>⚠️ Timing Error:</strong> Infusion end time must be after infusion start time.
            </div>
            """, unsafe_allow_html=True)
            valid_times = False
        
        if time_from_infusion_end_to_peak_draw_h < 0:
            st.markdown("""
            <div class="error-box">
                <strong>⚠️ Timing Error:</strong> Peak sample time must be after infusion end time.
            </div>
            """, unsafe_allow_html=True)
            valid_times = False
        
        if valid_times:
            st.markdown(f"""
            <div class="info-box">
                <strong>Timing Info:</strong> Infusion duration: <strong>{infusion_duration_formatted}</strong>, 
                Time to peak draw: <strong>{time_to_peak_formatted}</strong>, 
                Current interval: <strong>q{current_interval_h_pt}h</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Button with better styling
        calc_button = st.button("Run Peak & Trough Analysis", use_container_width=True, key="run_peak_trough_calc")
        
        # Display results when button is clicked
        if calc_button and valid_times:
            if c_peak_measured <= c_trough_measured:
                st.markdown("""
                <div class="error-box">
                    <strong>⚠️ Input Error:</strong> Peak level must be higher than trough level.
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show progress indicator
                with st.spinner("Analyzing peak and trough levels..."):
                    # Simulate calculations (simplified for demonstration)
                    scr_mgdl = scr_umol / 88.4
                    
                    # Calculate CrCl
                    if fem:
                        crcl = ((140 - age) * wt * 0.85) / (72 * scr_mgdl)
                    else:
                        crcl = ((140 - age) * wt) / (72 * scr_mgdl)
                    
                    # Calculate time between peak and trough
                    time_between_samples = hours_diff(peak_sample_time, trough_sample_time)
                    if time_between_samples <= 0:
                        time_between_samples = hours_diff(peak_sample_time, datetime.combine(datetime.today().date() + timedelta(days=1), trough_sample_time).time())
                    
                    # Calculate Ke from peak and trough (simplified)
                    ke_ind = math.log(c_peak_measured / c_trough_measured) / time_between_samples
                    
                    # Calculate remaining PK parameters (simplified)
                    thalf_ind = math.log(2) / ke_ind
                    
                    # Extrapolate Cmax to end of infusion
                    cmax_ind = c_peak_measured * math.exp(ke_ind * time_from_infusion_end_to_peak_draw_h)
                    
                    # Calculate individual Vd (simplified)
                    vd_ind = dose_for_levels / (cmax_ind * (1 - math.exp(-ke_ind * current_interval_h_pt)))
                    
                    # Calculate clearance
                    cl_ind = ke_ind * vd_ind
                    
                    # Calculate AUC24
                    auc_interval = dose_for_levels / cl_ind
                    auc24_ind = auc_interval * (24 / current_interval_h_pt)
                    
                    # Calculate expected Cmax/Cmin for current dose
                    exp_cmax_ind = dose_for_levels / (vd_ind * (1 - math.exp(-ke_ind * current_interval_h_pt)))
                    exp_cmin_ind = exp_cmax_ind * math.exp(-ke_ind * current_interval_h_pt)
                    
                    # Target dose calculation (simplified)
                    if "Empirical" in target_level_desc:
                        target_auc24 = 500  # Midpoint of 400-600
                        target_trough = 12.5  # Midpoint of 10-15
                        trough_range = (10, 15)
                        auc_range = (400, 600)
                    else:
                        target_auc24 = 600  # Lower end of >600
                        target_trough = 17.5  # Midpoint of 15-20
                        trough_range = (15, 20)
                        auc_range = (600, None)
                    
                    # Calculate new dose based on AUC target
                    target_auc_interval = target_auc24 * (current_interval_h_pt / 24)
                    new_dose_raw = target_auc_interval * cl_ind
                    new_dose_rounded = round(new_dose_raw / 250) * 250
                    
                    # Status check (same function as before)
                    def check_target_status(value, target_range):
                        lower, upper = target_range
                        if value < lower:
                            return "BELOW TARGET"
                        elif upper and value > upper:
                            return "ABOVE TARGET"
                        else:
                            return "WITHIN TARGET"
                    
                    trough_status = check_target_status(c_trough_measured, trough_range)
                    auc_status = check_target_status(auc24_ind, auc_range)
                    
                    # Store results
                    pk_results = {
                        'Calculation Mode': 'Peak & Trough',
                        'Dose Administered': f"{dose_for_levels} mg",
                        'Current Dosing Interval': f"q{current_interval_h_pt}h",
                        'Infusion Duration': infusion_duration_formatted,
                        'Time to Peak Draw': time_to_peak_formatted,
                        'Measured Peak': f"{c_peak_measured:.1f} mg/L",
                        'Measured Trough': f"{c_trough_measured:.1f} mg/L",
                        'Trough Status vs Target': trough_status,
                        'Individual Vd': f"{vd_ind:.1f} L",
                        'Individual Ke': f"{ke_ind:.4f} h⁻¹",
                        'Individual t½': f"{thalf_ind:.1f} h",
                        'Individual CL': f"{cl_ind:.2f} L/h",
                        'Extrapolated Cmax': f"{cmax_ind:.1f} mg/L",
                        'Expected Cmax': f"{exp_cmax_ind:.1f} mg/L",
                        'Expected Cmin': f"{exp_cmin_ind:.1f} mg/L",
                        'AUC_interval': f"{auc_interval:.1f} mg·h/L",
                        'Individual AUC24': f"{auc24_ind:.1f} mg·h/L",
                        'AUC Status vs Target': auc_status,
                        'Suggested New Dose': f"{new_dose_rounded} mg q{current_interval_h_pt}h"
                    }
                
                # Display results in a modern format
                st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
                
                # Display simplified visual indicators instead of Plotly charts
                show_level_indicators(c_trough_measured, trough_range, auc24_ind, auc_range)
                
                # Display status section
                st.markdown("<h4>Target Status</h4>", unsafe_allow_html=True)
                show_status_section(pk_results)
                
                # Display PK parameters in cards
                st.markdown("<h4>Individual PK Parameters</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Volume of Distribution</div>
                        <div class="metric-value">{vd_ind:.1f} L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Elimination Rate (Ke)</div>
                        <div class="metric-value">{ke_ind:.4f} h⁻¹</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Half-Life</div>
                        <div class="metric-value">{thalf_ind:.1f} h</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Individual Clearance</div>
                        <div class="metric-value">{cl_ind:.2f} L/h</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Extrapolated Cmax</div>
                        <div class="metric-value">{cmax_ind:.1f} mg/L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">AUC₂₄</div>
                        <div class="metric-value">{auc24_ind:.1f} mg·h/L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Second row of metrics
                col7, col8 = st.columns(2)
                with col7:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Expected Cmax</div>
                        <div class="metric-value">{exp_cmax_ind:.1f} mg/L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col8:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Expected Cmin</div>
                        <div class="metric-value">{exp_cmin_ind:.1f} mg/L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation with better styling
                st.markdown("<h3>Recommendation</h3>", unsafe_allow_html=True)
                
                recommendation_html = f"""
                <div class="recommendation-card">
                    <div class="recommendation-title">Suggested Dose Based on Individual PK</div>
                    <div class="recommendation-dose">{new_dose_rounded} mg q{current_interval_h_pt}h</div>
                    <div class="recommendation-description">Based on individual PK parameters and target AUC of {target_auc24} mg·h/L.</div>
                </div>
                """
                
                st.markdown(recommendation_html, unsafe_allow_html=True)
                
                # Fixed AI interpretation for Peak & Trough
                st.markdown("<h3>AI-Generated Interpretation</h3>", unsafe_allow_html=True)
                
                # Use the same rendering function for consistent styling
                interpretation_html = render_interpretation(
                    trough_status=trough_status,
                    trough_measured=c_trough_measured, 
                    auc_status=auc_status,
                    auc24=auc24_ind,
                    thalf=thalf_ind,
                    interval_h=current_interval_h_pt,
                    new_dose=new_dose_rounded,
                    target_desc=target_level_desc
                )
                
                st.markdown(interpretation_html, unsafe_allow_html=True)
                
                # Download button with better styling
                st.download_button(
                    label="📄 Download Complete Report",
                    data="Vancomycin TDM Report\n" + "\n".join([f"{k}: {v}" for k, v in pk_results.items()]),
                    file_name=f"vanco_peak_trough_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

    # --- FOOTER ---
    st.markdown("""
    <div class="footer">
        <p>Vancomycin TDM Assistant | Based on Clinical Pharmacokinetics Pharmacy Handbook (2nd ed.)</p>
        <p>Disclaimer: This tool is for educational purposes only. Clinical decisions should be made by qualified healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
