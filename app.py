import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Tafel Analysis Tool", layout="wide")
st.title("Web-Based Tafel Analysis")
st.markdown("Upload your Excel/CSV data to calculate Tafel Slope and Exchange Current Density.")

# --- 2. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Tafel Data (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # --- 3. DATA READING & PREPROCESSING ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file, header=None)
            
        # Rename columns for clarity (assuming Col 1=Voltage, Col 2=Current)
        # Note: Python uses 0-indexing, so column 0 is the first column.
        df.columns = ['V_raw_mV', 'J_raw'] if df.shape[1] >= 2 else ['V_raw_mV']
        
        # Extract data (replicating your MATLAB logic)
        V_raw_mV = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        J_raw = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        # Drop any NaN values that might have resulted from headers
        valid_idx = ~np.isnan(V_raw_mV) & ~np.isnan(J_raw)
        V_raw_mV = V_raw_mV[valid_idx]
        J_raw = J_raw[valid_idx]

        # --- UNITS & CONVERSION ---
        # User option: Does the file actually have headers?
        # Your MATLAB code assumed raw data immediately, but web apps often get headers.
        
        # MATLAB Logic: V_raw = V_raw_mV / 1000
        V_raw = V_raw_mV / 1000.0
        
        # MATLAB Logic: eta = abs(V_raw)
        eta = np.abs(V_raw)
        
        # MATLAB Logic: J_abs = abs(J_raw), log_J = log10(J_abs)
        J_abs = np.abs(J_raw)
        log_J = np.log10(J_abs)
        
        # Create a clean DataFrame for processing
        data = pd.DataFrame({'log_J': log_J, 'eta': eta, 'V_raw': V_raw})

        # --- 4. MODE SELECTION ---
        st.write("---")
        method = st.radio("Select Tafel Region Method:", ('Automatic (Recommended)', 'Manual (Expert)'))
        
        best_indices = []
        
        if method == 'Automatic (Recommended)':
            # --- AUTOMATIC LOGIC (Replicating your MATLAB loop) ---
            window_size = 6
            r2_thresh = 0.99
            slope_min = 0.03
            slope_max = 0.25
            
            best_r2 = -np.inf
            
            # Progress bar for user experience
            progress_bar = st.progress(0)
            
            # Loop through data
            # MATLAB: for i = 1:(length(log_J) - windowSize + 1)
            # Python ranges are 0-indexed and exclusive at the end
            num_points = len(data)
            
            for i in range(num_points - window_size + 1):
                # Update progress bar occasionally
                if i % 10 == 0:
                    progress_bar.progress(i / num_points)

                subset = data.iloc[i : i + window_size]
                x_win = subset['log_J']
                y_win = subset['eta']
                
                # Linear Regression
                slope, intercept, r_value, p_value, std_err = linregress(x_win, y_win)
                r2 = r_value ** 2
                
                # Check criteria
                if slope_min < slope < slope_max and r2 > r2_thresh:
                    if r2 > best_r2:
                        best_r2 = r2
                        best_indices = subset.index.tolist()
            
            progress_bar.empty() # Clear bar
            
            if not best_indices:
                st.error("No valid auto-detected Tafel region found. Try Manual Mode.")
            else:
                st.success(f"Automatic region found! (Best R²: {best_r2:.4f})")

       else:
            # --- MANUAL LOGIC (UPDATED: Select by Current X-Axis) ---
            st.write("Enter the Current Density range to fit (e.g., 1.5 to 5.0).")
            st.info("Tip: Enter values in the same units as your Excel file (e.g., mA or A).")
            
            col1, col2 = st.columns(2)
            with col1:
                # We default to 1.0 just to have a non-zero starting value
                j_start = st.number_input("Start Current Density", value=1.5)
            with col2:
                j_end = st.number_input("End Current Density", value=5.0)
            
            # Validation: Log(0) is impossible, so we block it
            if j_start == 0 or j_end == 0:
                st.error("Current cannot be zero for Tafel analysis (Log scale).")
                best_indices = []
            else:
                # 1. Convert user inputs to Log scale (to match the X-axis)
                # We use abs() to handle cases where users might type negative current
                log_start = np.log10(abs(j_start))
                log_end = np.log10(abs(j_end))
                
                # 2. Sort min/max so it doesn't matter which order you type them
                l_low = min(log_start, log_end)
                l_high = max(log_start, log_end)
                
                # 3. Filter data based on Log Current (X-axis) instead of Voltage
                mask = (data['log_J'] >= l_low) & (data['log_J'] <= l_high)
                best_indices = data[mask].index.tolist()
                
                if len(best_indices) == 0:
                    st.warning(f"No data found between {j_start} and {j_end}. Check if your file uses A or mA.")

        # --- 5. CALCULATIONS & PLOTTING ---
        if len(best_indices) > 0:
            # Extract Fit Data
            fit_data = data.loc[best_indices]
            x_fit = fit_data['log_J']
            y_fit = fit_data['eta']
            
            # Calculate Slope & Intercept
            slope_b, intercept_a, r_val, p_val, stderr = linregress(x_fit, y_fit)
            
            # Calculate Exchange Current Density (j0)
            # MATLAB: log_j0 = -intercept_a / slope_b; j0 = 10^log_j0;
            log_j0 = -intercept_a / slope_b
            j0 = 10 ** log_j0
            
            # --- DISPLAY RESULTS ---
            st.write("### Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tafel Slope (b)", f"{slope_b*1000:.1f} mV/dec")
            c2.metric("Exchange Current (j₀)", f"{j0:.2e} A/cm²")
            c3.metric("R-Squared", f"{r_val**2:.4f}")
            
            # --- PLOTTING (Plotly) ---
            fig = go.Figure()
            
            # 1. Full Data (Grey)
            fig.add_trace(go.Scatter(
                x=data['log_J'], y=data['eta'],
                mode='lines+markers', name='Raw Data',
                marker=dict(color='lightgrey', size=5),
                line=dict(color='lightgrey')
            ))
            
            # 2. Fitted Line (Red) extends slightly beyond selection for visibility
            # Create a line based on the equation y = mx + c
            y_pred = slope_b * x_fit + intercept_a
            
            fig.add_trace(go.Scatter(
                x=x_fit, y=y_pred,
                mode='lines', name='Tafel Fit',
                line=dict(color='red', width=3)
            ))

            # Update Layout to match your MATLAB style
            fig.update_layout(
                title="Tafel Analysis Plot",
                xaxis_title="Log (Current Density) [log(A/cm²)]",
                yaxis_title="Overpotential (η) [V]",
                yaxis=dict(autorange="reversed"), # Reverse Y axis like your code
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download Results
            res_df = pd.DataFrame({
                "Parameter": ["Tafel Slope (V/dec)", "Tafel Slope (mV/dec)", "Exchange Current (A/cm2)", "R2"],
                "Value": [slope_b, slope_b*1000, j0, r_val**2]
            })
            st.download_button("Download Result CSV", res_df.to_csv(index=False), "tafel_results.csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Make sure your file has Voltage (mV) in Col 1 and Current (A) in Col 2.")
