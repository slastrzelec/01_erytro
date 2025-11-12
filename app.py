import streamlit as st
import cv2
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

# --- Page settings (Must be at the very top) ---
st.set_page_config(
    page_title="Erythrocyte Analysis App",
    page_icon="🔬",
    layout="wide" # Use wide layout for professional look
)

# Zmienna do spójnego koloru akcentu (dla config.toml lub ręcznego ustawienia)
ACCENT_COLOR = "#B71C1C" 

# --- TITLE & UPLOAD INSTRUCTIONS ---
st.title("🔬 Erythrocyte Analysis App")
st.markdown("### Upload your image or use the sample one for testing")
st.markdown("---")
# ----------------------------------------------------------------------


# --- INTRODUCTORY SECTION ---
st.header("Why Erythrocyte Shape Matters? 🩸")
st.markdown(
    """
    <p style="text-align: justify;">
    The shape of a red blood cell (erythrocyte) is critical to its function. 
    Healthy erythrocytes have a **biconcave disc** shape, which maximizes the surface area for gas exchange. 
    Any **shape anomalies** (e.g., elongation, sickle shape, spherical) indicate 
    various conditions such as anemia, thalassemia, or other severe blood disorders. 
    Quantitative analysis of the Shape Factor is a crucial diagnostic tool.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --- COMBINED SECTION: ABSTRACT AND PUBLICATION DOWNLOAD ---

PUBLICATION_URL = "https://raw.githubusercontent.com/slastrzelec/01_erytro/main/publikacja%20SS%20APP.pdf"
FILE_NAME = "publication_SS_APP.pdf"

ABSTRACT_TEXT = """
    Functionalized carbon nanotubes are a group of nanomaterials with many potential applications in
    bionanomedicine. However, they can also be toxic, especially those functionalized with metal ions.
    Carbon nanotubes enter cells easily. The research focused on investigating the acute effects of multi-
    walled carbon nanotubes with attached Ni2+ ions (MWCNTs-Ni) on the functioning of red blood
    cells. The very low concentration of MWCNTs-Ni used did not cause changes in the size and shape of
    red blood cells, but it did affect the states of haemoglobin and its ability to reversibly bind oxygen.
    MWCNTs-Ni-treated red blood cells showed an increased affinity for O2, similar to that observed in red
    blood cells from essential hypertensive subjects. The results indicate a potential risk that MWCNTs-Ni
    may influence the development of hypertension.
"""

st.subheader("Want to Know More? Check Out My Publication 📖")

# Using columns: first for abstract, second (smaller) for button
col_abstract, col_button = st.columns([4, 1])

with col_abstract:
    # Using accent color
    st.markdown(
        f'<blockquote style="border-left: 5px solid {ACCENT_COLOR}; padding: 10px; margin: 0 0; text-align: justify;">'
        f'{ABSTRACT_TEXT.strip()}'
        f'</blockquote>',
        unsafe_allow_html=True
    )

with col_button:
    st.write("")
    st.write("") 
    try:
        response = requests.get(PUBLICATION_URL)
        response.raise_for_status()
        pdf_bytes = response.content
        
        st.download_button(
            label="⬇️ DOWNLOAD PDF",
            data=pdf_bytes,
            file_name=FILE_NAME,
            mime="application/pdf",
            use_container_width=True
        )

    except requests.exceptions.RequestException as e:
        st.error("❌ Error loading file for download.")

st.markdown("---")
# --- END COMBINED SECTION ---


# --- Core function: erythrocyte shape analysis ---
def get_erythrocyte_shape_factors(image, anomaly_threshold, min_axis_size):
    processed_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors_data = []
    anomalies_data = []
    
    for contour in contours:
        if len(contour) > 5: 
            try:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                
                minor_axis = min(axes)
                major_axis = max(axes)
                
                # Condition for minimum size (artifact filtering)
                if major_axis < min_axis_size or minor_axis < min_axis_size:
                    continue
                
                if minor_axis > 0:
                    shape_factor = major_axis / minor_axis
                    
                    # Calculating Ellipticity
                    ellipticity = 1 - (minor_axis / major_axis)
                    
                    ellipse_color = (0, 255, 0)  # Green = normal
                    
                    if shape_factor > 1.3 and shape_factor <= 1.5:
                        ellipse_color = (0, 255, 255)  # Yellow = moderately elongated
                    elif shape_factor > 1.5:
                        ellipse_color = (0, 0, 255)  # Red = highly elongated

                    if shape_factor <= anomaly_threshold:
                        # Storing all metrics
                        shape_factors_data.append({
                            "Erythrocyte Number": len(shape_factors_data) + 1,
                            "Shape Factor": shape_factor,
                            "Ellipticity": ellipticity,
                            "Major Axis": major_axis,
                            "Minor Axis": minor_axis,
                            "Area": area,
                            "Perimeter": perimeter
                        })
                        cv2.ellipse(processed_image, ellipse, ellipse_color, 2)
                        angle_rad = np.radians(orientation)
                        
                        # Drawing Axes (Major/Minor)
                        major_end_point_1 = (int(center[0] + major_axis/2 * np.cos(angle_rad)),
                                             int(center[1] + major_axis/2 * np.sin(angle_rad)))
                        major_end_point_2 = (int(center[0] - major_axis/2 * np.cos(angle_rad)),
                                             int(center[1] - major_axis/2 * np.sin(angle_rad)))
                        cv2.line(processed_image, major_end_point_1, major_end_point_2, (0, 0, 255), 1)
                        
                        minor_angle_rad = np.radians(orientation + 90)
                        minor_end_point_1 = (int(center[0] + minor_axis/2 * np.cos(minor_angle_rad)),
                                             int(center[1] + minor_axis/2 * np.sin(minor_angle_rad)))
                        minor_end_point_2 = (int(center[0] - minor_axis/2 * np.cos(minor_angle_rad)),
                                             int(center[1] - minor_axis/2 * np.sin(minor_angle_rad)))
                        cv2.line(processed_image, minor_end_point_1, minor_end_point_2, (255, 0, 0), 1)
                        
                        # Labeling the cell
                        cv2.putText(processed_image, str(len(shape_factors_data)),
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Storing all metrics for anomalies
                        anomalies_data.append({
                            "Erythrocyte Number": len(anomalies_data) + 1,
                            "Shape Factor": shape_factor,
                            "Ellipticity": ellipticity,
                            "Major Axis": major_axis,
                            "Minor Axis": minor_axis,
                            "Area": area,
                            "Perimeter": perimeter
                        })
                        cv2.ellipse(processed_image, ellipse, (255, 0, 255), 2) # Magenta color for anomalies
                        cv2.putText(processed_image, f"A{len(anomalies_data)}",
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            except cv2.error:
                continue 

    return processed_image, shape_factors_data, anomalies_data


# --- Streamlit UI ---

# --- Sidebar: image source selection ---
st.sidebar.header("Image Source")
use_default_image = st.sidebar.checkbox("Use sample image (C.jpg)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])

# --- Sidebar: analysis settings ---
st.sidebar.header("Analysis Settings")

# 1. SLIDER FOR MINIMUM SIZE
min_axis_size_slider = st.sidebar.slider(
    "Min. Axis Size (px) to filter artifacts", 
    min_value=10, 
    max_value=150, 
    value=50, 
    step=5,
    help="Minimum required length of the major and minor axes to consider a contour an erythrocyte (e.g., 50px)."
)

# 2. INPUT FOR CALIBRATION FACTOR (NOWOŚĆ)
calibration_factor = st.sidebar.number_input(
    "Calibration Factor (µm per pixel)",
    min_value=0.0, 
    max_value=10.0,
    value=0.0, 
    step=0.01,
    # Usunięto argument 'help'
)

# DODANY TEKST POD INPUTEM
st.sidebar.markdown(
    """
    <small>Wprowadź współczynnik konwersji: 1 piksel = X mikrometrów (µm). 
    Jeśli skala jest nieznana, pozostaw wartość **0.0**, aby użyć jednostek pikselowych (px).</small>
    """,
    unsafe_allow_html=True
)


anomaly_threshold_slider = st.sidebar.slider(
    "Anomaly detection threshold (Shape Factor >)", 
    min_value=1.5, 
    max_value=2.5, 
    value=1.7, 
    step=0.05
)

# --- Run button ---
run_button = st.sidebar.button("▶ Run Analysis")

# --- Main logic ---
if run_button:
    image_to_process = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.success("✅ Image uploaded successfully!")

    elif use_default_image:
        default_url = "https://raw.githubusercontent.com/slastrzelec/01_erytro/main/experminental_data_from_microscope/C.jpg"
        try:
            response = requests.get(default_url)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image_to_process = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            st.info("📷 Sample image loaded.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error loading image: {e}")

    if image_to_process is not None:
        # Pass both thresholds to the function
        processed_img, shape_factors, anomalies = get_erythrocyte_shape_factors(
            image_to_process, anomaly_threshold_slider, min_axis_size_slider
        )

        # OBLICZENIA KALIBRACJI
        def apply_calibration(df, factor):
            # Jeśli współczynnik kalibracji wynosi 0, nie dodajemy kolumn mikrometrycznych.
            if df.empty or factor == 0.0:
                return df
            
            # Przeliczanie długości (Axis, Perimeter)
            df['Major Axis (µm)'] = df['Major Axis'] * factor
            df['Minor Axis (µm)'] = df['Minor Axis'] * factor
            df['Perimeter (µm)'] = df['Perimeter'] * factor
            
            # Przeliczanie powierzchni (Area) - Kwadrat współczynnika
            df['Area (µm²)'] = df['Area'] * (factor ** 2)
            
            return df
        
        df_normal = pd.DataFrame(shape_factors)
        df_anomalies = pd.DataFrame(anomalies)

        # Kalibracja jest aplikowana tylko jeśli factor > 0.0
        df_normal = apply_calibration(df_normal, calibration_factor)
        df_anomalies = apply_calibration(df_anomalies, calibration_factor)
        
        
        # Creating DF for full table and visualization
        df_full = pd.concat([df_normal, df_anomalies], ignore_index=True)
        
        # Adding classification column for charts
        df_full['Classification'] = np.where(df_full['Shape Factor'] > anomaly_threshold_slider, 'Anomaly', 'Normal/Moderate')


        if not df_normal.empty or not df_anomalies.empty:
            
            # --- START: VISUAL IMPROVEMENT (Metrics and Image side-by-side) ---
            
            col_img, col_metrics = st.columns([3, 2]) # Wide layout for better visual separation

            with col_img:
                st.markdown("##### Processed Image")
                st.image(processed_img, channels="BGR")

            with col_metrics:
                if not df_normal.empty:
                    # Shape Factor and Ellipticity Metrics
                    avg_sf = df_normal['Shape Factor'].mean()
                    avg_ellipticity = df_normal['Ellipticity'].mean() 
                    std_sf = df_normal['Shape Factor'].std()

                    # Ustawienie jednostek
                    if calibration_factor > 0.0 and 'Area (µm²)' in df_normal.columns:
                        avg_major_um = df_normal['Major Axis (µm)'].mean()
                        avg_area_um2 = df_normal['Area (µm²)'].mean()
                        major_label = "Avg Major Axis (µm)"
                        area_label = "Avg Area (µm²)"
                        major_value = f"{avg_major_um:.2f}"
                        area_value = f"{avg_area_um2:,.2f}"
                    else:
                        avg_major_px = df_normal['Major Axis'].mean()
                        avg_area_px2 = df_normal['Area'].mean()
                        major_label = "Avg Major Axis (px)"
                        area_label = "Avg Area (px²)"
                        major_value = f"{avg_major_px:.2f}"
                        area_value = f"{avg_area_px2:,.2f}"
                        

                    st.markdown("##### Key Statistics (Normal Cells)")
                    
                    # Shape Factor and Ellipticity Metrics
                    col_met1, col_met2 = st.columns(2)
                    
                    with col_met1:
                        st.metric(label="Average Shape Factor", value=f"{avg_sf:.2f}", help="Mean ratio of the major axis to the minor axis.")
                    
                    with col_met2:
                        st.metric(label="Average Ellipticity", value=f"{avg_ellipticity:.2f}", help="Mean ellipticity (1 - Minor/Major Axis).")
                    
                    # Size Metrics (Now dynamic based on calibration_factor)
                    st.markdown("---") 
                    st.metric(label=major_label, value=major_value, help="Mean length of the major semi-axis.")
                    
                    st.metric(label=area_label, value=area_value, help="Mean cell area.")
                        
                    st.metric(label="Std. Deviation (SF)", value=f"{std_sf:.2f}", help="Standard deviation for the Shape Factor.")
                    st.metric(label="Total Cells Analyzed", value=len(shape_factors) + len(anomalies))
                else:
                    st.info("No normal cells detected below the threshold.")

            st.markdown("---")
            # --- END: VISUAL IMPROVEMENT ---

            # Zmienne dla wykresów, które uwzględniają jednostki µm
            if calibration_factor > 0.0 and 'Area (µm²)' in df_full.columns:
                area_col_name = 'Area (µm²)'
                major_axis_col_name = 'Major Axis (µm)'
                perimeter_col_name = 'Perimeter (µm)'
                area_unit = 'µm²'
                length_unit = 'µm'
            else:
                # Użycie pikseli, jeśli kalibracja wynosi 0 lub jest wyłączona
                area_col_name = 'Area'
                major_axis_col_name = 'Major Axis'
                perimeter_col_name = 'Perimeter'
                area_unit = 'px²'
                length_unit = 'px'


            # --- Scatter Plot for correlation ---
            st.subheader("🔬 Correlation Scatter Plot: Shape Factor vs. Area")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            
            color_map = {'Normal/Moderate': '#1f77b4', 'Anomaly': ACCENT_COLOR}
            
            for name, group in df_full.groupby('Classification'):
                ax_scatter.scatter(group[area_col_name], group['Shape Factor'], 
                                   label=name, 
                                   color=color_map[name], 
                                   alpha=0.6, 
                                   edgecolors='w', 
                                   linewidths=0.5)

            ax_scatter.axhline(y=anomaly_threshold_slider, color='r', linestyle='--', label=f'SF Threshold ({anomaly_threshold_slider:.2f})')
            
            ax_scatter.legend(title='Classification')
            ax_scatter.set_xlabel(f'Area ({area_unit})')
            ax_scatter.set_ylabel('Shape Factor (SF)')
            ax_scatter.set_title('Shape Factor vs. Area by Classification')
            ax_scatter.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_scatter)
            
            
            # --- Shape Factor Histogram ---
            st.markdown("---")
            st.subheader("📊 Shape Factor Distribution")
            fig_sf, ax_sf = plt.subplots(figsize=(10, 6)) 
            if not df_normal.empty:
                ax_sf.hist(df_normal['Shape Factor'], bins=15, alpha=0.7, color='#1f77b4', edgecolor='black', label='Normal (SF <= Threshold)')
            if not df_anomalies.empty:
                ax_sf.hist(df_anomalies['Shape Factor'], bins=15, alpha=0.7, color=ACCENT_COLOR, edgecolor='black', label='Anomaly (SF > Threshold)')
            
            ax_sf.axvline(x=anomaly_threshold_slider, color='r', linestyle='--', label=f'Threshold ({anomaly_threshold_slider:.2f}')
            
            ax_sf.legend()
            ax_sf.set_xlabel('Shape Factor (Major Axis / Minor Axis)')
            ax_sf.set_ylabel('Frequency')
            ax_sf.set_title('Distribution of Erythrocyte Shape Factors')
            ax_sf.grid(axis='y', alpha=0.5)
            st.pyplot(fig_sf)
            
            # --- Area Histogram ---
            st.markdown("---")
            st.subheader(f"📐 Area Distribution (Cell Area - {area_unit})")
            fig_area, ax_area = plt.subplots(figsize=(10, 6))
            
            if not df_full.empty:
                ax_area.hist(df_full[df_full['Classification'] == 'Normal/Moderate'][area_col_name], bins=20, alpha=0.7, color='#2ca02c', edgecolor='black', label='Normal/Moderate')
                ax_area.hist(df_full[df_full['Classification'] == 'Anomaly'][area_col_name], bins=20, alpha=0.7, color=ACCENT_COLOR, edgecolor='black', label='Anomaly')
            
            ax_area.legend()
            ax_area.set_xlabel(f'Area ({area_unit})')
            ax_area.set_ylabel('Frequency')
            ax_area.set_title(f'Distribution of Erythrocyte Area ({area_unit})')
            ax_area.grid(axis='y', alpha=0.5)
            st.pyplot(fig_area)

            # --- Perimeter Histogram ---
            st.markdown("---")
            st.subheader(f"🔗 Perimeter Distribution (Contour Perimeter - {length_unit})")
            fig_perim, ax_perim = plt.subplots(figsize=(10, 6))
            
            if not df_full.empty:
                ax_perim.hist(df_full[df_full['Classification'] == 'Normal/Moderate'][perimeter_col_name], bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black', label='Normal/Moderate')
                ax_perim.hist(df_full[df_full['Classification'] == 'Anomaly'][perimeter_col_name], bins=20, alpha=0.7, color=ACCENT_COLOR, edgecolor='black', label='Anomaly')
            
            ax_perim.legend()
            ax_perim.set_xlabel(f'Perimeter ({length_unit})')
            ax_perim.set_ylabel('Frequency')
            ax_perim.set_title(f'Distribution of Erythrocyte Perimeter ({length_unit})')
            ax_perim.grid(axis='y', alpha=0.5)
            st.pyplot(fig_perim)


            # --- Results Table (Updated with Micrometer units) ---
            st.markdown("---")
            st.subheader("📋 Detailed Results Table")
            
            # Tworzenie mapowania kolumn do wyświetlenia
            column_mapping = {
                "Erythrocyte Number": "Cell ID",
                "Shape Factor": "SF (Major/Minor)",
                "Ellipticity": "Ellipticity",
                "Classification": "Classification"
            }
            
            # Dodanie kolumn pikselowych i mikrometrycznych w zależności od dostępności (czyli calibration_factor > 0.0)
            if calibration_factor > 0.0 and 'Major Axis (µm)' in df_full.columns:
                column_mapping.update({
                    'Major Axis (µm)': 'Major Axis (µm)',
                    'Minor Axis (µm)': 'Minor Axis (µm)',
                    'Area (µm²)': 'Area (µm²)',
                    'Perimeter (µm)': 'Perimeter (µm)',
                    'Major Axis': 'Major Axis (px)', # Zachowujemy też wartości w px
                    'Minor Axis': 'Minor Axis (px)',
                    'Area': 'Area (px²)',
                    'Perimeter': 'Perimeter (px)'
                })
            else:
                # Jeśli kalibracja wyłączona, pokazujemy tylko wartości w pikselach
                column_mapping.update({
                    'Major Axis': 'Major Axis (px)',
                    'Minor Axis': 'Minor Axis (px)',
                    'Area': 'Area (px²)',
                    'Perimeter': 'Perimeter (px)'
                })
            
            # Zmiana nazw kolumn i wybór odpowiedniej kolejności do wyświetlenia
            df_display = df_full.rename(columns=column_mapping)
            
            # Definicja kolejności kolumn (priorytet dla µm, jeśli są dostępne)
            ordered_columns = [
                'Cell ID', 'Classification', 'SF (Major/Minor)', 'Ellipticity',
                'Major Axis (µm)', 'Minor Axis (µm)', 'Area (µm²)', 'Perimeter (µm)',
                'Major Axis (px)', 'Minor Axis (px)', 'Area (px²)', 'Perimeter (px)'
            ]
            
            # Filtrowanie kolumn, które faktycznie istnieją po kalibracji/bez niej
            cols_to_show = [col for col in ordered_columns if col in df_display.columns]

            st.dataframe(df_display[cols_to_show], use_container_width=True)
        else:
            st.warning("⚠️ No erythrocytes detected in the image based on contour analysis (check minimal axis size setting).")

    else:
        st.warning("⚠️ Please upload a file or select the sample image option.")