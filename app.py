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
    layout="wide" # Użycie szerokiego układu dla profesjonalnego wyglądu
)

# Zmienna do spójnego koloru akcentu (dla config.toml lub ręcznego ustawienia)
ACCENT_COLOR = "#B71C1C" 

# --- MOVED SECTION TO THE VERY TOP (TITLE & UPLOAD INSTRUCTIONS) ---
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

# Użycie kolumn: pierwsza na abstrakt, druga (mniejsza) na przycisk
col_abstract, col_button = st.columns([4, 1])

with col_abstract:
    # Użycie koloru akcentu
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


# --- Core function: erythrocyte shape analysis (Zaktualizowana do zapisu pólosi) ---
def get_erythrocyte_shape_factors(image, anomaly_threshold=1.7):
    processed_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors_data = []
    anomalies_data = []
    
    # Stała minimalnego rozmiaru
    MIN_AXIS_SIZE = 50 

    for contour in contours:
        if len(contour) > 5: 
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                # Minor i Major axis są teraz przechowywane
                minor_axis = min(axes)
                major_axis = max(axes)
                
                # NOWY WARUNEK: Pomijamy kontur, jeśli któraś z osi jest mniejsza niż MIN_AXIS_SIZE
                if major_axis < MIN_AXIS_SIZE or minor_axis < MIN_AXIS_SIZE:
                    continue
                
                if minor_axis > 0:
                    shape_factor = major_axis / minor_axis
                    ellipse_color = (0, 255, 0)  # Green = normal
                    
                    if shape_factor > 1.3 and shape_factor <= 1.5:
                        ellipse_color = (0, 255, 255)  # Yellow = moderately elongated
                    elif shape_factor > 1.5:
                        ellipse_color = (0, 0, 255)  # Red = highly elongated

                    if shape_factor <= anomaly_threshold:
                        # Zapisujemy długości półosi
                        shape_factors_data.append({
                            "Erythrocyte Number": len(shape_factors_data) + 1,
                            "Shape Factor": shape_factor,
                            "Major Axis": major_axis,
                            "Minor Axis": minor_axis
                        })
                        cv2.ellipse(processed_image, ellipse, ellipse_color, 2)
                        angle_rad = np.radians(orientation)
                        
                        # Rysowanie Osie (Major/Minor)
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
                        
                        # Etykietowanie komórki
                        cv2.putText(processed_image, str(len(shape_factors_data)),
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Zapisujemy długości półosi dla anomalii
                        anomalies_data.append({
                            "Erythrocyte Number": len(anomalies_data) + 1,
                            "Shape Factor": shape_factor,
                            "Major Axis": major_axis,
                            "Minor Axis": minor_axis
                        })
                        cv2.ellipse(processed_image, ellipse, (255, 0, 255), 2) # Kolor Magenta dla anomalii
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
        processed_img, shape_factors, anomalies = get_erythrocyte_shape_factors(image_to_process, anomaly_threshold_slider)

        st.subheader("📊 Analysis Results")

        df_normal = pd.DataFrame(shape_factors)
        df_anomalies = pd.DataFrame(anomalies)

        if not df_normal.empty or not df_anomalies.empty:
            
            # --- START: VISUAL IMPROVEMENT (Metrics and Image side-by-side) ---
            
            col_img, col_metrics = st.columns([3, 2]) # Szeroki układ dla lepszej wizualnej separacji

            with col_img:
                st.markdown("##### Processed Image")
                st.image(processed_img, channels="BGR")

            with col_metrics:
                if not df_normal.empty:
                    avg_sf = df_normal['Shape Factor'].mean()
                    med_sf = df_normal['Shape Factor'].median()
                    std_sf = df_normal['Shape Factor'].std()
                    
                    # OBLICZENIA DŁUGOŚCI PÓŁOSI
                    avg_major_axis = df_normal['Major Axis'].mean()
                    avg_minor_axis = df_normal['Minor Axis'].mean()
                    
                    st.markdown("##### Key Statistics (Normal Cells)")
                    
                    # Stara Metryka - Rozkład w 2 kolumnach
                    col_met1, col_met2 = st.columns(2)
                    
                    with col_met1:
                        st.metric(label="Average Shape Factor", value=f"{avg_sf:.2f}", help="Średnia proporcja głównej osi do mniejszej osi.")
                    
                    with col_met2:
                        st.metric(label="Median Shape Factor", value=f"{med_sf:.2f}", help="Mediana rozkładu.")
                    
                    # NOWE Metryki - Średnia długość półosi
                    st.markdown("---") 
                    st.metric(label="Avg Major Axis (px)", value=f"{avg_major_axis:.2f}", help="Średnia długość dużej półosi (w pikselach).")
                    st.metric(label="Avg Minor Axis (px)", value=f"{avg_minor_axis:.2f}", help="Średnia długość małej półosi (w pikselach).")
                        
                    st.metric(label="Std. Deviation (SF)", value=f"{std_sf:.2f}", help="Odchylenie standardowe dla Shape Factor.")
                    st.metric(label="Total Cells Analyzed", value=len(shape_factors) + len(anomalies))
                else:
                    st.info("No normal cells detected below the threshold.")

            st.markdown("---")
            # --- END: VISUAL IMPROVEMENT ---

            # --- Histogram ---
            st.subheader("📊 Shape Factor Distribution")
            fig, ax = plt.subplots(figsize=(10, 6)) # Zwiększony rozmiar figury dla szerokiego układu
            if not df_normal.empty:
                ax.hist(df_normal['Shape Factor'], bins=15, alpha=0.7, color='#1f77b4', edgecolor='black', label='Normal (SF <= Threshold)')
            if not df_anomalies.empty:
                ax.hist(df_anomalies['Shape Factor'], bins=15, alpha=0.7, color=ACCENT_COLOR, edgecolor='black', label='Anomaly (SF > Threshold)')
            
            ax.axvline(x=anomaly_threshold_slider, color='r', linestyle='--', label=f'Threshold ({anomaly_threshold_slider:.2f})')
            
            ax.legend()
            ax.set_xlabel('Shape Factor (Major Axis / Minor Axis)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Erythrocyte Shape Factors')
            ax.grid(axis='y', alpha=0.5)
            st.pyplot(fig)

            # --- Results Table (Zaktualizowana do wyświetlania długości pólosi) ---
            st.markdown("---")
            st.subheader("📋 Detailed Results Table")
            # Zmiana nazw kolumn dla lepszej czytelności
            df_full = pd.concat([df_normal, df_anomalies], ignore_index=True)
            df_full.rename(columns={
                "Erythrocyte Number": "Cell ID",
                "Shape Factor": "SF (Major/Minor)",
                "Major Axis": "Major Axis (px)",
                "Minor Axis": "Minor Axis (px)"
            }, inplace=True)
            
            # Dodanie kolumny klasyfikacji
            df_full['Classification'] = np.where(df_full['SF (Major/Minor)'] > anomaly_threshold_slider, 'Anomaly (SF > Threshold)', 'Normal/Moderate')

            st.dataframe(df_full, use_container_width=True)
        else:
            st.warning("⚠️ No erythrocytes detected in the image based on contour analysis.")

    else:
        st.warning("⚠️ Please upload a file or select the sample image option.")