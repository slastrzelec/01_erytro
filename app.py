import streamlit as st
import cv2
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

# --- Page settings ---
st.set_page_config(
    page_title="Erythrocyte Analysis App",
    page_icon="🔬"
)

# --- ZAKTUALIZOWANA SEKCJA WPROWADZAJĄCA Z WYJUSTOWANIEM ---
st.header("Dlaczego Kształt Erytrocytów Ma Znaczenie? 🩸")
st.markdown(
    """
    <p style="text-align: justify;">
    Kształt czerwonej krwinki (erytrocytu) jest krytyczny dla jej funkcji. 
    Zdrowe erytrocyty mają kształt <b>dwuwklęsłego dysku</b>, co maksymalizuje powierzchnię wymiany gazowej. 
    Wszelkie <b>anomalie kształtu</b> (np. wydłużenie, sierpowaty kształt, kulisty) wskazują na 
    różne schorzenia, takie jak niedokrwistość, talasemia, czy inne poważne zaburzenia krwi. 
    Analiza współczynnika kształtu (Shape Factor) jest kluczowym, kwantytatywnym narzędziem diagnostycznym.
    </p>
    """,
    unsafe_allow_html=True # Niezbędne do poprawnego renderowania HTML
)
st.markdown("---")
# --- KONIEC ZAKTUALIZOWANEJ SEKCJI ---


# --- Core function: erythrocyte shape analysis ---
def get_erythrocyte_shape_factors(image, anomaly_threshold=1.7):
    processed_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors_data = []
    anomalies_data = []

    for contour in contours:
        if len(contour) > 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                minor_axis = min(axes)
                major_axis = max(axes)
                
                if minor_axis > 0:
                    shape_factor = major_axis / minor_axis
                    ellipse_color = (0, 255, 0)  # Green = normal
                    if shape_factor > 1.3 and shape_factor <= 1.5:
                        ellipse_color = (0, 255, 255)  # Yellow = moderately elongated
                    elif shape_factor > 1.5:
                        ellipse_color = (0, 0, 255)  # Red = highly elongated

                    if shape_factor <= anomaly_threshold:
                        shape_factors_data.append({
                            "Erythrocyte Number": len(shape_factors_data) + 1,
                            "Shape Factor": shape_factor
                        })
                        cv2.ellipse(processed_image, ellipse, ellipse_color, 2)
                        angle_rad = np.radians(orientation)
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
                        cv2.putText(processed_image, str(len(shape_factors_data)),
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        anomalies_data.append({
                            "Erythrocyte Number": len(anomalies_data) + 1,
                            "Shape Factor": shape_factor
                        })
                        cv2.ellipse(processed_image, ellipse, (255, 0, 255), 2)
                        cv2.putText(processed_image, f"A{len(anomalies_data)}",
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            except cv2.error:
                continue

    return processed_image, shape_factors_data, anomalies_data


# --- Streamlit UI ---
st.title("🔬 Erythrocyte Analysis App")
st.markdown("### Upload your image or use the sample one for testing")

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
        st.image(processed_img, channels="BGR", caption="Processed Erythrocytes")

        df_normal = pd.DataFrame(shape_factors)
        df_anomalies = pd.DataFrame(anomalies)

        if not df_normal.empty or not df_anomalies.empty:
            if not df_normal.empty:
                avg = df_normal['Shape Factor'].mean()
                med = df_normal['Shape Factor'].median()
                std = df_normal['Shape Factor'].std()
                st.markdown("---")
                st.subheader("📈 Descriptive Statistics (Normal Cells)")
                st.info(f"**Average:** {avg:.2f}\n\n**Median:** {med:.2f}\n\n**Standard Deviation:** {std:.2f}")

            # --- Histogram ---
            st.markdown("---")
            st.subheader("📊 Shape Factor Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            if not df_normal.empty:
                ax.hist(df_normal['Shape Factor'], bins=15, alpha=0.6, color='b', edgecolor='black', label='Normal')
            if not df_anomalies.empty:
                ax.hist(df_anomalies['Shape Factor'], bins=15, alpha=0.6, color='m', edgecolor='black', label='Anomaly')
            ax.legend()
            ax.set_xlabel('Shape Factor')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            # --- Results Table ---
            st.markdown("---")
            st.subheader("📋 Results Table")
            st.dataframe(pd.concat([df_normal, df_anomalies], ignore_index=True))
        else:
            st.warning("⚠️ No erythrocytes detected.")

    else:
        st.warning("⚠️ Please upload a file or select the sample image option.")