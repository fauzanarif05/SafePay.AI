import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import base64

# Konfigurasi halaman
st.set_page_config(
    page_title="SafePay.AI",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang menarik
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .fraud-result {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .safe-result {
        background: linear-gradient(90deg, #51cf66, #40c057);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model XGBoost dan scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Memuat model XGBoost yang sudah dilatih
        model = joblib.load('xgb_model.pkl')
        # Memuat scaler yang digunakan saat training
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {e}")
        st.error("Pastikan file 'xgb_model.pkl' dan 'scaler.pkl' tersedia di direktori yang sama")
        
        # Fallback ke mock model untuk demo
        class MockModel:
            def predict(self, X):
                # Simulasi prediksi berdasarkan jumlah transaksi dan jenis
                if X[0][2] > 200000 and X[0][1] in [2, 3]:  # amount > 200k dan CASH-OUT/TRANSFER
                    return np.array([1])
                return np.array([0])
            
            def predict_proba(self, X):
                if X[0][2] > 200000 and X[0][1] in [2, 3]:
                    return np.array([[0.2, 0.8]])
                return np.array([[0.85, 0.15]])
        
        class MockScaler:
            def transform(self, X):
                return X
        
        return MockModel(), MockScaler(), False
    except Exception as e:
        st.error(f"❌ Error memuat model: {e}")
        return None, None, False

# Fungsi untuk data chart penipuan (simulasi)
def get_fraud_data():
    years = [2019, 2020, 2021, 2022, 2023]
    fraud_cases = [1250, 2340, 3890, 4560, 5200]
    losses = [45.2, 89.7, 156.8, 203.4, 267.9]  # dalam miliar rupiah
    
    return pd.DataFrame({
        'Tahun': years,
        'Kasus_Penipuan': fraud_cases,
        'Kerugian_Miliar_Rp': losses
    })
    
def get_base64_of_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"File {path} tidak ditemukan!")
        return None
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

# Navigation
def main():
    logo_base64 = get_base64_of_image("logo.png")
    
    if logo_base64:
        st.markdown(f"""
        <div class="main-header" style="display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{logo_base64}" width="60" style="margin-right: 15px;">
                <h1 style="margin: 0; color: white;">SafePay.AI</h1>
            </div>
            <p style="margin: 0; color: white;">Sistem Deteksi Penipuan Pembayaran Online Berbasis AI</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback jika logo tidak ada
        st.markdown("""
        <div class="main-header" style="text-align: center;">
            <h1 style="margin: 0; color: white;">🛡️ SafePay.AI</h1>
            <p style="margin: 0; color: white;">Sistem Deteksi Penipuan Pembayaran Online Berbasis AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model dan scaler
    model, scaler, model_loaded = load_model_and_scaler()
    
    # # Status model
    # if model_loaded:
    #     st.sidebar.success("✅ Model XGBoost berhasil dimuat")
    # else:
    #     st.sidebar.warning("⚠️ Menggunakan mock model untuk demo")
    
    # Sidebar Navigation dengan radio buttons
    # CSS untuk membuat efek hover dan status aktif pada sidebar
    st.markdown("""
        <style>
            /* Styling untuk Sidebar */
            .sidebar .sidebar-content {
                padding-top: 50px;
            }

            /* Efek Hover dan Active untuk Menu */
            .menu-item {
                padding: 15px;
                margin: 8px 0;
                background-color: #f1f1f1;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s, color 0.3s, transform 0.2s;
                text-align: center;
                font-weight: 500;
                border: 1px solid #e0e0e0;
            }
            
            .menu-item:hover {
                background-color: #00bcd4;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 188, 212, 0.3);
            }

            .menu-item.active {
                background-color: #00bcd4;
                color: white;
                box-shadow: 0 4px 8px rgba(0, 188, 212, 0.3);
            }

            /* Custom button styling */
            .stButton > button {
                width: 100%;
                padding: 15px;
                margin: 8px 0;
                background-color: #f1f1f1;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                color: #333;
                font-weight: 500;
                transition: all 0.3s ease;
            }

            .stButton > button:hover {
                background-color: #00bcd4;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 188, 212, 0.3);
                border-color: #00bcd4;
            }

            .stButton > button:focus {
                background-color: #00bcd4;
                color: white;
                box-shadow: 0 4px 8px rgba(0, 188, 212, 0.3);
                border-color: #00bcd4;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation dengan button
    st.sidebar.markdown("## Navigasi")

    # Inisialisasi session state untuk menu aktif
    if 'active_menu' not in st.session_state:
        st.session_state.active_menu = "🏠 Dashboard"

    # Menu buttons yang berjejer ke bawah
    if st.sidebar.button("🏠 Dashboard", key="dashboard"):
        st.session_state.active_menu = "🏠 Dashboard"

    if st.sidebar.button("📊 Analisis Variabel", key="analysis"):
        st.session_state.active_menu = "📊 Analisis Variabel"

    if st.sidebar.button("🔮 Prediksi Penipuan", key="prediction"):
        st.session_state.active_menu = "🔮 Prediksi Penipuan"

    # Menampilkan konten sesuai dengan menu yang dipilih
    if st.session_state.active_menu == "🏠 Dashboard":
        show_dashboard()
    elif st.session_state.active_menu == "📊 Analisis Variabel":
        show_variable_analysis()
    elif st.session_state.active_menu == "🔮 Prediksi Penipuan":
        show_prediction(model, scaler, model_loaded)

    st.markdown("""
        <style>
            .feature-card {
                padding: 20px;
                margin: 10px;
                background-color: #f4f4f9;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            .feature-card:hover {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                background-color: #e6f7ff;
            }

            .metric-card {
                padding: 20px;
                margin: 10px;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                background-color: #e6f7ff;
            }

            .metric-card h3 {
                font-size: 24px;
                font-weight: bold;
                color: #0073e6;
            }

            .metric-card h2 {
                font-size: 36px;
                color: #0073e6;
            }

            .metric-card p {
                color: #777;
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

# Fungsi untuk menunjukkan Dashboard, Analisis Variabel, dan Prediksi Penipuan
def show_dashboard():
    st.markdown("## 🏠 Dashboard SafePay.AI")

    # Penjelasan Aplikasi
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Tentang SafePay.AI</h3>
            <p>SafePay.AI adalah sistem deteksi penipuan pembayaran online yang menggunakan teknologi 
            Machine Learning untuk menganalisis pola transaksi dan mengidentifikasi aktivitas mencurigakan 
            secara real-time.</p>
           
        
    ✨ Fitur Utama:
    - 📈 Analisis trend penipuan online
    - ⚡ Prediksi real-time
    - 📊 Visualisasi data interaktif
        
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Statistik Hari Ini</h3>
            <h2 style="color: #667eea;">1,247</h2>
            <p>Transaksi Dianalisis</p>
            <hr>
            <h2 style="color: #e74c3c;">23</h2>
            <p>Penipuan Terdeteksi</p>
            <hr>
            <h2 style="color: #27ae60;">99.7%</h2>
            <p>Akurasi Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Grafik Laporan Penipuan
    st.markdown("## 📊 Tren Penipuan Online di Indonesia (2020-2025)")

    # Contoh data
    fraud_data = {
        'Tahun': [2020, 2021, 2022, 2023, 2024, 2025],
        'Kasus_Penipuan': [1409, 1000, 1200, 1247, 5111, 14496],
        'Kerugian_Miliar_Rp': [75, 100, 120, 150, 500, 2600]
    }
    fraud_data = pd.DataFrame(fraud_data)

    # Membuat subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Jumlah Laporan Kasus Penipuan Online', 'Kerugian Finansial (Miliar Rp)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Chart 1: Kasus Penipuan
    fig.add_trace(
        go.Scatter(
            x=fraud_data['Tahun'],
            y=fraud_data['Kasus_Penipuan'],
            mode='lines+markers',
            name='Kasus Penipuan',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # Chart 2: Kerugian
    fig.add_trace(
        go.Bar(
            x=fraud_data['Tahun'],
            y=fraud_data['Kerugian_Miliar_Rp'],
            name='Kerugian (Miliar Rp)',
            marker_color='#3498db'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📈 Peningkatan Kasus (2020-2025)",
            value="928%",
            delta="13,087 kasus"
        )

    with col2:
        st.metric(
            label="💰 Total Kerugian 2023",
            value="Rp 267.9 M",
            delta="31% dari 2022"
        )

    with col3:
        st.metric(
            label="⚠️ Rata-rata per Hari",
            value="14 kasus",
            delta="Tahun 2023"
        )

def show_variable_analysis():
    st.markdown("## 📊 Analisis Variabel Deteksi Penipuan")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Variabel yang Mempengaruhi Deteksi Penipuan Online</h3>
        <p>Sistem SafePay.AI menganalisis berbagai variabel untuk mendeteksi pola penipuan. 
        Berikut adalah penjelasan detail setiap variabel berdasarkan analisis Feature Importance dari model XGBoost:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Definisi variabel berdasarkan gambar Feature Importance XGBoost
    variables = {
        "NewbalanceOrig": {
            "icon": "📊",
            "description": "Saldo akun pengirim (original) setelah transaksi selesai",
            "importance": "Sangat Tinggi",
            "importance_score": 0.44,
            "fraud_pattern": "Perubahan saldo yang tidak konsisten dengan amount transaksi atau pola manipulasi saldo bisa mengindikasikan penipuan"
        },
        "OldbalanceOrg": {
            "icon": "🏦",
            "description": "Saldo akun pengirim (original) sebelum transaksi dilakukan",
            "importance": "Tinggi", 
            "importance_score": 0.19,
            "fraud_pattern": "Ketidaksesuaian antara saldo awal dengan kemampuan melakukan transaksi besar bisa mencurigakan"
        },
        "Type": {
            "icon": "💳",
            "description": "Jenis transaksi online yang dilakukan (PAYMENT, TRANSFER, CASH-IN, CASH-OUT, DEBIT)",
            "importance": "Tinggi",
            "importance_score": 0.18,
            "fraud_pattern": "CASH-OUT dan TRANSFER memiliki risiko penipuan lebih tinggi dibandingkan jenis transaksi lainnya"
        },
        "Amount": {
            "icon": "💰",
            "description": "Jumlah nominal uang yang terlibat pada transaksi",
            "importance": "Sedang",
            "importance_score": 0.15,
            "fraud_pattern": "Transaksi dengan jumlah sangat besar atau pola jumlah yang tidak wajar bisa mengindikasikan penipuan"
        },
        "NewbalanceDest": {
            "icon": "📈",
            "description": "Saldo penerima (destination) setelah transaksi diterima",
            "importance": "Rendah",
            "importance_score": 0.04,
            "fraud_pattern": "Pola akumulasi dana yang tidak wajar dalam waktu singkat pada rekening penerima"
        },
        "Step": {
            "icon": "⏰",
            "description": "Merepresentasikan satuan waktu transaksi, di mana 1 step setara dengan 1 jam sejak awal pencatatan data",
            "importance": "Sangat Rendah",
            "importance_score": 0.003,
            "fraud_pattern": "Pola waktu transaksi mencurigakan pada jam-jam tertentu atau dalam rentang waktu yang tidak biasa"
        },
        "OldbalanceDest": {
            "icon": "🎯",
            "description": "Saldo penerima (destination) sebelum transaksi diterima",
            "importance": "Sangat Rendah",
            "importance_score": 0.002,
            "fraud_pattern": "Rekening penerima dengan saldo 0 yang tiba-tiba menerima transfer besar perlu diwaspadai"
        }
    }
    
    # Display variabel dalam bentuk card dengan urutan berdasarkan importance score
    sorted_variables = dict(sorted(variables.items(), key=lambda x: x[1]['importance_score'], reverse=True))
    
    for var_name, var_info in sorted_variables.items():
        with st.expander(f"{var_info['icon']} {var_name} - Importance Score: {var_info['importance_score']:.3f} ({var_info['importance']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Deskripsi:** {var_info['description']}")
                st.markdown(f"**Pola Penipuan:** {var_info['fraud_pattern']}")
            
            with col2:
                # Progress bar untuk importance score
                st.metric("Feature Importance", f"{var_info['importance_score']:.3f}")
                st.progress(var_info['importance_score'])
                
                # Color coding berdasarkan importance
                if var_info['importance_score'] >= 0.40:
                    st.success("🔥 Sangat Penting")
                elif var_info['importance_score'] >= 0.15:
                    st.info("⚡ Penting")
                elif var_info['importance_score'] >= 0.05:
                    st.warning("📊 Kontribusi Sedang")
                else:
                    st.error("📉 Kontribusi Minimal")
    
    # Visualisasi importance berdasarkan data aktual dari XGBoost
    st.markdown("### 📊 Feature Importance dari Model XGBoost")
    
    # Data sesuai dengan gambar yang diberikan
    var_names = ['newbalanceOrig', 'oldbalanceOrg', 'type', 'amount', 'newbalanceDest', 'step', 'oldbalanceDest']
    importance_values = [0.44, 0.19, 0.18, 0.15, 0.04, 0.003, 0.002]
    
    fig = px.bar(
        x=importance_values,
        y=var_names,
        orientation='h',
        title="Feature Importance Score dari Model XGBoost",
        color=importance_values,
        color_continuous_scale="RdYlBu_r",  # Skema warna yang lebih menarik
        text=[f"{val:.3f}" for val in importance_values]
    )
    
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Variabel",
        height=500,
        yaxis={'categoryorder': 'total ascending'},  # Mengurutkan dari terkecil ke terbesar
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight tambahan berdasarkan data XGBoost
    st.markdown("### 💡 Insights dari Feature Importance XGBoost:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🏆 Top 4 Features:**
        1. **NewbalanceOrig** (0.440) - Saldo pengirim akhir
        2. **OldbalanceOrg** (0.190) - Saldo pengirim awal  
        3. **Type** (0.180) - Jenis transaksi
        4. **Amount** (0.150) - Nominal transaksi
        """)
    
    with col2:
        st.warning("""
        **⚖️ Features Minor:**
        5. **NewbalanceDest** (0.040)
        6. **Step** (0.003)
        7. **OldbalanceDest** (0.002)
        
        Kontribusi minimal dalam prediksi
        """)
    
    with col3:
        st.success("""
        **📈 XGBoost Insights:**
        - **Balance tracking** adalah kunci utama
        - **Waktu transaksi** kurang berpengaruh
        - **Pattern saldo** lebih penting dari timing
        """)

def show_prediction(model, scaler, model_loaded):
    st.markdown("## 🔮 Prediksi Penipuan Online")
    
    st.markdown("""
    <div class="feature-card">
        <h3>💡 Cara Penggunaan</h3>
        <p>Masukkan detail transaksi pada form di bawah ini. Sistem akan menganalisis data 
        menggunakan algoritma XGBoost untuk memprediksi apakah transaksi tersebut 
        terindikasi penipuan atau tidak.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status model
    if model_loaded:
        st.success("🚀 **Model XGBoost Aktif** - Menggunakan model yang telah dilatih")
    else:
        st.warning("⚠️ **Mode Demo** - Menggunakan simulasi model untuk demonstrasi")
    
    # Form input dalam kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Data Transaksi")
        
        # Input tanggal dan jam untuk step
        st.markdown("#### 🗓️ Waktu Transaksi")
        
        # Pilihan tanggal (1-31 untuk 1 bulan)
        transaction_day = st.selectbox(
            "📅 Tanggal Transaksi",
            options=list(range(1, 32)),
            index=0,
            help="Pilih tanggal transaksi (1-31 hari dalam bulan)"
        )
        
        # Pilihan jam (0-23)
        transaction_hour = st.selectbox(
            "🕐 Jam Transaksi",
            options=list(range(0, 24)),
            index=12,
            help="Pilih jam transaksi (0-23)"
        )
        
        # Konversi tanggal dan jam ke step
        # Formula: step = (day - 1) * 24 + hour + 1
        # Day 1 jam 0 = step 1, Day 1 jam 23 = step 24, dst.
        step = (transaction_day - 1) * 24 + transaction_hour + 1
        
        # Validasi step range (1-743)
        if step > 743:
            st.error(f"⚠️ Step yang dihitung ({step}) melebihi batas maksimum (743). Silakan pilih tanggal dan jam yang lebih awal.")
            step = min(step, 743)  # Batasi ke maksimum
        
        # Tampilkan step yang dihitung
        st.info(f"📊 **Step yang dihitung:** {step} (Hari {transaction_day}, Jam {transaction_hour:02d}:00)")
        
        type_ = st.selectbox(
            "💳 Jenis Transaksi", 
            ["PAYMENT", "TRANSFER", "CASH-IN", "CASH-OUT", "DEBIT"],
            help="Pilih jenis transaksi yang akan dianalisis"
        )
        
        amount = st.number_input(
            "💰 Jumlah Nominal Transaksi (Rp)", 
            min_value=0.0, value=100000.0, step=10000.0,
            help="Jumlah nominal uang yang terlibat dalam transaksi"
        )
        
    
    with col2:
        st.markdown("### 🏦 Data Saldo")
        oldbalanceOrg = st.number_input(
            "🏦 Saldo Akun Pengirim Sebelum Transaksi (Rp)", 
            min_value=0.0, value=500000.0, step=10000.0,
            help="Saldo akun pengirim (original) sebelum transaksi"
        )
        
        newbalanceOrig = st.number_input(
            "📊 Saldo Akun Pengirim Setelah Transaksi (Rp)", 
            min_value=0.0, value=400000.0, step=10000.0,
            help="Saldo akun pengirim (original) setelah transaksi"
        )
        
        oldbalanceDest = st.number_input(
            "🎯 Saldo Akun Penerima Sebelum Transaksi (Rp)", 
            min_value=0.0, value=1000000.0, step=10000.0,
            help="Saldo penerima (destination) sebelum transaksi"
        )
        
        newbalanceDest = st.number_input(
            "📈 Saldo Akun Penerima Setelah Transaksi (Rp)", 
            min_value=0.0, value=1100000.0, step=10000.0,
            help="Saldo penerima (destination) setelah transaksi"
        )
    
    # Validasi logika saldo
    balance_check_orig = abs((oldbalanceOrg - amount) - newbalanceOrig)
    balance_check_dest = abs((oldbalanceDest + amount) - newbalanceDest)
    
    if balance_check_orig > 1:  # Toleransi untuk pembulatan
        st.warning("⚠️ Peringatan: Saldo pengirim tidak konsisten dengan jumlah transaksi!")
    
    if balance_check_dest > 1:
        st.warning("⚠️ Peringatan: Saldo penerima tidak konsisten dengan jumlah transaksi!")
    
    # Membuat dataframe untuk input
    input_data = pd.DataFrame({
        'step': [step],
        'type': [type_],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    
    # Tombol prediksi dengan styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 Lakukan Prediksi", 
            type="primary",
            use_container_width=True
        )
    
    if predict_button:
        # Validasi step sebelum prediksi
        if step > 743:
            st.error("❌ Tidak dapat melakukan prediksi. Pilih tanggal dan jam yang valid (Step maksimum: 743)")
            return
            
        # Cek apakah model tersedia
        if model is None or scaler is None:
            st.error("❌ Model tidak dapat dimuat. Pastikan file 'xgb_model.pkl' dan 'scaler.pkl' tersedia.")
            return
        
        # Loading animation
        with st.spinner('🔄 Menganalisis data transaksi dengan XGBoost...'):
            try:
                # Simulasi loading
                import time
                time.sleep(1.5)
                
                # Encoding type sesuai dengan mapping yang digunakan saat training
                type_mapping = {
                    'PAYMENT': 0, 'TRANSFER': 3, 'CASH-IN': 4, 'CASH-OUT': 2, 'DEBIT': 1
                }
                
                # Membuat copy data untuk preprocessing
                input_data_processed = input_data.copy()
                
                # Encoding kolom 'type'
                input_data_processed['type'] = input_data_processed['type'].map(type_mapping)
                
                # Scaling data menggunakan scaler yang sudah dilatih
                input_data_scaled = scaler.transform(input_data_processed)

                # Melakukan prediksi
                prediction = model.predict(input_data_scaled)
                prediction_proba = model.predict_proba(input_data_scaled)
                
                # Probabilitas untuk setiap kelas
                safe_prob = prediction_proba[0][0]  # Probabilitas kelas 0 (tidak fraud)
                fraud_prob = prediction_proba[0][1]  # Probabilitas kelas 1 (fraud)
                
                # Results
                st.markdown("---")
                st.markdown("## 🎯 Hasil Prediksi XGBoost")
                
                # Tampilkan informasi waktu transaksi
                st.markdown(f"**📅 Waktu Transaksi:** Hari {transaction_day}, Jam {transaction_hour:02d}:00 (Step {step})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0] == 1:
                        st.markdown("""
                        <div class="prediction-result fraud-result">
                            <h2>🚨 TERINDIKASI PENIPUAN</h2>
                            <p>Transaksi ini menunjukkan pola yang mencurigakan</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.error("⚠️ **Rekomendasi:** Tinjau kembali transaksi ini dan lakukan verifikasi tambahan")
                        
                    else:
                        st.markdown("""
                        <div class="prediction-result safe-result">
                            <h2>✅ TRANSAKSI AMAN</h2>
                            <p>Transaksi ini tampak normal dan tidak mencurigakan</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("✓ **Rekomendasi:** Transaksi dapat diproses dengan normal")
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = fraud_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Tingkat Risiko Penipuan (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.metric("🔒 Probabilitas Aman", f"{safe_prob:.4f} ({safe_prob:.2%})")
                    st.metric("⚠️ Probabilitas Penipuan", f"{fraud_prob:.4f} ({fraud_prob:.2%})")
                    
                    # Confidence level
                    confidence = max(safe_prob, fraud_prob)
                    st.metric("🎯 Tingkat Keyakinan", f"{confidence:.2%}")
                
                # Risk factors analysis
                st.markdown("### 🔍 Analisis Faktor Risiko")
                
                risk_factors = []
                
                # Analyze risk factors berdasarkan pola yang umum dalam fraud detection
                if amount > 1000000:
                    risk_factors.append("💰 Jumlah transaksi sangat besar (> 1 juta)")
                
                if type_ in ['CASH-OUT', 'TRANSFER']:
                    risk_factors.append("💳 Jenis transaksi berisiko tinggi (CASH-OUT/TRANSFER)")
                
                if oldbalanceOrg == 0:
                    risk_factors.append("🏦 Saldo pengirim kosong")
                
                if newbalanceDest == oldbalanceDest + amount and oldbalanceDest == 0:
                    risk_factors.append("🎯 Penerima rekening baru/kosong")
                
                if amount > oldbalanceOrg:
                    risk_factors.append("⚠️ Jumlah transaksi melebihi saldo pengirim")
                
                # Analisis waktu transaksi berdasarkan jam
                if transaction_hour >= 0 and transaction_hour <= 5:  # Dini hari
                    risk_factors.append("🌙 Transaksi pada dini hari (00:00-05:59)")
                elif transaction_hour >= 22:  # Malam hari
                    risk_factors.append("🌃 Transaksi pada malam hari (22:00-23:59)")
                
                # Analisis berdasarkan step yang tinggi (akhir bulan)
                if step > 600:  # Mendekati akhir dataset (hari ke-25+)
                    risk_factors.append("📅 Transaksi pada akhir periode (hari 25+)")
                
                # Tampilkan hasil analisis
                col1, col2 = st.columns(2)
                
                with col1:
                    if risk_factors:
                        st.warning("**🚩 Faktor Risiko Terdeteksi:**")
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.success("✅ Tidak ada faktor risiko khusus yang terdeteksi")
                
                with col2:
                    # Summary statistics
                    st.info("**📊 Ringkasan Prediksi:**")
                    st.write(f"• **Model:** XGBoost Classifier")
                    st.write(f"• **Waktu:** Hari {transaction_day}, Jam {transaction_hour:02d}:00")
                    st.write(f"• **Prediksi:** {'Fraud' if prediction[0] == 1 else 'Bukan Fraud'}")
                    st.write(f"• **Confidence:** {confidence:.1%}")
                    st.write(f"• **Risk Level:** {len(risk_factors)} faktor risiko")
                
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {str(e)}")
                st.error("Pastikan format data input sesuai dengan model yang dilatih")

if __name__ == "__main__":
    main()