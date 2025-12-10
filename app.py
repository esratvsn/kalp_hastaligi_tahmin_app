import streamlit as st
import pickle
import numpy as np
import pandas as pd 
import base64 
from io import BytesIO 

# --- 1. MODEL VE SCALER YÜKLEME ---
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model veya Scaler dosyası bulunamadı. Lütfen 'heart_model.pkl' ve 'scaler.pkl' dosyalarının uygulama dizininde olduğundan emin olun.")
    st.stop()
except Exception as e:
    st.error(f"Model yüklenirken beklenmedik bir hata oluştu: {e}")
    st.stop()


# ----------------------------------------------------------------------
# 🔥 ÖZELLİK ÖNEM SKORLARI (Kısaltmaları açılmış, grafik için)
FEATURE_IMPORTANCE_SCORES = {
    'Talyum Testi (Thal)': 0.187,
    'Göğüs Ağrısı Tipi (CP)': 0.145,
    'Boyalı Damar Sayısı (CA)': 0.13,
    'Maks. Kalp Atış Hızı (Thalach)': 0.11,
    'ST Depresyonu (Oldpeak)': 0.08,
    'Yaş': 0.07,
    'Egzersizle Göğüs Ağrısı (Exang)': 0.065,
    'Kolesterol (Chol)': 0.06,
    'İstirahat Kan Basıncı (Trestbps)': 0.055,
    'Eğim (Slope)': 0.045,
    'Cinsiyet (Sex)': 0.038,
    'İstirahat EKG (Restecg)': 0.015,
    'Açlık Kan Şekeri (FBS)': 0.005,
}

df_importance = pd.DataFrame(
    list(FEATURE_IMPORTANCE_SCORES.items()), 
    columns=['Özellik', 'Önem Skoru']
).set_index('Özellik')
# ----------------------------------------------------------------------


# --- 2. SAYFA AYARLARI VE BAŞLIK ---
st.set_page_config(
    page_title="Kalp Hastalığı Tahmin Uygulaması",
    page_icon="❤️",
    layout="wide"
)

# Başlık ve Açıklama
st.markdown("<h1 style='text-align:center;color:#E50000;'>❤️ Kalp Hastalığı Risk Tahmin Aracı</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Makine öğrenimi destekli bu araç, girdiğiniz parametrelere göre kalp hastalığı riskinizi tahmin eder.</p>", unsafe_allow_html=True)

# Yan Çubuk (Sidebar) Kod Bloğu (Artık grafik burada yok!)
with st.sidebar:
    st.title("ℹ️ Kullanım Kılavuzu")
    st.info("Lütfen yandaki tüm alanları eksiksiz ve doğru bir şekilde doldurunuz. Model, bu verileri kullanarak risk analizi yapacaktır.")
    st.markdown("---")
    st.subheader("⚠️ Önemli Sorumluluk Reddi")
    st.caption("Bu uygulama bir tıbbi teşhis aracı DEĞİLDİR. Sonuçlar yalnızca bilgilendirme amaçlıdır ve profesyonel tıbbi tavsiyenin yerini tutmaz. Daima bir doktora danışın.")
    # Not: Önceden burada bulunan grafik kaldırıldı.


# --- 3. GÖRSEL VE FORM AYIRMA ---

col_img, col_form = st.columns([1, 2], gap="large")

# ----------------- GIF YÜKLEME VE ORTALAMA BÖLÜMÜ (Base64 ile) -----------------
with col_img:
    try:
        gif_file = "heart_beat.gif" 

        # Base64 Kodlama (GIF'in animasyonunu garanti eder)
        with open(gif_file, "rb") as f:
            contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")

        # HTML olarak yerleştirme
        st.markdown(
            f"""
            <p style='text-align:center; color:#E50000; font-weight:bold;'>Kardiyak Sağlık Analizi</p>
            <img src="data:image/gif;base64,{data_url}" width="350" style="display: block; margin-left: auto; margin-right: auto;"/>
            """, 
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"⚠️ Animasyon yüklenemedi. '{gif_file}' dosyasını klasöre ekleyin.")
        st.image("heart.png", caption="Kardiyak Sağlık Analizi (Yedek)", width=350)
    except Exception as e:
        st.error(f"GIF yüklenirken bir hata oluştu: {e}")
        
    st.markdown("---")
    
    # --- TAHMİN BUTONU ---
    if 'predict_button' not in st.session_state:
        st.session_state.predict_button = False

    if st.button("🔍 RİSKİ HESAPLA", type="primary", use_container_width=True):
        st.session_state.predict_button = True


# --- 4. GİRDİ FORMU (GRUPLANDIRILMIŞ VE TEMALI) ---
with col_form:
    st.subheader("🩺 Hasta Parametreleri Girişi")

    # A. TEMEL VE DEMOGRAFİK BİLGİLER
    with st.container(border=True):
        st.markdown("##### 👤 Temel ve Fiziksel Bilgiler")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Yaş", 18, 100, 50, help="Kişinin yaşı.")
        with c2:
            sex_option = st.selectbox("Cinsiyet", options=["Erkek (1)", "Kadın (0)"], index=0)
            sex = int(sex_option.split('(')[1].split(')')[0]) 
        with c3:
            cp = st.selectbox("Göğüs Ağrısı Tipi (CP)", [0, 1, 2, 3], index=0)
            st.caption("0=Tipik Anjina, 3=Asemptomatik")

    # B. BİYOKİMYASAL VE EKG SONUÇLARI
    with st.expander("🩸 Biyokimyasal ve EKG Verileri (Tıklayın)"):
        
        col4, col5 = st.columns(2)
        with col4:
            trestbps = st.number_input("İstirahat Kan Basıncı (mmHg)", 80, 200, 120, help="İstirahat sırasındaki kan basıncı.")
            
            fbs_label = "Açlık Kan Şekeri >120 mg/dl?"
            fbs_option = st.selectbox(fbs_label, options=["Hayır (0)", "Evet (1)"], index=0)
            fbs = int(fbs_option.split('(')[1].split(')')[0]) 

        with col5:
            chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200, help="Serum kolesterol seviyesi.")
            restecg = st.selectbox("İstirahat EKG Sonucu", [0, 1, 2])
            st.caption("0=Normal, 1=ST-T Bozukluğu, 2=Hipertrofi")

    # C. EGZERSİZ VE STRES TESTİ VERİLERİ
    with st.expander("📈 Kardiyak Stres Testi Verileri (Tıklayın)"):
        
        col6, col7 = st.columns(2)
        with col6:
            thalach = st.number_input("Maksimum Kalp Atış Hızı", 60, 220, 150, help="Egzersiz sırasında ulaşılan maksimum kalp atış hızı.")
            oldpeak = st.number_input("ST Depresyonu (Oldpeak)", 0.0, 6.5, 1.0, step=0.1, help="Egzersiz sonrası EKG'deki düşüş miktarı.")
            slope = st.selectbox("Eğim", [0, 1, 2], help="0=Yukarı eğimli, 1=Düz, 2=Aşağı eğimli.")
        
        with col7:
            exang_label = "Egzersizle Göğüs Ağrısı?"
            exang_option = st.selectbox(exang_label, options=["Hayır (0)", "Evet (1)"], index=0)
            exang = int(exang_option.split('(')[1].split(')')[0])

            ca = st.selectbox("Boyalı Damar Sayısı (CA)", [0, 1, 2, 3], help="Anjiyografide görülen büyük damar sayısı.")
            thal = st.selectbox("Talyum Stres Testi (Thal)", [1, 2, 3])
            st.caption("1=Normal, 2=Sabit Kusur, 3=Tersinebilir Kusur.")

        
# --- 5. TAHMİN VE SONUÇ GÖSTERİMİ ---

if st.session_state.predict_button:
    
    st.markdown("## ") 

    try:
        # NumPy dizisine dönüştürme
        values = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])

        values_scaled = scaler.transform(values)
        prediction = model.predict(values_scaled)[0]

        # --- DİNAMİK SONUÇ GÖSTERİMİ ---
        st.markdown("---")
        st.subheader("✅ Analiz Sonucu")
        
        if prediction == 1:
            st.error("❗ YÜKSEK RİSK TESPİT EDİLDİ", icon="🚨")
            st.markdown("""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid #E50000; background-color: #331a1a;">
                <p style='font-size: 18px;'>
                Girilen verilere göre **Kalp Hastalığı Riskiniz YÜKSEK** görünmektedir. 
                Bu durum, bir kardiyoloji uzmanına danışmanız gerektiğini gösterir.
                </p>
                <ul>
                    <li>Yaşam tarzı değişikliklerini (diyet ve egzersiz) değerlendirin.</li>
                    <li>Kan basıncı ve kolesterol seviyelerinizi düzenli olarak kontrol ettirin.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("✔ DÜŞÜK RİSK TESPİT EDİLDİ", icon="👍")
            st.markdown("""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid #38761D; background-color: #1a331a;">
                <p style='font-size: 18px;'>
                Girilen verilere göre **Kalp Hastalığı Riskiniz DÜŞÜK** görünmektedir. 
                Sağlıklı yaşam tarzınızı korumaya devam edin.
                </p>
                <ul>
                    <li>Yine de rutin sağlık kontrollerinizi aksatmayın.</li>
                    <li>Sağlıklı beslenmeye ve düzenli egzersiz yapmaya devam edin.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        
        # 🔥 GRAFİĞİ BURAYA KOYUYORUZ! (Tahmin sonucunun hemen altına)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Modelin Karar Mekanizması")
        st.caption("Bu tahminin yapılmasında modelin en çok öncelik verdiği özelliklerin sıralaması:")
        # Grafiği ana sütunun tamamına yerleştirin
        st.bar_chart(df_importance, color="#E50000") 
        st.caption("Kaynak: Random Forest modelinizden elde edilen özellik önem skorları.")


    except Exception as e:
        st.error("❌ Hesaplama sırasında bir sorun oluştu. Lütfen tüm alanları kontrol edin ve girdiğiniz değerlerin geçerli aralıkta olduğundan emin olun.")
        st.caption(f"Teknik Detay (Geliştirici için): {e}")

    # Hesaplama bittikten sonra butona basılma durumunu sıfırla
    st.session_state.predict_button = False


# --- 6. FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Developed by Esra Tavşan • 2025</p>", unsafe_allow_html=True)