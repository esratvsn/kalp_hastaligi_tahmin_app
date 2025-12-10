# ❤️ Kalp Hastalığı Risk Tahmin Uygulaması

Bu proje, makine öğrenimi (Machine Learning) algoritmaları kullanılarak, hasta verilerine dayanarak kalp hastalığı riskini tahmin eden interaktif bir web uygulamasıdır. Uygulama, hızlı prototipleme ve veri bilimi uygulamaları için Streamlit çatısı ile geliştirilmiştir.

## 🚀 Proje Amacı ve Kullanılan Teknolojiler

Bu uygulama, bir Random Forest sınıflandırma modelinin tahmin gücünü basit bir kullanıcı arayüzü ile birleştirir.

* **Web Çatısı:** Streamlit
* **Model:** Random Forest Classifier (Doğruluk: %90+)
* **Veri İşleme:** Pandas, NumPy
* **Model Kaydı:** `pickle` (heart_model.pkl ve scaler.pkl)
* **Görselleştirme:** Streamlit'in yerleşik grafik ve HTML/CSS yetenekleri.

## 💡 Temel Özellikler

* **Kullanıcı Dostu Arayüz:** Hastalık riskini gösteren renk kodlu dinamik sonuçlar.
* **Güven Skoru:** Modelin tahminine olan güvenini gösteren görselleştirme.
* **Model Şeffaflığı:** Tahminde en etkili olan özellikleri gösteren açıklayıcı Özellik Önem Grafiği.
* **Dinamik Görsel:** CSS/Base64 ile entegre edilmiş hareketli nabız (GIF) görseli.

## 🛠️ Yerel Çalıştırma Talimatları

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Gerekli Kütüphaneleri Kurun:** `requirements.txt` dosyasındaki tüm bağımlılıkları yükleyin.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Uygulamayı Başlatın:** Proje klasörünüzde terminali açın ve Streamlit ile uygulamayı çalıştırın.
    ```bash
    streamlit run app.py
    ```

---
*Developed by Esra Tavşan - 2025*
