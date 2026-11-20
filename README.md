# Dashboard sampah Jawa Barat

Dashboard Streamlit untuk mengeksplorasi data dan hierarchical clustering pengelolaan sampah pada 27 kota/kabupaten Jawa Barat dari `SOURCE_CODE_DATMIN_KELOMPOK1.ipynb`.

## Menjalankan secara lokal

Dari folder proyek, gunakan lingkungan `.venv` dan pasang dependensi sesuai `requirements.txt`:

Untuk lingkungan yang sudah disiapkan, klik dua kali `jalankan_dashboard.bat`.

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit. Versi paket untuk reproduksi dicatat dalam `requirements.txt`.

Untuk menyiapkan ulang di komputer lain dengan Python terpasang:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Isi dashboard

- **Ringkasan:** indikator dan gambaran umum data wilayah.
- **Eksplorasi wilayah:** peringkat dan perbandingan indikator antarwilayah.
- **Analisis klaster:** anggota, profil, visualisasi PCA, dan evaluasi jumlah klaster K=2–8; default K=3.
- **Data & metodologi:** tabel data, asal data, dan penjelasan metode serta keterbatasannya.

Filter wilayah hanya mengubah tampilan. Standardisasi, klasterisasi, PCA, dan evaluasi model tetap menggunakan seluruh 27 wilayah agar hasil tidak berubah hanya karena filter.

## Data dan reproduksi

`data/sampah_jabar_snapshot.csv` merupakan snapshot **data olahan dari output tersimpan notebook**, bukan data mentah. Tujuh fitur numerik berasal dari sel 16, sedangkan nama wilayah dan label referensi klaster berasal dari sel 22; nomor sel dimulai dari nol. `Progress_SA_num` sudah melalui imputasi median dalam notebook.

`data/provenance.json` mencatat sumber, SHA256 notebook, sel ekstraksi, dan keterbatasan data. Gunakan snapshot, notebook dengan hash yang sama, kode aplikasi, serta versi dependensi dalam `requirements.txt` untuk mereproduksi analisis. Notebook asli tidak diubah.

Analisis mengikuti standardisasi fitur dan hierarchical clustering Ward dengan jarak Euclidean. PCA memakai solver `full`. Keduanya deterministik dan tidak menggunakan seed acak. Metrik klaster merupakan evaluasi internal untuk analisis deskriptif; tidak ada split train/test atau cross-validation, dan hasilnya bukan ukuran kemampuan prediksi terhadap data baru. Label klaster bukan urutan kualitas wilayah.

## Batasan interpretasi

- Tahun data dan URL sumber tidak tersedia. Berkas mentah `data_sampah.csv` juga tidak disertakan.
- Makna satuan persentase terhadap sumber asli **belum terverifikasi**. Angka dipertahankan sesuai output notebook, tanpa dikalikan 100.
- `Timbulan_num` dan `jumlah_sampah_ton_per_hari` adalah dua kolom berbeda. Definisi detail kolom kedua belum terverifikasi; jangan menganggapnya otomatis sebagai volume sampah terkelola.
- `% S. Terkelola` dan `% S. Belum Terkelola` saling melengkapi. Keduanya dipertahankan untuk mengikuti notebook, sehingga dimensi tersebut mendapat bobot ganda dalam klasterisasi.
- Nilai snapshot mengikuti presisi tampilan output notebook; presisi data mentah belum dapat diperiksa.
- Peta tidak disertakan karena `Jabar_By_Kab.geojson` tidak tersedia.
