import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REGION = "Kota/Kabupaten"
FEATURES = [
    "% S. Terkelola",
    "% S. Belum Terkelola",
    "jumlah_sampah_ton_per_hari",
    "Jumlah_Unit",
    "Jumlah_Penduduk",
    "Timbulan_num",
    "Progress_SA_num",
]
LABELS = {
    "% S. Terkelola": "Sampah terkelola (% sesuai source)",
    "% S. Belum Terkelola": "Belum terkelola (% sesuai source)",
    "jumlah_sampah_ton_per_hari": "Jumlah sampah (ton/hari, kolom source)",
    "Jumlah_Unit": "Jumlah unit tercatat",
    "Jumlah_Penduduk": "Jumlah penduduk (jiwa)",
    "Timbulan_num": "Timbulan (ton/hari)",
    "Progress_SA_num": "Progress SA (nilai source)",
}
COLORS = ["#16735b", "#e0a247", "#678bbc", "#ad718a", "#80a44f", "#c46f4b", "#7b75ad", "#8d9e9a"]

st.set_page_config(page_title="Resik | Dashboard Sampah Jawa Barat", layout="wide", initial_sidebar_state="expanded")
st.html("""
<style>
    .stApp { background: #f6f7f3; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1560px; }
    h1, h2, h3 { color: #173e31; letter-spacing: -.035em; }
    h1 { font-size: 2.25rem !important; }
    h3 { font-size: 1.12rem !important; }
    [data-testid="stHeader"] { background: #f6f7f3e8; }
    [data-testid="stSidebar"] { background: #eaf0e6; border-right: 1px solid #dae2d6; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e0e7db; border-radius: 14px; padding: 19px 20px; min-height: 122px; }
    [data-testid="stMetricLabel"] { color: #6b7e72; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #193f31; font-size: 1.85rem; letter-spacing: -.045em; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 14px; border-color: #e0e7db; background: white; }
    [data-testid="stCaptionContainer"] { color: #718074; }
    .brand { font-size: 2rem; font-weight: 850; letter-spacing: -.08em; color: #175c42; line-height: 1.1; }
    .brand span { color: #d59b3a; }
    .brand-sub { color: #70816d; font-size: .67rem; letter-spacing: .18em; margin: 9px 0 28px; }
    .eyebrow { color: #728575; text-transform: uppercase; letter-spacing: .16em; font-size: .66rem; font-weight: 700; margin-bottom: 9px; }
    .hero { position: relative; overflow: hidden; background: #164f3b; padding: 27px 32px; border-radius: 17px; color: #eff5e9; margin-bottom: 23px; }
    .hero:after { content: ''; width: 210px; height: 210px; border: 34px solid #ffffff09; border-radius: 50%; position: absolute; right: 35px; top: -75px; pointer-events: none; }
    .hero .eyebrow { color: #bfd1ae; }
    .hero h2 { color: #f7f9ef; font-size: 1.9rem; font-weight: 650; margin: 0 0 8px; padding: 0; }
    .hero p { color: #c5d6c4; font-size: .9rem; margin: 0; max-width: 690px; }
    .chip { display: inline-block; background: #e7eddf; border: 1px solid #d6e0cd; border-radius: 30px; padding: 5px 12px; font-size: .72rem; color: #4f694f; }
    .note { background: #eef3e7; border-left: 3px solid #71994e; padding: 14px 17px; border-radius: 0 9px 9px 0; color: #476144; font-size: .84rem; line-height: 1.6; margin: 8px 0 18px; }
    .footer { color: #80907f; border-top: 1px solid #dee5d8; padding-top: 15px; margin-top: 32px; font-size: .72rem; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label { padding: 8px 10px; border-radius: 8px; }
    @media (max-width: 700px) { .block-container { padding: 1rem; } .hero { padding: 22px; } .hero h2 { font-size: 1.5rem; } }
</style>
""")


def id_number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


@st.cache_data
def load_snapshot():
    return pd.read_csv(ROOT / "data/sampah_jabar_snapshot.csv", encoding="utf-8-sig")


@st.cache_data
def analyze(snapshot, n_clusters):
    scaled = StandardScaler().fit_transform(snapshot[FEATURES])
    tree = linkage(scaled, method="ward")
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward").fit_predict(scaled)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(scaled)
    evaluation = []
    for k in range(2, 9):
        cut_labels = fcluster(tree, k, criterion="maxclust")
        wcss = sum(np.square(scaled[cut_labels == label] - scaled[cut_labels == label].mean(axis=0)).sum() for label in np.unique(cut_labels))
        evaluation.append({
            "K": k,
            "WCSS": wcss,
            "Silhouette": silhouette_score(scaled, cut_labels),
            "Davies–Bouldin": davies_bouldin_score(scaled, cut_labels),
            "Calinski–Harabasz": calinski_harabasz_score(scaled, cut_labels),
        })
    clustered = snapshot.copy()
    clustered["Klaster"] = [f"Klaster {label + 1}" for label in labels]
    clustered["PC1"] = coordinates[:, 0]
    clustered["PC2"] = coordinates[:, 1]
    return clustered, pd.DataFrame(evaluation), tree, pca.explained_variance_ratio_


def chart(frame, spec, height=320):
    st.vega_lite_chart(frame, {
        "height": height,
        "config": {
            "view": {"stroke": None},
            "axis": {"domain": False, "gridColor": "#edf0e8", "labelColor": "#6b7e70", "titleColor": "#6b7e70", "labelFontSize": 11, "titleFontWeight": "normal"},
            "legend": {"orient": "bottom", "title": None, "labelColor": "#607463", "symbolType": "circle"},
        },
        **spec,
    }, width="stretch", theme=None)


def cluster_color():
    return {"field": "Klaster", "type": "nominal", "scale": {"domain": cluster_names, "range": COLORS[:n_clusters]}, "legend": {"title": None}}


snapshot = load_snapshot()
provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8-sig"))

with st.sidebar:
    st.html('<div class="brand">resik<span>.</span></div><div class="brand-sub">WAWASAN SAMPAH JAWA BARAT</div>')
    page = st.radio("Navigasi", ["Ringkasan", "Eksplorasi wilayah", "Analisis klaster", "Data & metodologi"], label_visibility="collapsed", key="page")
    st.divider()
    st.markdown("**Filter tampilan**")
    region_type = st.selectbox("Jenis wilayah", ["Semua wilayah", "Kabupaten", "Kota"], key="region_type")
    selected_regions = st.multiselect("Pilih wilayah", sorted(snapshot[REGION]), placeholder="Semua kota / kabupaten", key="regions")
    with st.expander("Pengaturan klaster"):
        n_clusters = st.slider("Jumlah klaster (K)", 2, 8, 3, key="n_clusters")
        st.caption("Ward–Euclidean, tujuh fitur standar. Model selalu memakai seluruh 27 wilayah.")
    cluster_names = [f"Klaster {i + 1}" for i in range(n_clusters)]
    selected_clusters = st.multiselect("Klaster", cluster_names, placeholder="Semua klaster", key=f"clusters_{n_clusters}")
    st.caption("Pilihan kosong berarti semua. Filter hanya mengubah tampilan, bukan membentuk ulang model.")
    st.divider()
    st.html('<span class="chip">Snapshot notebook</span>')
    st.caption("Tahun data belum tercantum pada source. Angka bukan pemantauan waktu nyata.")

clustered, evaluation, tree, explained_variance = analyze(snapshot, n_clusters)
visible = clustered.copy()
if region_type != "Semua wilayah":
    visible = visible[visible[REGION].str.startswith(region_type + " ")]
if selected_regions:
    visible = visible[visible[REGION].isin(selected_regions)]
if selected_clusters:
    visible = visible[visible["Klaster"].isin(selected_clusters)]

st.html('<div class="eyebrow">OBSERVATORIUM LINGKUNGAN &nbsp; / &nbsp; JAWA BARAT</div>')
title_column, badge_column = st.columns([4, 1])
with title_column:
    st.title({"Ringkasan": "Dashboard pengelolaan sampah", "Eksplorasi wilayah": "Kenali setiap wilayah", "Analisis klaster": "Pola di balik pengelompokan", "Data & metodologi": "Data yang bisa ditelusuri"}[page])
with badge_column:
    st.html('<div style="text-align:right;padding-top:17px"><span class="chip">Hierarchical clustering</span></div>')
st.caption(f"{len(visible)} dari {len(snapshot)} wilayah ditampilkan · {n_clusters} klaster · Jawa Barat")

if visible.empty and page in ["Ringkasan", "Eksplorasi wilayah"]:
    st.info("Tidak ada wilayah yang cocok. Ubah atau kosongkan filter di sidebar.")
    st.stop()

if page == "Ringkasan":
    st.html('<div class="hero"><div class="eyebrow">DARI DATA, MENUJU PEMAHAMAN</div><h2>Memahami sampah. Menata langkah.</h2><p>Jelajahi timbulan, karakteristik wilayah, dan pola pengelolaan sampah kota/kabupaten di Jawa Barat.</p></div>')
    metrics = st.columns(4)
    metrics[0].metric("Wilayah dalam cakupan", id_number(len(visible)))
    metrics[1].metric("Timbulan · ton/hari", id_number(visible["Timbulan_num"].sum()))
    metrics[2].metric("Penduduk · jiwa", id_number(visible["Jumlah_Penduduk"].sum()))
    metrics[3].metric("Unit tercatat", id_number(visible["Jumlah_Unit"].sum()))
    st.write("")
    left, right = st.columns([1.65, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Wilayah dengan timbulan terbesar")
        st.caption("Sepuluh wilayah teratas dalam cakupan filter · ton per hari")
        top_regions = visible.nlargest(10, "Timbulan_num")
        chart(top_regions, {"mark": {"type": "bar", "cornerRadiusEnd": 4, "height": 17}, "encoding": {
            "y": {"field": REGION, "type": "nominal", "sort": "-x", "title": None, "axis": {"labelLimit": 195}},
            "x": {"field": "Timbulan_num", "type": "quantitative", "title": "Timbulan (ton/hari)"},
            "color": cluster_color(),
            "tooltip": [{"field": REGION}, {"field": "Timbulan_num", "title": "Ton/hari", "format": ",.0f"}, {"field": "Klaster"}],
        }}, 350)
    with right, st.container(border=True):
        st.subheader("Komposisi klaster")
        st.caption("Kelompok dengan karakteristik numerik serupa")
        counts = visible.groupby("Klaster").size().reset_index(name="Wilayah")
        chart(counts, {"mark": {"type": "arc", "innerRadius": 72, "outerRadius": 106, "cornerRadius": 5, "padAngle": .035}, "encoding": {
            "theta": {"field": "Wilayah", "type": "quantitative"}, "color": cluster_color(),
            "tooltip": [{"field": "Klaster"}, {"field": "Wilayah"}],
        }}, 350)
    largest = visible.loc[visible["Timbulan_num"].idxmax()]
    share = largest["Timbulan_num"] / visible["Timbulan_num"].sum() * 100
    st.html(f'<div class="note"><strong>Sorotan cakupan.</strong> {html.escape(largest[REGION])} memiliki timbulan terbesar: <strong>{id_number(largest["Timbulan_num"])} ton/hari</strong>, atau {id_number(share, 1)}% dari total wilayah yang sedang ditampilkan. Klaster menunjukkan kemiripan, bukan peringkat kinerja.</div>')
    with st.container(border=True):
        st.subheader("Sekilas profil klaster")
        st.caption("Rata-rata dihitung dari wilayah yang tampil. Angka dalam kolom persentase dipertahankan sesuai source.")
        profile = visible.groupby("Klaster").agg(Wilayah=(REGION, "size"), **{"Rerata timbulan (ton/hari)": ("Timbulan_num", "mean"), "Rerata terkelola (% source)": ("% S. Terkelola", "mean"), "Rerata jumlah unit": ("Jumlah_Unit", "mean")})
        st.dataframe(profile.round(2), width="stretch")

elif page == "Eksplorasi wilayah":
    st.caption("Bandingkan nilai antarwilayah, lalu telusuri satu wilayah secara lebih rinci.")
    control_left, control_right = st.columns([2, 1])
    metric = control_left.selectbox("Indikator perbandingan", FEATURES, index=5, format_func=LABELS.get, key="rank_metric")
    limit = control_right.slider("Jumlah wilayah pada grafik", 1, len(visible), min(10, len(visible)), key="rank_limit") if len(visible) > 1 else 1
    with st.container(border=True):
        chart(visible.nlargest(limit, metric), {"mark": {"type": "bar", "cornerRadiusEnd": 4, "height": 18}, "encoding": {
            "y": {"field": REGION, "type": "nominal", "sort": "-x", "title": None, "axis": {"labelLimit": 210}},
            "x": {"field": metric.replace(".", r"\."), "type": "quantitative", "title": LABELS[metric]},
            "color": cluster_color(),
            "tooltip": [{"field": REGION}, {"field": metric.replace(".", r"\."), "format": ",.3f"}, {"field": "Klaster"}],
        }}, max(200, limit * 29))
    selected_region = st.selectbox("Detail kota / kabupaten", sorted(visible[REGION]), key="detail_region")
    region = visible.set_index(REGION).loc[selected_region]
    st.subheader(selected_region)
    st.caption(f'{region["Klaster"]} · Label kelompok pada K={n_clusters}')
    columns = st.columns(4)
    for column, (label, feature, decimals) in zip(columns, [("Timbulan · ton/hari", "Timbulan_num", 0), ("Terkelola · % source", "% S. Terkelola", 2), ("Penduduk · jiwa", "Jumlah_Penduduk", 0), ("Unit tercatat", "Jumlah_Unit", 0)]):
        column.metric(label, id_number(region[feature], decimals))
    st.write("")
    details = pd.DataFrame({"Indikator": [LABELS[f] for f in FEATURES], "Nilai wilayah": [region[f] for f in FEATURES], "Rerata cakupan": [visible[f].mean() for f in FEATURES]})
    st.dataframe(details.round(3), hide_index=True, width="stretch")
    st.caption("Timbulan_num dan jumlah_sampah_ton_per_hari adalah dua kolom berbeda. Definisi hubungan keduanya belum tersedia; dashboard tidak menyamakannya.")

elif page == "Analisis klaster":
    st.html('<div class="note">Model deskriptif memakai seluruh <strong>27 wilayah</strong> dan tujuh fitur yang distandardisasi, mengikuti notebook. Filter hanya membatasi titik dan anggota yang ditampilkan. Nomor klaster bukan urutan baik–buruk.</div>')
    current = evaluation.set_index("K").loc[n_clusters]
    columns = st.columns(3)
    columns[0].metric("Silhouette · lebih tinggi lebih baik", id_number(current["Silhouette"], 4))
    columns[1].metric("Davies–Bouldin · lebih rendah lebih baik", id_number(current["Davies–Bouldin"], 4))
    columns[2].metric("Calinski–Harabasz · lebih tinggi lebih baik", id_number(current["Calinski–Harabasz"], 2))
    st.write("")
    left, right = st.columns([1.6, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Ruang kemiripan wilayah")
        variance = explained_variance.sum() * 100
        st.caption(f"Proyeksi PCA dua dimensi · {id_number(variance, 1)}% variansi fitur standar terwakili")
        chart(visible, {"mark": {"type": "circle", "size": 150, "opacity": .85, "stroke": "white", "strokeWidth": 1.5}, "encoding": {
            "x": {"field": "PC1", "type": "quantitative", "title": f"PC1 ({id_number(explained_variance[0] * 100, 1)}%)"},
            "y": {"field": "PC2", "type": "quantitative", "title": f"PC2 ({id_number(explained_variance[1] * 100, 1)}%)"},
            "color": cluster_color(), "tooltip": [{"field": REGION}, {"field": "Klaster"}, {"field": "Timbulan_num", "title": "Timbulan (ton/hari)"}],
        }}, 350)
    with right, st.container(border=True):
        st.subheader("Perbandingan jumlah klaster")
        eval_metric = st.selectbox("Metrik evaluasi", ["Silhouette", "Davies–Bouldin", "Calinski–Harabasz", "WCSS"], key="eval_metric")
        chart(evaluation, {"layer": [
            {"mark": {"type": "line", "point": {"filled": True, "size": 70}, "color": "#16735b", "strokeWidth": 2.5}, "encoding": {"x": {"field": "K", "type": "ordinal", "title": "Jumlah klaster"}, "y": {"field": eval_metric, "type": "quantitative", "scale": {"zero": False}}, "tooltip": [{"field": "K"}, {"field": eval_metric, "format": ".4f"}]}},
            {"transform": [{"filter": f"datum.K === {n_clusters}"}], "mark": {"type": "point", "filled": True, "size": 160, "color": "#d79e43"}, "encoding": {"x": {"field": "K", "type": "ordinal"}, "y": {"field": eval_metric, "type": "quantitative"}}},
        ]}, 274)
        best_silhouette = int(evaluation.loc[evaluation["Silhouette"].idxmax(), "K"])
        st.caption(f"Silhouette tertinggi pada K={best_silhouette}. Metrik lain dapat memilih K berbeda; tidak ada klaim satu K unggul pada semua ukuran.")
    st.subheader("Anggota dan profil klaster")
    profile_metric = st.selectbox("Indikator profil", FEATURES, index=5, format_func=LABELS.get, key="profile_metric")
    for group in cluster_names:
        members = visible[visible["Klaster"] == group]
        all_members = clustered[clustered["Klaster"] == group]
        with st.expander(f"{group} · {len(members)} tampil / {len(all_members)} anggota total"):
            if members.empty:
                st.caption("Tidak ada anggota dalam cakupan filter saat ini.")
            else:
                st.caption(f"Rata-rata {LABELS[profile_metric].lower()}: {id_number(members[profile_metric].mean(), 2)}")
                st.dataframe(members[[REGION, profile_metric]].sort_values(profile_metric, ascending=False), hide_index=True, width="stretch")
    with st.expander("Dendrogram seluruh wilayah"):
        structure = dendrogram(tree, labels=clustered[REGION].tolist(), no_plot=True)
        segments = pd.DataFrame([{"Cabang": i, "Urutan": j, "Posisi": x, "Jarak": y} for i, (xs, ys) in enumerate(zip(structure["icoord"], structure["dcoord"])) for j, (x, y) in enumerate(zip(xs, ys))])
        leaf_positions = [5 + 10 * i for i in range(len(structure["ivl"]))]
        names = dict(zip(leaf_positions, structure["ivl"]))
        expression = json.dumps(names, ensure_ascii=False) + "[datum.value]"
        chart(segments, {"mark": {"type": "line", "color": "#247552", "strokeWidth": 1.5}, "encoding": {
            "x": {"field": "Posisi", "type": "quantitative", "title": None, "axis": {"values": leaf_positions, "labelExpr": expression, "labelAngle": -70, "labelLimit": 170, "grid": False}},
            "y": {"field": "Jarak", "type": "quantitative", "title": "Jarak Ward"},
            "detail": {"field": "Cabang", "type": "nominal"}, "order": {"field": "Urutan"},
        }}, 320)
    with st.expander("Tabel evaluasi K=2–8"):
        st.dataframe(evaluation.round(4), hide_index=True, width="stretch")
        st.download_button("Unduh evaluasi CSV", evaluation.to_csv(index=False).encode("utf-8-sig"), "evaluasi_ward.csv", "text/csv")

else:
    st.caption("Snapshot data, hasil olahan, dan batas interpretasi disajikan bersama agar analisis dapat direproduksi.")
    st.subheader("Data wilayah")
    export_columns = [REGION, "Klaster"] + FEATURES
    st.dataframe(visible[export_columns], hide_index=True, width="stretch", height=420)
    st.download_button("Unduh data sesuai filter", visible[export_columns].to_csv(index=False).encode("utf-8-sig"), f"sampah_jabar_k{n_clusters}.csv", "text/csv", type="primary")
    st.write("")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.subheader("Audit snapshot penuh")
        audit = pd.DataFrame({"Kolom": snapshot.columns, "Tipe": snapshot.dtypes.astype(str).values, "Nilai kosong": snapshot.isna().sum().values})
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption(f"{len(snapshot)} baris · {snapshot[REGION].nunique()} wilayah unik · {int(snapshot.duplicated().sum())} baris duplikat · {int(np.isfinite(snapshot[FEATURES].to_numpy()).sum())} nilai fitur finite")
    with right, st.container(border=True):
        st.subheader("Cara analisis dibentuk")
        st.markdown("""
1. Snapshot tujuh fitur numerik diambil dari output sel 16, lalu nama wilayah dan label referensi dicocokkan melalui indeks pada output sel 22.
2. `StandardScaler` menstandarkan fitur seluruh wilayah untuk analisis deskriptif.
3. Hierarchical clustering menggunakan Ward dan jarak Euclidean; K awal mengikuti notebook, yaitu 3.
4. Evaluasi K=2–8 dihitung ulang dengan pemotongan linkage Ward. PCA hanya membantu visualisasi; model memakai seluruh tujuh fitur.

Tidak ada target prediksi, train/test, atau CV. Ward dan PCA dengan solver `full` bersifat deterministik, sehingga tidak memakai seed acak. Nilai metrik adalah evaluasi internal pada snapshot, bukan bukti kemampuan prediksi data baru.
""")
    st.subheader("Sumber dan batas interpretasi")
    st.markdown("""
- **Data olahan dari notebook.** CSV asli dan GeoJSON tidak disertakan. Snapshot berasal dari tabel output tersimpan, dengan presisi yang ditampilkan notebook. Tidak ada angka simulasi.
- **Tahun dan URL sumber belum tersedia.** Dashboard tidak menunjukkan tren waktu atau data terkini.
- **Persentase tetap pada skala source.** Contoh nilai 0,01 tetap 0,01 pada kolom bernama persen; tidak dikalikan 100. Makna satuan terhadap sumber asli belum terverifikasi.
- **Dua ukuran volume berbeda.** `Timbulan_num` berasal dari teks `Timbulan` dengan satuan tpd. `jumlah_sampah_ton_per_hari` tidak diasumsikan sebagai volume terkelola.
- **Progress SA sudah diimputasi.** Nilai mentah dan posisi input nonnumerik sebelum imputasi tidak bisa dipulihkan dari tabel snapshot. Arti SA dan jenis unit juga belum dijelaskan source.
- **Fitur persentase berpasangan.** Terkelola dan belum terkelola saling melengkapi, sehingga dimensi ini mendapat bobot ganda. Keduanya dipertahankan untuk mereproduksi notebook.
- **Peta belum tersedia.** Batas wilayah tidak direka; file GeoJSON asli diperlukan untuk visualisasi geografis.
""")
    if n_clusters == 3:
        agreement = adjusted_rand_score(snapshot["Klaster_Notebook"], clustered["Klaster"])
        st.caption(f"Kesesuaian partisi K=3 dengan label notebook: Adjusted Rand Index {id_number(agreement, 4)}. Nomor label antarimplementasi tidak memiliki makna ordinal.")
    with st.expander("Provenance dan jejak reproduksi"):
        st.json(provenance, expanded=False)
        st.code("SHA256 snapshot: " + hashlib.sha256((ROOT / "data/sampah_jabar_snapshot.csv").read_bytes()).hexdigest())
        st.caption("Versi dependency disimpan pada requirements.txt. Hash notebook sumber tercatat pada provenance.json.")

st.html('<div class="footer">RESIK. &nbsp; / &nbsp; Pengelolaan sampah Jawa Barat &nbsp; · &nbsp; Dashboard analisis Kelompok 1 &nbsp; · &nbsp; Snapshot output notebook</div>')
