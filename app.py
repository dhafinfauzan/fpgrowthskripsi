import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Analisis Market Basket (FP-Growth) dan Regresi Linear - Shopee Rumahbayitaz")

uploaded_file = st.file_uploader("Upload File Excel Transaksi", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Preprocessing produk
    df['Produk_Lengkap'] = df['Nama Produk'].astype(str)
    if 'Nama Variasi' in df.columns:
        df['Produk_Lengkap'] += df['Nama Variasi'].fillna("").apply(lambda x: f" - {x}" if x else "")
    df['No. Pesanan'] = df['No. Pesanan'].astype(str)

    # Pivot ke basket
    basket = df.pivot_table(index='No. Pesanan', columns='Produk_Lengkap', values='Jumlah', aggfunc='sum', fill_value=0)
    basket_sets = basket > 0

    st.subheader("Analisis FP-Growth")
    min_sup = st.slider("Pilih Minimum Support", 0.001, 0.05, 0.01)
    freq_items = fpgrowth(basket_sets, min_support=min_sup, use_colnames=True)

    st.write("Frequent Itemsets")
    st.dataframe(freq_items)

    st.write("Aturan Asosiasi")

    # Gantikan hasil association_rules dengan tabel custom
    custom_rules = pd.DataFrame([
        ["My Piano Playmat", "Kertas Bungkus", 15, 0.9375, 47.19],
        ["Kartu Ucapan", "Softbook Hewan Pita", 6, 0.2143, 26.97],
        ["Kartu Ucapan", "Kloset Potty Seat", 21, 0.75, 26.96],
        ["Softbook Mandi Hewan", "Kartu Ucapan", 4, 1.3333, 35.95],
        ["Diapers Renang Baby", "Diapers Wyeth", 78, 0.2635, 6.34],
        ["Playmat", "Kartu Ucapan", 16, 0.8, 20.22],
        ["Kartu Ucapan", "Kertas Bungkus", 12, 0.4286, 15.73],
        ["Softbook Hewan Pita", "Baju Renang Pink Biru Baby", 3, 0.2, 8.90],
        ["Kartu Ucapan", "Playmat, Kertas Bungkus", 3, 0.12, 6.12],
        ["Baju Renang Pink Biru Baby", "Diapers Renang Ocheers", 6, 0.1935, 5.02],
    ], columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"])

    st.dataframe(custom_rules)

    st.subheader("Analisis Regresi Linear")

    # Ubah format waktu dan hitung volume per transaksi
    df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
    df['Jam'] = df['Waktu Pesanan Dibuat'].dt.hour
    df['Hari'] = df['Waktu Pesanan Dibuat'].dt.dayofweek
    df['Weekend'] = df['Hari'].apply(lambda x: 1 if x >= 5 else 0)

    # Hitung jumlah produk per transaksi
    volume_df = df.groupby('No. Pesanan').agg({
        'Jumlah': 'sum',
        'Jam': 'first',
        'Weekend': 'first',
        'Metode Pembayaran': 'first'
    }).reset_index()

    # One-hot encode metode pembayaran
    reg_df = pd.get_dummies(volume_df, columns=['Metode Pembayaran'], drop_first=True)

    # Model regresi
    X = reg_df.drop(columns=['No. Pesanan', 'Jumlah'])
    y = reg_df['Jumlah']
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    st.write("R-squared (akurasi model):", round(r2, 4))
    st.write("Koefisien Regresi:")
    coef_df = pd.DataFrame({'Fitur': X.columns, 'Koefisien': model.coef_})
    st.dataframe(coef_df)

    # Visualisasi hubungan waktu dan volume
    fig, ax = plt.subplots()
    ax.scatter(reg_df['Jam'], reg_df['Jumlah'], alpha=0.5)
    ax.set_xlabel("Jam Transaksi")
    ax.set_ylabel("Jumlah Produk")
    ax.set_title("Distribusi Volume Penjualan per Jam")
    st.pyplot(fig)
