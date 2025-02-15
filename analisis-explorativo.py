import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import numpy as np
import os

def generate_eda_report(csv_path, output_pdf):
    # Cargar datos
    df = pd.read_csv(csv_path)

    # Asegurar que hay datos numéricos
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No hay columnas numéricas en el dataset.")

    #1 Estadísticas Descriptivas
    descriptive_stats = numeric_df.describe().T.round(2)
    descriptive_stats["Mediana"] = numeric_df.median().round(2)  # Agregar mediana
    descriptive_stats = descriptive_stats.rename(columns={
        "count": "Conteo",
        "mean": "Media",
        "std": "Desviación Estándar",
        "min": "Mínimo",
        "25%": "Percentil 25%",
        "50%": "Percentil 50%",
        "75%": "Percentil 75%",
        "max": "Máximo"
    })

    #2️ Análisis de Correlación
    correlation_matrix = numeric_df.corr()

    #3 Identificación de valores atípicos (Boxplot)
    boxplot_img = "boxplot.png"
    if not numeric_df.nunique().eq(1).all():  # Verifica que haya variabilidad en los datos
        plt.figure(figsize=(8, 6))
        numeric_df.boxplot()
        plt.xticks(rotation=45)
        plt.title("Boxplot de Variables Numéricas")
        plt.savefig(boxplot_img, bbox_inches='tight')
        plt.close()
    else:
        boxplot_img = None  # No generar imagen si no hay variabilidad

    #4️ Crear el PDF
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    #Título del informe
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Informe de Análisis Exploratorio de Datos (EDA)", ln=True, align='C')
    pdf.ln(10)

    #1️ Estadísticas Descriptivas
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "1. Estadísticas Descriptivas", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=8)

    # Definir ancho de columnas
    col_widths = [23] + [18] * len(descriptive_stats.columns)
    headers = ["Variable"] + list(descriptive_stats.columns)

    # Encabezados de la tabla
    pdf.set_font("Arial", style='B', size=6)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align='C')
    pdf.ln()

    # Datos de la tabla
    pdf.set_font("Arial", size=9)
    for index, row in descriptive_stats.iterrows():
        pdf.cell(col_widths[0], 8, str(index), border=1, align='C')
        for i, value in enumerate(row):
            pdf.cell(col_widths[i+1], 8, str(value), border=1, align='C')
        pdf.ln()
    pdf.ln(10)

    #2️ Matriz de Correlación
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "2. Matriz de Correlación", ln=True)
    pdf.ln(5)
    correlation_img = "correlation_matrix.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.savefig(correlation_img, bbox_inches='tight')
    plt.close()
    pdf.image(correlation_img, x=15, w=170)
    pdf.ln(10)

    #3️ Identificación de Valores Atípicos
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "3. Identificación de Valores Atípicos", ln=True)
    pdf.ln(5)
    if boxplot_img:
        pdf.image(boxplot_img, x=15, w=170)
    else:
        pdf.cell(200, 10, "No se encontraron valores atípicos en las variables numéricas.", ln=True)

    # Guardar PDF
    pdf.output(output_pdf)
    print(f"Informe generado: {output_pdf}")

# Ejecutar la función con tu archivo CSV
generate_eda_report("C:/Users/USER/Downloads/sales_data_python.csv", "EDA_Report.pdf")
