import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from fpdf import FPDF

# Cargar datos
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Manejo de valores nulos y duplicados
def clean_data(df):
    report = "" 
    
    # 1️ Identificar valores nulos
    missing_values = df.isnull().sum()
    report += "\n1️ Manejo de Valores Nulos:\n"
    report += str(missing_values[missing_values > 0]) + "\n"
    
    # Si hay valores nulos, se llenan con la mediana (para datos numéricos)
    df.fillna(df.median(numeric_only=True), inplace=True)
    report += "- Valores nulos en variables numéricas llenados con la mediana.\n"
    
    # 2️ Eliminar duplicados
    duplicates = df.duplicated().sum()
    report += "\n2️ Manejo de Duplicados:\n"
    report += f"- Se encontraron {duplicates} registros duplicados.\n"
    df.drop_duplicates(inplace=True)
    
    return df, report

# Normalización o estandarización
def scale_numeric_features(df, method='standard'):
    report = "\n3️ Estandarización:\n"
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
        report += "- Se aplicó estandarización (media 0, desviación 1).\n"
    else:
        scaler = MinMaxScaler()
        report += "- Se aplicó normalización (0 a 1).\n"
    
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df, report

# Codificación de variables categóricas
def encode_categorical_features(df):
    report = "\n4️ Codificación de Variables Categóricas:\n"
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_features:
        unique_values = df[col].nunique()
        if unique_values > 10:  # Si hay muchas categorías, usar Label Encoding
            df[col] = LabelEncoder().fit_transform(df[col])
            report += f"- '{col}' codificado con Label Encoding.\n"
        else:  # Si hay pocas categorías, usar One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            report += f"- '{col}' codificado con One-Hot Encoding.\n"
    
    return df, report

# Generar informe PDF
def generate_pdf_report(report_text, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Título del informe
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, "Informe de Preprocesamiento de Datos", ln=True, align='C')
    pdf.ln(10)

    # Eliminar emojis y caracteres especiales no admitidos
    report_text = report_text.encode('latin-1', 'ignore').decode('latin-1')

    # Configurar fuente y agregar contenido
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, report_text)

    # Guardar el PDF
    pdf.output(output_pdf)
    print(f"Informe generado correctamente: {output_pdf}")



# Ejecutar el proceso
csv_path = "C:/Users/USER/Downloads/sales_data_python.csv"
df = load_data(csv_path)
df, report1 = clean_data(df)
df, report2 = scale_numeric_features(df, method='standard')
df, report3 = encode_categorical_features(df)

# Guardar dataset preprocesado
df.to_csv("sales_data_cleaned.csv", index=False)

# Generar informe
final_report = report1 + report2 + report3
generate_pdf_report(final_report, "Preprocessing_Report.pdf")
