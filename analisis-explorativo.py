import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Cargar datos
df = pd.read_csv(r'C:\Users\USER\Downloads\sales_data_python.csv')

# Convertir la columna Date a tipo datetime
df['Date'] = pd.to_datetime(df['Date'])

# Exploración inicial
num_filas, num_columnas = df.shape
tipos_datos = df.dtypes
valores_nulos = df.isnull().sum()

descripcion = df.describe()

# Gráfico de distribución de ventas
plt.figure(figsize=(8, 5))
sns.histplot(df['Units_Sold'], bins=30, kde=True)
plt.title('Distribución de Unidades Vendidas')
plt.xlabel('Unidades Vendidas')
plt.ylabel('Frecuencia')
plt.savefig('units_sold_distribution.png')
plt.close()

# Matriz de correlación (eliminando columnas no numéricas)
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(6, 4))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.savefig('correlation_matrix.png')
plt.close()

# Crear el informe PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Informe de Análisis Exploratorio de Datos (EDA)', ln=True, align='C')
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 10)
pdf.cell(0, 10, f'Tamaño del dataset: {num_filas} filas, {num_columnas} columnas', ln=True)
pdf.cell(0, 10, 'Tipos de Datos:', ln=True)
pdf.multi_cell(0, 5, str(tipos_datos))
pdf.cell(0, 10, 'Valores Nulos:', ln=True)
pdf.multi_cell(0, 5, str(valores_nulos))
pdf.cell(0, 10, 'Estadísticas Descriptivas:', ln=True)
pdf.multi_cell(0, 5, str(descripcion))

pdf.image('units_sold_distribution.png', x=10, y=pdf.get_y() + 10, w=100)
pdf.ln(60)
pdf.image('correlation_matrix.png', x=10, y=pdf.get_y() + 10, w=100)

pdf.output('eda_report.pdf')
