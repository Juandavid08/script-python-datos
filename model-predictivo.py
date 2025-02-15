import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fpdf import FPDF

# Cargar datos preprocesados
df = pd.read_csv("sales_data_cleaned.csv")

# Usamos "Units_Sold" como variable objetivo (ventas)
X = df.drop(columns=["Units_Sold"])  # Variables independientes
y = df["Units_Sold"]  # Variable dependiente

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#1️ Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

#2️ Regresión Polinómica
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

#3️ Random Forest con búsqueda de hiperparámetros
rf_model = RandomForestRegressor(random_state=42)
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
rf_grid = GridSearchCV(rf_model, param_grid, cv=3, scoring="neg_mean_squared_error")
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Función para evaluar modelos
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae_linear, rmse_linear, r2_linear = evaluate_model(y_test, y_pred_linear)
mae_poly, rmse_poly, r2_poly = evaluate_model(y_test, y_pred_poly)
mae_rf, rmse_rf, r2_rf = evaluate_model(y_test, y_pred_rf)

# Comparación de modelos
results = f"""
Modelo               | MAE     | RMSE    | R2
-------------------------------------------------
Regresión Lineal     | {mae_linear:.2f} | {rmse_linear:.2f} | {r2_linear:.2f}
Regresión Polinómica | {mae_poly:.2f} | {rmse_poly:.2f} | {r2_poly:.2f}
Random Forest        | {mae_rf:.2f} | {rmse_rf:.2f} | {r2_rf:.2f}
"""

# Justificación del modelo final
best_model = "Random Forest" if r2_rf > r2_linear and r2_rf > r2_poly else "Regresión Lineal"
justification = f"\nEl modelo seleccionado es: {best_model} debido a su mejor desempeño en R² y menor error."

# Función para generar el informe PDF
def generate_pdf_report(report_text, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, "Informe de Modelo Predictivo", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    # Forzamos la codificación para evitar errores de caracteres no soportados
    report_text = report_text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 8, report_text)
    pdf.output(output_pdf)
    print(f"Informe generado correctamente: {output_pdf}")

final_report = "Construcción del modelo predictivo:\n" + results + "\n" + justification
generate_pdf_report(final_report, "Predictive_Model_Report.pdf")
