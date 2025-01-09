import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
import os

# Đọc toàn bộ dữ liệu mà không bị giới hạn
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load file CSV
file_path = 'D:\\AI\\real_estate_listings.csv'

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_csv(file_path, delimiter='\t')
        if len(data.columns) == 1:  # Not tab-separated
            data = pd.read_csv(file_path, delimiter=',')
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

data = load_data(file_path)
if data is None:
    raise Exception("Failed to load data. Please check the file path and format.")

# Hàm xử lý dữ liệu
def extract_land_area(area):
    if pd.isna(area):
        return np.nan
    try:
        area = area.split(' ')[0].replace('.', '').replace(',', '.')
        return float(area)
    except ValueError:
        return np.nan

def extract_price(price):
    if pd.isna(price):
        return np.nan
    price = price.lower().replace(' ', '')
    try:
        if 'tỷ' in price and 'triệu' in price:
            parts = price.split('tỷ')
            ty_part = float(parts[0]) * 1e9
            trieu_part = float(parts[1].replace('triệu', '').replace(',', '.')) * 1e6
            return ty_part + trieu_part
        elif 'tỷ' in price:
            return float(price.replace('tỷ', '').replace(',', '.')) * 1e9
        elif 'triệu' in price:
            return float(price.replace('triệu', '').replace(',', '.')) * 1e6
        return float(price.replace(',', '.'))
    except ValueError:
        return np.nan

# Hàm trích xuất quận từ vị trí  
def extract_district(location):
    if pd.isna(location):
        return 'Unknown'
    
    location = location.lower()  
    location = location.replace("phường", "").replace("xã", "").replace("huyện", "").replace("tp.", "").strip()
    
    location_parts = location.split(',')
    if len(location_parts) > 1:
        location = location_parts[-1].strip()
    
    if "quận" not in location and "huyện" not in location:
        location = "huyện " + location

    for district in location_advantage.keys():
        if district.lower() == location:
            return district
    
    return 'Unknown'

# Ánh xạ Location Advantage - chuyển tên quận về chữ thường
location_advantage = {
    'quận 1': 1.0, 'quận 3': 1.0, 'quận 5': 0.9, 'quận 10': 0.85,
    'quận 7': 0.8, 'quận 4': 0.8, 'quận 2': 0.85, 'quận 6': 0.7,
    'quận 8': 0.6, 'quận 9': 0.5, 'quận 12': 0.7, 'quận bình thạnh': 0.8,
    'quận gò vấp': 0.7, 'quận tân bình': 0.75, 'quận phú nhuận': 0.85,
    'huyện bình chánh': 0.5, 'huyện củ chi': 0.4, 'huyện nhà bè': 0.6,
    'quận thủ đức': 0.7, 'huyện cần giờ': 0.4, 'huyện hóc môn': 0.5,
    'quận bình tân': 0.6
}

# Ánh xạ Type of House Advantage
type_of_house_advantage = {
    'nhà phố': 1.0, 'biệt thự': 1.2, 'căn hộ': 0.8, 'nhà cấp 4': 0.6,
    'nhà trọ': 0.5, 'nhà vườn': 0.9, 'nhà xưởng': 0.7, 'nhà hẻm, ngõ': 0.7,
    'nhà mặt tiền': 1.1, 'biệt thự, villa': 1.2, 'đất thổ cư': 0.5
}

# Ánh xạ Legal Documents Advantage
legal_documents_advantage = {
    'sổ hồng': 1.0, 'sổ đỏ': 1.1, 'giấy tờ hợp lệ': 0.8
}

# Chuẩn hóa các giá trị trong cột Type of House
def normalize_type_of_house(type_of_house):
    if pd.isna(type_of_house):
        return 'Unknown'
    
    type_of_house = type_of_house.lower().strip()
    if 'hẻm' in type_of_house or 'ngõ' in type_of_house:
        return 'nhà hẻm, ngõ'
    elif 'mặt tiền' in type_of_house:
        return 'nhà mặt tiền'
    elif 'biệt thự' in type_of_house or 'villa' in type_of_house:
        return 'biệt thự, villa'
    elif 'đất thổ cư' in type_of_house:
        return 'đất thổ cư'
    return type_of_house

# Chuẩn hóa các giá trị trong cột Legal Documents
def normalize_legal_documents(legal_documents):
    if pd.isna(legal_documents):
        return 'Unknown'
    
    legal_documents = legal_documents.lower().strip()
    if 'sổ hồng' in legal_documents:
        return 'sổ hồng'
    elif 'sổ đỏ' in legal_documents:
        return 'sổ đỏ'
    elif 'giấy tờ hợp lệ' in legal_documents:
        return 'giấy tờ hợp lệ'
    return legal_documents

# Tiền xử lý dữ liệu
if 'Land Area' in data.columns:
    data['Land Area'] = data['Land Area'].apply(extract_land_area)
else:
    data['Land Area'] = np.nan

data['Price'] = data['Price'].apply(extract_price) if 'Price' in data.columns else np.nan
data['Bedrooms'] = data['Bedrooms'].str.extract(r'(\d+)').astype(float) if 'Bedrooms' in data.columns else np.nan
data['Toilets'] = data['Toilets'].str.extract(r'(\d+)').astype(float) if 'Toilets' in data.columns else np.nan
data['Location'] = data['Location'].fillna('Unknown') if 'Location' in data.columns else 'Unknown'
data['District'] = data['Location'].apply(extract_district)
print(f"Sample District values:\n{data['District'].unique()}")
data['Location_Advantage'] = data['District'].apply(lambda x: location_advantage.get(x, 0.5))
print(f"Sample Location_Advantage values:\n{data['Location_Advantage'].unique()}")

# Chuẩn hóa và gán giá trị lợi thế cho Type of House
data['Type of House'] = data['Type of House'].apply(normalize_type_of_house)
data['Type_of_House_Advantage'] = data['Type of House'].apply(lambda x: type_of_house_advantage.get(x, 0.5))
print(f"Sample Type_of_House_Advantage values:\n{data['Type_of_House_Advantage'].unique()}")

# Chuẩn hóa và gán giá trị lợi thế cho Legal Documents
data['Legal Documents'] = data['Legal Documents'].apply(normalize_legal_documents)
data['Legal_Documents_Advantage'] = data['Legal Documents'].apply(lambda x: legal_documents_advantage.get(x, 0.5))
print(f"Sample Legal_Documents_Advantage values:\n{data['Legal_Documents_Advantage'].unique()}")

# Định nghĩa đặc trưng
categorical_features = ['Location', 'Type of House', 'Legal Documents']
numeric_features = ['Land Area', 'Bedrooms', 'Toilets', 'Total Floors', 'Location_Advantage', 'Type_of_House_Advantage', 'Legal_Documents_Advantage']

# Tiền xử lý
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Xây dựng tập dữ liệu
X = data.drop('Price', axis=1)
y = data['Price']

X = preprocessor.fit_transform(X)
y = SimpleImputer(strategy='mean').fit_transform(y.values.reshape(-1, 1)).ravel()

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Mô hình RandomForestRegressor
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', RandomForestRegressor(n_estimators=100, random_state=1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Tính toán độ tin cậy của mô hình
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Percentage Error: {mape * 100:.2f}%')

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores).mean()
print(f'Cross-Validated RMSE: {cv_rmse}')

# Trực quan hóa kết quả
plt.scatter(y_test, y_pred)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Biểu đồ giá thực tế và giá dự đoán")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

# Biểu đồ phân phối lỗi
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel("Sai số (Giá thực tế - Giá dự đoán)")
plt.ylabel("Tần suất")
plt.title("Biểu đồ phân phối lỗi")
plt.show()

# Vẽ sơ đồ heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame({'Giá thực tế': y_test, 'Giá dự đoán': y_pred}).corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap của Giá thực tế và Giá dự đoán")
plt.show()

# Vẽ sơ đồ scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Scatter Plot của Giá thực tế và Giá dự đoán")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

# Tính toán tỷ lệ phần trăm dự đoán chính xác trong khoảng ±10%
accuracy_within_10_percent = np.mean(np.abs(errors) <= 0.1 * y_test) * 100
print(f'Tỷ lệ phần trăm dự đoán chính xác trong khoảng ±10%: {accuracy_within_10_percent:.2f}%')

# Kiểm tra overfitting
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train MSE: {train_mse}, Test MSE: {test_mse}')
print(f'Train R^2: {train_r2}, Test R^2: {test_r2}')

if train_mse < test_mse and train_r2 > test_r2:
    print("Mô hình có dấu hiệu bị overfitting.")
else:
    print("Mô hình không bị overfitting.")

# Giao diện Tkinter
class HousePricePredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dự đoán giá nhà đất")
        self.create_widgets()
        self.y_test = y_test
        self.y_pred = y_pred

    def create_widgets(self):
        fields = [
            ("Vị trí", "Location"),
            ("Loại nhà", "Type of House"),
            ("Diện tích (m²)", "Land Area"),
            ("Số phòng ngủ", "Bedrooms"),
            ("Số toilet", "Toilets"),
            ("Số tầng", "Total Floors"),
            ("Giấy tờ pháp lý", "Legal Documents")
        ]

        self.entries = {}
        for i, (label_text, field_name) in enumerate(fields):
            tk.Label(self, text=label_text).grid(row=i, column=0, padx=10, pady=5)
            entry = tk.Entry(self)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries[field_name] = entry

        self.predict_button = tk.Button(self, text="Dự đoán", command=self.predict)
        self.predict_button.grid(row=len(fields), column=0, columnspan=2, pady=10)

        self.plot_button = tk.Button(self, text="Vẽ biểu đồ", command=self.plot)
        self.plot_button.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)

        # Thêm nhãn để hiển thị tỷ lệ phần trăm dự đoán chính xác
        self.accuracy_label = tk.Label(self, text="")
        self.accuracy_label.grid(row=len(fields) + 2, column=0, columnspan=2, pady=10)

    def predict(self):
        try:
            input_data = {
                field: self.entries[field].get()
                for field in self.entries
            }
            location = input_data.get('Location', 'Unknown')  
            district = extract_district(location)   
            input_data['District'] = district 
            input_data['Location_Advantage'] = location_advantage.get(district, 0.5)  
            input_data['Type_of_House_Advantage'] = type_of_house_advantage.get(normalize_type_of_house(input_data['Type of House']), 0.5)
            input_data['Legal_Documents_Advantage'] = legal_documents_advantage.get(normalize_legal_documents(input_data['Legal Documents']), 0.5)
            input_data = {k: float(v) if k in numeric_features else v for k, v in input_data.items()}
            price = predict_house_price(input_data)
            messagebox.showinfo("Dự đoán giá nhà đất", f"Giá dự đoán: {price:,.2f} VND")

        except ValueError as e:
            messagebox.showerror("Lỗi", "Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra lại.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")

    def plot(self):
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel("Giá thực tế")
        plt.ylabel("Giá dự đoán")
        plt.title("Biểu đồ giá thực tế và giá dự đoán")
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red')
        plt.legend(["Đường chuẩn", "Dữ liệu dự đoán"])
        plt.show()

        # Biểu đồ phân phối lỗi
        errors = self.y_test - self.y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel("Sai số (Giá thực tế - Giá dự đoán)")
        plt.ylabel("Tần suất")
        plt.title("Biểu đồ phân phối lỗi")
        plt.show()

        # Vẽ sơ đồ heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(pd.DataFrame({'Giá thực tế': self.y_test, 'Giá dự đoán': self.y_pred}).corr(), annot=True, cmap='coolwarm')
        plt.title("Heatmap của Giá thực tế và Giá dự đoán")
        plt.show()

        # Vẽ sơ đồ scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=self.y_pred)
        plt.xlabel("Giá thực tế")
        plt.ylabel("Giá dự đoán")
        plt.title("Scatter Plot của Giá thực tế và Giá dự đoán")
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red')
        plt.show()

        # Tính toán tỷ lệ phần trăm dự đoán chính xác trong khoảng ±10%
        accuracy_within_10_percent = np.mean(np.abs(errors) <= 0.1 * self.y_test) * 100
        self.accuracy_label.config(text=f"Tỷ lệ phần trăm dự đoán chính xác trong khoảng ±10%: {accuracy_within_10_percent:.2f}%")

def predict_house_price(input_data):
    df_input = pd.DataFrame([input_data])
    df_input_transformed = preprocessor.transform(df_input)
    return pipeline.predict(df_input_transformed)[0]

if __name__ == "__main__":
    app = HousePricePredictorApp()
    app.mainloop()