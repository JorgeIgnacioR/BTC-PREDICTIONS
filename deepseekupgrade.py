import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import requests
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class EnhancedBitcoinPredictor:
    """
    Predictor mejorado de precios de Bitcoin con:
    - Arquitectura Bidirectional LSTM
    - Preprocesamiento avanzado
    - Entrenamiento adaptativo
    - Predicción multi-paso
    """
    
    def __init__(self, seq_length=60, epochs=50, batch_size=32):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._create_advanced_model()
        self.df = None
        self.training_history = []
        self.mape_values = []

    def _create_advanced_model(self):
        """Crea modelo Bidirectional LSTM con regularización"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True),
                          input_shape=(self.seq_length, 3)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.1),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizer=optimizer, 
                     loss='mean_absolute_percentage_error',
                     metrics=['mape'])
        return model

    def _preprocess_data(self, prices):
        """Preprocesamiento avanzado con diferenciación y features temporales"""
        # Diferenciación para hacer serie estacionaria
        diff = np.diff(prices, n=1)
        
        # Escalado adaptativo
        window_size = 365
        scaled_features = []
        
        for i in range(len(diff)):
            window = diff[max(0, i-window_size):i+1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_window = scaler.fit_transform(window.reshape(-1, 1))
            scaled_features.append(scaled_window[-1][0])
        
        # Dataframe con features temporales
        dates = self.df.index[1:]
        df_processed = pd.DataFrame({
            'price_diff': diff,
            'scaled_diff': scaled_features
        }, index=dates)
        
        df_processed['day'] = df_processed.index.day
        df_processed['month'] = df_processed.index.month
        
        return df_processed

    def create_sequences(self, data):
        """Crea secuencias multivariadas"""
        X, y = [], []
        features = ['scaled_diff', 'day', 'month']
        
        for i in range(len(data) - self.seq_length):
            seq = data[features].values[i:i + self.seq_length]
            target = data['price_diff'].values[i + self.seq_length]
            X.append(seq)
            y.append(target)
            
        return np.array(X), np.array(y)

    def train_dynamically(self, data):
        """Entrenamiento adaptativo con monitoreo en tiempo real"""
        callbacks = [
            EarlyStopping(patience=7, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        plt.ion()
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Monitor de Entrenamiento en Tiempo Real", fontsize=14)
        
        total_samples = len(data)
        batch_size = max(min(self.batch_size, total_samples//10), 1)
        
        for i in range(self.seq_length + 1, len(data)):
            window = data[max(0, i-365):i]
            
            self.scaler.fit(window[['price_diff']])
            scaled_diff = self.scaler.transform(window[['price_diff']])
            
            X, y = self.create_sequences(window)
            
            if len(X) < 1:
                continue
                
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            self.training_history.append(history.history)
            current_mape = history.history['val_mape'][-1]
            self.mape_values.append(current_mape)
            
            if i % 10 == 0:
                self._update_training_plots(fig, ax, data, i)
                
        plt.ioff()
        plt.show()

    def _update_training_plots(self, fig, ax, data, current_idx):
        """Actualiza los gráficos de monitoreo"""
        for a in ax:
            a.clear()
            
        # Gráfico de precios
        ax[0].plot(data.index[:current_idx], data['price_diff'][:current_idx], 
                  label='Diferencia Real', color='navy')
        ax[0].set_title('Evolución de Precios Diferenciados')
        ax[0].grid(True, alpha=0.3)
        
        # Gráfico de error
        ax[1].plot(self.mape_values, label='MAPE (%)', color='crimson')
        ax[1].set_title('Evolución del Error (MAPE)')
        ax[1].set_ylabel('Error Porcentual')
        ax[1].grid(True, alpha=0.3)
        
        # Gráfico de aprendizaje
        ax[2].semilogy([h['loss'][0] for h in self.training_history], 
                      label='Training Loss')
        ax[2].semilogy([h['val_loss'][0] for h in self.training_history], 
                      label='Validation Loss')
        ax[2].set_title('Curvas de Aprendizaje')
        ax[2].set_ylabel('Loss (log scale)')
        ax[2].grid(True, alpha=0.3)
        
        for a in ax:
            a.legend()
            a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def predict_next_window(self, data, steps=7):
        """Predicción multi-paso adaptativa"""
        predictions = []
        current_sequence = data[-self.seq_length:]
        
        for _ in range(steps):
            scaled_seq = self.scaler.transform(current_sequence)
            prediction = self.model.predict(scaled_seq.reshape(1, -1, 3))
            
            last_price = current_sequence[-1][0]
            predicted_diff = prediction[0][0]
            predicted_price = last_price + predicted_diff
            
            predictions.append(predicted_price)
            
            new_entry = np.array([[predicted_diff, 
                                 (self.df.index[-1] + pd.Timedelta(days=1)).day,
                                 (self.df.index[-1] + pd.Timedelta(days=1)).month]])
            current_sequence = np.vstack([current_sequence[1:], new_entry])
            
        return predictions

    def full_pipeline(self, file_path):
        """Ejecuta el pipeline completo"""
        raw_data = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        self.df = raw_data
        processed_data = self._preprocess_data(raw_data['price'].values)
        X, y = self.create_sequences(processed_data)
        self.train_dynamically(processed_data)
        return self.predict_next_window(X[-1], steps=7)

if __name__ == "__main__":
    # Configuración y ejecución principal
    predictor = EnhancedBitcoinPredictor(seq_length=60, epochs=100)
    predictions = predictor.full_pipeline("data/prices.csv")
    print(f"\nPredicciones para los próximos 7 días:")
    for i, pred in enumerate(predictions, 1):
        print(f"Día {i}: {pred:.2f} USD")