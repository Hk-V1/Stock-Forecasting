import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("forecasts", exist_ok=True)

class StockForecaster:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.sequence_length = 60
        
    def preprocess_data(self, df, price_column='Close'):
        """Preprocess stock data for LSTM training"""
        # Ensure date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        # Scale the price data
        prices = df[price_column].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        return scaled_prices, df
    
    def create_sequences(self, data, sequence_length):
        """Create input sequences for LSTM"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, model_type='LSTM', units=50, dropout_rate=0.2, learning_rate=0.001):
        """Build LSTM or GRU model"""
        model = Sequential()
        
        if model_type == 'LSTM':
            model.add(LSTM(units=units, return_sequences=True, input_shape=(self.sequence_length, 1)))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=units))
            model.add(Dropout(dropout_rate))
        else:  # GRU
            model.add(GRU(units=units, return_sequences=True, input_shape=(self.sequence_length, 1)))
            model.add(Dropout(dropout_rate))
            model.add(GRU(units=units, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(GRU(units=units))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, X_train, y_train, model_type='LSTM', **kwargs):
        """Train the model"""
        self.model = self.build_model(model_type=model_type, **kwargs)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32),
            validation_split=0.2,
            verbose=0,
            shuffle=False
        )
        
        return history
    
    def forecast(self, last_sequence, n_days):
        """Generate forecasts for n_days ahead"""
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_days):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            pred = self.model.predict(input_seq, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform predictions
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts)
        
        return forecasts.flatten()
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        self.model.save(f"{filepath}_model.h5")
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        self.model = load_model(f"{filepath}_model.h5")
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def initialize_session_state():
    """Initialize all session state variables"""
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = StockForecaster()
    if 'aug_data' not in st.session_state:
        st.session_state.aug_data = None
    if 'sep_data' not in st.session_state:
        st.session_state.sep_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    if 'price_column' not in st.session_state:
        st.session_state.price_column = None

def main():
    st.title("📈 Stock Price Forecasting Dashboard")
    st.markdown("*Predict future stock prices using LSTM/GRU neural networks*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Preview Data", "Explore Data", "Train Model", "Forecast Future Prices", "Metrics"]
    )
    
    if page == "Upload & Preview Data":
        upload_and_preview_page()
    elif page == "Explore Data":
        explore_data_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Forecast Future Prices":
        forecast_page()
    elif page == "Metrics":
        metrics_page()

def upload_and_preview_page():
    st.header("📁 Upload & Preview Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("August 2025 Data (Training)")
        aug_file = st.file_uploader(
            "Upload aug_2025.csv",
            type=['csv'],
            key='aug_upload'
        )
        
        if aug_file is not None:
            try:
                aug_data = pd.read_csv(aug_file)
                st.session_state.aug_data = aug_data
                
                st.success("✅ August data uploaded successfully!")
                st.write("**Data Preview:**")
                st.dataframe(aug_data.head())
                
                st.write("**Data Info:**")
                st.write(f"Shape: {aug_data.shape}")
                st.write(f"Columns: {list(aug_data.columns)}")
                
                # Check for required columns
                if 'Date' not in aug_data.columns:
                    st.warning("⚠️ 'Date' column not found. Please ensure your CSV has a 'Date' column.")
                if 'Close' not in aug_data.columns:
                    st.warning("⚠️ 'Close' column not found. Please ensure your CSV has a 'Close' column.")
                
            except Exception as e:
                st.error(f"Error reading August data: {str(e)}")
    
    with col2:
        st.subheader("September 2025 Data (Baseline)")
        sep_file = st.file_uploader(
            "Upload sep_2025.csv",
            type=['csv'],
            key='sep_upload'
        )
        
        if sep_file is not None:
            try:
                sep_data = pd.read_csv(sep_file)
                st.session_state.sep_data = sep_data
                
                st.success("✅ September data uploaded successfully!")
                st.write("**Data Preview:**")
                st.dataframe(sep_data.head())
                
                st.write("**Data Info:**")
                st.write(f"Shape: {sep_data.shape}")
                st.write(f"Columns: {list(sep_data.columns)}")
                
            except Exception as e:
                st.error(f"Error reading September data: {str(e)}")
    
    # Data validation
    if st.session_state.aug_data is not None and st.session_state.sep_data is not None:
        st.success("🎉 Both datasets uploaded! You can now proceed to explore the data.")

def explore_data_page():
    st.header("📊 Explore Data")
    
    if st.session_state.aug_data is None:
        st.warning("Please upload August data first.")
        return
    
    aug_data = st.session_state.aug_data.copy()
    
    # Preprocess August data
    if 'Date' in aug_data.columns:
        aug_data['Date'] = pd.to_datetime(aug_data['Date'])
        aug_data = aug_data.sort_values('Date')
    
    # Price column selection
    price_columns = [col for col in aug_data.columns if col.lower() in ['close', 'price', 'adj close', 'adjusted close']]
    if not price_columns:
        price_columns = [col for col in aug_data.select_dtypes(include=[np.number]).columns]
    
    price_column = st.selectbox("Select price column:", price_columns, index=0 if price_columns else None)
    
    if price_column:
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(aug_data))
        with col2:
            st.metric("Average Price", f"${aug_data[price_column].mean():.2f}")
        with col3:
            st.metric("Min Price", f"${aug_data[price_column].min():.2f}")
        with col4:
            st.metric("Max Price", f"${aug_data[price_column].max():.2f}")
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=aug_data['Date'] if 'Date' in aug_data.columns else range(len(aug_data)),
            y=aug_data[price_column],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Add baseline predictions if available
        if st.session_state.sep_data is not None:
            sep_data = st.session_state.sep_data.copy()
            if 'Date' in sep_data.columns:
                sep_data['Date'] = pd.to_datetime(sep_data['Date'])
            
            baseline_col = st.selectbox("Select baseline price column:", sep_data.select_dtypes(include=[np.number]).columns)
            if baseline_col:
                fig.add_trace(go.Scatter(
                    x=sep_data['Date'] if 'Date' in sep_data.columns else range(len(aug_data), len(aug_data) + len(sep_data)),
                    y=sep_data[baseline_col],
                    mode='lines',
                    name='Baseline Predictions',
                    line=dict(color='orange', width=2, dash='dash')
                ))
        
        # Add forecasts if available
        if st.session_state.forecasts:
            for period, forecast_data in st.session_state.forecasts.items():
                fig.add_trace(go.Scatter(
                    x=forecast_data['dates'],
                    y=forecast_data['prices'],
                    mode='lines+markers',
                    name=f'LSTM Forecast ({period} days)',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Stock Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                aug_data, 
                x=price_column, 
                nbins=30, 
                title="Price Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Daily returns
            if len(aug_data) > 1:
                aug_data['Daily_Return'] = aug_data[price_column].pct_change() * 100
                fig_returns = px.histogram(
                    aug_data.dropna(), 
                    x='Daily_Return', 
                    nbins=30, 
                    title="Daily Returns Distribution (%)"
                )
                st.plotly_chart(fig_returns, use_container_width=True)

def train_model_page():
    st.header("🤖 Train Model")
    
    if st.session_state.aug_data is None:
        st.warning("Please upload August data first.")
        return
    
    aug_data = st.session_state.aug_data.copy()
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type:", ["LSTM", "GRU"])
        units = st.slider("Hidden Units:", 32, 128, 50, 16)
        dropout_rate = st.slider("Dropout Rate:", 0.1, 0.5, 0.2, 0.1)
        sequence_length = st.slider("Sequence Length:", 30, 100, 60, 10)
    
    with col2:
        epochs = st.slider("Epochs:", 20, 200, 50, 10)
        batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning Rate:", [0.1, 0.01, 0.001, 0.0001], index=2)
    
    # Price column selection
    price_columns = [col for col in aug_data.columns if col.lower() in ['close', 'price', 'adj close', 'adjusted close']]
    if not price_columns:
        price_columns = [col for col in aug_data.select_dtypes(include=[np.number]).columns]
    
    price_column = st.selectbox("Select price column for training:", price_columns)
    
    if st.button("🚀 Start Training", type="primary"):
        if price_column:
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Preprocess data
                    forecaster = st.session_state.forecaster
                    forecaster.sequence_length = sequence_length
                    
                    scaled_data, processed_df = forecaster.preprocess_data(aug_data, price_column)
                    
                    # Create sequences
                    X, y = forecaster.create_sequences(scaled_data, sequence_length)
                    
                    # Reshape for LSTM
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    
                    # Train model
                    history = forecaster.train_model(
                        X, y,
                        model_type=model_type,
                        units=units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    # Save model
                    model_name = f"models/stock_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    forecaster.save_model(model_name)
                    
                    st.session_state.model_trained = True
                    st.session_state.training_history = history
                    st.session_state.price_column = price_column
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title('Model Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    
                    ax2.plot(history.history['mae'], label='Training MAE')
                    ax2.plot(history.history['val_mae'], label='Validation MAE')
                    ax2.set_title('Model MAE')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('MAE')
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please select a price column.")
    
    # Show training status
    if st.session_state.model_trained:
        st.success("🎯 Model is trained and ready for forecasting!")
    else:
        st.info("👆 Configure parameters above and click 'Start Training'")

def forecast_page():
    st.header("🔮 Forecast Future Prices")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
        return
    
    if st.session_state.aug_data is None:
        st.warning("Please upload August data first.")
        return
    
    # Forecast configuration
    st.subheader("Forecast Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_periods = st.multiselect(
            "Select forecast periods:",
            [7, 14, 30],
            default=[7, 14, 30]
        )
    
    with col2:
        start_date = st.date_input(
            "Forecast start date:",
            value=datetime.now().date()
        )
    
    if st.button("📈 Generate Forecasts", type="primary"):
        if forecast_periods:
            with st.spinner("Generating forecasts..."):
                try:
                    forecaster = st.session_state.forecaster
                    aug_data = st.session_state.aug_data.copy()
                    price_column = st.session_state.price_column
                    
                    # Preprocess data to get the last sequence
                    scaled_data, _ = forecaster.preprocess_data(aug_data, price_column)
                    last_sequence = scaled_data[-forecaster.sequence_length:]
                    
                    # Generate forecasts for each period
                    st.session_state.forecasts = {}
                    
                    for period in forecast_periods:
                        forecasts = forecaster.forecast(last_sequence, period)
                        
                        # Create forecast dates
                        forecast_dates = [start_date + timedelta(days=i) for i in range(period)]
                        
                        st.session_state.forecasts[period] = {
                            'prices': forecasts,
                            'dates': forecast_dates
                        }
                    
                    st.success("✅ Forecasts generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating forecasts: {str(e)}")
    
    # Display forecasts
    if st.session_state.forecasts:
        st.subheader("📊 Forecast Results")
        
        # Create comprehensive chart
        fig = go.Figure()
        
        # Add historical data
        aug_data = st.session_state.aug_data.copy()
        price_column = st.session_state.price_column
        
        if 'Date' in aug_data.columns:
            aug_data['Date'] = pd.to_datetime(aug_data['Date'])
            historical_x = aug_data['Date']
        else:
            historical_x = range(len(aug_data))
        
        fig.add_trace(go.Scatter(
            x=historical_x,
            y=aug_data[price_column],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecasts
        colors = ['red', 'green', 'purple']
        for i, (period, forecast_data) in enumerate(st.session_state.forecasts.items()):
            fig.add_trace(go.Scatter(
                x=forecast_data['dates'],
                y=forecast_data['prices'],
                mode='lines+markers',
                name=f'{period}-day Forecast',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Stock Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary table
        st.subheader("📋 Forecast Summary")
        
        summary_data = []
        for period, forecast_data in st.session_state.forecasts.items():
            summary_data.append({
                'Period': f'{period} days',
                'Start Price': f"${forecast_data['prices'][0]:.2f}",
                'End Price': f"${forecast_data['prices'][-1]:.2f}",
                'Price Change': f"${forecast_data['prices'][-1] - forecast_data['prices'][0]:.2f}",
                'Change %': f"{((forecast_data['prices'][-1] / forecast_data['prices'][0]) - 1) * 100:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Forecasts")
        
        for period, forecast_data in st.session_state.forecasts.items():
            forecast_df = pd.DataFrame({
                'Date': forecast_data['dates'],
                'Predicted_Price': forecast_data['prices']
            })
            
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {period}-day forecast",
                data=csv,
                file_name=f"forecast_{period}_days.csv",
                mime="text/csv",
                key=f"download_{period}"
            )
            
            # Save locally
            forecast_df.to_csv(f"forecasts/forecast_{period}_days.csv", index=False)

def metrics_page():
    st.header("📊 Model Performance Metrics")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
        return
    
    if not st.session_state.forecasts:
        st.warning("Please generate forecasts first.")
        return
    
    if st.session_state.sep_data is None:
        st.warning("Please upload September baseline data to compare metrics.")
        return
    
    sep_data = st.session_state.sep_data.copy()
    
    # Baseline column selection
    baseline_columns = [col for col in sep_data.select_dtypes(include=[np.number]).columns]
    baseline_column = st.selectbox("Select baseline prediction column:", baseline_columns)
    
    if baseline_column:
        st.subheader("🎯 Forecast vs Baseline Comparison")
        
        # Calculate metrics for each forecast period
        metrics_data = []
        
        for period, forecast_data in st.session_state.forecasts.items():
            if len(sep_data) >= period:
                # Get baseline predictions for the same period
                baseline_prices = sep_data[baseline_column].head(period).values
                forecast_prices = forecast_data['prices'][:period]
                
                # Calculate metrics (assuming baseline as "actual" for comparison)
                metrics = calculate_metrics(baseline_prices, forecast_prices)
                
                metrics_data.append({
                    'Forecast Period': f'{period} days',
                    'MAE': f"{metrics['MAE']:.4f}",
                    'RMSE': f"{metrics['RMSE']:.4f}",
                    'MAPE': f"{metrics['MAPE']:.2f}%"
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Comparison chart
            fig = go.Figure()
            
            for period, forecast_data in st.session_state.forecasts.items():
                if len(sep_data) >= period:
                    x_range = list(range(1, period + 1))
                    
                    # Baseline
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=sep_data[baseline_column].head(period),
                        mode='lines+markers',
                        name=f'Baseline ({period}d)',
                        line=dict(dash='dash')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=forecast_data['prices'][:period],
                        mode='lines+markers',
                        name=f'LSTM Forecast ({period}d)'
                    ))
            
            fig.update_layout(
                title="Forecast vs Baseline Comparison",
                xaxis_title="Days Ahead",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error analysis
            st.subheader("📈 Error Analysis")
            
            error_data = []
            for period, forecast_data in st.session_state.forecasts.items():
                if len(sep_data) >= period:
                    baseline_prices = sep_data[baseline_column].head(period).values
                    forecast_prices = forecast_data['prices'][:period]
                    errors = np.abs(baseline_prices - forecast_prices)
                    
                    error_data.extend([
                        {'Period': f'{period}d', 'Day': i+1, 'Error': err} 
                        for i, err in enumerate(errors)
                    ])
            
            if error_data:
                error_df = pd.DataFrame(error_data)
                fig_error = px.box(
                    error_df, 
                    x='Period', 
                    y='Error', 
                    title="Forecast Error Distribution by Period"
                )
                st.plotly_chart(fig_error, use_container_width=True)
        
        else:
            st.warning("No metrics available. Ensure baseline data has enough records for comparison.")

# Sidebar info
def sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Guide")
    st.sidebar.markdown("""
    1. **Upload Data**: Upload your CSV files
    2. **Explore**: Visualize historical data
    3. **Train**: Configure and train LSTM/GRU model
    4. **Forecast**: Generate future predictions
    5. **Metrics**: Compare with baseline
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Expected CSV Format")
    st.sidebar.markdown("""
    **Required columns:**
    - `Date`: Date column (YYYY-MM-DD)
    - `Close`: Closing price (or similar price column)
    
    **Optional columns:**
    - `Open`, `High`, `Low`, `Volume`
    """)
    
    # Safe access to session state
    model_trained = getattr(st.session_state, 'model_trained', False)
    forecasts = getattr(st.session_state, 'forecasts', {})
    
    if model_trained:
        st.sidebar.success("✅ Model trained!")
    else:
        st.sidebar.info("⏳ Model not trained yet")
    
    if forecasts:
        st.sidebar.success(f"✅ {len(forecasts)} forecasts generated!")

if __name__ == "__main__":
    # Initialize session state first
    initialize_session_state()
    sidebar_info()
    main()
