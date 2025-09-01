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
    import pandas as pd

    # Ensure date column is datetime
    if 'Date' in df.columns:
        try:
            # First try: strict day-first parsing (DD-MM-YYYY like 13-06-2025)
            df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
        except Exception:
            # Fallback: let pandas infer formats, but prefer dayfirst
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors="coerce")

        # Drop invalid dates if any (optional, or you can warn user)
        df = df.dropna(subset=['Date'])

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

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
            validation_split=kwargs.get('validation_split', 0.2),
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
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'baseline_data' not in st.session_state:
        st.session_state.baseline_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    if 'price_column' not in st.session_state:
        st.session_state.price_column = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None

def main():
    st.title("📈 Stock Price Forecasting Dashboard")
    st.markdown("*Predict future stock prices using LSTM/GRU neural networks*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Preview Data", "Explore Data", "Train Model", "Forecast Future Prices", "Model Performance"]
    )
    
    if page == "Upload & Preview Data":
        upload_and_preview_page()
    elif page == "Explore Data":
        explore_data_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Forecast Future Prices":
        forecast_page()
    elif page == "Model Performance":
        metrics_page()

def upload_and_preview_page():
    st.header("📁 Upload & Preview Data")
    
    st.markdown("Upload your historical stock data files (June, July, August) to train the forecasting model:")
    
    # Multiple file upload for training data
    st.subheader("📈 Historical Stock Data (June, July, August)")
    uploaded_files = st.file_uploader(
        "Upload CSV files for training",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload multiple CSV files containing historical stock data (June, July, August)"
    )
    
    if uploaded_files:
        combined_data = []
        file_info = []
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                combined_data.append(df)
                
                # Try to get date range
                date_range = "No Date column"
                if 'Date' in df.columns:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'])
                        date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
                    except:
                        date_range = "Invalid Date format"
                
                file_info.append({
                    'Filename': uploaded_file.name,
                    'Records': len(df),
                    'Columns': ', '.join(df.columns),
                    'Date Range': date_range
                })
                
                st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {str(e)}")
        
        if combined_data:
            # Combine all data
            try:
                full_data = pd.concat(combined_data, ignore_index=True)
                
                # Sort by date if available
                if 'Date' in full_data.columns:
                    full_data['Date'] = pd.to_datetime(full_data['Date'])
                    full_data = full_data.sort_values('Date').reset_index(drop=True)
                    full_data = full_data.drop_duplicates(subset=['Date'])
                
                st.session_state.training_data = full_data
                
                st.success(f"🎉 Combined dataset created with {len(full_data)} total records!")
                
                # Display file information
                st.subheader("📋 File Summary")
                file_df = pd.DataFrame(file_info)
                st.dataframe(file_df, use_container_width=True)
                
                # Preview combined data
                st.subheader("👀 Combined Data Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**First 5 records:**")
                    st.dataframe(full_data.head())
                
                with col2:
                    st.write("**Last 5 records:**")
                    st.dataframe(full_data.tail())
                
                # Data validation
                st.subheader("✅ Data Validation")
                validation_results = []
                
                if 'Date' in full_data.columns:
                    validation_results.append("✅ Date column found")
                    date_range = f"{full_data['Date'].min().strftime('%Y-%m-%d')} to {full_data['Date'].max().strftime('%Y-%m-%d')}"
                    validation_results.append(f"📅 Date range: {date_range}")
                else:
                    validation_results.append("❌ Date column missing")
                
                price_columns = [col for col in full_data.columns if col.lower() in ['close', 'price', 'adj close', 'adjusted close']]
                if price_columns:
                    validation_results.append(f"✅ Price columns found: {', '.join(price_columns)}")
                else:
                    validation_results.append("❌ No standard price columns found")
                
                for result in validation_results:
                    st.write(result)
                
            except Exception as e:
                st.error(f"Error combining data: {str(e)}")
    
    # Optional baseline data upload
    st.subheader("📊 Baseline Predictions (Optional)")
    st.markdown("Upload a CSV file with existing predictions to compare against your model:")
    
    baseline_file = st.file_uploader(
        "Upload baseline predictions CSV",
        type=['csv'],
        key='baseline_upload',
        help="Optional: Upload baseline predictions for comparison"
    )
    
    if baseline_file is not None:
        try:
            baseline_data = pd.read_csv(baseline_file)
            if 'Date' in baseline_data.columns:
                baseline_data['Date'] = pd.to_datetime(baseline_data['Date'])
            st.session_state.baseline_data = baseline_data
            
            st.success("✅ Baseline data uploaded successfully!")
            st.write("**Baseline Data Preview:**")
            st.dataframe(baseline_data.head())
            
        except Exception as e:
            st.error(f"Error reading baseline data: {str(e)}")
    
    # Show ready status
    if hasattr(st.session_state, 'training_data') and st.session_state.training_data is not None:
        st.success("🚀 Training data ready! You can now proceed to explore and train your model.")

def explore_data_page():
    st.header("📊 Explore Data")
    
    if st.session_state.training_data is None:
        st.warning("Please upload training data first.")
        return
    
    training_data = st.session_state.training_data.copy()
    
    # Preprocess training data
    if 'Date' in training_data.columns:
        training_data['Date'] = pd.to_datetime(training_data['Date'])
        training_data = training_data.sort_values('Date').reset_index(drop=True)
    
    # Price column selection
    price_columns = [col for col in training_data.columns if col.lower() in ['close', 'price', 'adj close', 'adjusted close']]
    if not price_columns:
        price_columns = [col for col in training_data.select_dtypes(include=[np.number]).columns]
    
    if not price_columns:
        st.error("No numeric columns found for price data.")
        return
    
    price_column = st.selectbox("Select price column:", price_columns, index=0)
    
    if price_column:
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(training_data))
        with col2:
            st.metric("Average Price", f"${training_data[price_column].mean():.2f}")
        with col3:
            st.metric("Min Price", f"${training_data[price_column].min():.2f}")
        with col4:
            st.metric("Max Price", f"${training_data[price_column].max():.2f}")
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=training_data['Date'] if 'Date' in training_data.columns else range(len(training_data)),
            y=training_data[price_column],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Add baseline predictions if available
        if st.session_state.baseline_data is not None:
            baseline_data = st.session_state.baseline_data.copy()
            if 'Date' in baseline_data.columns:
                baseline_data['Date'] = pd.to_datetime(baseline_data['Date'])
            
            baseline_cols = [col for col in baseline_data.select_dtypes(include=[np.number]).columns]
            if baseline_cols:
                baseline_col = st.selectbox("Select baseline price column:", baseline_cols)
                if baseline_col:
                    fig.add_trace(go.Scatter(
                        x=baseline_data['Date'] if 'Date' in baseline_data.columns else range(len(training_data), len(training_data) + len(baseline_data)),
                        y=baseline_data[baseline_col],
                        mode='lines',
                        name='Baseline Predictions',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
        
        # Add forecasts if available
        if st.session_state.forecasts:
            colors = ['red', 'green', 'purple', 'cyan', 'magenta']
            for i, (period, forecast_data) in enumerate(st.session_state.forecasts.items()):
                fig.add_trace(go.Scatter(
                    x=forecast_data['dates'],
                    y=forecast_data['prices'],
                    mode='lines+markers',
                    name=f'LSTM Forecast ({period} days)',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="Stock Price Analysis - Historical Data & Forecasts",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly breakdown if we have date data
        if 'Date' in training_data.columns:
            st.subheader("📅 Monthly Data Breakdown")
            training_data['Month'] = training_data['Date'].dt.strftime('%Y-%m')
            monthly_stats = training_data.groupby('Month')[price_column].agg(['count', 'mean', 'min', 'max']).round(2)
            monthly_stats.columns = ['Records', 'Avg Price', 'Min Price', 'Max Price']
            st.dataframe(monthly_stats, use_container_width=True)
        
        # Price distribution and volatility analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                training_data, 
                x=price_column, 
                nbins=30, 
                title="Price Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Daily returns
            if len(training_data) > 1:
                training_data['Daily_Return'] = training_data[price_column].pct_change() * 100
                fig_returns = px.histogram(
                    training_data.dropna(), 
                    x='Daily_Return', 
                    nbins=30, 
                    title="Daily Returns Distribution (%)"
                )
                st.plotly_chart(fig_returns, use_container_width=True)

def train_model_page():
    st.header("🤖 Train Model")
    
    if st.session_state.training_data is None:
        st.warning("Please upload training data first.")
        return
    
    training_data = st.session_state.training_data.copy()
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type:", ["LSTM", "GRU"])
        units = st.slider("Hidden Units:", 32, 128, 50, 16)
        dropout_rate = st.slider("Dropout Rate:", 0.1, 0.5, 0.2, 0.1)
        sequence_length = st.slider("Sequence Length (days):", 30, 100, 60, 10)
    
    with col2:
        epochs = st.slider("Training Epochs:", 20, 200, 50, 10)
        batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning Rate:", [0.1, 0.01, 0.001, 0.0001], index=2)
        validation_split = st.slider("Validation Split:", 0.1, 0.3, 0.2, 0.05)
    
    # Price column selection
    price_columns = [col for col in training_data.columns if col.lower() in ['close', 'price', 'adj close', 'adjusted close']]
    if not price_columns:
        price_columns = [col for col in training_data.select_dtypes(include=[np.number]).columns]
    
    if not price_columns:
        st.error("No numeric columns found for price data.")
        return
    
    price_column = st.selectbox("Select price column for training:", price_columns)
    
    # Data preparation preview
    if price_column:
        st.subheader("📋 Training Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(training_data))
        with col2:
            sequences_available = max(0, len(training_data) - sequence_length)
            st.metric("Training Sequences", sequences_available)
        with col3:
            train_size = int(sequences_available * (1 - validation_split))
            st.metric("Training/Validation", f"{train_size}/{sequences_available - train_size}")
    
    if st.button("🚀 Start Training", type="primary"):
        if price_column:
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Preprocess data
                    forecaster = st.session_state.forecaster
                    forecaster.sequence_length = sequence_length
                    
                    scaled_data, processed_df = forecaster.preprocess_data(training_data, price_column)
                    
                    # Create sequences
                    X, y = forecaster.create_sequences(scaled_data, sequence_length)
                    
                    if len(X) == 0:
                        st.error("Not enough data to create training sequences. Try reducing sequence length.")
                        return
                    
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
                        batch_size=batch_size,
                        validation_split=validation_split
                    )
                    
                    # Save model
                    model_name = f"models/stock_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    forecaster.save_model(model_name)
                    
                    st.session_state.model_trained = True
                    st.session_state.training_history = history
                    st.session_state.price_column = price_column
                    st.session_state.model_name = model_name
                    
                    st.success(f"✅ {model_type} model trained successfully!")
                    st.success(f"💾 Model saved as: {model_name}")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
                    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
                    ax1.set_title('Model Loss During Training')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss (MSE)')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
                    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
                    ax2.set_title('Model MAE During Training')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Mean Absolute Error')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Training summary
                    final_loss = history.history['loss'][-1]
                    final_val_loss = history.history['val_loss'][-1]
                    final_mae = history.history['mae'][-1]
                    final_val_mae = history.history['val_mae'][-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Training Loss", f"{final_loss:.6f}")
                    with col2:
                        st.metric("Final Validation Loss", f"{final_val_loss:.6f}")
                    with col3:
                        st.metric("Final Training MAE", f"{final_mae:.4f}")
                    with col4:
                        st.metric("Final Validation MAE", f"{final_val_mae:.4f}")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.exception(e)
        else:
            st.error("Please select a price column.")
    
    # Show training status
    if st.session_state.model_trained:
        st.success("🎯 Model is trained and ready for forecasting!")
        if st.session_state.model_name:
            st.info(f"📁 Model saved as: {st.session_state.model_name}")
    else:
        st.info("👆 Configure parameters above and click 'Start Training'")

def forecast_page():
    st.header("🔮 Forecast Future Prices")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
        return
    
    if st.session_state.training_data is None:
        st.warning("Please upload training data first.")
        return
    
    # Forecast configuration
    st.subheader("⚙️ Forecast Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Custom forecast periods
        st.write("**Select forecast periods (days):**")
        preset_periods = st.multiselect(
            "Preset periods:",
            [7, 14, 30, 60, 90],
            default=[7, 14, 30]
        )
        
        custom_period = st.number_input(
            "Custom period (days):",
            min_value=1,
            max_value=365,
            value=7,
            help="Enter a custom forecast period"
        )
        
        add_custom = st.button("Add Custom Period")
        if add_custom and custom_period not in preset_periods:
            preset_periods.append(custom_period)
            st.success(f"Added {custom_period} days to forecast periods")
        
        forecast_periods = sorted(preset_periods)
    
    with col2:
        # Get the last date from training data for forecast start
        if 'Date' in st.session_state.training_data.columns:
            last_date = st.session_state.training_data['Date'].max()
            default_start = last_date + timedelta(days=1)
        else:
            default_start = datetime.now().date()
        
        start_date = st.date_input(
            "Forecast start date:",
            value=default_start,
            help="Start date for forecasts"
        )
        
        # Show data info
        st.info(f"🗓️ Last training data point: {last_date.strftime('%Y-%m-%d') if 'Date' in st.session_state.training_data.columns else 'Unknown'}")
    
    if st.button("📈 Generate Forecasts", type="primary"):
        if forecast_periods:
            with st.spinner("Generating forecasts..."):
                try:
                    forecaster = st.session_state.forecaster
                    training_data = st.session_state.training_data.copy()
                    price_column = st.session_state.price_column
                    
                    # Preprocess data to get the last sequence
                    scaled_data, _ = forecaster.preprocess_data(training_data, price_column)
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
                    st.exception(e)
    
    # Display forecasts
    if st.session_state.forecasts:
        st.subheader("📊 Forecast Results")
        
        # Create comprehensive chart
        fig = go.Figure()
        
        # Add historical data
        training_data = st.session_state.training_data.copy()
        price_column = st.session_state.price_column
        
        if 'Date' in training_data.columns:
            training_data['Date'] = pd.to_datetime(training_data['Date'])
            historical_x = training_data['Date']
        else:
            historical_x = range(len(training_data))
        
        fig.add_trace(go.Scatter(
            x=historical_x,
            y=training_data[price_column],
            mode='lines',
            name='Historical Prices (June-August)',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecasts
        colors = ['red', 'green', 'purple', 'cyan', 'magenta']
        for i, (period, forecast_data) in enumerate(st.session_state.forecasts.items()):
            fig.add_trace(go.Scatter(
                x=forecast_data['dates'],
                y=forecast_data['prices'],
                mode='lines+markers',
                name=f'{period}-day Forecast',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Stock Price Forecasts for Upcoming Months",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary table
        st.subheader("📋 Forecast Summary")
        
        summary_data = []
        for period, forecast_data in st.session_state.forecasts.items():
            start_price = forecast_data['prices'][0]
            end_price = forecast_data['prices'][-1]
            price_change = end_price - start_price
            percent_change = ((end_price / start_price) - 1) * 100
            
            summary_data.append({
                'Period': f'{period} days',
                'Start Date': forecast_data['dates'][0].strftime('%Y-%m-%d'),
                'End Date': forecast_data['dates'][-1].strftime('%Y-%m-%d'),
                'Start Price': f"${start_price:.2f}",
                'End Price': f"${end_price:.2f}",
                'Price Change': f"${price_change:.2f}",
                'Change %': f"{percent_change:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Forecasts")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (period, forecast_data) in enumerate(st.session_state.forecasts.items()):
            forecast_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in forecast_data['dates']],
                'Predicted_Price': forecast_data['prices']
            })
            
            csv = forecast_df.to_csv(index=False)
            
            with [col1, col2, col3][i % 3]:
                st.download_button(
                    label=f"📥 Download {period}-day forecast",
                    data=csv,
                    file_name=f"forecast_{period}_days_{start_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key=f"download_{period}"
                )
        
        # Detailed forecast view
        st.subheader("🔍 Detailed Forecast View")
        
        selected_period = st.selectbox(
            "Select period for detailed view:",
            list(st.session_state.forecasts.keys())
        )
        
        if selected_period:
            forecast_data = st.session_state.forecasts[selected_period]
            detailed_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in forecast_data['dates']],
                'Day': range(1, len(forecast_data['dates']) + 1),
                'Predicted_Price': [f"${price:.2f}" for price in forecast_data['prices']],
                'Daily_Change': ['N/A'] + [f"${forecast_data['prices'][i] - forecast_data['prices'][i-1]:.2f}" for i in range(1, len(forecast_data['prices']))],
                'Cumulative_Change': [f"${price - forecast_data['prices'][0]:.2f}" for price in forecast_data['prices']]
            })
            
            st.dataframe(detailed_df, use_container_width=True)

def metrics_page():
    st.header("📊 Model Performance Metrics")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
        return
    
    # Training metrics from history
    if st.session_state.training_history:
        st.subheader("📈 Training Performance")
        
        history = st.session_state.training_history
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Training Loss", f"{history.history['loss'][-1]:.6f}")
        with col2:
            st.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.6f}")
        with col3:
            st.metric("Final Training MAE", f"{history.history['mae'][-1]:.4f}")
        with col4:
            st.metric("Final Validation MAE", f"{history.history['val_mae'][-1]:.4f}")
        
        # Training curves
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss During Training', 'MAE During Training')
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=list(range(1, len(history.history['loss'])+1)), 
                      y=history.history['loss'], 
                      name='Training Loss', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(1, len(history.history['val_loss'])+1)), 
                      y=history.history['val_loss'], 
                      name='Validation Loss', 
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # MAE plot
        fig.add_trace(
            go.Scatter(x=list(range(1, len(history.history['mae'])+1)), 
                      y=history.history['mae'], 
                      name='Training MAE', 
                      line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(range(1, len(history.history['val_mae'])+1)), 
                      y=history.history['val_mae'], 
                      name='Validation MAE', 
                      line=dict(color='red')),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison with baseline if available
    if st.session_state.baseline_data is not None and st.session_state.forecasts:
        st.subheader("🎯 Forecast vs Baseline Comparison")
        
        baseline_data = st.session_state.baseline_data.copy()
        baseline_columns = [col for col in baseline_data.select_dtypes(include=[np.number]).columns]
        
        if baseline_columns:
            baseline_column = st.selectbox("Select baseline prediction column:", baseline_columns)
            
            if baseline_column:
                # Calculate metrics for each forecast period
                metrics_data = []
                
                for period, forecast_data in st.session_state.forecasts.items():
                    if len(baseline_data) >= period:
                        # Get baseline predictions for the same period
                        baseline_prices = baseline_data[baseline_column].head(period).values
                        forecast_prices = forecast_data['prices'][:period]
                        
                        # Calculate metrics (treating baseline as reference)
                        mae = mean_absolute_error(baseline_prices, forecast_prices)
                        rmse = np.sqrt(mean_squared_error(baseline_prices, forecast_prices))
                        mape = np.mean(np.abs((baseline_prices - forecast_prices) / baseline_prices)) * 100
                        
                        metrics_data.append({
                            'Forecast Period': f'{period} days',
                            'MAE': f"{mae:.4f}",
                            'RMSE': f"{rmse:.4f}",
                            'MAPE': f"{mape:.2f}%",
                            'Avg Baseline Price': f"${np.mean(baseline_prices):.2f}",
                            'Avg Forecast Price': f"${np.mean(forecast_prices):.2f}"
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Comparison chart
                    fig = go.Figure()
                    
                    for period, forecast_data in st.session_state.forecasts.items():
                        if len(baseline_data) >= period:
                            x_range = list(range(1, period + 1))
                            
                            # Baseline
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=baseline_data[baseline_column].head(period),
                                mode='lines+markers',
                                name=f'Baseline ({period}d)',
                                line=dict(dash='dash'),
                                marker=dict(symbol='circle')
                            ))
                            
                            # Forecast
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=forecast_data['prices'][:period],
                                mode='lines+markers',
                                name=f'LSTM Forecast ({period}d)',
                                marker=dict(symbol='diamond')
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
                        if len(baseline_data) >= period:
                            baseline_prices = baseline_data[baseline_column].head(period).values
                            forecast_prices = forecast_data['prices'][:period]
                            errors = np.abs(baseline_prices - forecast_prices)
                            
                            error_data.extend([
                                {'Period': f'{period}d', 'Day': i+1, 'Absolute_Error': err, 'Relative_Error_%': (err/baseline_prices[i])*100} 
                                for i, err in enumerate(errors)
                            ])
                    
                    if error_data:
                        error_df = pd.DataFrame(error_data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_error = px.box(
                                error_df, 
                                x='Period', 
                                y='Absolute_Error', 
                                title="Absolute Forecast Error Distribution"
                            )
                            st.plotly_chart(fig_error, use_container_width=True)
                        
                        with col2:
                            fig_rel_error = px.box(
                                error_df, 
                                x='Period', 
                                y='Relative_Error_%', 
                                title="Relative Forecast Error Distribution (%)"
                            )
                            st.plotly_chart(fig_rel_error, use_container_width=True)
                
                else:
                    st.warning("No metrics available. Ensure baseline data has enough records for comparison.")
    
    # Forecast statistics
    if st.session_state.forecasts:
        st.subheader("📊 Forecast Statistics")
        
        stats_data = []
        for period, forecast_data in st.session_state.forecasts.items():
            prices = forecast_data['prices']
            stats_data.append({
                'Period': f'{period} days',
                'Min Price': f"${np.min(prices):.2f}",
                'Max Price': f"${np.max(prices):.2f}",
                'Mean Price': f"${np.mean(prices):.2f}",
                'Std Dev': f"${np.std(prices):.2f}",
                'Total Change': f"${prices[-1] - prices[0]:.2f}",
                'Volatility %': f"{(np.std(prices)/np.mean(prices))*100:.2f}%"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

# Sidebar info
def sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Guide")
    st.sidebar.markdown("""
    1. **Upload Data**: Upload June, July, August CSV files
    2. **Explore**: Visualize historical data trends
    3. **Train**: Configure and train LSTM/GRU model
    4. **Forecast**: Generate predictions for upcoming months
    5. **Performance**: Analyze model metrics and accuracy
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Expected CSV Format")
    st.sidebar.markdown("""
    **Required columns:**
    - `Date`: Date column (YYYY-MM-DD format)
    - `Close`: Closing price (or similar price column)
    
    **Optional columns:**
    - `Open`, `High`, `Low`, `Volume`, `Adj Close`
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Current Status")
    
    # Safe access to session state
    training_data = getattr(st.session_state, 'training_data', None)
    model_trained = getattr(st.session_state, 'model_trained', False)
    forecasts = getattr(st.session_state, 'forecasts', {})
    
    if training_data is not None:
        st.sidebar.success(f"✅ Training data loaded ({len(training_data)} records)")
    else:
        st.sidebar.info("⏳ Upload training data")
    
    if model_trained:
        st.sidebar.success("✅ Model trained!")
        if hasattr(st.session_state, 'model_name') and st.session_state.model_name:
            st.sidebar.info(f"📁 {os.path.basename(st.session_state.model_name)}")
    else:
        st.sidebar.info("⏳ Model not trained yet")
    
    if forecasts:
        st.sidebar.success(f"✅ {len(forecasts)} forecasts generated!")
        periods = list(forecasts.keys())
        st.sidebar.info(f"📊 Periods: {', '.join(map(str, periods))} days")
    else:
        st.sidebar.info("⏳ No forecasts generated")

def create_sample_data():
    """Create sample data for demonstration"""
    st.subheader("📝 Create Sample Data")
    st.markdown("Generate sample stock data for testing the application:")
    
    if st.button("🎲 Generate Sample Data"):
        # Create sample data for June, July, August
        date_ranges = [
            ('2024-06-01', '2024-06-30', 'June'),
            ('2024-07-01', '2024-07-31', 'July'),
            ('2024-08-01', '2024-08-31', 'August')
        ]
        
        all_data = []
        np.random.seed(42)  # For reproducible results
        
        base_price = 100
        for start_date, end_date, month in date_ranges:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate realistic stock price data
            n_days = len(dates)
            returns = np.random.normal(0.001, 0.02, n_days)  # Small daily returns with volatility
            
            prices = [base_price]
            for i in range(1, n_days):
                price = prices[-1] * (1 + returns[i])
                prices.append(max(price, 1))  # Ensure positive prices
            
            # Create OHLC data
            month_data = pd.DataFrame({
                'Date': dates,
                'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'High': [p * np.random.uniform(1.01, 1.05) for p in prices],
                'Low': [p * np.random.uniform(0.95, 0.99) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, n_days)
            })
            
            all_data.append(month_data)
            base_price = prices[-1]  # Continue from last price
        
        # Combine all months
        combined_data = pd.concat(all_data, ignore_index=True)
        st.session_state.training_data = combined_data
        
        st.success("✅ Sample data generated successfully!")
        st.write("**Generated Data Preview:**")
        st.dataframe(combined_data.head(10))
        
        # Create download buttons for individual months
        col1, col2, col3 = st.columns(3)
        
        for i, (data, (_, _, month)) in enumerate(zip(all_data, date_ranges)):
            csv = data.to_csv(index=False)
            with [col1, col2, col3][i]:
                st.download_button(
                    label=f"📥 Download {month} Data",
                    data=csv,
                    file_name=f"stock_data_{month.lower()}_2024.csv",
                    mime="text/csv",
                    key=f"sample_{month}"
                )

def load_saved_models():
    """Load previously saved models"""
    st.subheader("📂 Load Saved Model")
    
    model_files = []
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith("_model.h5"):
                model_name = file.replace("_model.h5", "")
                model_files.append(model_name)
    
    if model_files:
        selected_model = st.selectbox("Select a saved model:", model_files)
        
        if st.button("📥 Load Model"):
            try:
                model_path = f"models/{selected_model}"
                st.session_state.forecaster.load_model(model_path)
                st.session_state.model_trained = True
                st.session_state.model_name = model_path
                
                st.success(f"✅ Model loaded successfully: {selected_model}")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    else:
        st.info("No saved models found.")

if __name__ == "__main__":
    # Initialize session state first
    initialize_session_state()
    
    # Add sample data generator in sidebar
    with st.sidebar:
        sidebar_info()
        st.markdown("---")
        if st.button("🎲 Generate Sample Data"):
            # Create sample data for June, July, August
            date_ranges = [
                ('2024-06-01', '2024-06-30'),
                ('2024-07-01', '2024-07-31'),
                ('2024-08-01', '2024-08-31')
            ]
            
            all_data = []
            np.random.seed(42)
            base_price = 100
            
            for start_date, end_date in date_ranges:
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                n_days = len(dates)
                returns = np.random.normal(0.001, 0.02, n_days)
                
                prices = [base_price]
                for i in range(1, n_days):
                    price = prices[-1] * (1 + returns[i])
                    prices.append(max(price, 1))
                
                month_data = pd.DataFrame({
                    'Date': dates,
                    'Close': prices,
                    'Volume': np.random.randint(1000000, 5000000, n_days)
                })
                
                all_data.append(month_data)
                base_price = prices[-1]
            
            combined_data = pd.concat(all_data, ignore_index=True)
            st.session_state.training_data = combined_data
            st.sidebar.success("✅ Sample data generated!")
        
        # Load saved models section
        st.markdown("---")
        st.markdown("### 📂 Saved Models")
        load_saved_models()
    
    main()
