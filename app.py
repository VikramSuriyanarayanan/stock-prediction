import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data import fetch_data, prepare_sequences, get_recent_sequence, SUPPORTED_TICKERS
from model import train_model, predict_future

st.markdown("# 💹 Stock & FX Prediction Dashboard")
st.caption("A clean, modern dashboard for stock and USD/INR price prediction. Powered by LSTM neural networks.")
st.markdown("---")

# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Go to...",
    ("Stock Price Predictor", "USD to INR Prediction")
)

if page == "Stock Price Predictor":
    st.subheader('🔍 Stock Price Predictor')
    col1, col2 = st.columns([2,1])
    with col1:
        ticker = st.selectbox("Select a stock ticker:", SUPPORTED_TICKERS)
    with col2:
        predict_button = st.button("Predict Future Prices")

    if predict_button:
        with st.spinner("Fetching data and training model..."):
            # Try fetching 1-minute data for 1-hour prediction
            data_1h = fetch_data(ticker, period='2d', interval='1m')
            intraday_available = not data_1h.empty
            pred_1h = None
            if intraday_available:
                X_1h, y_1h, scaler_1h = prepare_sequences(data_1h, lookback=60)
                if X_1h.shape[0] > 0:
                    model_1h = train_model(X_1h, y_1h, epochs=5)
                    recent_seq_1h = get_recent_sequence(data_1h, lookback=60, scaler=scaler_1h)
                    pred_1h = predict_future(model_1h, recent_seq_1h, scaler_1h)
                else:
                    intraday_available = False

            # 1-day prediction (try 1-hour interval, fallback to daily)
            data_1d = fetch_data(ticker, period='60d', interval='1h')
            used_daily = False
            if data_1d.empty:
                data_1d = fetch_data(ticker, period='365d', interval='1d')
                used_daily = True
            if data_1d.empty:
                st.error("No 1-hour or daily interval data available for this ticker. Try again later or choose another ticker.")
                st.stop()
            lookback = 24 if not used_daily else 5  # Use 24 for 1-hour, 5 for 1-day
            X_1d, y_1d, scaler_1d = prepare_sequences(data_1d, lookback=lookback)
            if X_1d.shape[0] == 0:
                st.error("Not enough data to make a 1-day prediction. Try again later or with a different ticker.")
                st.stop()
            model_1d = train_model(X_1d, y_1d, epochs=5)
            recent_seq_1d = get_recent_sequence(data_1d, lookback=lookback, scaler=scaler_1d)
            pred_1d = predict_future(model_1d, recent_seq_1d, scaler_1d)

        # --- Prediction Results Card ---
        with st.container():
            st.subheader(f"📊 {ticker} Prediction Results")
            colA, colB = st.columns(2)
            with colA:
                if intraday_available and pred_1h is not None:
                    st.success(f"**Next 1 Hour:** ${pred_1h:.2f}")
                else:
                    st.info("Intraday (1-hour) prediction is not available due to missing or insufficient 1-minute data. Only the 1-day prediction is shown.")
                if 'used_daily' in locals() and used_daily:
                    st.info("1-day prediction is based on daily data (1d interval) due to missing or insufficient 1-hour interval data.")
                st.success(f"**Next 1 Day:** ${pred_1d:.2f}")
            with colB:
                st.markdown(
                    "**How are these numbers generated?**\nPredictions are made using an LSTM neural network trained on recent price data for the selected ticker.")

        # --- Interactive Historical & Prediction Chart with Plotly ---
        import plotly.graph_objs as go
        hist_df = data_1d.copy()
        hist_df = hist_df[-365:] if len(hist_df) > 365 else hist_df
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], mode='lines', name='History', line=dict(color='royalblue')))

        # Recursive LSTM predictions for 1 month, 1 year, 5 years, 10 years
        import numpy as np
        from datetime import timedelta
        forecast_horizons = {
            '1 Month': 21,      # ~21 trading days
            '1 Year': 252,      # ~252 trading days
            '5 Years': 252*5,   # ~1260 trading days
            '10 Years': 252*10  # ~2520 trading days
        }
        last_seq = hist_df['Close'].values[-lookback:]
        preds = {}
        seq = last_seq.copy()
        for label, steps in forecast_horizons.items():
            future = []
            curr_seq = seq.copy()
            for _ in range(steps):
                input_seq = scaler_1d.transform(curr_seq.reshape(-1, 1)).reshape(1, lookback, 1)
                pred_scaled = model_1d.predict(input_seq, verbose=0)
                pred = scaler_1d.inverse_transform(pred_scaled)[0, 0]
                future.append(pred)
                curr_seq = np.append(curr_seq[1:], pred)
            preds[label] = future

        # Add future predictions to plot
        last_date = hist_df.index[-1]
        for label, future in preds.items():
            future_dates = [last_date + timedelta(days=i+1) for i in range(len(future))]
            fig.add_trace(go.Scatter(x=future_dates, y=future, mode='lines', name=f'Prediction ({label})',
                                     line=dict(dash='dot')))

        fig.update_layout(title=f"{ticker} Historical and Predicted Prices",
                          xaxis_title='Date', yaxis_title='Price',
                          hovermode='x unified', template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Always show past 30 days daily close (fallback to last available)
    import yfinance as yf
    st.subheader("Past 30 Days Price History (Interactive)")
    data_30d = yf.download(ticker, period='30d', interval='1d', progress=False)
    periods = ['90d', '180d', '1y', '5y', 'max']
    actual_period = '30d'
    if data_30d.empty:
        for p in periods:
            data_30d = yf.download(ticker, period=p, interval='1d', progress=False)
            if not data_30d.empty:
                actual_period = p
                break
    if not data_30d.empty:
        import plotly.express as px
        fig2 = px.line(data_30d, x=data_30d.index, y='Close', title=f'{ticker} Daily Close - Last {actual_period}', labels={'Close':'Price ($)'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('No daily data available for this ticker.')

elif page == "USD to INR Prediction":
    from dotenv import load_dotenv
    import os
    from datetime import datetime
    from alpha_vantage.foreignexchange import ForeignExchange

    st.subheader('💱 USD to INR Prediction')
    load_dotenv()
    def get_api_key():
        return st.sidebar.text_input('Alpha Vantage API Key', value=os.getenv('ALPHA_VANTAGE_API_KEY', ''), type='password')
    api_key = get_api_key()

    usd_inr = 'USDINR=X'
    import yfinance as yf
    try:
        fx_data = yf.download(usd_inr, period='365d', interval='1d', progress=False)[['Close']].dropna()
    except Exception:
        fx_data = None

    # If yfinance fails, try Alpha Vantage
    if fx_data is None or fx_data.empty:
        if api_key:
            try:
                cc = ForeignExchange(key=api_key, output_format='pandas')
                av_data, _ = cc.get_currency_exchange_daily(from_symbol='USD', to_symbol='INR', outputsize='full')
                av_data = av_data.rename(columns={
                    '4. close': 'Close',
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '5. volume': 'Volume',
                })
                av_data = av_data[['Close']].sort_index()
                av_data.index = pd.to_datetime(av_data.index)
                # Only keep last 365 days
                av_data = av_data[av_data.index >= (datetime.now() - pd.Timedelta(days=365))]
                fx_data = av_data
            except Exception as e:
                st.info(f'Alpha Vantage error: {e}')
        else:
            fx_data = None

    if fx_data is not None and not fx_data.empty:
        import plotly.graph_objs as go
        st.subheader('USD/INR - Past 1 Year (Interactive)')
        fx_hist = fx_data[-365:] if len(fx_data) > 365 else fx_data
        fig_fx = go.Figure()
        fig_fx.add_trace(go.Scatter(x=fx_hist.index, y=fx_hist['Close'], mode='lines', name='History', line=dict(color='#21ce99')))

        # Dropdown for forecast horizon
        horizon_options = {
            '1 Day': 1,
            '1 Month': 21,
            '1 Year': 252,
            '5 Years': 252*5,
            '10 Years': 252*10
        }
        selected_horizon = st.selectbox("Select prediction horizon:", list(horizon_options.keys()), index=0)
        steps = horizon_options[selected_horizon]

        # LSTM recursive prediction for selected horizon only
        from data import prepare_sequences, get_recent_sequence
        from model import train_model, predict_future
        import numpy as np
        from datetime import timedelta
        lookback_fx = 5
        X_fx, y_fx, scaler_fx = prepare_sequences(fx_data, lookback=lookback_fx)
        last_seq_fx = fx_hist['Close'].values[-lookback_fx:]
        preds_fx = None
        if X_fx.shape[0] > 0:
            model_fx = train_model(X_fx, y_fx, epochs=5)
            future = []
            curr_seq = last_seq_fx.copy()
            for _ in range(steps):
                input_seq = scaler_fx.transform(curr_seq.reshape(-1, 1)).reshape(1, lookback_fx, 1)
                pred_scaled = model_fx.predict(input_seq, verbose=0)
                pred = scaler_fx.inverse_transform(pred_scaled)[0, 0]
                future.append(pred)
                curr_seq = np.append(curr_seq[1:], pred)
            preds_fx = future
            last_date = fx_hist.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(len(future))]
            fig_fx.add_trace(go.Scatter(x=future_dates, y=future, mode='lines', name=f'Prediction ({selected_horizon})', line=dict(dash='dot', color='#FFD700')))

        fig_fx.update_layout(title="USD/INR Historical and Predicted Rates",
                            xaxis_title='Date', yaxis_title='INR',
                            hovermode='x unified', template='plotly_dark', height=500)
        st.plotly_chart(fig_fx, use_container_width=True)

        with st.container():
            st.subheader('USD/INR Prediction Results')
            if preds_fx is not None:
                st.success(f"**Next 1 Day USD/INR:** ₹{preds_fx[0]:.2f}")
                st.info(f"**Next {selected_horizon}:** ₹{preds_fx[-1]:.2f}")
                st.warning("⚠️ Long-term predictions are highly uncertain and for illustration only. Please do not use for financial decisions.")
            else:
                st.info('Not enough data to predict USD/INR.')
    else:
        st.info('No USD/INR data available from yfinance or Alpha Vantage. Please check your API key or try again later.')
