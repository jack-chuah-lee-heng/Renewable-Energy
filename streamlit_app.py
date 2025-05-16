# streamlit_app.py
import streamlit as st

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Energy Trading MVP", layout="wide", initial_sidebar_state="expanded")

import os
import decimal
import random
import time
from datetime import datetime, timedelta, timezone
import psycopg2
import psycopg2.extras  # For dictionary cursor
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd  # For st.dataframe

# --- Configuration ---
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost:5432/energy_trading_mvp')
# ^^^ IMPORTANT: Replace with your actual PostgreSQL connection string

MOCK_ASSET_ID = 'BESS_001'
MOCK_MARKET_ID = 'LMP_Market_1'
STRATEGY_INTERVAL_SECONDS = 60  # How often the strategy *would* run if continuously active
DATA_REFRESH_INTERVAL_SECONDS = 5  # How often the dashboard UI tries to refresh


# --- Database Connection (Singleton for Streamlit) ---
@st.cache_resource  # Replaces st.experimental_singleton for resource management
def get_db_connection_manager():
    class ConnectionManager:
        def __init__(self):
            self._conn = None
            self._connect()

        def _connect(self):
            try:
                self._conn = psycopg2.connect(DATABASE_URL)
                # st.toast("DB Connected", icon="🔌") # Moved toast to avoid issues before set_page_config
            except psycopg2.Error as e:
                # st.error(f"Error connecting to database: {e}") # Moved error to avoid issues
                print(f"Error connecting to database: {e}")  # Log to console instead
                self._conn = None
                raise

        def get_connection(self):
            if self._conn is None or self._conn.closed:
                self._connect()
            return self._conn

        def close_connection(self):
            if self._conn and not self._conn.closed:
                self._conn.close()
                # st.toast("DB Disconnected", icon="🔌") # Avoid Streamlit calls here

    return ConnectionManager()


def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    db_manager = get_db_connection_manager()
    conn = db_manager.get_connection()
    if conn is None:
        st.error(
            "Database connection is not available.")  # This st.error is fine as it's within a function called after set_page_config
        return None

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            if commit:
                conn.commit()
                return True
            if fetchone:
                return cur.fetchone()
            if fetchall:
                return cur.fetchall()
            return None
    except psycopg2.Error as e:
        st.error(f"Database query error: {e}")
        if conn and not commit:
            try:
                conn.rollback()
            except psycopg2.Error as rb_err:
                st.error(f"Rollback failed: {rb_err}")
        return None


# --- Global State using st.session_state ---
# Initialize session state variables if they don't exist.
# This is one of the few things allowed before other Streamlit commands after set_page_config.
if 'strategy_active' not in st.session_state:
    st.session_state.strategy_active = False
if 'last_strategy_run_time' not in st.session_state:
    st.session_state.last_strategy_run_time = datetime.min.replace(tzinfo=timezone.utc)
if 'last_data_refresh' not in st.session_state:
    st.session_state.last_data_refresh = datetime.now(timezone.utc)
if 'initial_settings_loaded' not in st.session_state:
    st.session_state.initial_settings_loaded = False
if 'buy_threshold' not in st.session_state:
    st.session_state.buy_threshold = "20.00"
if 'sell_threshold' not in st.session_state:
    st.session_state.sell_threshold = "60.00"
if 'min_soc' not in st.session_state:
    st.session_state.min_soc = "10.00"
if 'max_soc' not in st.session_state:
    st.session_state.max_soc = "90.00"


# --- Database Helper Functions (Adapted for Streamlit) ---
def get_latest_asset_telemetry(asset_id=MOCK_ASSET_ID):
    query = 'SELECT * FROM asset_telemetry WHERE asset_id = %s ORDER BY timestamp DESC LIMIT 1'
    return execute_query(query, (asset_id,), fetchone=True)


def get_latest_market_price(market_id=MOCK_MARKET_ID):
    query = 'SELECT * FROM market_prices WHERE market_id = %s ORDER BY timestamp DESC LIMIT 1'
    return execute_query(query, (market_id,), fetchone=True)


def load_initial_strategy_settings_to_session_state(asset_id=MOCK_ASSET_ID):
    """Loads settings from DB and populates st.session_state for form inputs."""
    if not st.session_state.initial_settings_loaded:
        query = 'SELECT * FROM strategy_settings WHERE asset_id = %s LIMIT 1'
        settings = execute_query(query, (asset_id,), fetchone=True)
        if settings:
            st.session_state.buy_threshold = str(settings['buy_threshold_price_per_mwh'])
            st.session_state.sell_threshold = str(settings['sell_threshold_price_per_mwh'])
            st.session_state.min_soc = str(settings['min_soc_for_discharge'])
            st.session_state.max_soc = str(settings['max_soc_for_charge'])
            st.session_state.strategy_active = settings['is_strategy_active']
        else:
            st.warning(f"No settings found for {asset_id}. Using defaults and attempting to save.")
            try:
                save_strategy_settings(  # This function now calls st.toast/st.error, which is fine here
                    asset_id,
                    decimal.Decimal(st.session_state.buy_threshold),
                    decimal.Decimal(st.session_state.sell_threshold),
                    decimal.Decimal(st.session_state.min_soc),
                    decimal.Decimal(st.session_state.max_soc)
                )
            except Exception as e:
                st.error(f"Failed to save initial default settings: {e}")

        st.session_state.initial_settings_loaded = True
    return {
        'buy_threshold_price_per_mwh': decimal.Decimal(st.session_state.buy_threshold),
        'sell_threshold_price_per_mwh': decimal.Decimal(st.session_state.sell_threshold),
        'min_soc_for_discharge': decimal.Decimal(st.session_state.min_soc),
        'max_soc_for_charge': decimal.Decimal(st.session_state.max_soc),
        'is_strategy_active': st.session_state.strategy_active
    }


def save_strategy_settings(asset_id, buy_thresh, sell_thresh, min_soc, max_soc):
    """Saves strategy settings to the database."""
    if buy_thresh >= sell_thresh:
        st.error('Buy threshold must be less than sell threshold.')
        return False
    if not (0 <= min_soc <= 100 and 0 <= max_soc <= 100 and min_soc < max_soc):
        st.error('Invalid SoC limits. Ensure 0 <= min_soc < max_soc <= 100.')
        return False

    check_query = "SELECT id FROM strategy_settings WHERE asset_id = %s"
    existing = execute_query(check_query, (asset_id,), fetchone=True)

    if existing:
        query = """
                UPDATE strategy_settings
                SET buy_threshold_price_per_mwh  = %s, \
                    sell_threshold_price_per_mwh = %s,
                    min_soc_for_discharge        = %s, \
                    max_soc_for_charge           = %s
                WHERE asset_id = %s; \
                """
        params = (buy_thresh, sell_thresh, min_soc, max_soc, asset_id)
    else:
        query = """
                INSERT INTO strategy_settings
                (asset_id, buy_threshold_price_per_mwh, sell_threshold_price_per_mwh,
                 min_soc_for_discharge, max_soc_for_charge, is_strategy_active)
                VALUES (%s, %s, %s, %s, %s, %s); \
                """
        params = (asset_id, buy_thresh, sell_thresh, min_soc, max_soc, st.session_state.strategy_active)

    success = execute_query(query, params, commit=True)
    if success:
        st.toast("Settings saved successfully!", icon="✅")
        st.session_state.buy_threshold = str(buy_thresh)
        st.session_state.sell_threshold = str(sell_thresh)
        st.session_state.min_soc = str(min_soc)
        st.session_state.max_soc = str(max_soc)
    else:
        st.error("Failed to save settings.")
    return success


def log_trade_action(action_details):
    asset_id = action_details.get('asset_id', MOCK_ASSET_ID)
    query = """
            INSERT INTO trade_log
            (asset_id, action_type, soc_at_action_start, market_price_at_action,
             energy_mwh, power_kw, duration_seconds, estimated_cost_or_revenue, notes, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) RETURNING id; \
            """
    params = (
        asset_id, action_details['action_type'],
        action_details.get('soc_at_action_start'), action_details.get('market_price_at_action'),
        action_details.get('energy_mwh'), action_details.get('power_kw'),
        action_details.get('duration_seconds'), action_details.get('estimated_cost_or_revenue'),
        action_details.get('notes')
    )
    result = execute_query(query, params, fetchone=True, commit=True)
    if result:
        st.toast(f"Trade action '{action_details['action_type']}' logged.", icon="📝")
        update_revenue_summary(asset_id)
    else:
        st.error(f"Failed to log trade action: {action_details['action_type']}")
    return result


# --- Mock Data Generation (Same as Flask, but called differently) ---
def generate_mock_asset_data_point():
    soc = decimal.Decimal(random.uniform(5, 95)).quantize(decimal.Decimal('0.01'))
    power_flow_kw = decimal.Decimal(random.uniform(-50, 50)).quantize(decimal.Decimal('0.001'))
    return {'asset_id': MOCK_ASSET_ID, 'state_of_charge': soc, 'power_flow_kw': power_flow_kw, 'status': 'Online',
            'timestamp': datetime.now(timezone.utc)}


def generate_mock_market_price_point():
    price_per_mwh = decimal.Decimal(random.uniform(5, 150)).quantize(decimal.Decimal('0.01'))
    return {'market_id': MOCK_MARKET_ID, 'price_per_mwh': price_per_mwh, 'currency': 'USD',
            'timestamp': datetime.now(timezone.utc)}


def scheduled_insert_mock_data():
    asset_data = generate_mock_asset_data_point()
    execute_query(
        'INSERT INTO asset_telemetry (asset_id, state_of_charge, power_flow_kw, status, timestamp) VALUES (%s, %s, %s, %s, %s)',
        (asset_data['asset_id'], asset_data['state_of_charge'], asset_data['power_flow_kw'], asset_data['status'],
         asset_data['timestamp']),
        commit=True
    )
    market_price = generate_mock_market_price_point()
    execute_query(
        'INSERT INTO market_prices (market_id, price_per_mwh, currency, timestamp) VALUES (%s, %s, %s, %s)',
        (market_price['market_id'], market_price['price_per_mwh'], market_price['currency'], market_price['timestamp']),
        commit=True
    )


# --- Arbitrage Strategy Engine (Adapted for Streamlit) ---
def run_arbitrage_strategy(manual_trigger=False):
    if not st.session_state.strategy_active and not manual_trigger:
        st.info("Strategy is not active. Automatic run skipped.")
        return

    current_time = datetime.now(timezone.utc)
    if not manual_trigger and (current_time - st.session_state.last_strategy_run_time) < timedelta(
            seconds=STRATEGY_INTERVAL_SECONDS):
        return

    st.sidebar.info(f"Running Arbitrage Strategy for {MOCK_ASSET_ID}...")
    settings_from_db = load_initial_strategy_settings_to_session_state(MOCK_ASSET_ID)

    latest_telemetry = get_latest_asset_telemetry(MOCK_ASSET_ID)
    latest_market_price = get_latest_market_price(MOCK_MARKET_ID)

    if not latest_telemetry or not latest_market_price:
        st.sidebar.warning("Strategy: Missing telemetry or market price data.")
        return

    current_soc = decimal.Decimal(latest_telemetry['state_of_charge'])
    current_price = decimal.Decimal(latest_market_price['price_per_mwh'])

    buy_thresh = decimal.Decimal(st.session_state.buy_threshold)
    sell_thresh = decimal.Decimal(st.session_state.sell_threshold)
    min_soc_discharge = decimal.Decimal(st.session_state.min_soc)
    max_soc_charge = decimal.Decimal(st.session_state.max_soc)

    target_charge_kw = decimal.Decimal('50.000')
    target_discharge_kw = decimal.Decimal('50.000')

    action_type = 'IDLE_PRICE_UNFAVORABLE'
    notes = f"Price ${current_price:.2f} not meeting thresholds (B:${buy_thresh:.2f}/S:${sell_thresh:.2f}). SoC:{current_soc:.2f}%."
    power_to_use_kw = decimal.Decimal('0')
    energy_mwh = decimal.Decimal('0')
    cost_or_revenue = decimal.Decimal('0')
    interval_duration_hours = decimal.Decimal(STRATEGY_INTERVAL_SECONDS) / decimal.Decimal(3600)

    if current_price > sell_thresh and current_soc > min_soc_discharge:
        action_type = 'DISCHARGE'
        power_to_use_kw = target_discharge_kw
        energy_mwh = (power_to_use_kw * interval_duration_hours) / 1000
        cost_or_revenue = energy_mwh * current_price
        notes = f"Discharging: Price ${current_price:.2f} > Sell ${sell_thresh:.2f}. SoC: {current_soc:.2f}%."
    elif current_price < buy_thresh and current_soc < max_soc_charge:
        action_type = 'CHARGE'
        power_to_use_kw = target_charge_kw
        energy_mwh = (power_to_use_kw * interval_duration_hours) / 1000
        cost_or_revenue = -(energy_mwh * current_price)
        notes = f"Charging: Price ${current_price:.2f} < Buy ${buy_thresh:.2f}. SoC: {current_soc:.2f}%."
    elif current_soc <= min_soc_discharge and current_price > sell_thresh:
        action_type = 'IDLE_SOC_LIMIT'
        notes = f"Sell Price ok, but SoC {current_soc:.2f}% <= min limit {min_soc_discharge:.2f}%."
    elif current_soc >= max_soc_charge and current_price < buy_thresh:
        action_type = 'IDLE_SOC_LIMIT'
        notes = f"Buy Price ok, but SoC {current_soc:.2f}% >= max limit {max_soc_charge:.2f}%."

    st.sidebar.metric(label=f"Strategy Action ({action_type.replace('_', ' ')})", value=f"{notes[:50]}...")
    log_trade_action({
        'action_type': action_type, 'soc_at_action_start': current_soc,
        'market_price_at_action': current_price, 'energy_mwh': energy_mwh,
        'power_kw': power_to_use_kw if action_type == 'CHARGE' else (
            -power_to_use_kw if action_type == 'DISCHARGE' else decimal.Decimal('0')),
        'duration_seconds': STRATEGY_INTERVAL_SECONDS,
        'estimated_cost_or_revenue': cost_or_revenue, 'notes': notes
    })
    st.session_state.last_strategy_run_time = current_time


# --- Revenue Calculation (Adapted for Streamlit) ---
def update_revenue_summary(asset_id=MOCK_ASSET_ID):
    twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
    query = """
            SELECT COALESCE(SUM(CASE WHEN action_type = 'CHARGE' THEN energy_mwh ELSE 0 END), 0)                   as total_charged_mwh, \
                   COALESCE(SUM(CASE WHEN action_type = 'DISCHARGE' THEN energy_mwh ELSE 0 END), \
                            0)                                                                                     as total_discharged_mwh, \
                   COALESCE(SUM(CASE WHEN action_type = 'CHARGE' THEN estimated_cost_or_revenue ELSE 0 END), \
                            0)                                                                                     as total_cost_of_charge, \
                   COALESCE(SUM(CASE WHEN action_type = 'DISCHARGE' THEN estimated_cost_or_revenue ELSE 0 END), \
                            0)                                                                                     as total_revenue_from_discharge
            FROM trade_log \
            WHERE asset_id = %s \
              AND timestamp >= %s; \
            """
    summary_data = execute_query(query, (asset_id, twenty_four_hours_ago), fetchone=True)

    if summary_data:
        net_revenue_24h = decimal.Decimal(summary_data['total_revenue_from_discharge']) + decimal.Decimal(
            summary_data['total_cost_of_charge'])
        upsert_query = """
                       INSERT INTO revenue_summary
                       (asset_id, summary_period, total_energy_charged_mwh, total_energy_discharged_mwh,
                        total_cost_of_charge, total_revenue_from_discharge, net_revenue, calculation_timestamp)
                       VALUES (%s, 'LAST_24_HOURS', %s, %s, %s, %s, %s, \
                               CURRENT_TIMESTAMP) ON CONFLICT (asset_id, summary_period) DO \
                       UPDATE SET
                           total_energy_charged_mwh = EXCLUDED.total_energy_charged_mwh, \
                           total_energy_discharged_mwh = EXCLUDED.total_energy_discharged_mwh, \
                           total_cost_of_charge = EXCLUDED.total_cost_of_charge, \
                           total_revenue_from_discharge = EXCLUDED.total_revenue_from_discharge, \
                           net_revenue = EXCLUDED.net_revenue, calculation_timestamp = CURRENT_TIMESTAMP; \
                       """
        execute_query(upsert_query, (
            asset_id, summary_data['total_charged_mwh'], summary_data['total_discharged_mwh'],
            summary_data['total_cost_of_charge'], summary_data['total_revenue_from_discharge'], net_revenue_24h
        ), commit=True)
        return net_revenue_24h
    return decimal.Decimal('0.00')


# --- APScheduler Setup for Background Tasks ---
@st.cache_resource
def get_scheduler():
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(scheduled_insert_mock_data, 'interval', seconds=15, id='mock_data_job')
    return scheduler


# Start the scheduler if it's not already running
# This block is now after st.set_page_config and session_state initialization
bg_scheduler = get_scheduler()
if not bg_scheduler.running:
    try:
        bg_scheduler.start()
        # st.toast("Background data generator started.", icon="⚙️") # This can be moved to main part of app if needed
        print("Background data generator started.")
    except Exception as e:
        # st.error(f"Failed to start background scheduler: {e}")
        print(f"Failed to start background scheduler: {e}")

# --- Streamlit UI Layout ---
# st.set_page_config is now at the top

st.title("🔋 Energy Storage Automated Trading MVP")
st.caption(f"Mock Asset ID: {MOCK_ASSET_ID} | Mock Market ID: {MOCK_MARKET_ID}")

# Display initial connection toasts here, after set_page_config
try:
    get_db_connection_manager().get_connection()  # Ensure connection is attempted
    st.toast("DB Connection Verified", icon="🔌")
except Exception:
    st.error("Failed to establish initial DB connection. Check console and DATABASE_URL.")

current_db_settings = load_initial_strategy_settings_to_session_state(MOCK_ASSET_ID)

with st.sidebar:
    st.header("Strategy Control")

    strategy_is_on = st.toggle("Activate Automated Strategy", value=st.session_state.strategy_active,
                               key="strategy_toggle_ui")

    if strategy_is_on != st.session_state.strategy_active:
        st.session_state.strategy_active = strategy_is_on
        execute_query(
            'UPDATE strategy_settings SET is_strategy_active = %s WHERE asset_id = %s',
            (st.session_state.strategy_active, MOCK_ASSET_ID), commit=True
        )
        st.toast(f"Strategy {'activated' if st.session_state.strategy_active else 'deactivated'}.", icon="💡")
        if not st.session_state.strategy_active:
            log_trade_action({'action_type': 'IDLE_STRATEGY_OFF', 'notes': 'Strategy manually turned off by user.'})
        st.rerun()

    if st.button("Run Strategy Manually Now", use_container_width=True):
        run_arbitrage_strategy(manual_trigger=True)
        st.rerun()

    st.divider()
    st.header("Strategy Settings")
    with st.form("settings_form"):
        # Form inputs now directly use session_state for their values,
        # but assign to new keys for submission to avoid direct loop on session_state itself.
        form_buy_thresh_val = st.number_input(
            "Buy Threshold ($/MWh)",
            min_value=0.0, value=float(st.session_state.buy_threshold), format="%.2f", step=0.01,
            key="form_buy_thresh_widget"
        )
        form_sell_thresh_val = st.number_input(
            "Sell Threshold ($/MWh)",
            min_value=0.0, value=float(st.session_state.sell_threshold), format="%.2f", step=0.01,
            key="form_sell_thresh_widget"
        )
        form_min_soc_val = st.number_input(
            "Min SoC for Discharge (%)",
            min_value=0.0, max_value=100.0, value=float(st.session_state.min_soc), format="%.2f", step=0.01,
            key="form_min_soc_widget"
        )
        form_max_soc_val = st.number_input(
            "Max SoC for Charge (%)",
            min_value=0.0, max_value=100.0, value=float(st.session_state.max_soc), format="%.2f", step=0.01,
            key="form_max_soc_widget"
        )
        submitted = st.form_submit_button("Save Settings", use_container_width=True)
        if submitted:
            save_success = save_strategy_settings(
                MOCK_ASSET_ID,
                decimal.Decimal(str(form_buy_thresh_val)),
                decimal.Decimal(str(form_sell_thresh_val)),
                decimal.Decimal(str(form_min_soc_val)),
                decimal.Decimal(str(form_max_soc_val))
            )
            if save_success:
                # The save_strategy_settings function already updates the main session_state keys
                st.rerun()

telemetry = get_latest_asset_telemetry()
market_price = get_latest_market_price()
revenue_summary_db = execute_query(
    "SELECT net_revenue FROM revenue_summary WHERE asset_id = %s AND summary_period = 'LAST_24_HOURS' ORDER BY calculation_timestamp DESC LIMIT 1",
    (MOCK_ASSET_ID,), fetchone=True
)
net_revenue_24h = revenue_summary_db['net_revenue'] if revenue_summary_db else decimal.Decimal('0.00')

if st.session_state.strategy_active:
    run_arbitrage_strategy()

col1, col2, col3 = st.columns(3)
with col1:
    soc_val = telemetry['state_of_charge'] if telemetry else 0
    st.metric(label="Battery SoC", value=f"{soc_val:.1f}%" if telemetry else "N/A")
    st.progress(int(soc_val) if telemetry else 0)

    asset_status = telemetry['status'] if telemetry else "Unknown"
    st.metric(label="Asset Status", value=asset_status,
              help="Operational status of the BESS asset (Online, Offline, Fault).")

with col2:
    power_flow = telemetry['power_flow_kw'] if telemetry else 0
    direction = "Idle"
    if power_flow > 0.1:
        direction = "Charging"
    elif power_flow < -0.1:
        direction = "Discharging"
    st.metric(label="Power Flow", value=f"{abs(power_flow):.2f} kW" if telemetry else "N/A", delta=f"{direction}",
              delta_color="off")

    st.metric(label="Est. Revenue (24h)", value=f"${net_revenue_24h:.2f}")

with col3:
    price_val = market_price['price_per_mwh'] if market_price else 0
    st.metric(label="Market Price", value=f"${price_val:.2f}/MWh" if market_price else "N/A")
    price_time = market_price['timestamp'].strftime('%H:%M:%S %Z') if market_price and market_price[
        'timestamp'] else "N/A"
    st.caption(f"Price as of: {price_time}")

    strategy_status_text = "ACTIVE" if st.session_state.strategy_active else "STOPPED"
    st.metric(label="Strategy Status", value=strategy_status_text,
              help="Current status of the automated trading strategy.")

st.divider()

st.subheader("Recent Trade Activity")
trade_log_data = execute_query(
    'SELECT timestamp, action_type, soc_at_action_start, market_price_at_action, energy_mwh, power_kw, estimated_cost_or_revenue, notes FROM trade_log WHERE asset_id = %s ORDER BY timestamp DESC LIMIT 10',
    (MOCK_ASSET_ID,), fetchall=True
)

if trade_log_data:
    df_log = pd.DataFrame(trade_log_data)

    for col in ['soc_at_action_start', 'market_price_at_action', 'energy_mwh', 'power_kw', 'estimated_cost_or_revenue']:
        if col in df_log.columns:
            df_log[col] = df_log[col].apply(lambda x: f"{decimal.Decimal(x):.2f}" if x is not None else "N/A")
    if 'timestamp' in df_log.columns:
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(df_log, use_container_width=True, hide_index=True)
else:
    st.info("No trade activity logged recently.")

if (datetime.now(timezone.utc) - st.session_state.last_data_refresh) > timedelta(seconds=DATA_REFRESH_INTERVAL_SECONDS):
    st.session_state.last_data_refresh = datetime.now(timezone.utc)
    st.rerun()

st.caption(f"Page last refreshed: {st.session_state.last_data_refresh.strftime('%Y-%m-%d %H:%M:%S %Z')}")

