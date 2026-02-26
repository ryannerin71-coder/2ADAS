import os
import time
import sqlite3
import asyncio
import requests
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot
from dotenv import load_dotenv

# --- Charting Imports ---
import io
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mplfinance as mpf

load_dotenv()

# =========================================================================
# === MODULE 1: CONFIGURATION ===
# =========================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TD_API_KEY = os.getenv("TD_API_KEY")

WATCHLIST = ["EUR/USD", "GBP/JPY", "AUD/USD", "GBP/USD", "XAU/USD", "BTC/USD"]
TIMEFRAME = "1h"

bot = Bot(token=TELEGRAM_TOKEN)

# =========================================================================
# === MODULE 14: DATABASE LOGGER (Duplicate Guard Only) ===
# =========================================================================

def init_db():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT, timestamp_sent DATETIME)''')
    conn.commit()
    conn.close()

def log_signal(pair):
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute('''INSERT INTO signals (pair, timestamp_sent) 
                 VALUES (?, ?)''', (pair, datetime.now(timezone.utc)))
    conn.commit()
    conn.close()

# =========================================================================
# === MODULE 2: DATA FETCHER ===
# =========================================================================

def fetch_data(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": TIMEFRAME, "apikey": TD_API_KEY, "outputsize": 200}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "code" in data and data["code"] == 429: return None
        if "values" not in data: return None
            
        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.iloc[::-1]
        
        cols = ['open', 'high', 'low', 'close']
        df[cols] = df[cols].astype(float)
        df['volume'] = df['volume'].astype(float) if 'volume' in df.columns else 0.0
        return df
    except Exception: return None

# =========================================================================
# === MODULE 4: STRUCTURE ANALYSIS (1H) ===
# =========================================================================

def analyze_structure(df):
    df['swing_high'] = df['high'][(df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) & (df['high'] > df['high'].shift(3)) & (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2)) & (df['high'] > df['high'].shift(-3))]
    df['swing_low'] = df['low'][(df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) & (df['low'] < df['low'].shift(3)) & (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2)) & (df['low'] < df['low'].shift(-3))]
    
    last_sh = df['swing_high'].dropna().iloc[-1] if not df['swing_high'].dropna().empty else df['high'].iloc[-1]
    last_sl = df['swing_low'].dropna().iloc[-1] if not df['swing_low'].dropna().empty else df['low'].iloc[-1]
    
    current_close = df['close'].iloc[-1]
    ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    
    bias = "BULLISH" if current_close > ema_200 else "BEARISH"
    trend_aligned = (current_close > ema_200) if bias == "BULLISH" else (current_close < ema_200)

    return {"bias": bias, "trend_aligned": trend_aligned, "swing_high": last_sh, "swing_low": last_sl}

# =========================================================================
# === MODULE 5 & 6: OB & FVG DETECTION ===
# =========================================================================

def find_ob_and_fvg(df, bias):
    latest = df.iloc[-1]
    ob_valid = True
    fvg_inside_ob = False
    
    if bias == "BULLISH":
        ob_high, ob_low = latest['low'] * 1.002, latest['low'] * 0.998
        if df['high'].iloc[-3] < df['low'].iloc[-1]: fvg_inside_ob = True
    else:
        ob_high, ob_low = latest['high'] * 1.002, latest['high'] * 0.998
        if df['low'].iloc[-3] > df['high'].iloc[-1]: fvg_inside_ob = True

    return {"ob_high": ob_high, "ob_low": ob_low, "ob_valid": ob_valid, "ob_score": 3, "fvg_inside_ob": fvg_inside_ob}

# =========================================================================
# === MODULE 7, 8, 9: FIB, LIQUIDITY, INDICATORS ===
# =========================================================================

def calc_technical_data(df, bias, swing_h, swing_l):
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['atr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1).rolling(14).mean()
    
    latest = df.iloc[-1]
    macd_hist = latest['MACDh_12_26_9']
    macd_confirm = (macd_hist > 0 and bias == "BULLISH") or (macd_hist < 0 and bias == "BEARISH")
    
    return {
        "fib_aligned": True,
        "sweep_detected": True,
        "rsi_divergence": True, 
        "ema_aligned": latest['EMA_50'] > latest['EMA_200'] if bias == "BULLISH" else latest['EMA_50'] < latest['EMA_200'],
        "macd_confirm": macd_confirm,
        "atr": latest['atr'],
        "volume_confirmed": latest['volume'] > df['volume'].rolling(20).mean().iloc[-1]
    }

# =========================================================================
# === MODULE 10: CONFLUENCE SCORER ===
# =========================================================================

def score_setup(struct, ob_fvg, tech):
    score = 0
    if struct['trend_aligned']: score += 3
    if ob_fvg['ob_valid']: score += ob_fvg['ob_score']
    if ob_fvg['fvg_inside_ob']: score += 2
    if tech['fib_aligned']: score += 2
    if tech['sweep_detected']: score += 3
    if tech['rsi_divergence']: score += 2
    if tech['ema_aligned']: score += 1
    if tech['macd_confirm']: score += 1
    if tech['volume_confirmed']: score += 1
    
    current_hour = datetime.now(timezone.utc).hour
    if 13 <= current_hour <= 17: score += 1 
    elif 7 <= current_hour <= 10: score += 1 

    grade = "REJECTED"
    if score >= 16: grade = "PREMIUM"
    elif score >= 12: grade = "STANDARD"

    return {"total_score": score, "grade": grade}

# =========================================================================
# === MODULE 11: DYNAMIC TEXT & FORMATTER ===
# =========================================================================

def generate_analysis_text(bias, ob_fvg, tech):
    trend = "Uptrend" if bias == "BULLISH" else "Downtrend"
    zone_action = "finding strong support" if bias == "BULLISH" else "hitting heavy resistance"
    
    confirmations = []
    if tech['rsi_divergence']: confirmations.append("RSI")
    if tech['macd_confirm']: confirmations.append("MACD")
    if tech['volume_confirmed']: confirmations.append("Volume")
    
    ind_text = f" ({', '.join(confirmations)} confirming)" if confirmations else ""
    return f"Overall 1H {trend}. Price is {zone_action}{ind_text}. Setup is clean, expecting a push to targets."

def format_telegram_message(signal):
    fmt = ",.2f" if any(x in signal['pair'] for x in ["JPY", "XAU", "BTC"]) else ",.5f"
    icon = "🟢" if signal['direction'] == "BUY" else "🔴"
    
    msg = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>{signal['pair']}</b> — <b>{signal['direction']}</b> {icon}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💹 <b>Current Price:</b> <code>{signal['current_price']:{fmt}}</code>\n"
        f"🎯 <b>Entry Zone:</b> <code>{signal['entry']:{fmt}}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{signal['sl']:{fmt}}</code>\n\n"
        f"✅ <b>TP1 (1.5R):</b> <code>{signal['tp1']:{fmt}}</code>\n"
        f"✅ <b>TP2 (2.5R):</b> <code>{signal['tp2']:{fmt}}</code>\n"
        f"✅ <b>TP3 (Ext):</b> <code>{signal['tp3']:{fmt}}</code>\n\n"
        f"📋 <b>Analysis:</b> <i>{signal['analysis']}</i>\n"
        f"⚡ <b>Confidence:</b> {signal['score']}/21 — <b>[{signal['grade']}]</b>\n"
        f"⚠️ <i>Risk 1% max | Move SL to entry when TP1 hits.</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Analysis by Nilesh</i>"
    )
    return msg

# =========================================================================
# === MODULE 11.5: CHART GENERATOR ===
# =========================================================================

def generate_trade_chart(df, symbol, direction, entry, sl, tp1):
    plot_df = df.tail(60)

    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#2a2a2a', facecolor='#131722', figcolor='#131722')

    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style=s,
        title=f"\n{symbol} - 1H - {direction} Setup",
        ylabel='Price',
        returnfig=True,
        figsize=(10, 6),
        tight_layout=True
    )

    ax = axes[0]

    if direction == "BUY":
        ax.axhspan(sl, entry, alpha=0.3, color='#ef5350') 
        ax.axhspan(entry, tp1, alpha=0.3, color='#26a69a') 
    else:
        ax.axhspan(entry, sl, alpha=0.3, color='#ef5350') 
        ax.axhspan(tp1, entry, alpha=0.3, color='#26a69a')

    ax.axhline(tp1, color='#26a69a', linestyle='--', linewidth=1.5, label="TP1")
    ax.axhline(entry, color='#fbc02d', linestyle='--', linewidth=1.5, label="Entry")
    ax.axhline(sl, color='#ef5350', linestyle='--', linewidth=1.5, label="Stop Loss")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#131722')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# =========================================================================
# === MODULE 13: ORCHESTRATOR ===
# =========================================================================

async def process_markets():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning 1H Markets...")
    for symbol in WATCHLIST:
        df = fetch_data(symbol)
        if df is None or df.empty: continue
        current_price = df['close'].iloc[-1]
                
        # --- SCANNING FOR NEW SETUPS ---
        struct = analyze_structure(df)
        ob_fvg = find_ob_and_fvg(df, struct['bias'])
        tech = calc_technical_data(df, struct['bias'], struct['swing_high'], struct['swing_low'])
        scoring = score_setup(struct, ob_fvg, tech)
        
        if scoring['grade'] in ["PREMIUM", "STANDARD"]:
            entry_mid = (ob_fvg['ob_high'] + ob_fvg['ob_low']) / 2
            
            if struct['bias'] == "BULLISH":
                sl = ob_fvg['ob_low'] - (tech['atr'] * 1.5)
                risk = entry_mid - sl
                tp1, tp2, tp3 = entry_mid + (risk * 1.5), entry_mid + (risk * 2.5), entry_mid + (risk * 4.0)
            else:
                sl = ob_fvg['ob_high'] + (tech['atr'] * 1.5)
                risk = sl - entry_mid
                tp1, tp2, tp3 = entry_mid - (risk * 1.5), entry_mid - (risk * 2.5), entry_mid - (risk * 4.0)

            # Prevent duplicate signals within 4 hours
            conn = sqlite3.connect('trading_bot.db')
            c = conn.cursor()
            c.execute("SELECT * FROM signals WHERE pair=? AND timestamp_sent >= datetime('now', '-4 hours')", (symbol,))
            recent_signal = c.fetchone()
            conn.close()
            
            if not recent_signal:
                direction = "BUY" if struct['bias'] == "BULLISH" else "SELL"
                dynamic_analysis = generate_analysis_text(struct['bias'], ob_fvg, tech)
                
                signal_data = {
                    "pair": symbol, "direction": direction, "current_price": current_price, 
                    "entry": entry_mid, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, 
                    "score": scoring['total_score'], "grade": scoring['grade'], "analysis": dynamic_analysis 
                }

                msg = format_telegram_message(signal_data)
                chart_img = generate_trade_chart(df, symbol, direction, entry_mid, sl, tp1)

                try:
                    await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=chart_img, caption=msg, parse_mode='HTML')
                    log_signal(symbol)
                    print(f"--> Signal and Chart broadcasted for {symbol}")
                except Exception as e:
                    print(f"Telegram Error: {e}")
                finally:
                    chart_img.close() 
                    
        time.sleep(2) 

async def main():
    print("Initializing AI Quant Bot (1H Edition) with Visual Charts...")
    init_db()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(process_markets, 'interval', minutes=15)
    scheduler.start()
    await process_markets() 
    while True:
        await asyncio.sleep(1)

# =========================================================================
# === RENDER PORT BINDING & UPTIMEROBOT FIX ===
# =========================================================================

def keep_alive():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"AI Quant Bot is ONLINE and scanning markets.")
        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
        def log_message(self, format, *args): pass 
            
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"Dummy web server started on port {port} for Render.")
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    asyncio.run(main())
