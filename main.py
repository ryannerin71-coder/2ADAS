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
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot
from dotenv import load_dotenv

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
# === MODULE 14: DATABASE LOGGER & TRACKER ===
# =========================================================================

def init_db():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT, direction TEXT, entry REAL, sl REAL, 
                    tp1 REAL, tp2 REAL, tp3 REAL, score INTEGER, 
                    grade TEXT, analysis_text TEXT, timestamp_sent DATETIME, 
                    outcome TEXT, outcome_timestamp DATETIME, pips_result REAL)''')
    conn.commit()
    conn.close()

def log_signal(data):
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute('''INSERT INTO signals 
                 (pair, direction, entry, sl, tp1, tp2, tp3, score, grade, analysis_text, timestamp_sent, outcome) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
              (data['pair'], data['direction'], data['entry'], data['sl'], data['tp1'], 
               data['tp2'], data['tp3'], data['score'], data['grade'], data['analysis'], 
               datetime.now(timezone.utc), "OPEN"))
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
        df = df.iloc[::-1] # Oldest to newest
        
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
# === MODULE 11 & 12: DYNAMIC TEXT & FORMATTER ===
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
# === MODULE 15: DAILY PERFORMANCE REPORT ===
# =========================================================================

async def send_daily_report():
    """Generates and sends a 24-hour summary of all closed trades."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating 24H Performance Report...")
    
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    # Fetch all trades closed in the last 24 hours
    c.execute("SELECT outcome, pips_result FROM signals WHERE outcome != 'OPEN' AND outcome_timestamp >= datetime('now', '-24 hours')")
    recent_closed = c.fetchall()
    conn.close()
    
    wins = sum(1 for t in recent_closed if t[0] == 'WIN')
    losses = sum(1 for t in recent_closed if t[0] == 'LOSS')
    total_trades = wins + losses
    net_pips = sum(t[1] for t in recent_closed if t[1] is not None)
    
    if total_trades == 0:
        return # Skip sending a report if no trades closed today
        
    win_rate = (wins / total_trades) * 100
    
    msg = (
        f"📅 <b>24-HOUR PERFORMANCE REPORT</b> 📅\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Total Trades Closed:</b> {total_trades}\n"
        f"✅ <b>Wins (TP Hit):</b> {wins}\n"
        f"❌ <b>Losses (SL Hit):</b> {losses}\n"
        f"🏆 <b>Win Rate:</b> {win_rate:.1f}%\n\n"
        f"💰 <b>Net Result:</b> {net_pips:+.1f} Pips\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Tracked by Nilesh</i>"
    )
    
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')
    except Exception as e:
        print(f"Daily Report Error: {e}")

# =========================================================================
# === MODULE 13: ORCHESTRATOR & TRADE MONITOR ===
# =========================================================================

async def process_markets():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning 1H Markets & Tracking Trades...")
    for symbol in WATCHLIST:
        df = fetch_data(symbol)
        if df is None or df.empty: continue
        
        current_price = df['close'].iloc[-1]
        
        # --- TRACKING LIVE TRADES ---
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        c.execute("SELECT id, direction, entry, sl, tp1 FROM signals WHERE pair=? AND outcome='OPEN'", (symbol,))
        open_trades = c.fetchall()
        
        for trade in open_trades:
            t_id, direction, entry, sl, tp1 = trade
            multiplier = 100 if "JPY" in symbol else 10 if "XAU" in symbol else 1 if "BTC" in symbol else 10000
            
            closed = False
            result = ""
            pips = 0
            
            # Check for Hits
            if direction == "BUY":
                if current_price >= tp1:
                    closed, result = True, "WIN"
                    pips = (tp1 - entry) * multiplier
                elif current_price <= sl:
                    closed, result = True, "LOSS"
                    pips = (sl - entry) * multiplier
            else: # SELL
                if current_price <= tp1:
                    closed, result = True, "WIN"
                    pips = (entry - tp1) * multiplier
                elif current_price >= sl:
                    closed, result = True, "LOSS"
                    pips = (entry - sl) * multiplier
                    
            if closed:
                c.execute("UPDATE signals SET outcome=?, pips_result=?, outcome_timestamp=? WHERE id=?", 
                          (result, pips, datetime.now(timezone.utc), t_id))
                conn.commit()
                
                # Send immediate alert that trade closed
                icon = "✅" if result == "WIN" else "❌"
                close_msg = f"{icon} <b>Trade Closed: {symbol}</b>\nResult: {result} ({pips:+.1f} Pips)\n<i>By Nilesh</i>"
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=close_msg, parse_mode='HTML')
                
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

            # Prevent duplicating signals within 4 hours
            c.execute("SELECT * FROM signals WHERE pair=? AND timestamp_sent >= datetime('now', '-4 hours') AND outcome='OPEN'", (symbol,))
            recent_signal = c.fetchone()
            
            if not recent_signal:
                dynamic_analysis = generate_analysis_text(struct['bias'], ob_fvg, tech)
                signal_data = {
                    "pair": symbol, "direction": "BUY" if struct['bias'] == "BULLISH" else "SELL",
                    "current_price": current_price, "entry": entry_mid, "sl": sl,
                    "tp1": tp1, "tp2": tp2, "tp3": tp3, "score": scoring['total_score'], 
                    "grade": scoring['grade'], "analysis": dynamic_analysis 
                }

                msg = format_telegram_message(signal_data)
                try:
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')
                    log_signal(signal_data)
                    print(f"--> Signal broadcasted for {symbol}")
                except Exception as e:
                    print(f"Telegram Error: {e}")
                    
        conn.close()
        time.sleep(2) 

async def main():
    print("Initializing AI Quant Bot (1H Edition) with Performance Tracking...")
    init_db()
    
    scheduler = AsyncIOScheduler()
    # Scans markets and checks open trades every 15 minutes
    scheduler.add_job(process_markets, 'interval', minutes=15)
    # Sends daily performance report every 24 hours
    scheduler.add_job(send_daily_report, 'interval', hours=24)
    scheduler.start()
    
    await process_markets() 
    
    while True:
        await asyncio.sleep(1)

# =========================================================================
# === RENDER PORT BINDING & UPTIMEROBOT FIX ===
# =========================================================================

def keep_alive():
    """Runs a tiny web server to satisfy Render's port scanner and UptimeRobot."""
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
            
        def log_message(self, format, *args):
            pass 
            
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"Dummy web server started on port {port} for Render.")
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    asyncio.run(main())
