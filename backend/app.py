"""
QuantRep Backend — Powered by OpenChart
Run: python app.py
API will be available at http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import math
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = jsonify({'status': 'ok'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# ── OpenChart initialisation (lazy so startup is fast) ─────────────────────
_nse = None

def get_nse():
    global _nse
    if _nse is None:
        from openchart import NSEData
        _nse = NSEData()
        _nse.download()          # fetch master symbol list
    return _nse


# ── Black-Scholes helpers ───────────────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes(S, K, T, r, sigma, option_type='CE'):
    """Returns (price, delta, gamma, theta, vega)"""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == 'CE' else max(K - S, 0)
        return intrinsic, 0, 0, 0, 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'CE':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1
    gamma = norm_cdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-(S * norm_cdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm_cdf(d2 if option_type == 'CE' else -d2)) / 365
    vega  = S * norm_cdf(d1) * math.sqrt(T) / 100
    return round(price, 2), round(delta, 4), round(gamma, 6), round(theta, 2), round(vega, 2)

def implied_vol(market_price, S, K, T, r, option_type):
    """Newton-Raphson IV solver"""
    sigma = 0.2
    for _ in range(100):
        price, _, _, _, vega = black_scholes(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < 0.001:
            break
        if vega == 0:
            break
        sigma -= diff / (vega * 100)
        sigma = max(0.001, min(sigma, 5.0))
    return round(sigma * 100, 2)


# ── Expiry helpers ──────────────────────────────────────────────────────────
def get_weekly_expiry(ref_date=None):
    """Return next Thursday (Nifty weekly expiry)"""
    d = ref_date or datetime.date.today()
    days_ahead = 3 - d.weekday()   # Thursday = 3
    if days_ahead <= 0:
        days_ahead += 7
    return d + datetime.timedelta(days=days_ahead)

def get_monthly_expiry(ref_date=None):
    """Last Thursday of current month"""
    d = ref_date or datetime.date.today()
    last = datetime.date(d.year, d.month,
                         [31,28,31,30,31,30,31,31,30,31,30,31][d.month-1])
    if d.month == 2 and ((d.year%4==0 and d.year%100!=0) or d.year%400==0):
        last = datetime.date(d.year, d.month, 29)
    while last.weekday() != 3:
        last -= datetime.timedelta(days=1)
    return last

def dte(expiry_date):
    """Days to expiry"""
    return max((expiry_date - datetime.date.today()).days, 0)

def option_symbol(index, expiry, strike, opt_type):
    """Build NFO symbol string e.g. NIFTY25FEB24400CE"""
    idx_map = {'NIFTY 50': 'NIFTY', 'BANKNIFTY': 'BANKNIFTY',
               'FINNIFTY': 'FINNIFTY', 'SENSEX': 'SENSEX'}
    idx = idx_map.get(index, 'NIFTY')
    yy  = str(expiry.year)[2:]
    mm  = expiry.strftime('%b').upper()
    return f"{idx}{yy}{mm}{int(strike)}{opt_type}"


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.datetime.now().isoformat()})


# ── 1. Spot price ─────────────────────────────────────────────────────────
@app.route('/api/spot')
def spot():
    """GET /api/spot?symbol=NIFTY 50"""
    symbol  = request.args.get('symbol', 'Nifty 50')
    end     = datetime.datetime.now()
    start   = end - datetime.timedelta(days=5)
    try:
        nse  = get_nse()
        data = nse.historical(symbol=symbol, exchange='NSE',
                               start=start, end=end, interval='1d')
        if data is None or data.empty:
            return jsonify({'error': 'No data'}), 404
        row  = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else row
        chng = round(float(row['close']) - float(prev['close']), 2)
        pct  = round(chng / float(prev['close']) * 100, 2)
        return jsonify({
            'symbol' : symbol,
            'spot'   : round(float(row['close']), 2),
            'open'   : round(float(row['open']),  2),
            'high'   : round(float(row['high']),  2),
            'low'    : round(float(row['low']),   2),
            'change' : chng,
            'pct'    : pct,
            'date'   : str(row.name.date()) if hasattr(row.name, 'date') else str(row.name)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 2. Spot OHLCV history ─────────────────────────────────────────────────
@app.route('/api/history')
def history():
    """GET /api/history?symbol=Nifty 50&days=365&interval=1d"""
    symbol   = request.args.get('symbol',   'Nifty 50')
    days     = int(request.args.get('days', 365))
    interval = request.args.get('interval', '1d')
    end      = datetime.datetime.now()
    start    = end - datetime.timedelta(days=days)
    try:
        nse  = get_nse()
        data = nse.historical(symbol=symbol, exchange='NSE',
                               start=start, end=end, interval=interval)
        if data is None or data.empty:
            return jsonify({'error': 'No data'}), 404
        records = []
        for idx, row in data.iterrows():
            records.append({
                'date'  : str(idx.date()) if hasattr(idx, 'date') else str(idx),
                'open'  : round(float(row['open']),   2),
                'high'  : round(float(row['high']),   2),
                'low'   : round(float(row['low']),    2),
                'close' : round(float(row['close']),  2),
                'volume': int(row.get('volume', 0))
            })
        return jsonify({'symbol': symbol, 'interval': interval, 'data': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 3. Options chain (live) ───────────────────────────────────────────────
@app.route('/api/chain')
def chain():
    """
    GET /api/chain?symbol=NIFTY 50&expiry_type=weekly
    Builds a synthetic options chain using:
      - Spot from OpenChart
      - B-S pricing for all strikes ±10 from ATM (step = lot_step)
    """
    symbol      = request.args.get('symbol',      'NIFTY 50')
    expiry_type = request.args.get('expiry_type', 'weekly')
    r           = 0.065   # risk-free rate

    # index config
    cfg = {
        'NIFTY 50'  : {'step': 50,  'lot': 75,  'base_iv': 14.0},
        'BANKNIFTY' : {'step': 100, 'lot': 30,  'base_iv': 15.5},
        'FINNIFTY'  : {'step': 50,  'lot': 40,  'base_iv': 14.8},
        'SENSEX'    : {'step': 100, 'lot': 10,  'base_iv': 13.5},
    }.get(symbol, {'step': 50, 'lot': 75, 'base_iv': 14.0})

    expiry = get_weekly_expiry() if expiry_type == 'weekly' else get_monthly_expiry()
    T      = dte(expiry) / 365

    try:
        # — get live spot —
        nse    = get_nse()
        end    = datetime.datetime.now()
        start  = end - datetime.timedelta(days=5)
        data   = nse.historical(symbol=symbol, exchange='NSE',
                                  start=start, end=end, interval='1d')
        spot   = round(float(data.iloc[-1]['close']), 2)
    except Exception as e:
        return jsonify({'error': f'Could not fetch spot: {e}'}), 500

    # round spot to nearest step for ATM
    atm    = round(spot / cfg['step']) * cfg['step']
    step   = cfg['step']
    iv_base = cfg['base_iv'] / 100

    strikes = [atm + i * step for i in range(-10, 11)]
    rows = []
    for K in strikes:
        moneyness = abs(K - spot) / spot
        # smile: OTM options have higher IV
        iv_smile  = iv_base * (1 + 1.5 * moneyness)

        c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(spot, K, T, r, iv_smile, 'CE')
        p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(spot, K, T, r, iv_smile * 1.03, 'PE')

        c_iv = implied_vol(c_price, spot, K, T, r, 'CE') if c_price > 0.5 else 0
        p_iv = implied_vol(p_price, spot, K, T, r, 'PE') if p_price > 0.5 else 0

        rows.append({
            'strike'     : K,
            'is_atm'     : K == atm,
            'call': {
                'ltp'    : c_price,
                'iv'     : c_iv,
                'delta'  : round(c_delta,  4),
                'gamma'  : round(c_gamma,  6),
                'theta'  : round(c_theta,  2),
                'vega'   : round(c_vega,   2),
                'oi'     : max(0, int((10 - abs(K - atm) / step) * 50000 * (1 + 0.3 * (1 - moneyness)))),
                'volume' : max(0, int((10 - abs(K - atm) / step) * 12000)),
            },
            'put': {
                'ltp'    : p_price,
                'iv'     : p_iv,
                'delta'  : round(p_delta,  4),
                'gamma'  : round(p_gamma,  6),
                'theta'  : round(p_theta,  2),
                'vega'   : round(p_vega,   2),
                'oi'     : max(0, int((10 - abs(K - atm) / step) * 48000 * (1 + 0.3 * (1 - moneyness)))),
                'volume' : max(0, int((10 - abs(K - atm) / step) * 11000)),
            }
        })

    total_call_oi = sum(r['call']['oi'] for r in rows)
    total_put_oi  = sum(r['put']['oi']  for r in rows)
    pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0

    # max pain = strike where total option writers lose least
    max_pain_strike = atm
    min_pain = float('inf')
    for K in strikes:
        pain = sum(max(k - K, 0) * row['call']['oi'] + max(K - k, 0) * row['put']['oi']
                   for k, row in [(r['strike'], r) for r in rows])
        if pain < min_pain:
            min_pain = pain
            max_pain_strike = K

    return jsonify({
        'symbol'    : symbol,
        'spot'      : spot,
        'atm'       : atm,
        'expiry'    : str(expiry),
        'dte'       : dte(expiry),
        'pcr'       : pcr,
        'max_pain'  : max_pain_strike,
        'chain'     : rows
    })


# ── 4. Strategy Payoff ────────────────────────────────────────────────────
@app.route('/api/payoff', methods=['POST'])
def payoff():
    """
    POST /api/payoff
    Body: { symbol, spot, legs: [{type, strike, action, premium, lots}] }
    Returns payoff curve points.
    """
    body   = request.json or {}
    spot   = float(body.get('spot', 24000))
    legs   = body.get('legs', [])
    step   = float(body.get('step', 50))
    rng    = float(body.get('range', 0.06))   # ±6% from spot

    low  = int(spot * (1 - rng) / step) * int(step)
    high = int(spot * (1 + rng) / step) * int(step) + int(step)
    prices = list(range(low, high + int(step), int(step)))

    points = []
    for p in prices:
        pnl = 0
        for leg in legs:
            K      = float(leg['strike'])
            prem   = float(leg['premium'])
            lots   = int(leg.get('lots', 1))
            action = leg['action'].upper()   # BUY / SELL
            opt    = leg['type'].upper()     # CE / PE
            mult   = 1 if action == 'BUY' else -1
            lot_sz = int(leg.get('lot_size', 75))

            intrinsic = max(p - K, 0) if opt == 'CE' else max(K - p, 0)
            leg_pnl   = mult * (intrinsic - prem) * lots * lot_sz
            pnl      += leg_pnl
        points.append({'price': p, 'pnl': round(pnl, 2)})

    max_profit = max(pt['pnl'] for pt in points)
    max_loss   = min(pt['pnl'] for pt in points)

    # breakeven(s)
    breakevens = []
    for i in range(1, len(points)):
        if (points[i-1]['pnl'] < 0) != (points[i]['pnl'] < 0):
            # linear interpolation
            x0, y0 = points[i-1]['price'], points[i-1]['pnl']
            x1, y1 = points[i]['price'],   points[i]['pnl']
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            breakevens.append(round(be, 2))

    return jsonify({
        'points'    : points,
        'max_profit': round(max_profit, 2),
        'max_loss'  : round(max_loss,   2),
        'breakevens': breakevens,
        'spot'      : spot
    })


# ── 5. Backtest ───────────────────────────────────────────────────────────
@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    POST /api/backtest
    Body: {
      symbol, start, end, expiry_type,
      entry_time, exit_time,
      sl_pct, target_pct, lots,
      legs: [{type, strike_offset, action, lots}]
    }
    Uses OpenChart to fetch real Nifty daily closes, then prices each
    leg using Black-Scholes on entry and exit day.
    """
    body        = request.json or {}
    symbol      = body.get('symbol',      'Nifty 50')
    start_str   = body.get('start',       '2024-01-01')
    end_str     = body.get('end',         '2024-12-31')
    sl_pct      = float(body.get('sl_pct',     50)) / 100
    target_pct  = float(body.get('target_pct', 30)) / 100
    lots        = int(body.get('lots', 1))
    legs_cfg    = body.get('legs', [
        {'type': 'CE', 'strike_offset': 0,  'action': 'SELL'},
        {'type': 'PE', 'strike_offset': 0,  'action': 'SELL'},
    ])

    index_cfg = {
        'Nifty 50'   : {'step': 50,  'lot': 75,  'iv': 0.14, 'ticker': 'Nifty 50'},
        'BANKNIFTY'  : {'step': 100, 'lot': 30,  'iv': 0.155,'ticker': 'BANKNIFTY'},
        'FINNIFTY'   : {'step': 50,  'lot': 40,  'iv': 0.148,'ticker': 'FINNIFTY'},
    }
    cfg = index_cfg.get(symbol, index_cfg['Nifty 50'])

    start_dt = datetime.datetime.strptime(start_str, '%Y-%m-%d')
    end_dt   = datetime.datetime.strptime(end_str,   '%Y-%m-%d')
    r        = 0.065

    try:
        nse  = get_nse()
        data = nse.historical(symbol=cfg['ticker'], exchange='NSE',
                               start=start_dt, end=end_dt, interval='1d')
        if data is None or data.empty:
            return jsonify({'error': 'No historical data'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    trades = []
    equity = []
    cumulative_pnl = 0

    closes = list(zip(
        [str(i.date()) if hasattr(i,'date') else str(i) for i in data.index],
        [float(v) for v in data['close']]
    ))

    # weekly Thursday grouping
    def next_thursday(d_str):
        d = datetime.date.fromisoformat(d_str)
        days = (3 - d.weekday()) % 7
        return d + datetime.timedelta(days=days or 7)

    processed_expiries = set()

    for i, (date_str, spot) in enumerate(closes):
        expiry = next_thursday(date_str)
        key    = str(expiry)
        if key in processed_expiries:
            continue

        # find exit day (Thursday close or next available)
        exit_spot = spot
        exit_date = date_str
        for j in range(i, min(i+8, len(closes))):
            d = datetime.date.fromisoformat(closes[j][0])
            if d >= expiry:
                exit_spot = closes[j][1]
                exit_date = closes[j][0]
                break

        processed_expiries.add(key)
        step = cfg['step']
        atm  = round(spot / step) * step
        T_entry = max((expiry - datetime.date.fromisoformat(date_str)).days / 365, 1/365)
        T_exit  = max((expiry - datetime.date.fromisoformat(exit_date)).days / 365, 0)

        total_entry_premium = 0
        total_exit_cost     = 0
        leg_details         = []

        for leg in legs_cfg:
            K       = atm + leg.get('strike_offset', 0)
            opt     = leg['type']
            action  = leg['action'].upper()
            iv      = cfg['iv'] * (1 + 0.5 * abs(leg.get('strike_offset', 0)) / (atm * 0.02 + 1))

            entry_price, *_ = black_scholes(spot,      K, T_entry, r, iv, opt)
            exit_price,  *_ = black_scholes(exit_spot, K, T_exit,  r, iv * 0.9, opt)

            mult = 1 if action == 'SELL' else -1
            leg_pnl = mult * (entry_price - exit_price) * lots * cfg['lot']

            total_entry_premium += entry_price
            total_exit_cost     += exit_price
            leg_details.append({
                'type': opt, 'strike': K, 'action': action,
                'entry': round(entry_price, 2), 'exit': round(exit_price, 2)
            })

        pnl = sum(
            (1 if l['action']=='SELL' else -1) * (l['entry'] - l['exit']) * lots * cfg['lot']
            for l in leg_details
        )

        # SL / Target check
        total_collected = sum(l['entry'] for l in leg_details if l['action']=='SELL')
        exit_reason = 'Time Exit'
        if total_collected > 0:
            if pnl < -total_collected * sl_pct * lots * cfg['lot']:
                pnl = -total_collected * sl_pct * lots * cfg['lot']
                exit_reason = 'SL Hit'
            elif pnl > total_collected * target_pct * lots * cfg['lot']:
                pnl = total_collected * target_pct * lots * cfg['lot']
                exit_reason = 'Target Hit'

        cumulative_pnl += pnl
        trades.append({
            'date'       : date_str,
            'exit_date'  : exit_date,
            'expiry'     : key,
            'spot'       : round(spot, 2),
            'exit_spot'  : round(exit_spot, 2),
            'premium'    : round(total_entry_premium, 2),
            'exit_cost'  : round(total_exit_cost, 2),
            'pnl'        : round(pnl, 2),
            'exit_reason': exit_reason,
            'legs'       : leg_details
        })
        equity.append({'date': date_str, 'cumulative_pnl': round(cumulative_pnl, 2)})

    if not trades:
        return jsonify({'error': 'No trades generated'}), 400

    pnls      = [t['pnl'] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    win_rate  = round(len(wins) / len(pnls) * 100, 1) if pnls else 0
    avg_win   = round(sum(wins)   / len(wins),   2) if wins   else 0
    avg_loss  = round(sum(losses) / len(losses), 2) if losses else 0

    # max drawdown
    peak = 0
    max_dd = 0
    running = 0
    for p in pnls:
        running += p
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)

    # Sharpe (annualised, assuming weekly returns)
    import statistics
    weekly_ret = pnls
    sharpe = 0
    if len(weekly_ret) > 1:
        avg_r = statistics.mean(weekly_ret)
        std_r = statistics.stdev(weekly_ret)
        sharpe = round((avg_r / std_r) * math.sqrt(52), 2) if std_r else 0

    return jsonify({
        'summary': {
            'total_pnl'  : round(total_pnl, 2),
            'total_trades': len(trades),
            'win_rate'   : win_rate,
            'avg_win'    : avg_win,
            'avg_loss'   : avg_loss,
            'max_drawdown': round(max_dd, 2),
            'sharpe'     : sharpe,
            'profit_factor': round(-sum(wins)/sum(losses), 2) if losses and sum(losses)!=0 else 0,
        },
        'trades' : trades,
        'equity' : equity
    })


# ── 6. Search symbols ─────────────────────────────────────────────────────
@app.route('/api/search')
def search():
    q        = request.args.get('q', '')
    exchange = request.args.get('exchange', 'NFO')
    try:
        nse     = get_nse()
        results = nse.search(q, exchange=exchange)
        return jsonify(json.loads(results.to_json(orient='records')))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🚀 QuantRep Backend starting on http://localhost:5000\n")
    app.run(debug=True, port=5000)
