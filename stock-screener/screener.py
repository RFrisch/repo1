"""
Stock Screener - Identifies stocks trading near their 52-week lows
Outputs results to data.json for the frontend to display
"""

import yfinance as yf
import json
from datetime import datetime
from pathlib import Path
import concurrent.futures

# Get repo root for output path (one level up from script)
REPO_ROOT = Path(__file__).parent.parent

# S&P 500 sample tickers (expand as needed)
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CMCSA',
    'NFLX', 'XOM', 'VZ', 'INTC', 'T', 'PFE', 'MRK', 'PEP', 'KO', 'ABT',
    'CVX', 'WMT', 'CSCO', 'ABBV', 'CRM', 'AVGO', 'ACN', 'TMO', 'MCD', 'COST',
    'NKE', 'DHR', 'NEE', 'LLY', 'WFC', 'BMY', 'UPS', 'ORCL', 'PM', 'TXN',
    'AMD', 'HON', 'QCOM', 'IBM', 'LOW', 'AMGN', 'CAT', 'GE', 'BA', 'SBUX',
    'GS', 'BLK', 'MMM', 'MDLZ', 'GILD', 'CVS', 'AXP', 'ISRG', 'BKNG', 'INTU',
    'DE', 'TGT', 'SYK', 'MO', 'SPGI', 'ZTS', 'CI', 'CB', 'PLD', 'BDX',
    'SCHW', 'NOW', 'CME', 'SO', 'DUK', 'TJX', 'CL', 'USB', 'D', 'APD'
]

# Threshold: stock is "cheap" if within X% of 52-week low
THRESHOLD_PERCENT = 15


def fetch_stock_data(ticker):
    """Fetch stock data and calculate proximity to 52-week low."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 20:
            return None

        current_price = hist['Close'].iloc[-1]
        low_52week = hist['Low'].min()
        high_52week = hist['High'].max()

        # Calculate how far above the 52-week low (as percentage)
        pct_above_low = ((current_price - low_52week) / low_52week) * 100

        # Calculate position in 52-week range (0% = at low, 100% = at high)
        range_position = ((current_price - low_52week) / (high_52week - low_52week)) * 100

        info = stock.info

        return {
            'ticker': ticker,
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/A'),
            'current_price': round(current_price, 2),
            'low_52week': round(low_52week, 2),
            'high_52week': round(high_52week, 2),
            'pct_above_low': round(pct_above_low, 2),
            'range_position': round(range_position, 2),
            'pe_ratio': info.get('trailingPE'),
            'market_cap': info.get('marketCap'),
            'dividend_yield': info.get('dividendYield')
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def format_market_cap(value):
    """Format market cap for display."""
    if value is None:
        return 'N/A'
    if value >= 1e12:
        return f'${value/1e12:.2f}T'
    if value >= 1e9:
        return f'${value/1e9:.2f}B'
    if value >= 1e6:
        return f'${value/1e6:.2f}M'
    return f'${value:,.0f}'


def main():
    print(f"Fetching data for {len(TICKERS)} stocks...")

    # Fetch data in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_stock_data, ticker): ticker for ticker in TICKERS}
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result:
                results.append(result)

    # Filter stocks near 52-week low
    cheap_stocks = [s for s in results if s['pct_above_low'] <= THRESHOLD_PERCENT]
    cheap_stocks.sort(key=lambda x: x['pct_above_low'])

    # Format for output
    for stock in cheap_stocks:
        stock['market_cap_fmt'] = format_market_cap(stock['market_cap'])
        stock['pe_ratio_fmt'] = f"{stock['pe_ratio']:.2f}" if stock['pe_ratio'] else 'N/A'
        stock['dividend_yield_fmt'] = f"{stock['dividend_yield']:.2f}%" if stock['dividend_yield'] else 'N/A'

    output = {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'threshold_percent': THRESHOLD_PERCENT,
        'total_screened': len(results),
        'stocks_found': len(cheap_stocks),
        'stocks': cheap_stocks
    }

    # Save to JSON (in repo root for GitHub Pages)
    output_path = REPO_ROOT / 'data.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Found {len(cheap_stocks)} stocks within {THRESHOLD_PERCENT}% of 52-week low")
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
