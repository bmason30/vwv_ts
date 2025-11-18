"""
Divergence Detection Debug Script
Tests the divergence detection with various symbols and shows internal details
"""

import pandas as pd
import yfinance as yf
from analysis.divergence import calculate_divergence_score, detect_advanced_divergences
from config.settings import get_momentum_divergence_config

def test_divergence_detection(symbol='TSLA', period='3mo'):
    """Test divergence detection with detailed output."""
    print(f"\n{'='*80}")
    print(f"Testing Divergence Detection: {symbol} ({period})")
    print(f"{'='*80}\n")

    # Fetch data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    print(f"✓ Data fetched: {len(data)} bars")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")

    # Get config
    config = get_momentum_divergence_config()
    print(f"\n✓ Configuration:")
    print(f"  Lookback period: {config['lookback_period']} bars")
    print(f"  Min swing distance: {config['min_swing_distance']} bars")
    print(f"  Peak prominence: {config['peak_prominence']*100}%")

    # Calculate RSI for context
    close = data['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    print(f"\n✓ Current indicators:")
    print(f"  Price: ${close.iloc[-1]:.2f}")
    print(f"  RSI: {current_rsi:.2f}")

    # Test divergence detection
    print(f"\n{'='*80}")
    print("Running Divergence Detection...")
    print(f"{'='*80}\n")

    # Calculate with technicals (dummy values)
    technicals = {
        'rsi_14': current_rsi,
        'mfi_14': 50,
        'stochastic_k': 50,
        'williams_r': -50
    }

    result = calculate_divergence_score(data, technicals, use_advanced=True)

    print(f"✓ Detection complete:")
    print(f"  Score: {result['score']}")
    print(f"  Status: {result['status']}")
    print(f"  Total divergences: {result['total_divergences']}")
    print(f"  Bullish: {result['bullish_count']}")
    print(f"  Bearish: {result['bearish_count']}")
    print(f"  Regular: {result.get('regular_count', 0)}")
    print(f"  Hidden: {result.get('hidden_count', 0)}")

    if 'error' in result:
        print(f"\n⚠️  Error: {result['error']}")

    if result['divergences']:
        print(f"\n{'='*80}")
        print("Detected Divergences:")
        print(f"{'='*80}\n")

        for i, div in enumerate(result['divergences'], 1):
            print(f"{i}. {div['type'].upper()} on {div['oscillator'].upper()}")
            print(f"   Strength: {div['strength']}")
            print(f"   Score: {div['score']:+.1f}")
            print(f"   Description: {div['description']}")
            if 'price_vals' in div:
                print(f"   Price values: {div['price_vals']}")
            if 'osc_vals' in div:
                print(f"   Oscillator values: {div['osc_vals']}")
            print()
    else:
        print(f"\nℹ️  No divergences detected in the current period.")
        print(f"   This is normal - divergences are relatively rare signals.")
        print(f"   Try:")
        print(f"   1. Testing with more volatile symbols (e.g., COIN, MARA)")
        print(f"   2. Testing during market reversals")
        print(f"   3. Adjusting lookback_period or peak_prominence")

    # Additional diagnostics
    print(f"\n{'='*80}")
    print("Advanced Diagnostics:")
    print(f"{'='*80}\n")

    from scipy.signal import find_peaks
    import numpy as np

    lookback = config['lookback_period']
    recent_data = data.tail(lookback)
    price_series = recent_data['Close']

    # Find price peaks
    price_peaks, _ = find_peaks(
        price_series.values,
        distance=config['min_swing_distance'],
        prominence=config['peak_prominence'] * price_series.mean()
    )

    price_troughs, _ = find_peaks(
        -price_series.values,
        distance=config['min_swing_distance'],
        prominence=config['peak_prominence'] * price_series.mean()
    )

    print(f"✓ Price analysis (last {lookback} bars):")
    print(f"  Peaks found: {len(price_peaks)}")
    print(f"  Troughs found: {len(price_troughs)}")
    print(f"  Need at least 2 peaks/troughs for divergence")

    if len(price_peaks) < 2 and len(price_troughs) < 2:
        print(f"\n⚠️  Insufficient peaks/troughs detected!")
        print(f"   Possible solutions:")
        print(f"   1. Increase lookback_period (current: {lookback})")
        print(f"   2. Decrease peak_prominence (current: {config['peak_prominence']})")
        print(f"   3. Decrease min_swing_distance (current: {config['min_swing_distance']})")

    # RSI analysis
    from analysis.divergence import calculate_rsi_series
    rsi_series = calculate_rsi_series(recent_data['Close'], period=14)

    rsi_peaks, _ = find_peaks(
        rsi_series.dropna().values,
        distance=config['min_swing_distance']
    )

    rsi_troughs, _ = find_peaks(
        -rsi_series.dropna().values,
        distance=config['min_swing_distance']
    )

    print(f"\n✓ RSI analysis (last {lookback} bars):")
    print(f"  Peaks found: {len(rsi_peaks)}")
    print(f"  Troughs found: {len(rsi_troughs)}")
    print(f"  Current RSI: {current_rsi:.2f}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test with multiple symbols
    symbols = ['TSLA', 'COIN', 'MARA', 'SPY', 'QQQ']

    for symbol in symbols:
        try:
            test_divergence_detection(symbol, period='3mo')
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}\n")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nDivergence detection is working if:")
    print("✓ No syntax errors")
    print("✓ Peak detection finds peaks/troughs")
    print("✓ Divergences appear on volatile symbols")
    print("\nNo divergences is NORMAL when:")
    print("• Market is trending steadily")
    print("• No recent reversals")
    print("• Oscillators and price move in sync")
    print("\nTo increase sensitivity, edit config/settings.py:")
    print("  'peak_prominence': 0.01  (reduce from 0.02)")
    print("  'lookback_period': 30     (increase from 20)")
