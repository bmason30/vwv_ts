"""
Diagnostic Test for Data Fetcher - Run this in your terminal
This will tell us exactly what's happening with the data fetching
"""
import sys
import pandas as pd
from datetime import datetime

print("=" * 60)
print("DATA FETCHER DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from data.fetcher import get_market_data_enhanced, is_etf
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check if the function has the fix
print("\n2. Checking function signature...")
import inspect
sig = inspect.signature(get_market_data_enhanced)
print(f"   Parameters: {list(sig.parameters.keys())}")

# Test 3: Simple data fetch without debug
print("\n3. Testing simple fetch (no debug)...")
try:
    data = get_market_data_enhanced('AAPL', period='1mo', show_debug=False)
    if data is not None:
        print(f"✅ Fetch successful")
        print(f"   Data shape: {data.shape}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Columns: {list(data.columns)}")
    else:
        print("❌ Fetch returned None")
except Exception as e:
    print(f"❌ Fetch failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Fetch with debug to see messages
print("\n4. Testing fetch WITH debug messages...")
try:
    # This should show debug messages in terminal
    data = get_market_data_enhanced('AAPL', period='3mo', show_debug=False)  # False for terminal test
    if data is not None:
        print(f"✅ Fetch successful")
        print(f"   Data shape: {data.shape}")
        
        # Check if period is actually being used
        days_of_data = len(data)
        print(f"   Days of data: {days_of_data}")
        
        if days_of_data < 30:
            print("   ⚠️  Warning: Very little data - possible issue")
        elif days_of_data > 180:
            print("   ⚠️  Warning: Too much data for 3mo - period parameter may not be working")
        else:
            print("   ✅ Data amount looks correct for 3 months")
    else:
        print("❌ Fetch returned None")
except Exception as e:
    print(f"❌ Fetch failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: ETF detection
print("\n5. Testing ETF detection...")
try:
    aapl_is_etf = is_etf('AAPL')
    spy_is_etf = is_etf('SPY')
    print(f"   AAPL is ETF: {aapl_is_etf} (should be False)")
    print(f"   SPY is ETF: {spy_is_etf} (should be True)")
    
    if not aapl_is_etf and spy_is_etf:
        print("✅ ETF detection working correctly")
    else:
        print("❌ ETF detection has issues")
except Exception as e:
    print(f"❌ ETF detection failed: {e}")

# Test 6: Verify the period bug is fixed
print("\n6. Testing if period parameter is actually used...")
try:
    data_1mo = get_market_data_enhanced('SPY', period='1mo', show_debug=False)
    data_6mo = get_market_data_enhanced('SPY', period='6mo', show_debug=False)
    
    if data_1mo is not None and data_6mo is not None:
        len_1mo = len(data_1mo)
        len_6mo = len(data_6mo)
        
        print(f"   1mo data: {len_1mo} days")
        print(f"   6mo data: {len_6mo} days")
        
        if len_6mo > len_1mo:
            print("✅ Period parameter IS being used correctly!")
        else:
            print("❌ CRITICAL: Period parameter NOT being used (still hardcoded)")
    else:
        print("❌ Could not fetch data for comparison")
except Exception as e:
    print(f"❌ Period test failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 60)
