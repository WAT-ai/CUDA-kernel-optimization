import sys
import io

error_buf = io.StringIO()
sys.stderr = error_buf

try:
    from strategies import generated_optimization_strategy
except Exception:
    print("Error: ", error_buf.getvalue())

def pipeline():
    return

def get_performance_metric():
    return