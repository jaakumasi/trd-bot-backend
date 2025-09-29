"""
Example of how to support time strings like "1h", "30m", etc.
This is just for demonstration - not implemented in your current system
"""

def parse_time_string(time_str: str) -> int:
    """Convert time strings like '1h', '30m', '24h' to seconds"""
    if isinstance(time_str, int):
        return time_str  # Already in seconds
    
    time_str = time_str.lower().strip()
    
    if time_str.endswith('s'):
        return int(time_str[:-1])
    elif time_str.endswith('m'):
        return int(time_str[:-1]) * 60
    elif time_str.endswith('h'):
        return int(time_str[:-1]) * 3600
    elif time_str.endswith('d'):
        return int(time_str[:-1]) * 86400
    else:
        # If no unit, assume seconds
        return int(time_str)

# Examples:
print(parse_time_string("1h"))    # 3600
print(parse_time_string("30m"))   # 1800
print(parse_time_string("2d"))    # 172800
print(parse_time_string("3600"))  # 3600