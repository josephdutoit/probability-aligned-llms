import re

def get_prob(ans: str):
    # Match various probability formats
    pattern = r'(\d+(?:\.\d+)?\s*%?)\s*(?:/\d+)?|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b'
    matches = re.findall(pattern, ans, re.IGNORECASE)
    
    if not matches:
        return None
    
    last_match = matches[-1]
    
    try:
        if '%' in last_match:
            return float(last_match.replace('%', '').strip()) / 100.0
        elif re.match(r'^\d+(\.\d+)?$', last_match):
            return float(last_match)
        return None
        # TODO: Could add word-to-number conversion here
    except ValueError:
        return None

def check_range(prob: float):
    return 0.0 <= prob <= 1.0

def compute_rewards(ans1, ans2):
    prob1 = get_prob(ans1)
    prob2 = get_prob(ans2)
    
    if prob1 is None or prob2 is None:
        return -1.0  
    
    if not (check_range(prob1) and check_range(prob2)):
        return -1.0  
    
    if prob1 + prob2 != 1.0:
        return -1.0 # Mutually exclusive probabilities must sum to 1
    
    return 1.0 #Valid probabilities
    