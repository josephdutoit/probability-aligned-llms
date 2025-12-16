import re

def get_prob(ans: str):
    # Match various probability formats
    pattern = r'<answer>(\d+(?:\.\d+)?\s*%?)\s*(?:/\d+)?|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b</answer>'
    matches = re.findall(pattern, ans, re.IGNORECASE)
    
    if not matches:
        return None
    
    last_match = matches[-1]
    
    try:
        if '%' in last_match:
            return float(last_match.replace('%', '').strip()) / 100.0
        elif '/' in last_match:
            num, den = last_match.split('/')
            return float(num) / float(den)
        elif re.match(r'^\d+(\.\d+)?$', last_match):
            return float(last_match)
        return None
        # TODO: Could add word-to-number conversion here
    except (ValueError, ZeroDivisionError):
        return None

def check_range(prob: float):
    return 0.0 <= prob <= 1.0

def compute_rewards(ans1, ans2):
    print(f"Computing rewards for answers: '{ans1}' and '{ans2}'")
    prob1 = get_prob(ans1)
    prob2 = get_prob(ans2)
    reward1 = 0.0
    reward2 = 0.0

    if prob1 is None:
        reward1 = -1.0  
    else:
        reward1 += 1.0
    if prob2 is None:
        reward2 = -1.0  
    else:
        reward2 += 1.0
        
    if prob1 is not None and not check_range(prob1):
        reward1 -= 1.0 
    if prob2 is not None and not check_range(prob2):
        reward2 -= 1.0 
    
    if prob1 is not None and prob2 is not None and prob1 + prob2 != 1.0:
        reward1 -= 1.0
        reward2 -= 1.0
    
    # Format reward
    if re.search(r'<think>.*?</think><answer>.*?</answer>', ans1, re.DOTALL):
        reward1 += 1.0
    if re.search(r'<think>.*?</think><answer>.*?</answer>', ans2, re.DOTALL):
        reward2 += 1.0
    
    return reward1, reward2
    