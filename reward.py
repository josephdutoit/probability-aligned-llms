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
        return -1.0

def check_range(prob: float):
    return 0.0 <= prob <= 1.0

def compute_rewards(ans1, ans2):
    rewards1 = [0.0] * len(ans1)
    rewards2 = [0.0] * len(ans2)
    probs1 = []
    probs2 = []
    for i in range(len(ans1)):
 
        a1 = ans1[i]
        a2 = ans2[i]
        print(f"Computing rewards for answers: '{a1}' and '{a2}'")
        
        # Format reward
        if re.search(r'<think>.*?</think><answer>.*?</answer>', a1, re.DOTALL):
            rewards1[i] += 1.0
        else:
            rewards1[i] += -2.0

        if re.search(r'<think>.*?</think><answer>.*?</answer>', a2, re.DOTALL):
            rewards2[i] += 1.0
        else:
            rewards2[i] += -2.0
        
        if rewards1[i] == 1.0 and rewards2[i] == 1.0:
            prob1 = get_prob(a1)
            prob2 = get_prob(a2)
            probs1.append(prob1)
            probs2.append(prob2)

    # continue only if all answers have the correct format
    if len(probs1) != len(ans1) or len(probs2) != len(ans2):
        return rewards1, rewards2

    # check the range contraints 
    for i in range(len(ans1)):
        if check_range(probs1[i]):
            rewards1[i] += 1.0
        else:
            rewards1[i] += -1.0
  
    for i in range(len(ans2)):
        if check_range(probs2[i]):
            rewards2[i] += 1.0
        else:
            rewards2[i] += -1.0

    # ensure that all probabilities match and are not None
    prob1 = probs1[0] if all(p == probs1[0] for p in probs1) else None
    prob2 = probs2[0] if all(p == probs2[0] for p in probs2) else None

    if prob1 is not None:
        for i in range(len(ans1)):
            rewards1[i] += 1.0

    if prob2 is not None:
        for i in range(len(ans2)):
            rewards2[i] += 1.0

    # do not continue unless all probabilities match and are not None
    if prob1 is None or prob2 is None:
        return rewards1, rewards2

    # do not continue unless both probabilities are in range
    if not check_range(prob1) or not check_range(prob2):
        return rewards1, rewards2

    # check the sum constraint
    if prob1 + prob2 == 1.0:
        for i in range(len(ans1)):
            rewards1[i] += 1.0
            rewards2[i] += 1.0  
    
    return rewards1, rewards2
    