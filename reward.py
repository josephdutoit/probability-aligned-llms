import re
import torch
# def get_prob(ans: str):
#     # Match various probability formats
#     pattern = r'<answer>(\d+(?:\.\d+)?\s*%?)\s*(?:/\d+)?|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b</answer>'
#     matches = re.findall(pattern, ans, re.IGNORECASE)
    
#     if not matches:
#         return None
    
#     last_match = matches[-1]
    
#     try:
#         if '%' in last_match:
#             return float(last_match.replace('%', '').strip()) / 100.0
#         elif '/' in last_match:
#             num, den = last_match.split('/')
#             return float(num) / float(den)
#         elif re.match(r'^\d+(\.\d+)?$', last_match):
#             return float(last_match)
#         return None
#         # TODO: Could add word-to-number conversion here
#     except (ValueError, ZeroDivisionError):
#         return -1.0

def get_prob(ans: str):
    m = re.match(
        r'.*<answer>\s*\d+(\.\d+)?\s*</answer>$',
        ans
    )
    if not m:
        return -1.0
    return float(m.group(1))


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
        
        if re.search(r'.*<answer>.*?</answer>', a1, re.DOTALL) and len(re.findall(r'<answer>.*?</answer>', a1)) == 1:
            rewards1[i] += 1.0
        
        if re.search(r'.*<answer>.*?</answer>', a2, re.DOTALL) and len(re.findall(r'<answer>.*?</answer>', a2)) == 1:
            rewards2[i] += 1.0

        # Format reward
        # if re.search(r'^<think>[\s\S]*?</think>\s*<answer>\s*\d+(\.\d+)?\s*</answer>$', a1, re.DOTALL):
        #     rewards1[i] += 1.0

        # if re.search(r'^<think>[\s\S]*?</think>\s*<answer>\s*\d+(\.\d+)?\s*</answer>$', a2, re.DOTALL):
        #     rewards2[i] += 1.0
        
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
  
    for i in range(len(ans2)):
        if check_range(probs2[i]):
            rewards2[i] += 1.0

    # ensure that all probabilities match and are not None
    # prob1 = probs1[0] if all(p == probs1[0] for p in probs1) else None
    # prob2 = probs2[0] if all(p == probs2[0] for p in probs2) else None
    var1 = torch.var(torch.tensor(probs1))
    var2 = torch.var(torch.tensor(probs2))  
    if var1.item() <= .01:
        for i in range(len(ans1)):
            rewards1[i] += 1.0
        prob1 = sum(probs1) / len(probs1)
    else:
        prob1 = None    

    if var2.item() <= .01:
        for i in range(len(ans2)):
            rewards2[i] += 1.0
        prob2 = sum(probs2) / len(probs2)
    else:
        prob2 = None

    # do not continue unless all probabilities match and are not None
    if prob1 is None or prob2 is None:
        return rewards1, rewards2

    # do not continue unless both probabilities are in range
    if not check_range(prob1) or not check_range(prob2):
        return rewards1, rewards2

    else:
        for i in range(len(ans1)):
            rewards1[i] += 1.0
            rewards2[i] += 1.0

    # check the sum constraint
    if abs(prob1 + prob2 - 1.0) <= 0.01:
        for i in range(len(ans1)):
            rewards1[i] += 1.0
            rewards2[i] += 1.0  
    
    return rewards1, rewards2
    