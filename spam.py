
Base_keyword_weights = {
    "free": 6, "win": 6, "winner":6, "act now":5, "buy now":5, "loan":5,
    "credit": 4, "cash":5, "click":4, "congratulations":5, "money":4,
    "risk free":5, "offer":4, "guarantee":4, "limited time":5,
    "urgent reply":5, "no risk":5
}
Base_weights = {
    "suspicious_link":1 , "short_link": 2, "many_links":1, 
    "excess_caps":1, "many_exclaims": 1, "mixed_symbols":1,
    "unknown_sender":1, "trusted_sender": -1, "empty_body": 1
} 

url_shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl","ow.ly", "is.gd"]
trusted_domains = ["gmail.com","yahoo.com","outlook.com","hotmail.com"]
stopwords = set(["the", "and", "is", "in", "at","of","a","an","to","for","on","that","this","with"])

learning_rate = 3.0
threshold = 2.5


def clean_text(txt):
    if not txt:
        return ""
    out = []
    
    for char in txt:
        if char.isalnum() or char.isspace() or char in "/:,-?&=#_%=~":
            out.append(char)

    return "".join(out).lower()

def tokenize(txt):
    bits = clean_text(txt).split()
    return [b for b in bits if b not in stopwords]

def simple_stem(w):
    for suf in ("ing", "ed", "ly", "s" ,"es"):
        if w.endswith(suf) and len(w) > len(suf) +1:
            return w[:-len(suf)]
    return w


def extract_features(subject,body,sender):
    subj = subject or ""
    body = body or ""
    mash = subj + " " + body
    toks = tokenize(mash)
    f = {}
    joined = " ".join(toks).lower()

    for phrase in ("act now","buy now", "limited time", "risk free","urgent reply","no risk"):
        f["kw:" + phrase] = 1 if phrase in joined else 0
    
    for t in toks:
        st = simple_stem(t)
        for kw in Base_keyword_weights:
            if simple_stem(kw) == st:
                f["kw:" + kw] = f.get("kw:" + kw, 0) + 1

    f["short_link"] = 1 if any(s in mash for s in url_shorteners) else 0
    f["many_links"] = 1 if mash.count("http") >= 3 else 0
    f["suspicious_link"] = 1 if any(s in mash.lower() for s in ["http://", "https://","www."]) and any(char.isdigit() for char in mash) else 0

    caps = sum(1 for c in mash if c.isupper())
    letters = sum(1 for c in mash if c.isalpha())
    f["excess_caps"] = 1 if letters and caps/ letters > 0.3 else 0
    f["many_exclaims"] = 1 if mash.count("!") >= 3 else 0
    f["mixed_symbols"] = 1 if sum(1 for c in mash if not c.isalnum() and not c.isspace()) > 6 else 0

    sender = (sender or "").lower()
    f["trusted_sender"] = 1 if any(d in sender for d in trusted_domains) else 0
    f["unknown_sender"] = 0 if f["trusted_sender"] else 1
    f["empty_body"] = 1 if not mash.strip() else 0

    for k in Base_keyword_weights:
        f.setdefault("kw:" + k, 0)
    for k in Base_weights:
        f.setdefault(k,0)
    
    return f

class BrainCell:
    def __init__(self):
        self.syn = {("kw:"+ k): float(v) for k,v in Base_keyword_weights.items()}
        self.syn.update({k: float(v) for k,v in Base_weights.items()})
        self.syn["bias"] = 0.0

    def raw_score(self,f):
        s = self.syn["bias"]
        for k, x in f.items():
            s+= self.syn.get(k,0.0) * x
        return s
    
    def learn(self, f, label):
        t = 1 if label.lower() == "spam" else -1
        s = self.raw_score(f)
        if (t == 1 and s <= threshold) or (t == -1 and s > threshold):

            for k, x in f.items():
                self.syn[k] = self.syn.get(k, 0.0) + learning_rate * t * x
            if label.lower() == "spam":
               self.syn["bias"] += learning_rate * 0.5


    def classify(self,f):
        s = self.raw_score(f)
        return "spam" if s>= threshold else "notspam"
    
detector = BrainCell()

spam_train = [
    ("win cash now","click here","unknown@spam.biz"),  
    ("act now", "buy now limited time offer", "promo@deals.com"),  
    ("congratulations","you won a prize","lottery@scam.com"),  
    ("risk free","get loan today","money@loan.biz"),  
    ("urgent reply needed", "verify your account now", "security@fake.com"),  
    ("you won", "claim your free gift card", "winner@scam.net"),  
    ("limited time", "act fast before offer expires", "deals@spam.org"),  
    ("free money", "no credit check required", "loan@spam.biz"),  
    ("click here now", "100% guaranteed winner", "prize@fake.com"),  
    ("congratulations winner", "cash prize waiting", "notify@scam.com")
        ]
ham_train = [
    ("meeting schedule", "please find attached", "boss@gmail.com"),  
    ("project update", "our report is ready", "colleague@outlook.com"),  
    ("lunch plan","where to eat?","friend@yahoo.com"),  
    ("team call","join the meeting","teammate@gmail.com"),  
    ("quarterly review", "let's discuss your progress", "manager@yahoo.com"),  
    ("weekend plans", "are you free on saturday", "buddy@hotmail.com"),  
    ("document review", "please check the attached file", "coworker@outlook.com"),  
    ("birthday party", "you're invited to celebrate", "friend@gmail.com"),  
    ("conference call", "dial in at 2pm today", "admin@yahoo.com"),  
    ("status update", "here is the latest information", "team@gmail.com") 
        ]

for _ in range(100):
    for subj, body, sender in spam_train:
        detector.learn(extract_features(subj, body, sender), "spam")
    for subj, body, sender in ham_train:
        detector.learn(extract_features(subj, body, sender), "notspam")

dataset = [(s, b, e, "spam") for s, b, e in spam_train] + \
           [(s, b, e, "notspam") for s, b, e in ham_train]


def evaluate_detailed_accuracy(dataset):
    correct_spam = 0
    total_spam = 0
    correct_ham = 0
    total_ham = 0

    for subj, body, sender, label in dataset:
        feats = extract_features(subj,body, sender)
        pred = detector.classify(feats)

        if label == "spam":
            total_spam += 1
            if pred == "spam":
                correct_spam += 1
        
        else:
            total_ham += 1
            if pred == "notspam":
                correct_ham += 1

    spam_acc = (correct_spam / total_spam * 100) if total_spam > 0 else 0
    ham_acc = (correct_ham / total_ham * 100) if total_ham > 0 else 0

    print(f"Your spam detector correctly identified {spam_acc}% of spam emails as spam.")  
    print(f"Your spam detector correctly identified {ham_acc}% of ham emails as not spam")

    return spam_acc, ham_acc

def is_spam(subject, body, sender="", learn_label = None):
    feats = extract_features(subject, body, sender)
    if learn_label:
        detector.learn(feats, learn_label)
    return detector.classify(feats)

if __name__ == "__main__":
    try: 
        subject = input().strip()
        body = input().strip()
        sender = input().strip()
        result = detector.classify(extract_features(subject, body, sender))
        print(result) 
    except EOFError:
        print("notspam")
