
Base_keyword_weights = {
    "free": 5, "win": 5,"winner":5, "act now":4, "buy now": 4,"loan": 4
    ,"credit": 3, "cash":4, "click":3, "congratulations": 4, "money":3
    ,"risk free": 4, "offer": 3, "guarantee": 3, "limited time": 4   
}

Base_weights = {
    "suspicious_link":1 , "short_link": 2, "many_links":1, 
    "excess_caps":1, "many_exclaims": 1, "mixed_symbols":1,
    "unknown_sender":1, "trusted_sender": -1, "empty_body": 1,
    "attachment_unknown_sender": 1
} 

url_shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl","ow.ly", "is.gd"]
trusted_domains = ["gmail.com","yahoo.com","outlook.com","hotmail.com"]
stopwords = set(["the", "and", "is", "in", "at","of","a","an","to","for","on","that","this","with"])

learning_rate = 1.5
threshold = -10.0


def clean_text(txt):
    if not txt:
        return ""
    out = []
    
    for char in txt:
        if char.isalnum() or char.isspace() or char in "/:,-?&=#_%=~":
            out.append(char)
        else:
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

    for phrase in ("act now","buy now", "limited time", "risk free","urgent reply", "no risk"):
        f["kw:" + phrase] = 1 if phrase in joined else 0
    
    for t in toks:
        st = simple_stem(t)
        for kw in Base_keyword_weights:
            if simple_stem(kw) == st:
                f["kw:" + kw] = f.get("kw:" + kw, 0) + 1

    f["short_link"] = 1 if any(s in mash for s in url_shorteners) else 0
    f["many_links"] = 1 if mash.count("http") >= 3 else 0
    f["suspicious_link"] = 1 if any(char.isdigit() for char in mash) else 0

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
        self.syn["bias"] = 5.0

    def raw_score(self,f):
        s = self.syn["bias"]
        for k, x in f.items():
            s+= self.syn.get(k,0.0) * x
        return s
    
    def learn(self, f, label):
        t = 1 if label.lower() == "spam" else -1
        s = self.raw_score(f)
        if t * s <= 0:
            for k,x in f.items():
                self.syn[k] = self.syn.get(k,0.0) + learning_rate * t * x
            self.syn["bias"] += learning_rate *t 

    # def train_once(self):
 
    #     for subj,body,sender in spam_train:
    #         f = extract_features(subj,body,sender)
    #         for k,x in f.items():
    #             self.syn[k] = self.syn.get(k, 0.0) + learning_rate *x
    #         self.syn["bias"] += learning_rate 
    #     for subj, body, sender in ham_train:
    #         f = extract_features(subj, body, sender)
    #         for k, x in f.items():
    #             self.syn[k] = self.syn.get(k, 0.0) - learning_rate * x
    #         self.syn["bias"] -= learning_rate


    def classify(self,f):
        s = self.raw_score(f)
        return "spam" if s>= threshold else "notspam"
    
detector = BrainCell()
# detector.train_once()

spam_train = [
            ("win cash now","click here","unknown@spam.biz"),
            ("act now", "buy now limited time offer", "promo@deals.com"),
            ("congratulations","you won a prize","lottery@scam.com"),
            ("risk free","get loan today","money@loan.biz")
        ]
ham_train = [
            ("meeting schedule", "please find attached", "boss@gmail.com"),
            ("project update", "our report is ready", "colleague@outlook.com"),
            ("lunch plan","where to eat?","friend@yahoo.com"),
            ("team call","join the meeting","teammate@gmail.com")
        ]
for _ in range(50):
    for subj, body,sender in spam_train + ham_train:
        label = "spam" if (subj,body,sender) in spam_train else "notspam"
        feats =extract_features(subj,body,sender)
        detector.learn(feats, label)

dataset = [(s,b,e,"spam") for s,b,e in spam_train] + [(s,b,e,"notspam") for s,b,e in ham_train]

def evaluate_accuracy(dataset, threshold_value=None):
    global threshold
    if threshold_value is not None:
        threshold = threshold_value

    correct = 0
    total = len(dataset)
    for subj, body, sender, label in dataset:
        feats = extract_features(subj,body, sender)
        pred = detector.classify(feats)
        if (pred =="spam" and label== "spam") or (pred== "notspam" and label == "notspam"):
            correct +=1
    return correct/ total * 100
# detector.learn(extract_features("win cash now","click here", "unknown@spam.biz"), "spam")
# detector.learn(extract_features("act now","buy now limited time offer", "promo@deals.com"), "spam")
# detector.learn(extract_features("meeting schedule","please find attached", "boss@gmail.com"), "notspam")
# detector.learn(extract_features("project update", "our report is ready", "colleague@outlook.com"), "notspam")

def is_spam(subject, body, sender="", learn_label = None):
    feats = extract_features(subject, body, sender)
    if learn_label:
        detector.learn(feats, learn_label)
    return detector.classify(feats)

# f_spam = extract_features("win cash now", "click here", "unknown@spam.biz")
# f_ham = extract_features("project update", "our report is ready", "colleague@outlook.com")
# print("spam score:", detector.raw_score(f_spam))
# print("ham score:", detector.raw_score(f_ham))

if __name__ == "__main__":
    
    try: 
        subject = input().strip()
        body = input().strip()
        sender = input().strip()
        print(is_spam(subject, body, sender))
    except EOFError:
        print("notspam")
