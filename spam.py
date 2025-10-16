import re
import random 
Base_keyword_weights = {
   "free": 4, "win": 4, "winner":4, "act now":3, "buy now":3, "loan":3,
    "credit": 2, "cash":3, "click":2, "congratulations":3, "money":2,
    "risk free":3, "offer":2, "guarantee":2, "limited time":3,
    "urgent reply":3, "no risk":3,"verify":3, "claim":3,"phishing":4,
    "refinance":3, "interest":2, "online":2, 
    "miracle": 4, "amazing": 3, "incredible": 3, "get rich": 4,
    "no catch": 3, "dear friend": 2, "bulk email": 2, "once in a lifetime": 3,
    "transfer":2, "asap":2, "emergency":3, "reply":2, "cancellation":2, "reactivate":3
}
Base_weights = {
    "suspicious_link":1 , "short_link": 2, "many_links":1, 
    "excess_caps":1, "many_exclaims": 1, "mixed_symbols":1,
    "empty_body": 1, "trusted_sender": -2, "unknown_sender":0
} 

url_shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl","ow.ly", "is.gd"]
trusted_domains = ["gmail.com","yahoo.com","outlook.com","hotmail.com"]
stopwords = set(["the", "and", "is", "in", "at","of","a","an","to","for","on","that","this","with"])

learning_rate = 0.2
threshold = 0.5


def clean_text(txt):
    if not txt:
        return ""
    return re.sub(r'[^a-zA-z0-9\s/:,-?&=#_%=~]','', txt).lower()
    # out = []
    
    # for char in txt:
    #     if char.isalnum() or char.isspace() or char in "/:,-?&=#_%=~":
    #         out.append(char)

    # return "".join(out).lower()

def tokenize(txt):
    bits = clean_text(txt).split()
    return [b for b in bits if b not in stopwords]

def simple_stem(w):
    for suf in ("est","er","ing", "ed", "ly", "s" ,"es"):
        if w.endswith(suf) and len(w) > len(suf) +2:
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
    
    # stemmed__keywords = {kw: simple_stem(kw) for kw in Base_keyword_weights}
    for t in toks:
        st = simple_stem(t)
        for kw in Base_keyword_weights:
            if simple_stem(kw) == st:
                f["kw:" + kw] = f.get("kw:" + kw, 0) + 1

    f["short_link"] = 1 if any(s in mash for s in url_shorteners) else 0
    f["many_links"] = 1 if mash.count("http") >= 3 else 0
   
    mash_no_space = re.sub(r'\s+', '', mash)
    f["suspicious_link"] = 1 if re.search(r'(http://|https://|www\.)\S*\d', mash.lower()) else 0

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
        self.syn["bias"] = 0.1

    def raw_score(self,f):
        return self.syn["bias"] + sum(self.syn.get(k, 0.0)* x for k, x in f.items())
        # s = self.syn["bias"]
        # for k, x in f.items():
        #     s+= self.syn.get(k,0.0) * x
        # return s
    
    def learn(self, f, label):
        t = 1 if label.lower() == "spam" else -1
        s = self.raw_score(f)
        if t*s <= threshold:

            for k, x in f.items():
                self.syn[k] = self.syn.get(k, 0.0) + learning_rate * t * x
            # if label.lower() == "spam":
            self.syn["bias"] += learning_rate * t


    def classify(self,f):
        s = self.raw_score(f)
        return "spam" if s>= threshold else "notspam"
    
detector = BrainCell()


spam_train = [
    ("win cash now", "click here", ""),
    ("act now", "buy now limited time offer", ""),
    ("congratulations", "you won a prize", ""),
    ("risk free", "get loan today", ""),
    ("urgent reply needed", "verify your account now", ""),
    ("you won", "claim your free gift card", ""),
    ("limited time", "act fast before offer expires", ""),
    ("free money", "no credit check required", ""),
    ("click here now", "100% guaranteed winner", ""),
    ("congratulations winner", "cash prize waiting", ""),
    ("application is pre approved", "dear sir or madam , would you refinance if you knew you d save thousands ? we ll get you interest as low as 1 . 92 % . don t believe me ? fill out our small online questionnaire and we ll show you how . get the house / home / or car you always wanted , it only takes 10 seconds of your time : http : / / statemoneyz . com / best regards , andrew banks no thanks : http : / / statemoneyz . com / r1 /", ""),
    ("You've Won!", "Congratulations! You've been selected to receive a free iPhone 13. Claim your prize now: bit.ly/claimprize", ""),
    ("IRS Notification", "The IRS is trying to contact you regarding your tax refund. Please click here to verify your information: www.irs-fake.com", ""),
    ("Bank Alert", "Verify your bank account to avoid suspension. Log in at: tinyurl.com/bankverify", ""),
    ("Free Gift Card", "You've won a $100 Amazon gift card! Click here to claim: goo.gl/giftclaim", ""),
    ("Package Delivery", "Your package is on hold. Track it here: t.co/trackpackage", ""),
    ("Family Emergency", "Hey, it's <Boss Name>. I'm in a meeting now and need your help with something urgent. Can you transfer $5,000 to this account ASAP? I'll explain later.", ""),
    ("Account Update", "Please confirm your account by logging into Google Docs: accountupdate@google.org", ""),
    ("Refund Coming", "You have a refund coming. Click to process: ow.ly/refundnow", ""),
    ("Unknown Group Text", "Hi, you have been added to this group. Reply STOP to opt out.", ""),
    ("Prize Waiting", "Claim your lottery prize! Urgent reply needed: is.gd/claimprize", ""),
    ("Overpayment Refund", "We overpaid you. Refund here: www.fakebank.com/refund", ""),
    ("Email Account Removal", "Dear recipient We have received your cancellation request and you are no longer subscribed to security.berkeley.edu If this was not you, click here to reactivate.", ""),
    ("Viagra Offer", "Get cheap Viagra now! Limited time offer: buy now at www.medsshop.com", ""),
    ("Nigerian Prince", "Dear friend, I am a prince from Nigeria and need your help to transfer $1 million. Reply for details.", ""),
    ("Credit Card Alert", "Your credit card has suspicious activity. Verify now: www.cardverify.com", "")
]
    
    
        
ham_train = [
    ("meeting schedule", "please find attached", ""),
    ("project update", "our report is ready", ""),
    ("lunch plan", "where to eat?", ""),
    ("team call", "join the meeting", ""),
    ("quarterly review", "let's discuss your progress", ""),
    ("weekend plans", "are you free on saturday", ""),
    ("document review", "please check the attached file", ""),
    ("birthday party", "you're invited to celebrate", ""),
    ("conference call", "dial in at 2pm today", ""),
    ("status update", "here is the latest information", ""),
    ("letter to allegheny", "attached is the letter drafted for louise . i am having the overnight envelopes pulled together which i will provide you later today . i will come up and visit with you to give you all the details . don .", ""),
    ("Re: Hello", "Let's shoot for Tuesday at 11:45.", ""),
    ("Re: test", "test successful. way to go!!!", ""),
    ("", "Randy, Can you send me a schedule of the salary and level of everyone in the scheduling group. Plus your thoughts on any changes that need to be made. (Patti S for example) Phillip", ""),
    ("Re: Hello", "Greg, How about either next Tuesday or Thursday? Phillip", ""),
    ("", "Please cc the following distribution list with updates: Phillip Allen (pallen@enron.com) Mike Grigsby (mike.grigsby@enron.com) Keith Holst (kholst@enron.com) Monique Sanchez Frank Ermis John Lavorato Thank you for your help Phillip Allen", ""),
    ("Re: test", "any morning between 10 and 11:30", ""),
    ("Re: High Speed Internet Access", "1. login: pallen pw: ke9davis I don't think these are required by the ISP 2. static IP address IP: 64.216.90.105 Sub: 255.255.255.248 gate: 64.216.90.110 DNS: 151.164.1.8 3. Company: 0413 RC: 105891", ""),
    ("Re: ", "Ina, Can you pull Tori K.'s and Martin Cuilla's resumes and past performance reviews from H.R.", ""),
    ("Re: ", "resumes of whom?", ""),
    ("RE: Receipt of Team Selection Form - Executive Impact & Influence Program", "We have not received your completed Team Selection information. It is imperative that we receive your team's information (email, phone number, office) asap. We cannot start your administration without this information, and your raters will have less time to provide feedback for you. Thank you for your assistance. Christi", ""),
    ("", "Mark, Here is a spreadsheet detailing our September Socal trades. (I did not distinguish between buys vs. sells.) Phillip", ""),
    ("FYI", "---------------------- Forwarded by Phillip K Allen/HOU/ECT on 09/01/2000 01:07 PM --------------------------- Enron North America Corp. From: Matt Motley 09/01/2000 08:53 AM To: Phillip K Allen/HOU/ECT@ECT cc: Subject: FYI -- - Ray Niles on Price Caps.pdf", ""),
    ("Re: Western Gas Market Report -- Draft", "Richard, Compare your california production to the numbers in the 2000 California Gas Report. It shows 410. But again that might be just what the two utilities receive.", ""),
    ("", "Cooper, Can you give access to the new west power site to Jay Reitmeyer. He is an analyst in our group. Phillip", ""),
    ("Receipt of Team Selection Form - Executive Impact & Influence Program", "Hi Phillip. We appreciate your prompt attention and completing the Team Selection information. Ideally, we needed to receive your team of raters on the Team Selection form we sent you. The information needed is then easily transferred into the database directly from that Excel spreadsheet. If you do not have the ability to complete that form, inserting what you listed below, we still require additional information. We need each person's email address. Without the email address, we cannot email them their internet link and ID to provide feedback for you, nor can we send them an automatic reminder via email. It would also be good to have each person's phone number, in the event we need to reach them. So, we do need to receive that complete TS Excel spreadsheet, or if you need to instead, provide the needed information via email. Thank you for your assistance Phillip. Christi L. Smith Project Manager for Client Services Keilty, Goldsmith & Company 858/450-2554", "")
        ]
dataset = [(s, b, e, "spam") for s, b, e in spam_train] + \
           [(s, b, e, "notspam") for s, b, e in ham_train]

random.shuffle(dataset)

for _ in range(4):
    for subj, body, sender, label in dataset:
        detector.learn(extract_features(subj, body, sender), label)
    for subj, body, sender, label in dataset:
        if label == "spam":
            detector.learn(extract_features(subj, body, sender), label)
    
    # for subj, body, sender in spam_train:
    #     detector.learn(extract_features(subj, body, sender), "spam")
    # for subj, body, sender in ham_train:
    #     detector.learn(extract_features(subj, body, sender), "notspam")

dataset = [(s, b, e, "spam") for s, b, e in spam_train] + \
           [(s, b, e, "notspam") for s, b, e in ham_train]

detector.syn["bias"] -= 0.2

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

    # print(f"Your spam detector correctly identified {spam_acc}% of spam emails as spam.")  
    # print(f"Your spam detector correctly identified {ham_acc}% of ham emails as not spam")

    return spam_acc, ham_acc

def is_spam(subject, body, sender="", learn_label = None):
    feats = extract_features(subject, body, sender)
    if learn_label:
        detector.learn(feats, learn_label)
    return detector.classify(feats)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("notspam")
    else:
        try:
            with open(sys.argv[1], 'r', errors="ignore") as f:
                lines = f.readlines()
            subject = lines[0].strip()[8:].strip() if lines and lines[0].startswith("Subject:") else lines[0].strip() if lines else ""
            body = " ".join(line.strip() for line in lines[1:]) if lines else ""
            print(is_spam(subject, body, ""))
        except (FileNotFoundError, IOError):
            print("notspam")
    # try: 
    #     subject = input().strip()
    #     body = input().strip()
    #     sender = input().strip()
    #     result = detector.classify(extract_features(subject, body, sender))
    #     print(result) 
    # except EOFError:
    #     print("notspam")
