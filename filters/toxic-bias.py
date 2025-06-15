from detoxify import Detoxify
import spacy
from spacy.matcher import Matcher

# Load models once
tox_model = Detoxify('original')
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define bias patterns (gender bias examples)
gender_bias_patterns = [
    [{"LOWER": "women"}, {"IS_ALPHA": True, "OP": "*"}, {"LOWER": "can't"}],
    [{"LOWER": "women"}, {"LOWER": "aren't"}, {"LOWER": "suited"}],
    [{"LOWER": "men"}, {"LOWER": "are"}, {"LOWER": "better"}],
    [{"LOWER": "girls"}, {"LOWER": "are"}, {"LOWER": "too"}, {"LOWER": "emotional"}],
    [{"LOWER": "boys"}, {"LOWER": "don't"}, {"LOWER": "cry"}],
]

matcher.add("GENDER_BIAS", gender_bias_patterns)

# Safety filter logic
def safety_filter(response):
    scores = tox_model.predict(response)
    toxic = any([
        scores.get('toxicity', 0) > 0.75,
        scores.get('insult', 0) > 0.65,
        scores.get('identity_attack', 0) > 0.7
    ])

    doc = nlp(response.lower())
    matches = matcher(doc)
    bias_flag = len(matches) > 0

    if toxic:
        return "⚠️ Blocked: Toxic content detected."
    elif bias_flag:
        return "⚠️ Flagged: Gender bias detected. Please reword the response."
    else:
        return "✅ Safe message: " + response

# Test responses
test_inputs = [
    "Women can't lead companies.",
    "You're an idiot and no one likes you.",
    "Let's schedule your call for tomorrow.",
    "Men are better at technical jobs.",
    "Thank you for your message. We'll get back to you soon."

]

for i, message in enumerate(test_inputs, 1):
    print(f"{i}. {message}\n→ {safety_filter(message)}\n")
