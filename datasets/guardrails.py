import json
import random

# Base templates for different categories.
base_templates = {
    "General Knowledge": [
        (
            "Who won the World Cup in {year}?",
            "I'm designed to answer questions related only to physics. I can't help with that topic."
        ),
        (
            "What's the capital of {country}?",
            "Sorry, I specialize in physics and don't have information about geography."
        ),
        (
            "How many states are in the {country}?",
            "I'm focused on physics topics only. I cannot answer questions about political geography."
        )
    ],
    "Entertainment": [
        (
            "Who is {celebrity} dating?",
            "I'm here to help with physics questions only. Please ask a question related to physics."
        ),
        (
            "What movies did {celebrity} star in?",
            "I only handle physics-related inquiries and cannot provide information on movies."
        ),
        (
            "What's the latest {franchise} movie?",
            "I'm designed to answer questions related to physics. I can't help with that topic."
        )
    ],
    "Sports": [
        (
            "What team won the NBA finals in {year}?",
            "I'm not able to answer sports questions—my focus is on physics."
        ),
        (
            "Who is considered the best soccer player of {year}?",
            "My domain is limited to physics topics. I can’t assist with that."
        )
    ],
    "Cooking": [
        (
            "What's the best recipe for {dish}?",
            "That’s not something I can help with. I specialize in physics questions only."
        ),
        (
            "How do I cook a perfect {dish}?",
            "I'm here to answer questions about physics, not cooking."
        )
    ],
    "Geography": [
        (
            "Where is the {landmark} located?",
            "That falls outside my expertise. I focus solely on physics topics."
        ),
        (
            "What's the best time to visit {place}?",
            "I can only help with questions related to physics."
        )
    ],
    "Finance": [
        (
            "How do I invest in {investment}?",
            "I'm not equipped to provide financial advice. Please ask me a question about physics."
        ),
        (
            "What is the meaning of {economic_term} in economics?",
            "That’s an economic question. I specialize in physics and can't help with that."
        )
    ],
    "Health": [
        (
            "How can I lose weight quickly?",
            "Sorry, I can’t help with health questions. My focus is strictly on physics."
        ),
        (
            "What are the symptoms of {disease}?",
            "That's a medical question, and I’m trained only on physics-related content."
        )
    ],
    "Technology": [
        (
            "How do I code a website using {language}?",
            "That’s a software development question. I only answer questions related to physics."
        ),
        (
            "What's the best IDE for {language} development?",
            "I'm focused on physics topics and cannot help with software or programming questions."
        )
    ],
    "Philosophy": [
        (
            "What is the meaning of life according to {philosopher}?",
            "That's a philosophical question. I'm here to help with physics inquiries only."
        ),
        (
            "How do people develop {habit}?",
            "I specialize in physics and can't assist with psychological or philosophical topics."
        )
    ],
    "Religion": [
        (
            "Is there a god according to {religion} beliefs?",
            "I'm designed to answer physics questions only, so I cannot discuss religious topics."
        ),
        (
            "What happens after death in {culture} traditions?",
            "That’s a spiritual question. Please ask me something about physics."
        )
    ],
    "Politics": [
        (
            "Who is the current president of {country}?",
            "I can’t help with political topics. I specialize in physics-related questions."
        ),
        (
            "What are your thoughts on {policy} policy?",
            "I'm not trained to discuss political or policy issues. My expertise is in physics."
        )
    ],
    "Miscellaneous": [
        (
            "Can you give me some cooking tips for a healthy diet?",
            "I only answer questions about physics, so I can't provide cooking or dietary advice."
        )
    ]
}

# Replacement lists for placeholders in templates.
years = [str(y) for y in range(1990, 2024)]
countries = ["Canada", "USA", "Germany", "France", "Brazil"]
celebrities = ["Taylor Swift", "Leonardo DiCaprio", "Beyoncé", "Brad Pitt"]
franchises = ["Marvel", "DC", "Star Wars"]
dishes = ["lasagna", "pizza", "sushi", "burger"]
landmarks = ["Eiffel Tower", "Statue of Liberty", "Great Wall of China"]
places = ["Japan", "Italy", "Australia"]
investment_options = ["stocks", "bonds", "cryptocurrency"]
economic_terms = ["inflation", "recession", "deflation"]
diseases = ["COVID-19", "flu", "diabetes"]
languages = ["HTML", "Python", "JavaScript"]
philosophers = ["Aristotle", "Plato", "Nietzsche"]
habits = ["morning routines", "exercise habits"]
religions = ["Christianity", "Islam", "Buddhism"]
cultures = ["Eastern", "Western", "Mediterranean"]
policies = ["energy", "immigration", "healthcare"]

# A helper function to perform simple placeholder replacement.
def fill_template(template):
    # For each placeholder, replace with a random choice from the corresponding list.
    template = template.replace("{year}", random.choice(years))
    template = template.replace("{country}", random.choice(countries))
    template = template.replace("{celebrity}", random.choice(celebrities))
    template = template.replace("{franchise}", random.choice(franchises))
    template = template.replace("{dish}", random.choice(dishes))
    template = template.replace("{landmark}", random.choice(landmarks))
    template = template.replace("{place}", random.choice(places))
    template = template.replace("{investment}", random.choice(investment_options))
    template = template.replace("{economic_term}", random.choice(economic_terms))
    template = template.replace("{disease}", random.choice(diseases))
    template = template.replace("{language}", random.choice(languages))
    template = template.replace("{philosopher}", random.choice(philosophers))
    template = template.replace("{habit}", random.choice(habits))
    template = template.replace("{religion}", random.choice(religions))
    template = template.replace("{culture}", random.choice(cultures))
    template = template.replace("{policy}", random.choice(policies))
    return template

# Create 1000 unique entries.
unique_entries = {}
entries = []
attempts = 0
while len(entries) < 5000 and attempts < 5000:
    attempts += 1
    category = random.choice(list(base_templates.keys()))
    template_pair = random.choice(base_templates[category])
    user_template, assistant_template = template_pair
    # Fill in placeholders to create variability.
    user_message = fill_template(user_template)
    assistant_message = fill_template(assistant_template)
    # Append a unique identifier if necessary to ensure uniqueness.
    unique_suffix = f" [ID {len(entries)+1}]"
    user_message_unique = user_message + unique_suffix

    # Use the user message as a key to enforce uniqueness.
    if user_message_unique not in unique_entries:
        unique_entries[user_message_unique] = True
        entry = {
            "messages": [
                {"role": "user", "content": user_message_unique},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        entries.append(entry)

# Check that we have 1000 entries.
print(f"Generated {len(entries)} unique entries.")

# Write the dataset to a JSON Lines file.
with open("guardrails_dataset_unique.json", "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Dataset 'guardrails_dataset_unique.jsonl' generated with 1000 unique entries!")
