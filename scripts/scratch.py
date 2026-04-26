with open('app.py', 'r', encoding='utf-8') as f:
    c = f.read()

old_choices = '["nvidia/nemotron-4-340b-instruct:free", "z-ai/glm-4.5-air:free", "minimax/minimax-m2.5:free", "google/gemini-2.5-flash:free", "meta-llama/llama-3.3-70b-instruct:free"]'
new_choices = '["openrouter/free", "meta-llama/llama-3.1-8b-instruct:free", "google/gemini-2.0-flash-lite-preview-02-05:free", "deepseek/deepseek-r1-distill-llama-70b:free", "nvidia/llama-3.1-nemotron-70b-instruct:free"]'

c = c.replace(old_choices, new_choices)
c = c.replace('value="nvidia/nemotron-4-340b-instruct:free"', 'value="openrouter/free"')
c = c.replace('value="z-ai/glm-4.5-air:free"', 'value="openrouter/free"')
c = c.replace('value="minimax/minimax-m2.5:free"', 'value="openrouter/free"')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(c)

print('Models updated in app.py')

# Also fix risk_agent.py default model
with open('agents/council/risk_agent.py', 'r', encoding='utf-8') as f:
    r = f.read()
r = r.replace('model_name: str = "nvidia/nemotron-4-340b-instruct:free"', 'model_name: str = "openrouter/free"')
with open('agents/council/risk_agent.py', 'w', encoding='utf-8') as f:
    f.write(r)

# fraud_agent.py
with open('agents/council/fraud_agent.py', 'r', encoding='utf-8') as f:
    r = f.read()
r = r.replace('model_name: str = "z-ai/glm-4.5-air:free"', 'model_name: str = "openrouter/free"')
with open('agents/council/fraud_agent.py', 'w', encoding='utf-8') as f:
    f.write(r)

# policy_agent.py
with open('agents/council/policy_agent.py', 'r', encoding='utf-8') as f:
    r = f.read()
r = r.replace('model_name: str = "minimax/minimax-m2.5:free"', 'model_name: str = "openrouter/free"')
with open('agents/council/policy_agent.py', 'w', encoding='utf-8') as f:
    f.write(r)

print("Default models updated in head agents.")
