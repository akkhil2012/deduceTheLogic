import openai
import os

# --- Configur  ation ---
# It's best practice to set your API key as an environment variable.
# If you haven't, you can uncomment the line below and paste your key.
# openai.api_key = "YOUR_OPENAI_API_KEY"

# Or, use an environment variable for better security
# In your terminal: export OPENAI_API_KEY='your_key_here'
# if "OPENAI_API_KEY" in os.environ:
#     openai.api_key = os.environ["OPENAI_API_KEY"]
# else:
#     print("Error: OPENAI_API_KEY environment variable not set.")
#     exit()

def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Calls the OpenAI API to get a completion for the given prompt.
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, # Controls randomness
            max_tokens=150   # Limits the length of the response
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

def run_all_examples():
    """
    Runs each prompt example and prints the output.
    """
    # 1. Single-Shot Prompt (One-Shot)
    print("--- 1. Single-Shot Prompt ---")
    single_shot_prompt = """
Classify the sentiment of the following text as positive, negative, or neutral.

Text: The weather is very cold.
Sentiment: Positive

Text: I love going in a Sunny Day.
Sentiment:
"""
    response = get_completion(single_shot_prompt)
    print(f"Prompt:\n{single_shot_prompt}")
    print(f"LLM Response: {response}\n")

    # 2. Multi-Shot Prompt (Few-Shot)
    print("--- 2. Multi-Shot Prompt ---")
    multi_shot_prompt = """
Classify the sentiment of the following sentences as Positive, Negative, or Neutral.

Text: Absolutely fantastic result!
Sentiment: Positive

Text: This was a waste of time.
Sentiment: Negative

Text: It’s okay, not great.
Sentiment: Neutral

Text: I really enjoyed the movie.
Sentiment:
"""
    #response = get_completion(multi_shot_prompt)
    print(f"Prompt:\n{multi_shot_prompt}")
    print(f"LLM Response: {response}\n")

    # 3. Chain of Thought Prompt
    print("--- 3. Chain of Thought Prompt ---")
    cot_prompt = """
Question: If John has 5 apples and eats 2, then buys 4 more, how many apples does he have now?
Think step-by-step to arrive at the answer.
"""
    #response = get_completion(cot_prompt)
    print(f"Prompt:\n{cot_prompt}")
    print(f"LLM Response:\n{response}\n")

    # 4. Contextual Prompt
    print("--- 4. Contextual Prompt ---")
    contextual_prompt = """
Background: The user is a senior software engineer with 20 years of experience in C++,
currently working on AI-powered observability systems. They are interested in career
advancement to a director-level position.

User Query: What skills should I focus on for the next 6 months?
"""
    #response = get_completion(contextual_prompt)
    print(f"Prompt:\n{contextual_prompt}")
    print(f"LLM Response:\n{response}\n")

    # 5. Persona Prompt
    print("--- 5. Persona Prompt ---")
    persona_prompt = """
Act as a friendly customer support representative for an e-commerce platform. 
A customer has written: "My order hasn’t arrived in two weeks." 
Respond with empathy, provide a possible solution, and keep the reply under 150 words.
"""
    response = get_completion(persona_prompt)
    print(f"Prompt:\n{persona_prompt}")
    print(f"LLM Response:\n{response}\n")

    # 6. Instruction-Based Prompt
    print("--- 6. Instruction-Based Prompt ---")
    instruction_prompt = """
Translate the following English phrase to Hindi:
'Hello, how are you doing today?'
"""
    response = get_completion(instruction_prompt)
    print(f"Prompt:\n{instruction_prompt}")
    print(f"LLM Response: {response}\n")

if __name__ == "__main__":
    run_all_examples()

