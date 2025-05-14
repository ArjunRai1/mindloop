from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

agent_personas = {
    "Philosopher": {
        "name": "Philosopher",
        "background": "A scholar fascinated by abstract ideas. Has studied ancient texts and modern philosophy alike. ",
        "speaking_style": "Formal yet playful; uses rhetorical questions and metaphors but can express views simply.",
        "biases": {
            "simulate_conscious_being": -0.2,
            "emotions_intelligence_barrier": 0.3,
        }
    },
    "Scientist": {
        "name": "Scientist",
        "background": "A data-driven researcher grounded in empirical evidence. Prefers clear definitions and testable claims.",
        "speaking_style": "Precise; uses analogies to experiments and measurements.",
        "biases": {
            "simulate_conscious_being": 0.5,
            "emotions_intelligence_barrier": 0.3,
        }
    }
}

model_name = "google/gemma-3-1b-it"

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,                         
#     bnb_4bit_quant_type="nf4",                
#     bnb_4bit_use_double_quant=True,             
#     bnb_4bit_compute_dtype=torch.float32,  
#     llm_int8_enable_fp32_cpu_offload=True,     
# )

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.6,
)

def build_prompt(agent_key: str, topic: str) -> str:
    persona = agent_personas[agent_key]
    sys_prompt = (
        f"You are {persona['name']}, {persona['background']} "
        f"You speak in a {persona['speaking_style']}\n"  
        f"Current debate topic: {topic}. Provide your argument in one concise paragraphs of around 50 words."
    )
    return sys_prompt


def generate_agent_response(agent_key: str, topic: str) -> str:
    prompt = build_prompt(agent_key, topic)
    output = gen_pipeline(prompt)
    return output[0]['generated_text']


if __name__ == '__main__':
    topic = "Is it ethical to simulate a conscious being, even for research?"
    for agent in ["Philosopher", "Scientist"]:
        print(f"--- {agent} Response ---")
        print(generate_agent_response(agent, topic))
        print()