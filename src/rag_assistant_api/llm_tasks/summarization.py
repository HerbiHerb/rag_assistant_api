import openai


def summarize_text(text: str, model_name: str, temperature: float = 0.2):
    system_msg = f"""Fasse den Text der vom Nutzer übergeben wurde bitte so präzise wie möglich zusammen. Berücksichtige alle wichtigen Informationen 
    in der Zusammenfassung, vor allem dann, wenn es sich um wissenschaftliche Fakten handelt."""
    chat_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]
    response = openai.chat.completions.create(
        messages=chat_messages, model=model_name, temperature=temperature
    )
    return response
