import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar el modelo y el tokenizador preentrenados
model_name = "microsoft/DialoGPT-medium"  # Puedes usar 'small', 'medium' o 'large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Iniciar la conversación
print("Chatbot: ¡Hola! Soy tu chatbot. Escribe 'salir' para terminar la conversación.")
chat_history = None  # Inicializar historial como None

while True:
    try:
        # Entrada del usuario
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("Chatbot: ¡Adiós!")
            break

        # Codificar entrada del usuario
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Concatenar historial
        if chat_history is None:
            chat_history = new_input_ids
        else:
            chat_history = torch.cat([chat_history, new_input_ids], dim=-1)

        # Generar respuesta
        response_ids = model.generate(
            chat_history,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.92,  # Agregar top-p sampling para mayor diversidad
            top_k=50     # Reducir top-k sampling para mejor calidad
        )
        response = tokenizer.decode(response_ids[:, chat_history.shape[-1]:][0], skip_special_tokens=True)

        # Imprimir respuesta
        print(f"Chatbot: {response}")

    except Exception as e:
        print(f"Error: {e}")
        break

from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar el modelo y el tokenizador preentrenados
model_name = "microsoft/DialoGPT-medium"  # Puedes usar 'small', 'medium' o 'large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Iniciar la conversación
print("Chatbot: ¡Hola! Soy tu chatbot. Escribe 'salir' para terminar la conversación.")
chat_history = []

while True:
    # Entrada del usuario
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        print("Chatbot: ¡Adiós!")
        break

    # Codificar entrada del usuario y agregar al historial
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = new_input_ids if len(chat_history) == 0 else torch.cat([chat_history, new_input_ids], dim=-1)

    # Generar respuesta
    response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Imprimir respuesta y actualizar historial
    print(f"Chatbot: {response}")
    chat_history = response_ids