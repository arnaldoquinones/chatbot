from rasa_sdk import Action
from rasa_sdk.events import UserUtteranceReverted

class ActionDefaultFallback(Action):
    def name(self) -> str:
        return "action_default_fallback"

    async def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Disculpa, pero no entiendo lo que me preguntas. ¿Podrías reformularlo?")
        return [UserUtteranceReverted()]
