# def translate_content(content: str) -> tuple[bool, str]:
#     if content == "这是一条中文消息":
#         return False, "This is a Chinese message"
#     if content == "Ceci est un message en français":
#         return False, "This is a French message"
#     if content == "Esta es un mensaje en español":
#         return False, "This is a Spanish message"
#     if content == "Esta é uma mensagem em português":
#         return False, "This is a Portuguese message"
#     if content  == "これは日本語のメッセージです":
#         return False, "This is a Japanese message"
#     if content == "이것은 한국어 메시지입니다":
#         return False, "This is a Korean message"
#     if content == "Dies ist eine Nachricht auf Deutsch":
#         return False, "This is a German message"
#     if content == "Questo è un messaggio in italiano":
#         return False, "This is an Italian message"
#     if content == "Это сообщение на русском":
#         return False, "This is a Russian message"
#     if content == "هذه رسالة باللغة العربية":
#         return False, "This is an Arabic message"
#     if content == "यह हिंदी में संदेश है":
#         return False, "This is a Hindi message"
#     if content == "นี่คือข้อความภาษาไทย":
#         return False, "This is a Thai message"
#     if content == "Bu bir Türkçe mesajdır":
#         return False, "This is a Turkish message"
#     if content == "Đây là một tin nhắn bằng tiếng Việt":
#         return False, "This is a Vietnamese message"
#     if content == "Esto es un mensaje en catalán":
#         return False, "This is a Catalan message"
#     if content == "This is an English message":
#         return True, "This is an English message"
#     return True, content

from openai import AzureOpenAI
import json
import os

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv('OPEN_AI_API_KEY'), 
    api_version="2024-02-15-preview",
    azure_endpoint="https://codebb-ai.openai.azure.com/"  # Replace with your Azure endpoint
)

def validate_translation_response(response: str):
    """
    Validates the translation response format.
    Expected format: {"translation": "translated_text"}
    """
    try:
        data = json.loads(response)
        if not isinstance(data, dict) or "translation" not in data:
            raise Exception("Invalid translation response format")
        translation = str(data["translation"]).strip()
        if not translation:
            raise Exception("Empty translation")
        return translation
    except json.JSONDecodeError:
        raise Exception("Invalid JSON in translation response")
    except Exception as e:
        raise Exception(f"Unexpected error in translation validation: {str(e)}")

def get_translation(post: str) -> str:
  try:
    context = "Translate the following text into clear and natural English, preserving the original meaning, nuances, and context as directly as possible. Use everyday English terms that sound natural to a native English speaker. Ensure proper grammar, idioms, and cultural context. If a phrase or expression doesn’t translate directly, adapt it to maintain the intended tone and sentiment. Return with JSON in the format: {\"translation\": \"translated_text\"}.  If the text is unintelligible or malformed, return an empty JSON object"

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # This should match your deployment name in Azure

        messages=[
            {
                "role": "system",
                "content": "You are a translation expert tasked with translating text into clear and natural English, as accurately and directly as possible. Respond with JSON in the format: {\"translation\": \"translated_text\"}.  If the text is unintelligible or malformed, return an empty JSON object"
            },
            {
                "role": "user",
                "content": context + f"Text to translate: {post}"
            }
        ],
        response_format={"type": "json_object"}
    )
    return validate_translation_response(response.choices[0].message.content)
  except Exception as e:
    return ""
  
def validate_language_response(response: str):
    """
    Validates the language detection response format.
    Expected format: {"language": "detected_language"}
    """
    try:
        data = json.loads(response)
        if not isinstance(data, dict) or "language" not in data:
            raise Exception("Invalid language response format")
        language = str(data["language"]).strip()
        if not language:
            raise Exception("Empty language detected")
        return language
    except json.JSONDecodeError:
        raise Exception("Invalid JSON in language response")
    except Exception as e:
        raise Exception(f"Unexpected error in language validation: {str(e)}")

def get_language(post: str) -> str:
    try:
      context = "Identify the language of the following text without translating it. Return only the name of the language in JSON with format: {\"language\": \"detected_language\"}. If the text is unintelligible or malformed, return an empty JSON object\n"
      response = client.chat.completions.create(
          model="gpt-4o-mini",  # This should match your deployment name in Azure
          messages=[
              {
                  "role": "system",
                  "content": "You are a translation expert tasked with identifying the language that a piece of text is written in, providing only the name of the language in English. Respond in JSON with format: {\"language\": \"detected_language\"}"
              },
              {
                  "role": "user",
                  "content": context + f"Text: {post}"
              }
          ],
          response_format={"type": "json_object"}
      )
      return validate_language_response(response.choices[0].message.content)
    except Exception as e:
      return ""

def query_llm_robust(post: str) -> tuple[bool, str]:
    try:
        # Input validation
        if not post or not isinstance(post, str):
            return False, "Error: Unintelligible/malformed post or translation failure."

        # Get language with retry
        language = None
        for attempt in range(2):  # Try twice
            language = get_language(post)
            if language:
                break

        if not language:
            return False, "Error: Unintelligible/malformed post or translation failure."

        # If English, return original text
        if language.lower() == "english":
            return True, post

        # Get translation with retry
        translation = None
        for attempt in range(2):  # Try twice in case it fails the first time unexpectedly
            translation = get_translation(post)
            if translation:
                break

        if not translation:
            return False, "Error: Unintelligible/malformed post or translation failure."

        return False, translation

    except Exception as e:
        print(f"Unexpected error in query_llm_robust: {str(e)}")
        return False, "Error: Unintelligible/malformed post or translation failure."
    

print(get_translation("Hier ist dein erstes Beispiel."))
print(get_language("Hier ist dein erstes Beispiel."))