from src.translator import query_llm_robust, client
from unittest.mock import patch
import json
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
  # Compute embeddings for both expected answer and LLM response
  embeddings_expected = model.encode(expected_answer, convert_to_tensor=True)
  embeddings_response = model.encode(llm_response, convert_to_tensor=True)

  # Calculate cosine similarity between embeddings
  similarity_score = util.cos_sim(embeddings_expected, embeddings_response).item()
  return similarity_score

def is_valid_translation(expected_answer: str, llm_response: str) -> bool:
    similarity_score = eval_single_response_translation(expected_answer, llm_response)
    return similarity_score > 0.9

def test_chinese():
    is_english, translated_content = query_llm_robust("这是一条中文消息")
    assert is_english == False
    assert is_valid_translation("This is a Chinese message.", translated_content)

def test_english_example_1():
    is_english, translated_content = query_llm_robust("This is an example sentence in English.")
    assert is_english == True
    assert is_valid_translation("This is an example sentence in English.", translated_content)

def test_english_example_2():
    is_english, translated_content = query_llm_robust("Learning new skills is always rewarding.")
    assert is_english == True
    assert is_valid_translation("Learning new skills is always rewarding.", translated_content)

def test_english_example_3():
    is_english, translated_content = query_llm_robust("Can you help me understand this concept?")
    assert is_english == True
    assert is_valid_translation("Can you help me understand this concept?", translated_content)

def test_english_example_4():
    is_english, translated_content = query_llm_robust("It's a beautiful day outside.")
    assert is_english == True
    assert is_valid_translation("It's a beautiful day outside.", translated_content)

def test_english_example_5():
    is_english, translated_content = query_llm_robust("What time does the event start?")
    assert is_english == True
    assert is_valid_translation("What time does the event start?", translated_content)

def test_spanish_example():
    is_english, translated_content = query_llm_robust("Este es un ejemplo en español.")
    assert is_english == False
    assert is_valid_translation("This is an example in Spanish.", translated_content)

def test_french_example():
    is_english, translated_content = query_llm_robust("Ceci est une phrase en français.")
    assert is_english == False
    assert is_valid_translation("This is a sentence in French.", translated_content)

def test_japanese_example():
    is_english, translated_content = query_llm_robust("今日は良い天気ですね。")
    assert is_english == False
    assert is_valid_translation("The weather is nice today.", translated_content)

def test_german_example():
    is_english, translated_content = query_llm_robust("Wie geht es dir heute?")
    assert is_english == False
    assert is_valid_translation("How are you today?", translated_content)

def test_russian_example():
    is_english, translated_content = query_llm_robust("Это прекрасный день для прогулки.")
    assert is_english == False
    assert is_valid_translation("It is a beautiful day for a walk.", translated_content)

def test_unintelligible_example_1():
    is_english, translated_content = query_llm_robust("asldkfjasdf!")
    assert is_english == False
    assert is_valid_translation("Error: Unintelligible/malformed post or translation failure.", translated_content)

def test_unintelligible_example_2():
    is_english, translated_content = query_llm_robust("23lkj4nvlwe")
    assert is_english == False
    assert is_valid_translation("Error: Unintelligible/malformed post or translation failure.", translated_content)

def test_unintelligible_example_3():
    is_english, translated_content = query_llm_robust("##@!!weroie")
    assert is_english == False
    assert is_valid_translation("Error: Unintelligible/malformed post or translation failure.", translated_content)

def test_unintelligible_example_4():
    is_english, translated_content = query_llm_robust("blargl fzz#")
    assert is_english == False
    assert is_valid_translation("Error: Unintelligible/malformed post or translation failure.", translated_content)

def test_unintelligible_example_5():
    is_english, translated_content = query_llm_robust("wkejfw@#2309")
    assert is_english == False
    assert is_valid_translation("Error: Unintelligible/malformed post or translation failure.", translated_content)

@patch.object(client.chat.completions, 'create')
def test_unexpected_language(mocker):
    # we mock the model's response to return a random non-JSON message
    mocker.return_value.choices[0].message.content = "I don't understand your request"

    assert query_llm_robust("Hier ist dein erstes Beispiel.") == (False, "Error: Unintelligible/malformed post or translation failure.")

@patch.object(client.chat.completions, 'create')
def test_english_input(mocker):
    """Test that English input is returned as-is"""
    # Mock language detection to return English in JSON format
    mocker.side_effect = [
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"language": "english"})})()})()]})(),
    ]

    result = query_llm_robust("This is an English sentence.")
    assert result == (True, "This is an English sentence.")

@patch.object(client.chat.completions, 'create')
def test_successful_translation(mocker):
    """Test successful translation from German to English"""
    # Mock language detection to return German, then translation in JSON format
    mocker.side_effect = [
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"language": "german"})})()})()]})(),
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"translation": "This is your first example."})})()})()]})()
    ]

    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert result == (False, "This is your first example.")

@patch.object(client.chat.completions, 'create')
def test_empty_input(mocker):
    """Test handling of empty input"""
    result = query_llm_robust("")
    assert result == (False, "Error: Unintelligible/malformed post or translation failure.")

    result = query_llm_robust(None)
    assert result == (False, "Error: Unintelligible/malformed post or translation failure.")

@patch.object(client.chat.completions, 'create')
def test_failed_language_detection(mocker):
    """Test handling of failed language detection"""
    # Mock language detection to return empty JSON responses twice
    mocker.side_effect = [
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"language": ""})})()})()]})(),
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"language": ""})})()})()]})()
    ]

    result = query_llm_robust("こんにちは")
    assert result == (False, "Error: Unintelligible/malformed post or translation failure.")

@patch.object(client.chat.completions, 'create')
def test_failed_translation(mocker):
    """Test handling of failed translation"""
    # Mock successful language detection but failed translation with JSON
    mocker.side_effect = [
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"language": "japanese"})})()})()]})(),
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"translation": ""})})()})()]})(),
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"translation": ""})})()})()]})()
    ]

    result = query_llm_robust("こんにちは")
    assert result == (False, "Error: Unintelligible/malformed post or translation failure.")

@patch.object(client.chat.completions, 'create')
def test_malformed_json_response(mocker):
    """Test handling of malformed JSON response"""
    # Mock language detection to return invalid JSON
    mocker.side_effect = [
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': '{"language": "german"'})()})()]})()
    ]

    result = query_llm_robust("Hallo Welt")
    assert result == (False, "Error: Unintelligible/malformed post or translation failure.")

@patch.object(client.chat.completions, 'create')
def test_misnamed_json_field_response(mocker):
    """Test handling of misnamed fields in JSON response"""
    # Mock language detection to return invalid JSON
    mocker.side_effect = [
        type('obj', (), {'choices': [type('obj', (), {'message': type('obj', (), {'content': json.dumps({"name": "german"})})()})()]})()
    ]

    result = query_llm_robust("Hallo Welt")
    assert result == (False, "Error: Unintelligible/malformed post or translation failure.")