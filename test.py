import requests
import re

def reload_resources():
    try:
        response = requests.get('http://127.0.0.1:5000/reload_resource')
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Parse the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
    
def ask_question(query):
    try: #http://127.0.0.1:5000
        response = requests.get("http://127.0.0.1:5000/ask", params={"query": query})
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
if __name__ == "__main__":
    
    # result = reload_resources()
    # if result:
    #     print(result)
    
   questions = [
    "What is Software Development?",
]

for question in questions:
    response = ask_question(question)
    
    # if response:
    #     answer_text = response['answer'][0] if isinstance(response['answer'], list) else response['answer']
    #     query = response.get('query', question)
    #     metadata = response['answer'][1] if isinstance(response['answer'], list) and isinstance(response['answer'][1], dict) else {}

    #     print(f"Q: {query}\nA: {answer_text.strip()}\n")
        
    #     if metadata:
    #         print("⏳ Response Metadata:")
    #         for key, value in metadata.items():
    #             print(f"   - {key.replace('_', ' ').capitalize()}: {value:.4f}")
    if response:
        # Extract the answer text from the response
        answer_text = response['answer'][0] if isinstance(response['answer'], list) else response['answer']
        
        # Clean up the answer text by removing <html>, </html>, and <title>...</title>
        # answer_text_cleaned = re.sub(r"```html>|```", "", answer_text)
        # answer_text_cleaned = re.sub(r"(?i)<title>.*?</title>", "", answer_text_cleaned)
        # Remove everything before <!DOCTYPE html> and after </body></html>
        answer_text_cleaned = re.sub(r"^(.*?)<!DOCTYPE html>", "<!DOCTYPE html>", answer_text)
        answer_text_cleaned = re.sub(r"(?i)<title>.*?</title>", "", answer_text_cleaned)
        answer_text_cleaned = re.sub(r"</body></html>(.*?)$", "</body></html>", answer_text_cleaned)
        answer_text_cleaned = re.sub(r"```", "", answer_text)

        # Extract the query and metadata
        query = response.get('query', question)
        metadata = response['answer'][1] if isinstance(response['answer'], list) and isinstance(response['answer'][1], dict) else {}

        # Print the cleaned-up answer and other details
        print(f"Q: {query}\nA: {answer_text_cleaned.strip()}\n")

        if metadata:
            print("⏳ Response Metadata:")
            for key, value in metadata.items():
                print(f"   - {key.replace('_', ' ').capitalize()}: {value:.4f}")
        print("\n" + "="*50 + "\n")  # Separator for readability
        
        
# import re

# response = ask_question(question)

# if response:
#     # Extract the answer text from the response
#     answer_text = response['answer'][0] if isinstance(response['answer'], list) else response['answer']
    
#     # Clean up the answer text by removing <html>, </html>, and <title>...</title>
#     answer_text_cleaned = re.sub(r"(?i)<html.*?>|</html>", "", answer_text)
#     answer_text_cleaned = re.sub(r"(?i)<title>.*?</title>", "", answer_text_cleaned)

#     # Extract the query and metadata
#     query = response.get('query', question)
#     metadata = response['answer'][1] if isinstance(response['answer'], list) and isinstance(response['answer'][1], dict) else {}

#     # Print the cleaned-up answer and other details
#     print(f"Q: {query}\nA: {answer_text_cleaned.strip()}\n")

#     if metadata:
#         print("⏳ Response Metadata:")
#         for key, value in metadata.items():
#             print(f"   - {key.replace('_', ' ').capitalize()}: {value:.4f}")
