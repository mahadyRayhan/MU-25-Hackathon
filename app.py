import os
import re
import cv2  # OpenCV library
import base64
from PIL import Image
import io
import time
import json
import os
import uuid
import re
from flask import Flask, request, jsonify, send_from_directory
from Q_A import GeminiQuestion_and_Answering

# Optional: Check that the API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    raise Exception("GOOGLE_API_KEY is not set. Please check your .env file.")

app = Flask(__name__)

# Initialize the GeminiQuestion_and_Answering system at startup
gemini_qa = GeminiQuestion_and_Answering()
gemini_qa.load_resources(load_resource=True)

# --- Configuration for image saving ---
WEBCAM_IMAGE_FOLDER = 'emo_images'
if not os.path.exists(WEBCAM_IMAGE_FOLDER):
    os.makedirs(WEBCAM_IMAGE_FOLDER)
WEBCAM_IMAGE_FILENAME = 'webcam_latest.png'
WEBCAM_IMAGE_PATH = os.path.join(WEBCAM_IMAGE_FOLDER, WEBCAM_IMAGE_FILENAME)

JSON_TREE_PATH = "QA_viz.json"

emotions = ["neutral", "confused"]
current_emotion_index = 0

# --- Function to build question tree ---
THRESHOLD = 0.2  # Adjust as needed for stricter or looser matching
# Define a set of common English stopwords to filter out during tokenization.
STOPWORDS = {
    "what", "is", "your", "a", "an", "the", "and", "or", "but", "if",
    "of", "for", "to", "in", "with", "on", "at", "from", "as", "it",
    "are", "this", "that", "was", "were", "be", "been", "has", "have", "had"
}

def tokenize(text):
    tokens = set(re.findall(r'\w+', text.lower()))
    filtered_tokens = {token for token in tokens if token not in STOPWORDS}
    return filtered_tokens

def context_overlap_ratio(question, node):
    node_text = node.get('question', '') + " " + node.get('answer', '')
    question_tokens = tokenize(question)
    node_tokens = tokenize(node_text)
    if not question_tokens:
        return 0.0
    common_tokens = question_tokens.intersection(node_tokens)
    return len(common_tokens) / len(question_tokens)

def find_deepest_context_node(node, question, threshold):
    ratio_here = context_overlap_ratio(question, node)
    if ratio_here < threshold:
        # Not in context with this node at all
        return None, 0.0
    
    # Node is in context; see if there's a child that is also in context, possibly with a better ratio.
    best_node = node
    best_ratio = ratio_here
    
    for child in node.get('children', []):
        candidate_node, candidate_ratio = find_deepest_context_node(child, question, threshold)
        # Only consider children that are themselves in context
        if candidate_node is not None and candidate_ratio >= threshold:
            # Prefer the child if it meets threshold and is deeper
            # We'll pick the child with the highest ratio among siblings.
            if candidate_ratio > best_ratio:
                best_node = candidate_node
                best_ratio = candidate_ratio

    return best_node, best_ratio

def find_best_placement_in_forest(forest, question, threshold):
    best_node = None
    best_ratio = 0.0
    
    for root in forest:
        node, ratio = find_deepest_context_node(root, question, threshold)
        if ratio > best_ratio:
            best_node = node
            best_ratio = ratio
    
    return best_node, best_ratio

def update_json_tree(file_path, question, answer, threshold=THRESHOLD):
    # Load existing tree data; initialize as an empty list if the file is absent or invalid.
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                tree = json.load(f)
                if not isinstance(tree, list):
                    tree = []
        except json.JSONDecodeError:
            tree = []
    else:
        tree = []

    # Create a new node with a unique identifier and an empty children list.
    new_id = str(uuid.uuid4())
    new_node = {
        "id": new_id,
        "question": question,
        "answer": answer,
        "children": []
    }

    # Find the best placement among all roots
    best_node, best_ratio = find_best_placement_in_forest(tree, question, threshold)

    if best_node is not None and best_ratio >= threshold:
        # Attach under the best matching node
        best_node.setdefault('children', []).append(new_node)
        print(f"Placed '{question}' under parent '{best_node['question']}' (overlap ratio={best_ratio:.2f}).")
    else:
        # No suitable parent found => new root node
        tree.append(new_node)
        if best_node is None:
            print(f"No suitable parent found for '{question}'; created new root.")
        else:
            print(f"Overlap ratio below threshold ({best_ratio:.2f}); created new root for '{question}'.")

    # Write the updated tree back to the JSON file with pretty printing.
    with open(file_path, 'w') as f:
        json.dump(tree, f, indent=4)

    return new_id

# --- Function to build question tree ---

def get_emotion():
    """Returns the current emotion (cycles through emotions list)."""
    global current_emotion_index # Use the global index
    emotion = emotions[current_emotion_index]
    current_emotion_index = (current_emotion_index + 1) % len(emotions) # Move to the next emotion, cycle back to 0 if needed
    return emotion # **Return the emotion value**

def capture_webcam_frame():
    """Captures a single frame from the webcam and saves it.""" # Modified to capture single frame
    save_folder = WEBCAM_IMAGE_FOLDER # Use the configured folder
    image_filename = WEBCAM_IMAGE_FILENAME # Use the configured filename

    # Open the webcam (0 is the default webcam on most systems)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None # Indicate failure

    try:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            return None # Indicate failure

        # Save the captured image
        cv2.imwrite(os.path.join(save_folder, image_filename), frame) # Save to configured path

        print(f"Image saved as {os.path.join(save_folder, image_filename)}") # Print save message
        return image_filename # Return filename on success

    except Exception as e:
        print(f"Error in capture_webcam_frame: {e}") # More specific error message
        return None # Indicate failure
    finally:
        # Release the webcam
        if 'cap' in locals() and cap.isOpened(): # Check if cap was successfully opened before releasing
            cap.release()

# Serve the HTML file
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/reload_resource", methods=["GET"])
def reload_resource():
    try:
        gemini_qa.load_resources(load_resource=True)  # Reload the resources
        return jsonify({"detail": "Resources reloaded successfully."}), 200
    except Exception as e:
        return jsonify({"detail": f"Failed to reload resources: {str(e)}"}), 500


# ADD THIS ROUTE TO SERVE STATIC FILES
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/get_current_emotion', methods=['GET'])
def get_current_emotion_route():
    """Endpoint to get the current emotion."""
    emotion_value = get_emotion()  # Call the CORRECTED get_emotion function
    print(f"Current emotion from route: {emotion_value}") # Keep print for server-side logging if you want
    return jsonify({"emotion": emotion_value}), 200

@app.route("/ask", methods=["GET"])
def ask_question():
    query = request.args.get('query')
    if not query:
        return jsonify({"detail": "Query not provided"}), 400
    try:
        answer_result = gemini_qa.get_answer(query)
        
        if isinstance(answer_result, tuple):
            answer_text = answer_result[0]
        elif isinstance(answer_result, dict):
            answer_text = answer_result.get('text', str(answer_result)) 
        else:
            answer_text = str(answer_result)

        answer_text = answer_text[8:]
        answer_text = re.sub(r"^(.*?)<!DOCTYPE html>", "<!DOCTYPE html>", answer_text)
        answer_text = re.sub(r"(?i)<h1.*?>.*?</h1>", "", answer_text, flags=re.DOTALL)
        answer_text = re.sub(r"</body></html>(.*?)$", "</body></html>", answer_text)
        answer_text = re.sub(r"```", "", answer_text)
        
        update_json_tree(JSON_TREE_PATH, query, answer_text)
        
        return jsonify({"query": query, "answer": answer_text}), 200
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


@app.route('/send_confused_query', methods=['POST'])
def send_confused_query():
    print("SEND CONFUSED QUERY")
    data = request.get_json() # Get JSON data from the request body
    if not data or 'question' not in data or 'emotion' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid request data'}), 400

    query = data['question']
    emotion = data['emotion']
    
    if not query:
        return jsonify({"detail": "Query not provided"}), 400
    try:
        answer_result = gemini_qa.get_answer(query, emotion) # Get the result from GeminiQuestion_and_Answering
        
        # --- Ensure we only send the text part of the answer ---
        if isinstance(answer_result, tuple): # Check if it's a tuple (common if you return text and some other info)
            answer_text = answer_result[0] # Assuming the first element is the text
        elif isinstance(answer_result, dict): # Check if it's a dictionary (maybe with 'text' key?)
            answer_text = answer_result.get('text', str(answer_result)) # Try to get 'text', if not, stringify the whole dict as fallback
        else: # If it's already a string or something else, try to convert to string as fallback
            answer_text = str(answer_result)

        answer_text = answer_text[8:]
        answer_text = re.sub(r"^(.*?)<!DOCTYPE html>", "<!DOCTYPE html>", answer_text)
        answer_text = re.sub(r"(?i)<h1.*?>.*?</h1>", "", answer_text, flags=re.DOTALL)
        answer_text = re.sub(r"</body></html>(.*?)$", "</body></html>", answer_text)
        answer_text = re.sub(r"```", "", answer_text)
        answer_text = "Detailted answer: " + answer_text
        
        print("CONFUSED answer: ", answer_text)
        
        return jsonify({"query": query, "answer": answer_text, "emotion": emotion}), 200 # Send back ONLY the text answer
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- New endpoint to get webcam image URL ---
# @app.route("/get_webcam_image_url", methods=["GET"]) # Webcam image URL endpoint
# def get_webcam_image_url():
#     try:
#         image_filename = capture_webcam_frame() # Capture and save image
#         if image_filename:
#             image_url = f"/emo_images/{image_filename}?{int(time.time())}" # Create image URL
#             return jsonify({"image_url": image_url}), 200
#         else:
#             return jsonify({"detail": "Failed to capture webcam image"}), 500
#     except Exception as e:
#         return jsonify({"detail": str(e)}), 500
    
if __name__ == "__main__":
    
    print("HTML file created. Starting the server...")
    app.run(host="0.0.0.0", port=5000)