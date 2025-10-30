import joblib
from pyscript import display, aio, document, create_proxy
import asyncio

# These are the 20 category names
CATEGORY_NAMES = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

# --- Global variables ---
vectorizer = None
model = None

# --- Get references to all our HTML elements ---
user_input = document.getElementById("user-input")
classify_button = document.getElementById("classify-button")
result_area = document.getElementById("result-area")
result_text = document.getElementById("result-text")


async def load_model():
    """Loads the .pkl files from the server."""
    global vectorizer, model
    
    print("Loading model files...")
    
    try:
        # Fetch and load the vectorizer
        response = await aio.open_url('./vectorizer.pkl')
        model_bytes = await response.read()
        vectorizer = joblib.load(model_bytes)
        
        # Fetch and load the model
        response = await aio.open_url('./model.pkl')
        model_bytes = await response.read()
        model = joblib.load(model_bytes)
        
        print("Model and vectorizer loaded successfully!")
        
        # Let the user know the app is ready
        classify_button.innerText = "Classify Text"
        classify_button.disabled = False
        
    except Exception as e:
        print(f"Error loading model: {e}")
        classify_button.innerText = "Error loading model"
        classify_button.style.backgroundColor = "red"


def predict_and_display(event):
    """
    This function runs when the button is clicked.
    It gets the text, runs the prediction, and shows the result.
    """
    global vectorizer, model
    
    text = user_input.value
    
    # Check if the user actually typed something
    if not text.strip():
        result_text.innerText = "Please enter some text."
        result_area.style.display = "block"
        return

    # Show a "loading" message
    classify_button.innerText = "Classifying..."
    classify_button.disabled = True
    result_area.style.display = "none"

    try:
        # We must wrap the text in a list
        text_list = [text]
        
        # 1. Convert the text to numbers
        text_vectorized = vectorizer.transform(text_list)
        
        # 2. Get the prediction
        prediction_array = model.predict(text_vectorized)
        
        # 3. Get the number (e.g., 13)
        prediction_index = prediction_array[0]
        
        # 4. Look up the category name
        category_name = CATEGORY_NAMES[prediction_index]
        
        # 5. Show the result
        result_text.innerText = category_name
        result_area.style.display = "block"

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        result_text.innerText = "An error occurred."
        result_area.style.display = "block"

    # Reset the button
    classify_button.innerText = "Classify Text"
    classify_button.disabled = False

# ==================================
# --- Main script execution ---
# ==================================
print("Starting main.py script...")

# 1. Disable the button while we load
classify_button.innerText = "Loading Model..."
classify_button.disabled = True

# 2. Create a "proxy" for our function
# This lets JavaScript's "click" event call our Python function
click_proxy = create_proxy(predict_and_display)

# 3. Attach the Python function to the button's 'click' event
classify_button.addEventListener("click", click_proxy)
print("Click listener attached to button.")

# 4. Start loading the model in the background
asyncio.ensure_future(load_model())
print("Model loading has been started.")
