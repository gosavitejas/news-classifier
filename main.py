import joblib
from pyscript import display, aio, document
import asyncio

# These are the 20 category names from our dataset
# We need them here to show the user the name, not just the number (e.g., "sci.med")
CATEGORY_NAMES = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]

# We are setting up "global" variables to hold our loaded model
# We set them to "None" to begin with
vectorizer = None
model = None

async def load_model():
    """
    This function loads our .pkl files from the server.
    'aio.open' is a special PyScript function to fetch files.
    """
    global vectorizer, model
    
    print("Loading model files...")
    
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
    button = document.getElementById("classify-button")
    button.innerText = "Classify Text"
    button.disabled = False

def predict(text):
    """
    This function runs the actual prediction.
    It's called by our JavaScript.
    """
    global vectorizer, model
    
    # We must wrap the text in a list, because the vectorizer expects a list
    text_list = [text]
    
    # 1. Convert the text to numbers using our loaded "dictionary"
    text_vectorized = vectorizer.transform(text_list)
    
    # 2. Get the prediction from our loaded "brain"
    prediction_array = model.predict(text_vectorized)
    
    # 3. The prediction is a number (e.g., 13), so get the first item
    prediction_index = prediction_array[0]
    
    # 4. Use the number to look up the category name
    category_name = CATEGORY_NAMES[prediction_index]
    
    print(f"Prediction: {category_name}")
    return category_name

# This is the main part of our script.
# We tell PyScript to run the 'load_model' function as soon as it can.
# We also temporarily disable the button so no one clicks it before the model is loaded.
button = document.getElementById("classify-button")
button.innerText = "Loading Model..."
button.disabled = True
asyncio.ensure_future(load_model())