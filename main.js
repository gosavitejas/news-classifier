// This file connects our HTML to our Python code

// 1. Get references to our HTML elements
import { interpreter } from "pyscript";
const classifyButton = document.getElementById("classify-button");
const userInput = document.getElementById("user-input");
const resultArea = document.getElementById("result-area");
const resultText = document.getElementById("result-text");

// 2. This is the main function that runs when the button is clicked
async function onClassifyClick() {
    console.log("Button clicked!");
    
    // Get the text from the text box
    const text = userInput.value;
    
    // Check if the user actually typed something
    if (text.trim().length === 0) {
        alert("Please enter some text to classify.");
        return;
    }
    
    // Show a "loading" message
    classifyButton.innerText = "Classifying...";
    classifyButton.disabled = true;
    resultArea.style.display = "none";

    try {
        // 3. This is the magic!
        // We get our Python file (main.py)
        const mainPy = interpreter.globals.get('main');
        
        // We call the 'predict' function *inside* main.py
        // and give it the user's text
        const prediction = mainPy.predict(text);

        // 4. Show the result
        resultText.innerText = prediction; // e.g., "sci.med"
        resultArea.style.display = "block"; // Make the result area visible

    } catch (error) {
        console.error("An error occurred:", error);
        alert("An error occurred during classification. Check the console.");
    }

    // 5. Reset the button
    classifyButton.innerText = "Classify Text";
    classifyButton.disabled = false;
}

// 6. Tell the browser: "When the 'classifyButton' is clicked,
//    run our 'onClassifyClick' function."
classifyButton.addEventListener("click", onClassifyClick);
