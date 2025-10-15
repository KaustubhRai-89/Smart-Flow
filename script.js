// --- Configuration ---
// Your Gemini API key is integrated here.
const GEMINI_API_KEY = "AIzaSyDKNZEH8UOlC-rTTvTLRCa1psHVqBAVRD8";

// --- DOM Element Selection ---
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const mobileMenu = document.getElementById("mobile-menu");
const imageUpload = document.getElementById("image-upload");
const fileNameDisplay = document.getElementById("file-name");
const processBtn = document.getElementById("process-btn");
const btnText = document.getElementById("btn-text");
const btnLoader = document.getElementById("btn-loader");
const demoOutput = document.getElementById("demo-output");
const promptInput = document.getElementById("prompt");

// --- State Management ---
let uploadedImageBase64 = null;
let uploadedImageType = null;

// --- Event Listeners ---

// Mobile Menu Toggle
mobileMenuBtn.addEventListener("click", () => {
  mobileMenu.classList.toggle("hidden");
});

// Close mobile menu on link click
const mobileLinks = mobileMenu.querySelectorAll("a");
mobileLinks.forEach((link) => {
  link.addEventListener("click", () => {
    mobileMenu.classList.add("hidden");
  });
});

// Image Upload Handler
imageUpload.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) {
    uploadedImageBase64 = null;
    uploadedImageType = null;
    fileNameDisplay.textContent = "";
    return;
  }

  fileNameDisplay.textContent = `File selected: ${file.name}`;
  const reader = new FileReader();
  reader.onloadend = () => {
    // Result includes the full data URL prefix, so we split it
    const base64String = reader.result.split(",")[1];
    uploadedImageBase64 = base64String;
    uploadedImageType = file.type;
  };
  reader.readAsDataURL(file);
});

// Process Button Click Handler
processBtn.addEventListener("click", handleAnalysis);

// --- Core Functions ---

/**
 * Handles the analysis process when the button is clicked.
 */
async function handleAnalysis() {
  const apiKey = GEMINI_API_KEY; // Use the hardcoded key
  const userPrompt = promptInput.value.trim();

  // 1. Validate Inputs
  if (apiKey === "YOUR_API_KEY_HERE" || !apiKey) {
    displayError("API Key is missing. Please contact the site administrator.");
    return;
  }
  if (!userPrompt && !uploadedImageBase64) {
    displayError(
      "Please enter a query or upload an image to start the analysis."
    );
    return;
  }

  // 2. Set Loading State
  setLoadingState(true);

  try {
    // 3. Call Gemini API
    const result = await callGeminiApi(
      apiKey,
      userPrompt,
      uploadedImageBase64,
      uploadedImageType
    );

    // 4. Display Result
    const text = result.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text) {
      // Replace newlines with <br> for HTML rendering
      displayResult(text.replace(/\n/g, "<br>"));
    } else {
      displayError(
        "Received an empty response from the API. The prompt may have been blocked."
      );
      console.error("API Response was empty or malformed:", result);
    }
  } catch (error) {
    // 5. Display Error
    console.error("Error calling Gemini API:", error);
    displayError(
      `An error occurred: ${error.message}. Check the console for more details.`
    );
  } finally {
    // 6. Unset Loading State
    setLoadingState(false);
  }
}

/**
 * Calls the Google Gemini API with the provided data.
 * @param {string} apiKey - The user's Gemini API key.
 * @param {string} prompt - The text prompt from the user.
 * @param {string|null} imageBase64 - The base64 encoded image string.
 * @param {string|null} imageType - The MIME type of the image (e.g., 'image/jpeg').
 * @returns {Promise<Object>} - The JSON response from the API.
 */
async function callGeminiApi(apiKey, prompt, imageBase64, imageType) {
  const model = "gemini-2.5-flash-preview-05-20"; // Use the appropriate model
  const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const parts = [];

  // Add text prompt if it exists
  if (prompt) {
    parts.push({ text: prompt });
  }

  // Add image data if it exists
  if (imageBase64 && imageType) {
    parts.push({
      inlineData: {
        mimeType: imageType,
        data: imageBase64,
      },
    });
  }

  const payload = {
    contents: [{ parts: parts }],
    // Optional: Add safetySettings or generationConfig here if needed
  };

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorBody = await response.json();
    throw new Error(
      `API Error: ${response.status} ${response.statusText} - ${
        errorBody.error?.message || "Unknown error"
      }`
    );
  }

  return response.json();
}

// --- UI Helper Functions ---

/**
 * Sets the loading state for the UI.
 * @param {boolean} isLoading - True to show loader, false to show text.
 */
function setLoadingState(isLoading) {
  if (isLoading) {
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");
    processBtn.disabled = true;
    demoOutput.innerHTML = `<p class="text-sky-400 text-center">Contacting the model... This may take a moment.</p>`;
  } else {
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
    processBtn.disabled = false;
  }
}

/**
 * Displays an error message in the output area.
 * @param {string} message - The error message to display.
 */
function displayError(message) {
  demoOutput.innerHTML = `<div class="error-message">${message}</div>`;
}

/**
 * Displays the successful result from the API.
 * @param {string} resultText - The formatted text to display.
 */
function displayResult(resultText) {
  demoOutput.innerHTML = `
        <h4 class="text-lg font-bold text-sky-400 mb-2">Analysis Complete</h4>
        <div class="space-y-4 mt-4 text-slate-300">${resultText}</div>
    `;
}
