import React, { useState } from "react";
import "./App.css";

function App() {
  const [inputText, setInputText] = useState("");
  const [translation, setTranslation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API_URL =
    process.env.REACT_APP_API_URL ||
    "https://eng2tamil-production.up.railway.app/translate";

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError("Please enter an English sentence to translate.");
      return;
    }

    setLoading(true);
    setError("");
    setTranslation("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: inputText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      const raw = data.translation;

      // Get tokens before the first <end>
      const firstEndIndex = raw.indexOf("<end>");
      const beforeEndRaw =
        firstEndIndex !== -1 ? raw.slice(0, firstEndIndex).trim() : raw;
      const beforeTokensArr = beforeEndRaw.split(/\s+/);

      // Get tokens after the last <end>
      const lastEndIndex = raw.lastIndexOf("<end>");
      const afterEndRaw =
        lastEndIndex !== -1 ? raw.slice(lastEndIndex + 5).trim() : "";
      const afterTokensArr = afterEndRaw.split(/\s+/);

      // Utility: Remove duplicates and filter out "end"
      const cleanTokens = (arr) => {
        const seen = new Set();
        return arr.filter((token) => {
          if (token === "end" || seen.has(token)) return false;
          seen.add(token);
          return true;
        });
      };

      // Clean and process
      const uniqueAfter = cleanTokens(afterTokensArr);
      const uniqueBefore = cleanTokens(beforeTokensArr);

      // Build output
      const beforeTokens = uniqueBefore.join(" ");
      const lastToken = uniqueAfter.length
        ? uniqueAfter[uniqueAfter.length - 1]
        : "";

      setTranslation(`${lastToken} ${beforeTokens} `);
    } catch (err) {
      setError(`Error translating: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>English to Tamil Translator</h1>
      <textarea
        rows="4"
        cols="50"
        placeholder="Enter English text here..."
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        disabled={loading}
        style={{ resize: "vertical", fontSize: "1em", padding: "10px" }}
      />
      <br />
      <button
        onClick={handleTranslate}
        disabled={loading || !inputText.trim()}
        style={{
          padding: "10px 20px",
          fontSize: "1em",
          cursor: loading || !inputText.trim() ? "not-allowed" : "pointer",
          backgroundColor: loading || !inputText.trim() ? "#ccc" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
        }}
      >
        {loading ? "Translating..." : "Translate"}
      </button>
      {error && <p style={{ color: "red", marginTop: "10px" }}>{error}</p>}
      {translation && (
        <>
          <h2>Filtered Translation Output:</h2>
          <pre
            style={{
              fontSize: "1.2em",
              color: "#444",
              background: "#f9f9f9",
              padding: "10px",
              borderRadius: "5px",
              whiteSpace: "pre-wrap",
            }}
          >
            {translation}
          </pre>
        </>
      )}
    </div>
  );
}

export default App;
