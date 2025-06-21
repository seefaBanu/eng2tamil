import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [translation, setTranslation] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError('Please enter an English sentence to translate.');
      return;
    }

    setLoading(true);
    setError('');
    setTranslation('');

    try {
      const response = await fetch('https://eng2tamil-production.up.railway.app/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: inputText }), // Changed 'text' to 'sentence'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      setTranslation(data.translation); // Changed 'tamil_translation' to 'translation'
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
        style={{ resize: 'vertical', fontSize: '1em', padding: '10px' }}
      />
      <br />
      <button
        onClick={handleTranslate}
        disabled={loading || !inputText.trim()}
        style={{
          padding: '10px 20px',
          fontSize: '1em',
          cursor: loading || !inputText.trim() ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Translating...' : 'Translate'}
      </button>
      {error && (
        <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>
      )}
      {translation && (
        <>
          <h2>Tamil Translation:</h2>
          <p
            style={{
              fontSize: '1.2em',
              color: '#444',
              background: '#f9f9f9',
              padding: '10px',
              borderRadius: '5px',
            }}
          >
            {translation}
          </p>
        </>
      )}
    </div>
  );
}

export default App;