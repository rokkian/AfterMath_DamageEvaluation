const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3000;

// OpenAI API Key
const api_key = "API_KEY";

// Function to encode the image
function encodeImage(imagePath) {
  const image = fs.readFileSync(imagePath);
  return image.toString('base64');
}

app.use(express.json());

app.post('/analyze-image', async (req, res) => {
  try {
    const imagePath = req.body.imagePath; // Expect the image path in the request body
    const base64Image = encodeImage(imagePath);

    const headers = {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${api_key}`
    };

    const payload = {
      "model": "gpt-4o-mini",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "here is a photo return json output structured pairs building_type could be private or public and damage_severity could be none, minor, moderate or severe and it could return as pairs at it has on the  picture... the return of the block should be a json with as much as pairs that there is different building in the picture"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": `data:image/jpeg;base64,${base64Image}`
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    };

    const response = await axios.post("https://api.openai.com/v1/chat/completions", payload, { headers });
    res.json(response.data);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'An error occurred while processing the request' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});