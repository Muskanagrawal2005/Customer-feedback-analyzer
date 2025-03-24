const express = require('express');
const mongoose = require('mongoose');
const { HfInference } = require('@huggingface/inference');
const path = require('path');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Initialize Hugging Face client
const client = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/feedbackDB')
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Feedback Schema
const feedbackSchema = new mongoose.Schema({
    text: String,
    sentiment: String,
    sentimentScore: Number,
    topics: [String],
    timestamp: { type: Date, default: Date.now }
});

const Feedback = mongoose.model('Feedback', feedbackSchema);

// Middleware
app.use(express.json());
app.use(express.static('public'));
app.set('view engine', 'ejs');

// Routes
app.get('/', (req, res) => {
    res.render('index');
});

app.get('/summary', (req, res) => {
    res.render('summary');
});

app.post('/submit-feedback', async (req, res) => {
    try {
        const { feedback } = req.body;
        
        // Analyze sentiment using Hugging Face
        const sentimentResult = await client.textClassification({
            model: "cardiffnlp/twitter-roberta-base-sentiment",
            inputs: feedback,
            provider: "hf-inference",
        });

        // Process sentiment result
        let sentiment = sentimentResult[0].label;
        const sentimentScore = parseFloat(sentimentResult[0].score);

        // Normalize sentiment values to match our frontend expectations
        if (sentiment === 'LABEL_0') sentiment = 'Negative';
        else if (sentiment === 'LABEL_1') sentiment = 'Neutral';
        else if (sentiment === 'LABEL_2') sentiment = 'Positive';

        console.log('Sentiment analysis result:', {
            sentiment,
            sentimentScore,
            rawResult: sentimentResult[0]
        });

        // Extract topics (simple implementation)
        const topics = extractTopics(feedback);

        // Save feedback to database
        const newFeedback = new Feedback({
            text: feedback,
            sentiment: sentiment,
            sentimentScore: sentimentScore,
            topics: topics
        });

        await newFeedback.save();
        console.log('Saved feedback:', newFeedback);

        res.json({
            success: true,
            sentiment: sentiment,
            score: sentimentScore,
            topics: topics
        });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({
            success: false,
            error: 'Error submitting feedback'
        });
    }
});

app.get('/get-summary', async (req, res) => {
    try {
        const feedbacks = await Feedback.find();
        console.log('Found feedbacks:', feedbacks.length);
        
        // Calculate sentiment counts
        const sentiment_counts = {
            Positive: 0,
            Negative: 0,
            Neutral: 0
        };

        let total_score = 0;
        let valid_scores = 0;

        feedbacks.forEach(feedback => {
            // Count sentiments
            sentiment_counts[feedback.sentiment] = (sentiment_counts[feedback.sentiment] || 0) + 1;
            
            // Calculate total score
            const score = parseFloat(feedback.sentimentScore);
            if (!isNaN(score)) {
                total_score += score;
                valid_scores++;
            }
        });

        console.log('Sentiment counts:', sentiment_counts);
        console.log('Total score:', total_score);
        console.log('Valid scores count:', valid_scores);

        // Calculate average sentiment score
        const average_sentiment_score = valid_scores > 0 ? total_score / valid_scores : 0;
        console.log('Average sentiment score:', average_sentiment_score);

        const response = {
            total_feedback: feedbacks.length,
            sentiment_counts,
            average_sentiment_score: parseFloat(average_sentiment_score.toFixed(2))
        };

        console.log('Sending summary response:', response);
        res.json(response);
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({
            success: false,
            error: 'Error fetching summary'
        });
    }
});

app.get('/get-all-feedback', async (req, res) => {
    try {
        const feedbacks = await Feedback.find().sort({ timestamp: -1 });
        console.log('Found feedbacks for all-feedback:', feedbacks.length);
        res.json({ feedbacks });
    } catch (error) {
        console.error('Error in get-all-feedback:', error);
        res.status(500).json({ error: 'Error fetching feedbacks' });
    }
});

// Helper function to extract topics
function extractTopics(text) {
    // Simple topic extraction based on common words
    const commonTopics = ['product', 'service', 'price', 'quality', 'delivery', 'support', 'website', 'app'];
    const words = text.toLowerCase().split(/\s+/);
    const topics = new Set();
    
    words.forEach(word => {
        if (commonTopics.includes(word)) {
            topics.add(word);
        }
    });

    return Array.from(topics);
}

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
}); 