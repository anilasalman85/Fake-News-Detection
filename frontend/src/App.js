import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BarChart3, Brain, AlertCircle, CheckCircle, Info, TrendingUp, Activity, Database, Image as ImageIcon, Search, Shield, Newspaper } from 'lucide-react';

// CSS Styles
const styles = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }

  .header {
    text-align: center;
    color: white;
    margin-bottom: 30px;
  }

  .header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
  }

  .header p {
    font-size: 1.1rem;
    opacity: 0.9;
  }

  .tabs {
    display: flex;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 5px;
    margin-bottom: 30px;
    backdrop-filter: blur(10px);
    flex-wrap: wrap;
  }

  .tab {
    flex: 1;
    min-width: 150px;
    padding: 15px 20px;
    background: transparent;
    border: none;
    color: white;
    font-size: 1rem;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .tab.active {
    background: rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  }

  .tab:hover {
    background: rgba(255, 255, 255, 0.15);
  }

  .content {
    background: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    max-height: 80vh;
    overflow-y: auto;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
  }

  .stat-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
  }

  .stat-card h3 {
    font-size: 2rem;
    margin-bottom: 5px;
  }

  .stat-card p {
    opacity: 0.9;
    font-size: 0.9rem;
  }

  .model-info {
    background: #f8f9fa;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
  }

  .model-info h3 {
    color: #333;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
  }

  .info-item {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    background: white;
    border-radius: 8px;
    border-left: 4px solid #667eea;
  }

  .textarea {
    width: 100%;
    min-height: 200px;
    padding: 15px;
    border: 2px solid #e1e5e9;
    border-radius: 8px;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
    transition: border-color 0.3s ease;
  }

  .textarea:focus {
    outline: none;
    border-color: #667eea;
  }

  .button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 15px 30px;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: transform 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 15px;
  }

  .button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
  }

  .button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .result-card {
    margin-top: 20px;
    padding: 25px;
    border-radius: 12px;
    border: 2px solid #e1e5e9;
  }

  .result-fake {
    background: #fff5f5;
    border-color: #f56565;
  }

  .result-real {
    background: #f0fff4;
    border-color: #48bb78;
  }

  .result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
  }

  .result-title {
    font-size: 1.2rem;
    font-weight: bold;
  }

  .result-fake .result-title {
    color: #e53e3e;
  }

  .result-real .result-title {
    color: #38a169;
  }

  .confidence-bar {
    background: #e2e8f0;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    margin: 10px 0;
  }

  .confidence-fill {
    height: 100%;
    transition: width 0.5s ease;
  }

  .confidence-fake {
    background: linear-gradient(90deg, #f56565, #e53e3e);
  }

  .confidence-real {
    background: linear-gradient(90deg, #48bb78, #38a169);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 15px;
    margin: 15px 0;
  }

  .metric {
    text-align: center;
    padding: 10px;
    background: rgba(255,255,255,0.5);
    border-radius: 8px;
  }

  .metric-value {
    font-size: 1.1rem;
    font-weight: bold;
    color: #2d3748;
  }

  .metric-label {
    font-size: 0.8rem;
    color: #718096;
    margin-top: 2px;
  }

  .risk-factors {
    margin-top: 15px;
  }

  .risk-item {
    background: #fed7d7;
    color: #c53030;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 5px 0;
    font-size: 0.9rem;
  }

  .loading {
    text-align: center;
    padding: 40px;
    color: #718096;
  }

  .error {
    background: #fed7d7;
    color: #c53030;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
  }

  .confusion-matrix {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    max-width: 300px;
    margin: 20px auto;
  }

  .matrix-cell {
    background: #4299e1;
    color: white;
    padding: 20px;
    text-align: center;
    border-radius: 8px;
    font-weight: bold;
  }

  .matrix-cell.diagonal {
    background: #38a169;
  }

  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
  }

  .image-button {
    background: white;
    border: 2px solid #e1e5e9;
    border-radius: 12px;
    padding: 15px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: center;
  }

  .image-button:hover {
    border-color: #667eea;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
  }

  .image-button.active {
    border-color: #667eea;
    background: #f0f4ff;
  }

  .image-display {
    margin-top: 20px;
    text-align: center;
  }

  .image-display img {
    max-width: 100%;
    height: auto;
    border-radius: 12px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
  }

  .dataset-section {
    background: #f8f9fa;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
  }

  .dataset-section h3 {
    color: #333;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .json-display {
    background: #1a202c;
    color: #e2e8f0;
    padding: 20px;
    border-radius: 8px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.9rem;
    overflow-x: auto;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
  }

  @media (max-width: 768px) {
    .container {
      padding: 10px;
    }
    
    .header h1 {
      font-size: 2rem;
    }
    
    .content {
      padding: 20px;
    }
    
    .tabs {
      flex-direction: column;
    }
    
    .stats-grid {
      grid-template-columns: 1fr;
    }
    
    .image-grid {
      grid-template-columns: 1fr;
    }
  }
`;

// Real images from the backend/images directory
const realImages = [
  { 
    name: 'Confusion Matrix', 
    filename: 'confusion_matrix.png',
    description: 'Model prediction accuracy visualization'
  },
  { 
    name: 'Data Distribution', 
    filename: 'data_distribution.png',
    description: 'Distribution of fake and real news articles'
  },
  { 
    name: 'ROC Curve', 
    filename: 'roc_curve.png',
    description: 'Receiver Operating Characteristic curve showing model performance'
  },
  { 
    name: 'Text Length by Label', 
    filename: 'text_length_by_label.png',
    description: 'Distribution of text lengths for fake and real news'
  },
  { 
    name: 'Fake Keywords by Label', 
    filename: 'fake_keywords_by_label.png',
    description: 'Most common keywords in fake vs real news'
  },
  { 
    name: 'Sentiment by Label', 
    filename: 'sentiment_by_label.png',
    description: 'Sentiment analysis distribution for fake and real news'
  }
];

// Main App Component
const App = () => {
  const [activeTab, setActiveTab] = useState('detector');
  const [newsText, setNewsText] = useState('');
  const textareaRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelStats, setModelStats] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState({
    analysis: {
      total_samples: 95615,
      fake_samples: 23481 + 35028,
      real_samples: 37106,
      fake_percentage: (((23481 + 35028) / 95615) * 100).toFixed(1),
      real_percentage: ((37106 / 95615) * 100).toFixed(1),
      unique_sources: 2,
      date_range: "2016-2022",
      model_metrics: {
        accuracy: 96.82,
        auc_roc: 0.9953,
        cv_accuracy: "0.9653 ± 0.0019",
        cv_auc: "0.9945 ± 0.0005"
      }
    },
    datasets: {
      "WELFake_Dataset.csv": {
        rows: 72134,
        columns: 4,
        missing_percentage: 0.21,
        duplicate_rows: 0,
        columns_info: ["title", "text", "label"],
        unique_titles: 62347,
        unique_texts: 62718
      },
      "Fake.csv": {
        rows: 23481,
        columns: 4,
        missing_percentage: 0.0,
        duplicate_rows: 3,
        columns_info: ["title", "text", "subject", "date"],
        unique_titles: 17903,
        unique_texts: 17455
      }
    },
    summary: {
      description: "Comprehensive fake news detection dataset combining WELFake dataset and additional labeled news articles. The dataset provides a balanced representation of authentic and fabricated news content, with extensive text analysis and feature engineering.",
      preprocessing_steps: [
        "Text cleaning and normalization",
        "Removal of HTML tags and special characters",
        "Tokenization and lemmatization",
        "Stop words removal",
        "TF-IDF vectorization (8,010 features)",
        "Feature engineering (text length, sentiment scores, etc.)",
        "Label encoding and standardization"
      ],
      model_details: {
        algorithm: "Ensemble (Voting Classifier)",
        features_used: 8010,
        training_samples: 50496,
        test_samples: 12625,
        cross_validation_folds: 5,
        hyperparameter_tuning: "Grid Search with Cross-validation",
        metrics: {
          accuracy: 96.82,
          precision: 97.4,
          recall: 96.8,
          f1_score: 97.1,
          auc_roc: 0.9953
        },
        confusion_matrix: {
          true_positives: 6733,
          false_positives: 226,
          true_negatives: 5490,
          false_negatives: 176
        }
      }
    }
  });

  const API_BASE_URL = 'http://localhost:5000';

  useEffect(() => {
    fetchHealthStatus();
    fetchModelStats();
    loadDatasetInfo();
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setNewsText(textareaRef.current.value);
    }, 300);

    return () => clearTimeout(timer);
  }, []);

  const fetchHealthStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setHealthStatus(data);
    } catch (err) {
      console.error('Failed to fetch health status:', err);
      setHealthStatus({
        server_status: 'unknown',
        model_loaded: false,
        components: {
          explainer: false,
          vader_analyzer: false
        }
      });
    }
  };

  const fetchModelStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      const data = await response.json();
      setModelStats(data);
    } catch (err) {
      console.error('Failed to fetch model stats:', err);
      setModelStats({
        accuracy: 0,
        feature_count: 0,
        training_samples: 0,
        model_type: 'Fallback',
        confusion_matrix: [[0, 0], [0, 0]],
        cross_validation: {
          mean: 0,
          std: 0,
          scores: [0, 0, 0, 0, 0]
        }
      });
    }
  };

  const loadDatasetInfo = async () => {
    try {
      setDatasetInfo(prevState => ({...prevState}));
    } catch (err) {
      console.error('Failed to load dataset info:', err);
    }
  };

  const analyzeNews = async () => {
    const text = textareaRef.current.value;
    if (!text.trim()) {
      setError('Please enter some news text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      setPrediction({
        prediction: result.prediction || 'Unknown',
        confidence: result.confidence || 0.5,
        fake_probability: result.fake_probability || 0.5,
        real_probability: result.real_probability || 0.5,
        certainty: result.certainty || 0.5,
        sentiment: result.sentiment || 'neutral',
        text_stats: result.text_stats || {
          word_count: text.split(' ').length,
          character_count: text.length,
          exclamation_count: (text.match(/!/g) || []).length,
          caps_ratio: (text.match(/[A-Z]/g) || []).length / text.length
        },
        risk_factors: result.risk_factors || []
      });
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Analysis failed: ${err.message}`);
      setPrediction({
        prediction: 'Real',
        confidence: 0.5,
        fake_probability: 0.5,
        real_probability: 0.5,
        certainty: 0.5,
        sentiment: 'neutral',
        text_stats: {
          word_count: text.split(' ').length,
          character_count: text.length,
          exclamation_count: (text.match(/!/g) || []).length,
          caps_ratio: (text.match(/[A-Z]/g) || []).length / text.length
        },
        risk_factors: ['Using fallback analysis due to server error']
      });
    } finally {
      setLoading(false);
    }
  };

  const DetectorTab = () => (
    <div>
      <div className="model-info">
        <h3><Brain size={20} />News Detector</h3>
        <p style={{color: '#718096', marginBottom: '20px'}}>
          Enter the news text you want to analyze below. Our AI model will evaluate its authenticity.
        </p>
      </div>

      <textarea
        ref={textareaRef}
        className="textarea"
        placeholder="Paste your news article here..."
      />

      <button
        className="button"
        onClick={analyzeNews}
        disabled={loading}
      >
        {loading ? (
          <>
            <Activity size={18} />
            Analyzing...
          </>
        ) : (
          <>
            <Brain size={18} />
            Analyze News
          </>
        )}
      </button>

      {error && (
        <div className="error">
          <AlertCircle size={18} style={{marginRight: '8px'}} />
          {error}
        </div>
      )}

      {prediction && (
        <div className={`result-card ${prediction.prediction.toLowerCase() === 'fake' ? 'result-fake' : 'result-real'}`}>
          <div className="result-header">
            {prediction.prediction.toLowerCase() === 'fake' ? (
              <AlertCircle size={24} style={{color: '#e53e3e'}} />
            ) : (
              <CheckCircle size={24} style={{color: '#38a169'}} />
            )}
            <span className="result-title">
              {prediction.prediction} News
            </span>
          </div>

          <div className="confidence-bar">
            <div
              className={`confidence-fill ${prediction.prediction.toLowerCase() === 'fake' ? 'confidence-fake' : 'confidence-real'}`}
              style={{width: `${prediction.confidence * 100}%`}}
            />
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <div className="metric-value">{prediction.prediction}</div>
              <div className="metric-label">Prediction</div>
            </div>
            <div className="metric">
              <div className="metric-value">{prediction.sentiment}</div>
              <div className="metric-label">Sentiment</div>
            </div>
          </div>

          {prediction.risk_factors && prediction.risk_factors.length > 0 && (
            <div className="risk-factors">
              <h4 style={{marginBottom: '10px', color: '#2d3748'}}>Risk Factors:</h4>
              {prediction.risk_factors.map((factor, index) => (
                <div key={index} className="risk-item">
                  {factor}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );

  const ModelStatsTab = () => (
    <div style={{display: 'flex', flexDirection: 'column', gap: '20px'}}>
      <div className="stats-grid" style={{gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px'}}>
        <div className="stat-card" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'}}>
          <h3 style={{color: 'white'}}>96.82%</h3>
          <p style={{color: 'white'}}>Model Accuracy</p>
        </div>
        <div className="stat-card" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'}}>
          <h3 style={{color: 'white'}}>95,615</h3>
          <p style={{color: 'white'}}>Total Samples</p>
        </div>
        <div className="stat-card" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'}}>
          <h3 style={{color: 'white'}}>8,010</h3>
          <p style={{color: 'white'}}>Total Features</p>
        </div>
        <div className="stat-card" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'}}>
          <h3 style={{color: 'white'}}>0.9953</h3>
          <p style={{color: 'white'}}>AUC-ROC Score</p>
        </div>
      </div>

      <div className="model-info">
        <h3><Brain size={20} />Model Information</h3>
        <div className="info-grid" style={{gridTemplateColumns: 'repeat(2, 1fr)', gap: '15px'}}>
          <div className="info-item">
            <span>Model Name</span>
            <span>Optimized Fake News Detection System</span>
          </div>
          <div className="info-item">
            <span>Model Type</span>
            <span>Ensemble (Voting Classifier)</span>
          </div>
          <div className="info-item">
            <span>Training Date</span>
            <span>2025-06-09 21:48:22</span>
          </div>
          <div className="info-item">
            <span>Version</span>
            <span>2.0</span>
          </div>
        </div>
      </div>

      <div className="model-info">
        <h3><Database size={20} />Dataset Overview</h3>
        <div className="info-grid" style={{gridTemplateColumns: 'repeat(2, 1fr)', gap: '15px'}}>
          <div className="info-item">
            <span>Training Samples</span>
            <span>50,496 (80.0%)</span>
          </div>
          <div className="info-item">
            <span>Testing Samples</span>
            <span>12,625 (20.0%)</span>
          </div>
          <div className="info-item">
            <span>Fake News</span>
            <span>34,791 (55.1%)</span>
          </div>
          <div className="info-item">
            <span>Real News</span>
            <span>28,330 (44.9%)</span>
          </div>
        </div>
      </div>

      <div className="model-info">
        <h3><BarChart3 size={20} />Model Performance</h3>
        <div className="info-grid" style={{gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px'}}>
          <div className="info-item">
            <span>Accuracy</span>
            <span>96.82%</span>
          </div>
          <div className="info-item">
            <span>Precision</span>
            <span>0.9605</span>
          </div>
          <div className="info-item">
            <span>Recall</span>
            <span>0.9689</span>
          </div>
          <div className="info-item">
            <span>F1-Score</span>
            <span>0.9647</span>
          </div>
          <div className="info-item">
            <span>Specificity</span>
            <span>0.9675</span>
          </div>
          <div className="info-item">
            <span>AUC-ROC</span>
            <span>0.9953</span>
          </div>
        </div>
      </div>

      <div className="model-info">
        <h3><TrendingUp size={20} />Confusion Matrix</h3>
        <div className="confusion-matrix" style={{maxWidth: '400px', margin: '20px auto'}}>
          <div className="matrix-cell diagonal">6,733</div>
          <div className="matrix-cell">226</div>
          <div className="matrix-cell">176</div>
          <div className="matrix-cell diagonal">5,490</div>
        </div>
        <div style={{textAlign: 'center', fontSize: '0.9rem', color: '#718096', marginTop: '10px'}}>
          <p>True Negatives (Correct Fake): 6,733 | False Positives (Incorrect Real): 226</p>
          <p>False Negatives (Incorrect Fake): 176 | True Positives (Correct Real): 5,490</p>
        </div>
      </div>

      <div className="model-info">
        <h3><Activity size={20} />Cross-Validation Results</h3>
        <div className="info-grid" style={{gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px'}}>
          <div className="info-item">
            <span>Accuracy</span>
            <span>0.9653 ± 0.0019</span>
          </div>
          <div className="info-item">
            <span>AUC-ROC</span>
            <span>0.9945 ± 0.0005</span>
          </div>
          <div className="info-item">
            <span>F1-Score</span>
            <span>0.9616 ± 0.0021</span>
          </div>
        </div>
      </div>

      <div className="model-info">
        <h3><Brain size={20} />Individual Model Performance</h3>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px'}}>
          <div className="info-grid" style={{gap: '10px'}}>
            <h4 style={{color: '#4a5568', padding: '10px 0'}}>Logistic Regression</h4>
            <div className="info-item">
              <span>Accuracy</span>
              <span>95.98%</span>
            </div>
            <div className="info-item">
              <span>AUC-ROC</span>
              <span>0.9925</span>
            </div>
            <div className="info-item">
              <span>F1-Score</span>
              <span>0.9553</span>
            </div>
          </div>
          <div className="info-grid" style={{gap: '10px'}}>
            <h4 style={{color: '#4a5568', padding: '10px 0'}}>Random Forest</h4>
            <div className="info-item">
              <span>Accuracy</span>
              <span>92.73%</span>
            </div>
            <div className="info-item">
              <span>AUC-ROC</span>
              <span>0.9826</span>
            </div>
            <div className="info-item">
              <span>F1-Score</span>
              <span>0.9201</span>
            </div>
          </div>
          <div className="info-grid" style={{gap: '10px'}}>
            <h4 style={{color: '#4a5568', padding: '10px 0'}}>XGBoost</h4>
            <div className="info-item">
              <span>Accuracy</span>
              <span>96.39%</span>
            </div>
            <div className="info-item">
              <span>AUC-ROC</span>
              <span>0.9949</span>
            </div>
            <div className="info-item">
              <span>F1-Score</span>
              <span>0.9600</span>
            </div>
          </div>
        </div>
      </div>

      <div className="model-info">
        <h3><Info size={20} />Feature Engineering</h3>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px'}}>
          <div>
            <h4 style={{color: '#4a5568', marginBottom: '10px'}}>Numerical Features</h4>
            <div style={{display: 'flex', flexWrap: 'wrap', gap: '8px'}}>
              {[
                'Text Length',
                'Title Length',
                'Word Count',
                'Exclamation Count',
                'Question Count',
                'Caps Ratio',
                'Compound',
                'Pos',
                'Neg',
                'Fake Keyword Count'
              ].map((feature, index) => (
                <span key={index} style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  padding: '4px 10px',
                  borderRadius: '12px',
                  fontSize: '0.8rem'
                }}>
                  {feature}
                </span>
              ))}
            </div>
          </div>
          <div>
            <h4 style={{color: '#4a5568', marginBottom: '10px'}}>TF-IDF Configuration</h4>
            <div className="info-grid" style={{gap: '10px'}}>
              <div className="info-item">
                <span>Max Features</span>
                <span>8,000</span>
              </div>
              <div className="info-item">
                <span>N-gram Range</span>
                <span>(1, 2)</span>
              </div>
              <div className="info-item">
                <span>Min Doc Frequency</span>
                <span>2</span>
              </div>
              <div className="info-item">
                <span>Max Doc Frequency</span>
                <span>0.8</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const VisualizationTab = () => (
    <div>
      <div className="model-info">
        <h3><ImageIcon size={20} />Analysis Visualizations</h3>
        <p style={{color: '#718096', marginBottom: '20px'}}>
          Detailed visualizations showing various aspects of the fake news detection analysis.
        </p>
      </div>

      <div className="image-grid" style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '30px',
        padding: '20px 0'
      }}>
        {realImages.map((image, index) => (
          <div key={index} className="model-info" style={{
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '15px'
          }}>
            <div>
              <h4 style={{
                color: '#2d3748', 
                marginBottom: '8px', 
                fontSize: '1.1rem',
                fontWeight: '600'
              }}>
                {image.name}
              </h4>
              <p style={{
                color: '#718096',
                fontSize: '0.9rem',
                lineHeight: '1.4'
              }}>
                {image.description}
              </p>
            </div>
            
            <div style={{
              position: 'relative',
              borderRadius: '12px',
              overflow: 'hidden',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              background: '#f8fafc',
              padding: '10px'
            }}>
              <img 
                src={`/images/${image.filename}`}
                alt={image.name}
                style={{
                  width: '100%',
                  height: 'auto',
                  borderRadius: '8px',
                  display: 'block'
                }}
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.parentElement.innerHTML = `
                    <div style="
                      padding: 20px;
                      text-align: center;
                      color: #e53e3e;
                      background: #fff5f5;
                      border-radius: 8px;
                      border: 1px dashed #fc8181;
                    ">
                      Image not found: ${image.filename}
                    </div>
                  `;
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const DatasetInfoTab = () => (
    <div>
      {/* Dataset Analysis */}
      <div className="dataset-section">
        <h3><Database size={20} />Dataset Analysis</h3>
        <div className="stats-grid">
          <div className="stat-card" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'}}>
            <h3 style={{color: 'white'}}>{datasetInfo.analysis.total_samples.toLocaleString()}</h3>
            <p style={{color: 'white'}}>Total Samples</p>
          </div>
          <div className="stat-card" style={{background: 'linear-gradient(135deg, #f6ad55 0%, #ed8936 100%)'}}>
            <h3 style={{color: 'white'}}>{datasetInfo.analysis.fake_percentage}%</h3>
            <p style={{color: 'white'}}>Fake News ({datasetInfo.analysis.fake_samples.toLocaleString()})</p>
          </div>
          <div className="stat-card" style={{background: 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)'}}>
            <h3 style={{color: 'white'}}>{datasetInfo.analysis.real_percentage}%</h3>
            <p style={{color: 'white'}}>Real News ({datasetInfo.analysis.real_samples.toLocaleString()})</p>
          </div>
          <div className="stat-card" style={{background: 'linear-gradient(135deg, #4299e1 0%, #3182ce 100%)'}}>
            <h3 style={{color: 'white'}}>{datasetInfo.analysis.model_metrics.accuracy}%</h3>
            <p style={{color: 'white'}}>Model Accuracy</p>
          </div>
        </div>
        <div style={{marginTop: '15px', padding: '15px', background: '#f7fafc', borderRadius: '8px', fontSize: '0.9rem', color: '#4a5568'}}>
          <p><strong>Dataset Overview:</strong> Our dataset combines {datasetInfo.analysis.unique_sources} major sources with {datasetInfo.analysis.total_samples.toLocaleString()} total samples. The data spans from {datasetInfo.analysis.date_range}, providing a comprehensive view of news content patterns.</p>
        </div>
      </div>

      {/* Dataset Details */}
      <div className="dataset-section">
        <h3><Info size={20} />Dataset Details</h3>
        <div className="info-grid" style={{gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px'}}>
          {Object.entries(datasetInfo.datasets).map(([name, data]) => (
            <div key={name} className="model-info" style={{padding: '20px'}}>
              <h4 style={{
                color: '#2d3748', 
                marginBottom: '15px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <Newspaper size={18} />
                {name}
              </h4>
              <div className="info-grid" style={{gap: '10px'}}>
                <div className="info-item">
                  <span>Total Entries</span>
                  <span>{data.rows.toLocaleString()}</span>
                </div>
                <div className="info-item">
                  <span>Unique Titles</span>
                  <span>{data.unique_titles.toLocaleString()}</span>
                </div>
                <div className="info-item">
                  <span>Unique Texts</span>
                  <span>{data.unique_texts.toLocaleString()}</span>
                </div>
                <div className="info-item">
                  <span>Missing Data</span>
                  <span>{data.missing_percentage}%</span>
                </div>
              </div>
              <div style={{marginTop: '15px'}}>
                <h5 style={{color: '#4a5568', marginBottom: '8px', fontSize: '0.9rem'}}>Available Features:</h5>
                <div style={{display: 'flex', flexWrap: 'wrap', gap: '6px'}}>
                  {data.columns_info.map((col, index) => (
                    <span key={index} style={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      color: 'white',
                      padding: '4px 10px',
                      borderRadius: '12px',
                      fontSize: '0.8rem'
                    }}>
                      {col}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Performance */}
      <div className="dataset-section">
        <h3><Brain size={20} />Model Performance</h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          marginBottom: '30px'
        }}>
          {Object.entries(datasetInfo.summary.model_details.metrics).map(([metric, value]) => (
            <div key={metric} style={{
              background: 'white',
              padding: '20px',
              borderRadius: '12px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              textAlign: 'center'
            }}>
              <h4 style={{
                color: '#2d3748',
                marginBottom: '8px',
                textTransform: 'capitalize'
              }}>
                {metric.replace(/_/g, ' ')}
              </h4>
              <p style={{
                fontSize: '1.5rem',
                fontWeight: '600',
                color: '#4a5568'
              }}>
                {typeof value === 'number' ? value.toFixed(1) + '%' : value}
              </p>
            </div>
          ))}
        </div>

        {/* Confusion Matrix */}
        <div className="model-info" style={{padding: '20px', marginBottom: '20px'}}>
          <h4 style={{color: '#2d3748', marginBottom: '15px'}}>Confusion Matrix</h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '10px',
            maxWidth: '400px',
            margin: '0 auto'
          }}>
            <div style={{
              background: '#48bb78',
              color: 'white',
              padding: '15px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{fontSize: '1.5rem', fontWeight: '600'}}>
                {datasetInfo.summary.model_details.confusion_matrix.true_positives}
              </div>
              <div style={{fontSize: '0.8rem', opacity: 0.9}}>True Positives</div>
            </div>
            <div style={{
              background: '#f56565',
              color: 'white',
              padding: '15px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{fontSize: '1.5rem', fontWeight: '600'}}>
                {datasetInfo.summary.model_details.confusion_matrix.false_positives}
              </div>
              <div style={{fontSize: '0.8rem', opacity: 0.9}}>False Positives</div>
            </div>
            <div style={{
              background: '#f56565',
              color: 'white',
              padding: '15px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{fontSize: '1.5rem', fontWeight: '600'}}>
                {datasetInfo.summary.model_details.confusion_matrix.false_negatives}
              </div>
              <div style={{fontSize: '0.8rem', opacity: 0.9}}>False Negatives</div>
            </div>
            <div style={{
              background: '#48bb78',
              color: 'white',
              padding: '15px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{fontSize: '1.5rem', fontWeight: '600'}}>
                {datasetInfo.summary.model_details.confusion_matrix.true_negatives}
              </div>
              <div style={{fontSize: '0.8rem', opacity: 0.9}}>True Negatives</div>
            </div>
          </div>
        </div>

        {/* Model Details */}
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px'}}>
          <div className="model-info" style={{padding: '20px'}}>
            <h4 style={{color: '#2d3748', marginBottom: '15px'}}>Training Details</h4>
            <div style={{display: 'flex', flexDirection: 'column', gap: '10px'}}>
              <div className="info-item">
                <span>Algorithm</span>
                <span>{datasetInfo.summary.model_details.algorithm}</span>
              </div>
              <div className="info-item">
                <span>Features</span>
                <span>{datasetInfo.summary.model_details.features_used.toLocaleString()}</span>
              </div>
              <div className="info-item">
                <span>Training Samples</span>
                <span>{datasetInfo.summary.model_details.training_samples.toLocaleString()}</span>
              </div>
              <div className="info-item">
                <span>Test Samples</span>
                <span>{datasetInfo.summary.model_details.test_samples.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="model-info" style={{padding: '20px'}}>
            <h4 style={{color: '#2d3748', marginBottom: '15px'}}>Preprocessing Pipeline</h4>
            <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
              {datasetInfo.summary.preprocessing_steps.map((step, index) => (
                <div key={index} style={{
                  background: '#f7fafc',
                  padding: '10px 15px',
                  borderRadius: '8px',
                  borderLeft: '3px solid #667eea',
                  fontSize: '0.9rem',
                  color: '#4a5568'
                }}>
                  {step}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <style>{styles}</style>
      <div className="container">
        <div className="header">
          <h1>
            <Shield size={32} style={{marginRight: '10px', verticalAlign: 'middle'}} />
            Fake News Detector AI
          </h1>
          <p>Advanced AI-powered news authenticity analysis</p>
        </div>

        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'detector' ? 'active' : ''}`}
            onClick={() => setActiveTab('detector')}
          >
            <Brain size={18} />
            News Detector
          </button>
          <button 
            className={`tab ${activeTab === 'stats' ? 'active' : ''}`}
            onClick={() => setActiveTab('stats')}
          >
            <BarChart3 size={18} />
            Model Stats
          </button>
          <button 
            className={`tab ${activeTab === 'visualizations' ? 'active' : ''}`}
            onClick={() => setActiveTab('visualizations')}
          >
            <ImageIcon size={18} />
            Visualizations
          </button>
          <button 
            className={`tab ${activeTab === 'dataset' ? 'active' : ''}`}
            onClick={() => setActiveTab('dataset')}
          >
            <Database size={18} />
            Dataset Info
          </button>
        </div>

        <div className="content">
          {activeTab === 'detector' && <DetectorTab />}
          {activeTab === 'stats' && <ModelStatsTab />}
          {activeTab === 'visualizations' && <VisualizationTab />}
          {activeTab === 'dataset' && <DatasetInfoTab />}
        </div>
      </div>
    </>
  );
};

export default App;