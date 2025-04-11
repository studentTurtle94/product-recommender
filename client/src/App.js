import React, { useState } from 'react';
import './App.css';
import ProductCard from './components/ProductCard';
import axios from 'axios';
import { Container, Row, Col, Form, Button, Alert } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [query, setQuery] = useState('');
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [alternativeSearches, setAlternativeSearches] = useState([]);
  const [refinedProducts, setRefinedProducts] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setSelectedFile(null);
      setImagePreview(null);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim() && !selectedFile) return;

    setLoading(true);
    setError(null);
    setProducts([]);
    setRefinedProducts([]);
    setAlternativeSearches([]);

    try {
      let response;
      const limit = 10; // Define limit

      if (selectedFile) {
        // Use recommend-multimodal endpoint if image is present
        const endpoint = `${API_BASE_URL}/recommend-multimodal`;
        const formData = new FormData();
        formData.append('query_text', query);
        formData.append('limit', limit);
        formData.append('image_file', selectedFile);

        response = await axios.post(endpoint, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        console.log('Recommend Multimodal Response:', response.data);

      } else {
        // Use recommend endpoint if only text is present
        const endpoint = `${API_BASE_URL}/recommend`;
        response = await axios.get(endpoint, {
          params: {
            query: query,
            limit: limit
          }
        });
        console.log('Recommend Text Response:', response.data);
      }

      // The response structure from /recommend and /recommend-multimodal 
      // should ideally be consistent (e.g., { products: [...], alternative_searches: [...] })
      const sortedProducts = (response.data.products || []).sort((a, b) => (b.rating || 0) - (a.rating || 0));
      setProducts(sortedProducts); 
      setRefinedProducts(response.data.products || []); // Assuming refined results are in 'products'
      setAlternativeSearches(response.data.alternative_searches || []);
      setError(null);
    } catch (err) {
      console.error('Recommendation search error:', err);
      setError('Failed to fetch recommendations. Please try again.');
      setProducts([]);
      setRefinedProducts([]);
      setAlternativeSearches([]);
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedFile(null);
    setImagePreview(null);
    const fileInput = document.getElementById('image-upload');
    if (fileInput) {
      fileInput.value = '';
    }
  }

  return (
    <div className="App">
      <div className="hero-section">
        <Container>
          <Row className="justify-content-center text-center">
            <Col md={10} lg={8}>
              <h1>Discover Your Perfect Style</h1>
              <p className="subtitle">
                Search through our curated collection of fashion items using natural language or upload an image
              </p>
              
              <div className="search-container">
                <Form onSubmit={handleSearch}>
                  <Form.Group className="d-flex gap-2 mb-3">
                    <Form.Control
                      type="text"
                      placeholder="Describe your style or refine your image search..."
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                    <Button 
                      type="submit" 
                      className="search-button" 
                      disabled={loading}
                    >
                      {loading ? 'Searching...' : 'Search'}
                    </Button>
                  </Form.Group>
                  <Form.Group className="mb-3 d-flex align-items-center gap-3">
                    <Form.Label htmlFor="image-upload" className="btn btn-outline-secondary mb-0">
                      {selectedFile ? 'Change Image' : 'Upload Image'}
                    </Form.Label>
                    <Form.Control 
                       id="image-upload"
                       type="file" 
                       accept="image/png, image/jpeg" 
                       onChange={handleFileChange} 
                       style={{ display: 'none' }}
                    />
                    {imagePreview && (
                      <div className="image-preview-container d-flex align-items-center gap-2">
                        <img src={imagePreview} alt="Preview" className="image-preview" />
                        <span className="file-name">{selectedFile?.name}</span>
                        <Button variant="outline-danger" size="sm" onClick={clearImage}>X</Button>
                      </div>
                    )}
                    {!imagePreview && <span className="text-muted">Optional: Upload an image for visual search</span>}
                  </Form.Group>
                </Form>
              </div>
            </Col>
          </Row>
        </Container>
      </div>

      <Container className="results-container">
        {error && (
          <Alert variant="danger" className="mb-4">
            {error}
          </Alert>
        )}
        
        {alternativeSearches.length > 0 && (
          <div className="alternative-searches">
            <h5>Recommended searches</h5>
            <div className="d-flex flex-wrap gap-2">
              {alternativeSearches.map((search, index) => (
                <Button 
                  key={index} 
                  variant="outline-secondary" 
                  onClick={() => {
                    setQuery(search);
                    setLoading(true);
                    axios.get(`${API_BASE_URL}/search`, {
                      params: {
                        query: search,
                        limit: 10
                      }
                    })
                    .then(response => {
                      const sortedProducts = response.data.products.sort((a, b) => (b.rating || 0) - (a.rating || 0));
                      setProducts(sortedProducts);
                      setRefinedProducts(response.data.products);
                      setAlternativeSearches(response.data.alternative_searches || []);
                      setError(null);
                      setLoading(false);
                    })
                    .catch(err => {
                      console.error('Search error:', err);
                      setError('Failed to fetch search results. Please try again.');
                      setProducts([]);
                      setRefinedProducts([]);
                      setAlternativeSearches([]);
                      setLoading(false);
                    });
                  }}
                >
                  {search}
                </Button>
              ))}
            </div>
          </div>
        )}
        
        {loading ? (
          <div className="text-center py-5">
            <div className="loading-text">Finding the perfect matches for you...</div>
          </div>
        ) : products.length > 0 ? (
          <Row className="g-4">
            {products.map((product, index) => (
              <Col key={product.id || index} md={6} lg={4}>
                <div className="h-100">
                  <ProductCard 
                    product={product} 
                    explanation={product.explanation}
                  />
                </div>
              </Col>
            ))}
          </Row>
        ) : query && !loading && (
          <Alert variant="info" className="text-center">
            No products found matching your search. Try different keywords or check out our recommended searches.
          </Alert>
        )}
      </Container>
    </div>
  );
}

export default App;
