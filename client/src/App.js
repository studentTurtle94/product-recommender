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

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`http://localhost:8000/search`, {
        params: {
          query: query,
          limit: 10
        }
      });
      
      setProducts(response.data.products || []);
      setAlternativeSearches(response.data.alternative_searches || []);
    } catch (err) {
      console.error('Search error:', err);
      setError('Failed to fetch search results. Please try again.');
      setProducts([]);
      setAlternativeSearches([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="hero-section">
        <Container>
          <Row className="justify-content-center text-center">
            <Col md={10} lg={8}>
              <h1>Discover Your Perfect Style</h1>
              <p className="subtitle">
                Search through our curated collection of fashion items using natural language
              </p>
              
              <div className="search-container">
                <Form onSubmit={handleSearch}>
                  <Form.Group className="d-flex gap-2">
                    <Form.Control
                      type="text"
                      placeholder="Try 'casual summer dress' or 'formal business attire'..."
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
                    axios.get(`http://localhost:8000/search`, {
                      params: {
                        query: search,
                        limit: 10
                      }
                    })
                    .then(response => {
                      setProducts(response.data.products || []);
                      setAlternativeSearches(response.data.alternative_searches || []);
                    })
                    .catch(err => {
                      console.error('Search error:', err);
                      setError('Failed to fetch search results. Please try again.');
                      setProducts([]);
                      setAlternativeSearches([]);
                    })
                    .finally(() => {
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
