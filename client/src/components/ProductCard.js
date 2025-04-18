import React, { useState, useEffect } from 'react';
import './ProductCard.css';

const ProductCard = ({ product, explanation }) => {
  // Log the number of images
  console.log(`Product: ${product?.title}, Images: ${product?.images?.length ?? 0}`);

  const [reviewData, setReviewData] = useState(null);
  const [showReviews, setShowReviews] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  useEffect(() => {
    // Fetch review data if available
    if (product.has_reviews) {
      fetchReviews(product.id);
    }
  }, [product.id, product.has_reviews]);

  const fetchReviews = async (productId) => {
    try {
      const response = await fetch(`http://localhost:8000/reviews/${productId}`);
      if (response.ok) {
        const data = await response.json();
        setReviewData(data);
      }
    } catch (error) {
      console.error("Failed to fetch reviews:", error);
    }
  };

  const formatPrice = (price) => {
    return price ? `$${price.toFixed(2)}` : 'Price not available';
  };

  const renderSentimentBar = (score) => {
    // Convert -1 to 1 score into a percentage (0-100%)
    const percentage = ((score + 1) / 2) * 100;
    const barColor = score > 0.3 ? 'green' : score < -0.3 ? 'red' : 'orange';
    
    return (
      <div className="sentiment-bar-container">
        <div 
          className="sentiment-bar" 
          style={{ width: `${percentage}%`, backgroundColor: barColor }}
        />
      </div>
    );
  };

  const toggleReviews = () => {
    setShowReviews(!showReviews);
  };

  // --- Image Navigation ---
  const hasImages = product.images && Array.isArray(product.images) && product.images.length > 0;

  const nextImage = () => {
    if (hasImages) {
      setCurrentImageIndex((prevIndex) => (prevIndex + 1) % product.images.length);
    }
  };

  const prevImage = () => {
    if (hasImages) {
      setCurrentImageIndex((prevIndex) => (prevIndex - 1 + product.images.length) % product.images.length);
    }
  };
  // --- End Image Navigation ---

  return (
    <div className="product-card">
      {/* Image Display */}
      {hasImages && (
        <div className="product-image-container">
          <img 
            src={product.images[currentImageIndex]?.large || product.images[currentImageIndex]?.thumb || 'placeholder.jpg'}
            alt={`${product.title} - Image ${currentImageIndex + 1}`} 
            className="product-image" 
          />
          {product.images.length > 1 && (
            <div className="image-nav">
              <button onClick={prevImage} className="arrow left-arrow">&lt;</button>
              <button onClick={nextImage} className="arrow right-arrow">&gt;</button>
            </div>
          )}
        </div>
      )}
      {/* End Image Display */}

      <div className="product-title">{product.title}</div>
      <div className="product-price">{formatPrice(product.price)}</div>
      <div className="product-category">
        {product.main_category || (product.categories && product.categories[0]) || 'Uncategorized'}
      </div>
      
      {explanation && (
        <div className="product-explanation">
          <p><strong>Why this matches:</strong> {explanation}</p>
        </div>
      )}
      
      <div className="product-description">
        {product.description ? (
          product.description.slice(0, 150) + (product.description.length > 150 ? '...' : '')
        ) : (
          'No description available'
        )}
      </div>
      
      {product.has_reviews && (
        <div className="product-reviews">
          <button 
            className="reviews-toggle"
            onClick={toggleReviews}
          >
            {showReviews ? 'Hide Reviews' : 'Show Reviews'}
          </button>
          
          {showReviews && reviewData && (
            <div className="reviews-section">
              <div className="review-summary">
                <p><strong>{reviewData.review_count}</strong> customer reviews</p>
                
                {reviewData.overall_sentiment !== undefined && (
                  <div className="sentiment">
                    <p>Overall sentiment: {reviewData.overall_sentiment > 0.5 ? 'Very Positive' : 
                      reviewData.overall_sentiment > 0 ? 'Positive' : 
                      reviewData.overall_sentiment > -0.5 ? 'Mixed' : 'Negative'}</p>
                    {renderSentimentBar(reviewData.overall_sentiment)}
                  </div>
                )}
                
                {reviewData.usage_contexts && reviewData.usage_contexts.length > 0 && (
                  <div className="usage-contexts">
                    <p><strong>Customers use this for:</strong> {reviewData.usage_contexts.slice(0, 2).join(', ')}</p>
                  </div>
                )}
                
                {reviewData.aspect_sentiments && Object.keys(reviewData.aspect_sentiments).length > 0 && (
                  <div className="aspect-sentiments">
                    <p><strong>Product qualities:</strong></p>
                    <div className="aspects-grid">
                      {Object.entries(reviewData.aspect_sentiments).map(([aspect, score]) => (
                        <div key={aspect} className="aspect-item">
                          <div className="aspect-name">{aspect}</div>
                          {renderSentimentBar(score)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {reviewData.representative_reviews && reviewData.representative_reviews.length > 0 && (
                  <div className="representative-reviews">
                    <p><strong>Sample review:</strong></p>
                    <div className="review-text">
                      "{reviewData.representative_reviews[0].text.slice(0, 200)}
                      {reviewData.representative_reviews[0].text.length > 200 ? '...' : ''}"
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ProductCard; 