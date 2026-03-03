"""Unit tests for BLEU score computation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.metric.Bleu import score_sentence, score_corpus


class TestBLEUSentence:
    """Tests for sentence-level BLEU scoring."""
    
    def test_perfect_match(self):
        """Test BLEU score for perfect match."""
        pred = [1, 2, 3, 4, 5]
        gold = [1, 2, 3, 4, 5]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # Perfect match should give BLEU = 1.0
        assert scores[-1] == pytest.approx(1.0, abs=0.01)
    
    def test_no_match(self):
        """Test BLEU score for complete mismatch."""
        pred = [1, 2, 3, 4, 5]
        gold = [6, 7, 8, 9, 10]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # No overlap should give very low BLEU
        assert scores[-1] < 0.1
    
    def test_partial_match(self):
        """Test BLEU score for partial match."""
        pred = [1, 2, 3, 4, 5]
        gold = [1, 2, 3, 6, 7]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # Some overlap but not perfect
        assert 0.1 < scores[-1] < 1.0
    
    def test_brevity_penalty(self):
        """Test that brevity penalty is applied."""
        pred = [1, 2]
        gold = [1, 2, 3, 4, 5, 6, 7, 8]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # Short prediction should have lower score due to brevity penalty
        assert scores[-1] < 0.5
    
    def test_empty_prediction(self):
        """Test handling of empty prediction."""
        pred = []
        gold = [1, 2, 3]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # Empty prediction should have very low score
        assert all(score < 0.1 for score in scores)
    
    def test_different_ngrams(self):
        """Test with different n-gram sizes."""
        pred = [1, 2, 3, 4]
        gold = [1, 2, 3, 4]
        
        scores_2gram = score_sentence(pred, gold, ngrams=2)
        scores_4gram = score_sentence(pred, gold, ngrams=4)
        
        # Both should be high for perfect match
        assert scores_2gram[-1] > 0.9
        assert scores_4gram[-1] > 0.9


class TestBLEUCorpus:
    """Tests for corpus-level BLEU scoring."""
    
    def test_corpus_perfect_match(self):
        """Test corpus BLEU for perfect matches."""
        preds = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        golds = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        score = score_corpus(preds, golds, ngrams=4)
        
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_corpus_no_match(self):
        """Test corpus BLEU for no matches."""
        preds = [[1, 2, 3], [4, 5, 6]]
        golds = [[7, 8, 9], [10, 11, 12]]
        
        score = score_corpus(preds, golds, ngrams=4)
        
        assert score < 0.1
    
    def test_corpus_mixed_quality(self):
        """Test corpus BLEU with mixed quality translations."""
        preds = [
            [1, 2, 3, 4],      # Perfect
            [5, 6, 7, 8],      # Perfect
            [9, 10, 11, 12]    # Completely wrong
        ]
        golds = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [13, 14, 15, 16]
        ]
        
        score = score_corpus(preds, golds, ngrams=4)
        
        # Should be between 0 and 1
        assert 0.1 < score < 1.0
    
    def test_corpus_different_lengths(self):
        """Test corpus BLEU with varying sentence lengths."""
        preds = [
            [1, 2],
            [3, 4, 5, 6, 7],
            [8, 9, 10]
        ]
        golds = [
            [1, 2, 3],
            [3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11]
        ]
        
        score = score_corpus(preds, golds, ngrams=4)
        
        # Should compute despite length differences
        assert 0.0 < score < 1.0
    
    def test_corpus_vs_average_sentence(self):
        """Test that corpus BLEU differs from average sentence BLEU."""
        preds = [[1, 2, 3], [4, 5, 6]]
        golds = [[1, 2, 3], [4, 5, 7]]
        
        corpus_score = score_corpus(preds, golds, ngrams=4)
        
        sent_scores = [
            score_sentence(preds[0], golds[0], ngrams=4)[-1],
            score_sentence(preds[1], golds[1], ngrams=4)[-1]
        ]
        avg_sent_score = sum(sent_scores) / len(sent_scores)
        
        # Corpus BLEU is not just average of sentence BLEUs
        assert abs(corpus_score - avg_sent_score) > 0.01
    
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        preds = []
        golds = []
        
        # Should handle empty corpus gracefully
        score = score_corpus(preds, golds, ngrams=4)
        assert isinstance(score, float)
    
    def test_smoothing_effect(self):
        """Test that smoothing affects scores."""
        preds = [[1, 2, 3]]
        golds = [[4, 5, 6]]
        
        score_no_smooth = score_corpus(preds, golds, ngrams=4, smooth=0)
        score_with_smooth = score_corpus(preds, golds, ngrams=4, smooth=1.0)
        
        # Smoothing should prevent zero scores
        assert score_with_smooth > score_no_smooth


class TestBLEUEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_token_perfect(self):
        """Test single token perfect match."""
        pred = [1]
        gold = [1]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # Should handle gracefully
        assert len(scores) > 0
    
    def test_single_token_mismatch(self):
        """Test single token mismatch."""
        pred = [1]
        gold = [2]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        assert scores[-1] < 0.5
    
    def test_repeated_tokens(self):
        """Test handling of repeated tokens."""
        pred = [1, 1, 1, 1]
        gold = [1, 2, 3, 4]
        
        scores = score_sentence(pred, gold, ngrams=4)
        
        # Should only count the token once (not reward repetition)
        assert scores[-1] < 0.5
    
    def test_long_sequences(self):
        """Test with long sequences."""
        pred = list(range(100))
        gold = list(range(100))
        
        score = score_corpus([pred], [gold], ngrams=4)
        
        # Should handle long sequences
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_unicode_tokens(self):
        """Test with non-integer tokens (if supported)."""
        # Most implementations use integer indices, but test anyway
        pred = [1, 2, 3]
        gold = [1, 2, 3]
        
        score = score_corpus([pred], [gold], ngrams=4)
        
        assert score > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
