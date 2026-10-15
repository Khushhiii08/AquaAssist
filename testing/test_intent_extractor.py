import unittest
import sys
import os

# Ensure the src directory is in the path so python can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.intent_extractor import extract_semantic_triplets, extract_hypothesis
from src.diagnostic_engine import detect_conflicting_measurements

class TestAquaAssistPipeline(unittest.TestCase):
    
    def test_english_surface_gasping_observation(self):
        query = "My shrimp are gasping at the surface, but I have not measured the dissolved oxygen."
        triplets, hypothesis = extract_semantic_triplets(query)
        
        # Verify polarity and symptoms
        self.assertEqual(triplets["subject"], "shrimp")
        self.assertIn("surface gasping / swimming near surface", triplets["symptoms"])
        self.assertEqual(triplets["certainty"], "definitive")
        self.assertIn("surface gasping", hypothesis)

    def test_negative_polarity_eating(self):
        query = "My shrimp are not eating and staying at the bottom."
        triplets, hypothesis = extract_semantic_triplets(query)
        
        # Verify negative polarity is strictly preserved
        self.assertEqual(triplets["raw_polarity"], "negative")
        self.assertIn("not eating", triplets["symptoms"])
        self.assertIn("not eating", hypothesis)

    def test_telugu_vernacular_with_uncertainty(self):
        query = "నా చెరువులో shrimp పైకి వస్తున్నాయి, oxygen తక్కువగా ఉందేమో."
        triplets, hypothesis = extract_semantic_triplets(query)
        
        # Verify bilingual recognition and uncertainty marker ("ఉందేమో")
        self.assertEqual(triplets["subject"], "shrimp")
        self.assertIn("surface gasping / swimming near surface", triplets["symptoms"])
        self.assertEqual(triplets["certainty"], "uncertain (may be)")
        self.assertIn("uncertain (may be)", hypothesis)

    def test_conflict_detection_gate(self):
        # Test conflicting DO measurements in a single prompt
        conflicting_query = "My dissolved oxygen is 7 mg/L, but another sensor says DO is 2 mg/L."
        has_conflict = detect_conflicting_measurements(conflicting_query)
        
        self.assertTrue(has_conflict)

    def test_non_conflicting_measurement(self):
        # Test normal single-value input
        normal_query = "My dissolved oxygen is 2.5 mg/L."
        has_conflict = detect_conflicting_measurements(normal_query)
        
        self.assertFalse(has_conflict)

if __name__ == "__main__":
    unittest.main()