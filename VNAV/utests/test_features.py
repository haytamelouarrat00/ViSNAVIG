import unittest
import numpy as np
from VNAV.features.sift_matcher import SIFTMatcher

try:
    from VNAV.features.xfeat_matcher import XFeatMatcher
    HAS_XFEAT = True
except ImportError:
    HAS_XFEAT = False

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create two simple dummy images that are identical
        self.img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        # Add a block to create features
        self.img1[40:60, 40:60, :] = 255
        
        self.img2 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        self.img2[40:60, 40:60, :] = 255

    def test_sift_matcher(self):
        matcher = SIFTMatcher()
        pts1, pts2 = matcher.match(self.img1, self.img2)
        self.assertEqual(pts1.shape[1], 2)
        self.assertEqual(pts2.shape[1], 2)
        self.assertEqual(pts1.shape[0], pts2.shape[0])

    @unittest.skipUnless(HAS_XFEAT, "XFeat library not found in the sibling VS directory.")
    def test_xfeat_matcher(self):
        matcher = XFeatMatcher()
        pts1, pts2 = matcher.match(self.img1, self.img2)
        self.assertEqual(pts1.shape[1], 2)
        self.assertEqual(pts2.shape[1], 2)
        self.assertEqual(pts1.shape[0], pts2.shape[0])

if __name__ == '__main__':
    unittest.main()
