import unittest
from unittest.mock import patch
import numpy as np
from VNAV.utilities.visualization import visualize_side_by_side, visualize_overlay

class TestVisualization(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_visualize_side_by_side(self, mock_show):
        # Create dummy images (one RGB, one grayscale)
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Call the function, it should not block because plt.show is mocked
        visualize_side_by_side(img1, img2, "Black RGB", "White Grayscale")
        
        # Verify that plt.show was called to render the figure
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_visualize_overlay(self, mock_show):
        # Create dummy images of different shapes to test resizing logic as well
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((120, 120, 3), dtype=np.uint8) * 255
        
        # Call the function
        visualize_overlay(img1, img2, "Rendered", "Real")
        
        # Verify that plt.show was called to render the figure
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
