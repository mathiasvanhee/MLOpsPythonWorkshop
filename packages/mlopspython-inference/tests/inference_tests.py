import unittest
from unittest.mock import Mock
from io import BytesIO
import numpy as np
from mlopspython_inference.inference_pillow import Inference, ImageLoaderInterface, ModelInterface


class TestInference(unittest.TestCase):
    def setUp(self):
        # Step 1: Mock the image loader interface
        self.mock_image_loader = Mock(spec=ImageLoaderInterface)

        # Simulate the behavior of loading an image and converting it to array
        self.mock_image_loader.load_img.return_value = np.ones((1, 224, 224, 3))

        # Step 2: Mock the model interface
        self.mock_model = Mock(spec=ModelInterface)

        # Simulate a prediction with some fake values
        self.mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])

        # Step 3: Mock logger (optional if needed)
        self.mock_logger = Mock()

        # Step 4: Create the Inference object with mocked dependencies
        self.inference = Inference(
            logger=self.mock_logger,
            model=self.mock_model,
            image_loader=self.mock_image_loader
        )

    def test_inference_execute(self):
        # Step 5: Simulate an input image file
        test_image = BytesIO(b'test image data')

        # Step 6: Execute the inference process
        result = self.inference.execute(test_image)

        # Step 7: Check if the prediction and values are as expected
        self.assertEqual(result['prediction'], 'Dog')
        self.assertEqual(result['values'], [0.1, 0.7, 0.2])

        # Step 8: Verify that the correct methods were called with expected parameters
        self.mock_image_loader.load_img.assert_called_once_with(test_image)
        self.mock_model.predict.assert_called_once_with(np.ones((1, 224, 224, 3)))


if __name__ == '__main__':
    unittest.main()
