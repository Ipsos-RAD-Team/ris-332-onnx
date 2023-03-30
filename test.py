import main    # The code to test
import unittest   # The test framework

class Test_unittesting(unittest.TestCase):
    def test_main(self):
        self.assertEqual(main.unittest(), [8, 12, 15])

if __name__ == '__main__':
    unittest.main()

# Tests in pytest