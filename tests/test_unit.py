#python -m unittest tests/test_unit.py 

import src.h2s_unit as u
from unittest import TestCase

class TestPredict(TestCase):
    def test_suma(self):
        self.assertAlmostEqual(u.suma(1,1),2,1e-5)

if __name__ == '__main__':
    unittest.main()