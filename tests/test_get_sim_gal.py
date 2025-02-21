#python -m unittest tests/test_unit.py 

from unittest import TestCase
import src.h2s_get_sim_gal as gsg

class TestPredict(TestCase):
    def test_suma(self):
        self.assertAlmostEqual(gsg.suma(1,1),2,1e-5)

if __name__ == '__main__':
    unittest.main()