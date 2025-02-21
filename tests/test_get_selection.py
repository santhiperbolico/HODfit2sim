#python -m unittest tests/test_unit.py 

from unittest import TestCase
import src.h2s_io as u

class TestPredict(TestCase):
    def test_get_selection(self):
        self.assertAlmostEqual(u.get_selection('prueba.txt','resultados.txt','txt', [2], [4], [10]),[0,1,2],1e-5)

if __name__ == '__main__':
    unittest.main()