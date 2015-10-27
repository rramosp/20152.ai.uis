import numpy as np
input_svm="""\
{"n_rows": 3, "n_cols": 4, "value":5 }
{"n_rows": 4, "n_cols": 2, "value":3.45 }
"""

output_svm="""\
[[ 5.  5.  5.  5.]
 [ 5.  5.  5.  5.]
 [ 5.  5.  5.  5.]]
[[ 3.45  3.45]
 [ 3.45  3.45]
 [ 3.45  3.45]
 [ 3.45  3.45]]
"""

input_ssd="""\
{'A': np.array([[1, 2], [3, 4]]), 'B': np.array([[0, 3], [4, 6]]) }
{'A': np.array([[ 1,  2,  5,  6], [ 3,  4, 10,  1]]), 'B': np.array([[ 0,  3,  5,  7], [ 4,  6,  9, -1]])}
"""

output_ssd="""\
7
13
"""

input_ms="""\
{'A': np.array([[1,2,3],[3,4,5]])}
{'A': np.array([[1,2,3],[3,4,5]]), 'eje': 'columnas'}
{'A': np.array([[1,2,3],[3,4,5]]), 'eje': 'filas'}
{'A': np.array([1,2,3]) }
{'A': np.array([[[ 1, 7], [ 5, 9]], [[1, 1], [ 2,  0]]]) } 
"""

output_ms="""\
(array([ 2.,  3.,  4.]), array([ 1.,  1.,  1.]))
(array([ 2.,  3.,  4.]), array([ 1.,  1.,  1.]))
(array([ 2.,  4.]), array([ 0.81649658,  0.81649658]))
[None, None]
[None, None]
"""

input_sel="""\
{'V': np.array([-4,5,-1,2,3,10,-2]), 'umbral': 0}
{'V': np.array([-4,5,-1,2,3,10,-2]), 'umbral': -2}
{'V': np.array([-4,5,-1,2,3,10,-2]), 'umbral': 12}
"""

output_sel="""\
[ 5  2  3 10]
[ 5 -1  2  3 10]
[]
"""

from checks import *
max_score = 4
score     = 0
score += check_function(single_value_matrix, input_svm, output_svm) if "single_value_matrix" in locals() else 0
score += check_function(sum_squared_difference, input_ssd, output_ssd) if "sum_squared_difference"  in locals() else 0
score += check_function(seleccion_mayor, input_sel, output_sel) if "seleccion_mayor"  in locals() else 0
score += check_function(means_stds, input_ms, output_ms) if "means_stds" in locals() else 0

print "---"
print "calificacion: %d/%d (%.0f"%(score, max_score, score*100/max_score)+"%)"

