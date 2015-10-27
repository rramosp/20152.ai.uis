import numpy as np
input_sm="""\
{ 'a': np.array([1,2,3]), 'b': np.array([4,5,6]) }
"""

output_sm="""\
[5 7 9]
"""

from checks import *
max_score = 1
score     = 0
score += check_function(suma_matrices, input_sm, output_sm) if "suma_matrices" in locals() else 0

print "---"
print "calificacion: "+str(score*100/max_score)+"%"
