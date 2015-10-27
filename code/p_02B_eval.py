import numpy as np
from haar import  *
M = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [ 5, 7, 5, 5, 5, 5, 5, 5, 5, 5], [ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], [ 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [10,10,10,10,10,10,10,10,10,10]])
Mi = get_integral(M)

output_haar_1 = -248
output_haar_2 = 2
output_haar_3 = 108
output_haar_4 = 112
output_haar_5 = 2

input_ssum_1   = {'topleft_row':1, 'topleft_col':2, 'height':4, 'width':6}
output_ssum_1  = 84

from checks import *
max_score = 3
score     = 0

score_haar = 0
score_haar += extract_haar(haar_1, M) == output_haar_1 if "haar_1" in locals() else 0
score_haar += extract_haar(haar_2, M) == output_haar_2 if "haar_2" in locals() else 0
score_haar += extract_haar(haar_3, M) == output_haar_3 if "haar_3" in locals() else 0
score_haar += extract_haar(haar_4, M) == output_haar_4 if "haar_4" in locals() else 0
score_haar += extract_haar(haar_5, M) == output_haar_5 if "haar_5" in locals() else 0
if score_haar==5:
   print "haar features: CORRECTO!!"
   score += 1
else:
   print "haar features: INCORRECTO!! verifica tu codigo"

if "get_submatrix_sum_using_integral" in locals():
  score_ssum = 0
  score_ssum += get_submatrix_sum_using_integral(None, Mi, **input_ssum_1) == output_ssum_1
  if score_ssum==1:
     print "get_submatrix_sum_using_integral CORRECTO!!"
     score += 1
  else:
     print "get_submatrix_sum_using_integral: INCORRECTO!! verifica tu codigo"


if "d_haar" in locals():
   if d_haar.shape[0]==1500 and d_haar.shape[1]==200 and np.sum(d_haar)==-473660366.0:
      print "make_haar_dataset_for_MNIST CORRECTO!!"
      score += 1
else:
   print "make_haar_dataset_for_MNIST INCORRECTO!! verifica tu codigo"


print "---"
print "calificacion: %d/%d (%.0f"%(score, max_score, score*100/max_score)+"%)"

