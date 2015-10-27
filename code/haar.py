import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def get_integral(M):
    """ 
    assumes one channel grayscale so that len(img.shape)=2
    """
    r = np.zeros(M.shape)
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            r[y,x] += r[y-1,x] if y>0 else 0
            r[y,x] += r[y,x-1] if x>0 else 0
            r[y,x] -= r[y-1,x-1] if x>0 and y>0 else 0
            r[y,x] += M[y,x]
    return r.astype(int)

def create_sample_matrix(N):
    M = []
    for i in range(N):
        M.append(np.ones(N)*(i+1))

    M = np.array(M).reshape((N,N)).astype(int)
    M[4,1] += 2
    return M

def get_submatrix_sum(image, integral, topleft_row, topleft_col, height, width):
    return np.sum(image[topleft_row:topleft_row+height, topleft_col:topleft_col+width])

def relative_2_absolute_coords(num_rows, num_cols, 
                               topleft_row_rel, topleft_col_rel, 
                               height_rel, width_rel):

    tlr    = int(round(num_rows*topleft_row_rel))
    tlc    = int(round(num_cols*topleft_col_rel))
    height = int(round(num_rows*height_rel))
    width  = int(round(num_cols*width_rel))
    
    if tlr < 0 or tlr+height > num_rows or\
       tlc < 0 or tlc+width  > num_cols:
        return None,None,None,None
    
    return tlr, tlc, height, width

def show_haar_feature(haar_feature):
    N = 100
    a = np.zeros((N,N))
    for i in haar_feature:
        tlr, tlc = i["topleft_row_rel"]*N, i["topleft_col_rel"]*N
        h, w     = i["height_rel"]*N, i["width_rel"]*N
        if i["op"]=="add":
            a[tlr:tlr+h,tlc:tlc+w]=50
        elif i["op"]=="sub":
            a[tlr:tlr+h,tlc:tlc+w]+=150
    plt.imshow(a, cmap = plt.cm.Greys_r, vmin=0, vmax=255)
    plt.xticks([]); plt.yticks([])
    
    
def show_haar_features(haar_features):
    fig = plt.figure(figsize=(12,5))
    for i in range(len(haar_features)):
        fig.add_subplot(len(haar_features)/7+1,7,i+1)
        show_haar_feature(haar_features[i])
        plt.title(str(i+1))
        
        
def shift_scale_haar(haar_feature, scale_width=1., scale_height=1., shift_rows=0., shift_cols=0.):
    r = []
    for i in haar_feature:
        b = {}
        b["op"] = i["op"]
        b["topleft_row_rel"] = i["topleft_row_rel"] * scale_height + shift_rows
        b["topleft_col_rel"] = i["topleft_col_rel"] * scale_width + shift_cols
        b["height_rel"]      = i["height_rel"] * scale_height
        b["width_rel"]       = i["width_rel"]  * scale_width
        r.append(b)
    return r

def extract_haar(haar_feature, image=None, integral=None, submatrix_sum_function=get_submatrix_sum ):
    img_h = integral.shape[0] if integral is not None else image.shape[0]
    img_w = integral.shape[1] if integral is not None else image.shape[1]
    result = 0
    for i in haar_feature:
        tlr_rel, tlc_rel = i["topleft_row_rel"], i["topleft_col_rel"]
        h_rel, w_rel     = i["height_rel"], i["width_rel"]
        tlr, tlc, h, w   = relative_2_absolute_coords(img_h, img_w, tlr_rel, tlc_rel, h_rel, w_rel)
        if tlr!=None:
            sub_matrix_sum = submatrix_sum_function(image, integral, tlr, tlc, h, w)
            if i["op"]=="add" and not sub_matrix_sum is None:
                result += sub_matrix_sum
            elif i["op"]=="sub" and not sub_matrix_sum is None:
                result -= sub_matrix_sum
        else:
            return 0
    return result

def get_haar_feature_with_transforms(haar_feature, image, integral, submatrix_sum_function=get_submatrix_sum, nb_scales=5, nb_shifts=5):

    l_scale_height = np.linspace(0,1,nb_scales)
    l_scale_width  = np.linspace(0,1,nb_scales)
    l_shift_cols   = np.linspace(0,1,nb_shifts)
    l_shift_rows   = np.linspace(0,1,nb_shifts)
    features = []
    c=0
    for scale_height in l_scale_height:
        for scale_width in l_scale_width:
            for shift_rows in l_shift_rows:
                for shift_cols in l_shift_cols:                
                    if scale_height + shift_rows < 1.0 and scale_width + shift_cols < 1.0 and\
                       scale_height!=0 and scale_width!=0:
                        h = extract_haar(shift_scale_haar(haar_feature, 
                                                          scale_height=scale_height,
                                                          scale_width=scale_width,
                                                          shift_rows=shift_rows,
                                                          shift_cols=shift_cols),
                                         image, integral,
                                         submatrix_sum_function=submatrix_sum_function)
                        features.append(h)
                        c += 1
    return features

def get_haar_features(haar_features, image, integral, submatrix_sum_function=get_submatrix_sum, nb_scales=5, nb_shifts=5 ):
    r = []
    for hf in haar_features:
        r += (get_haar_feature_with_transforms(hf, image, integral, submatrix_sum_function, nb_scales, nb_shifts))
    return r

