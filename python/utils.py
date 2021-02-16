import numpy as np

def test_validity(listOfVecs):
    for vec in listOfVecs:
        if (vec is None):
            return False
    return True

def pick_valid_side(left, right):
    if (not test_validity([left]) and not test_validity([right])):
        return None
    if (not test_validity([left]) and test_validity([right])):
        return right
    if (test_validity([left]) and not test_validity([right])):
        return left
    if (test_validity([left]) and test_validity([right])):
        return (left + right) / 2.0

def interp(val, x1, x2, y1, y2):
    """
        Assumes val is in range [x1, x2]. x1 maps to y1 and
        x2 maps to y2. Computes what val would map to if
        we linearly interpolated between y1 and y2
    """
    if (val < x1 or val > x2):
        raise ValueError("val not in [x1, x2] range")
    return ((val - x1)/(x2 - x1)) * (y2 - y1) + y1

def camel_to_train(str): 
    res = [str[0].lower()] 
    for c in str[1:]: 
        if c in ('ABCDEFGHIJKLMNOPQRSTUVWXYZ'): 
            res.append('-') 
            res.append(c.lower()) 
        else: 
            res.append(c) 
      
    return (''.join(res)).replace(" ", "")
      
if __name__ == "__main__":
    val = 4.5
    x1 = 4
    x2 = 5
    print(interp(val, x1, x2, 10, 100))
