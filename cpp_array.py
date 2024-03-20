import re

def convert_to_cpp_string(tensor_string):
    # Extract the numerical values from the tensor string
    values = re.findall(r'[-+]?\d*\.\d+|\d+', tensor_string)
    
    # Get the dimensions of the weight matrix
    dim1 = int([0])
    dim2 = int(values[1])
    
    # Convert the values into a 2D list
    matrix = [values[i:i+dim2] for i in range(2, len(values), dim2)]
    
    # Generate the C++ string representation
    cpp_string = f"float w1[{dim1}][{dim2}] = {{"
    for row in matrix:
        cpp_string += "{"
        cpp_string += ",".join(row)
        cpp_string += "},"
    cpp_string = cpp_string.rstrip(",")  # Remove the trailing comma
    cpp_string += "};"
    
    return cpp_string

# Example usage
tensor_string = "tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])"
cpp_string = convert_to_cpp_string(tensor_string)
print(cpp_string)
