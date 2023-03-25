import numpy as np


def partial_dMF(x, mf_definition, partial_parameter):
    """
    Calculates the partial derivative of a membership function at a point x.

    Parameters
    ----------
    x : float
        Point at which to calculate the partial derivative.
    mf_definition : tuple
        Tuple containing the name of the membership function and its parameters.
    partial_parameter : str
        String indicating which parameter to take the partial derivative with respect to.

    Returns
    -------
    float
        The partial derivative of the membership function at the given point and with respect to the given parameter.
    
    Raises
    ------
    ValueError
        If the partial_parameter is not a valid parameter for the given membership function.

    """
    
    # Ensure that the input value is a float or a NumPy array of floats
    if not isinstance(x, (float, np.ndarray)):
        raise TypeError("Input value must be a float or a NumPy array of floats.")
    
    # Ensure that the membership function definition is a tuple with two elements
    if not isinstance(mf_definition, tuple) and len(mf_definition) != 2:
        raise TypeError("Membership function definition must be a tuple with two elements.")
    
    # Ensure that the partial parameter is a string
    if not isinstance(partial_parameter, str):
        raise TypeError("Partial parameter must be a string.")
    
    
    mf_name = mf_definition[0]
    mf_params = mf_definition[1]

    if mf_name == 'gaussmf':

        sigma = mf_params['sigma']
        mean = mf_params['mean']

        if partial_parameter == 'sigma':
            result = (2./sigma**3) * np.exp(-(((x-mean)**2)/(sigma)**2))*(x-mean)**2
        elif partial_parameter == 'mean':
            result = (2./sigma**2) * np.exp(-(((x-mean)**2)/(sigma)**2))*(x-mean)
        else:
           raise ValueError(f"Invalid partial parameter '{partial_parameter}' for Gaussian membership function.")


    elif mf_name == 'gbellmf':

        a = mf_params['a']
        b = mf_params['b']
        c = mf_params['c']

        if partial_parameter == 'a':
            result = (2. * b * np.power((c-x),2) * np.power(np.absolute((c-x)/a), ((2 * b) - 2))) / \
                (np.power(a, 3) * np.power((np.power(np.absolute((c-x)/a),(2*b)) + 1), 2))
        elif partial_parameter == 'b':
            result = -1 * (2 * np.power(np.absolute((c-x)/a), (2 * b)) * np.log(np.absolute((c-x)/a))) / \
                (np.power((np.power(np.absolute((c-x)/a), (2 * b)) + 1), 2))
        elif partial_parameter == 'c':
            result = (2. * b * (c-x) * np.power(np.absolute((c-x)/a), ((2 * b) - 2))) / \
                (np.power(a, 2) * np.power((np.power(np.absolute((c-x)/a),(2*b)) + 1), 2))
        else:
           raise ValueError(f"Invalid partial parameter '{partial_parameter}' for bell-shaped membership function.")


    elif mf_name == 'sigmf':

        b = mf_params['b']
        c = mf_params['c']

        if partial_parameter == 'b':
            result = -1 * (c * np.exp(c * (b + x))) / \
                np.power((np.exp(b*c) + np.exp(c*x)), 2)
        elif partial_parameter == 'c':
            result = ((x - b) * np.exp(c * (x - b))) / \
                np.power((np.exp(c * (x - c))) + 1, 2)
        else:
           raise ValueError(f"Invalid partial parameter '{partial_parameter}' for sigmoidal membership function.")
           
    
    else:
        raise ValueError(f"Invalid membership function name '{mf_name}'.")


    return result