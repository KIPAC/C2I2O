import numpy as np

from c2i2o.utility import conv_utils


def test_convert_dict_to_2d_array():
    """Test function for convert_dict_to_2d_array"""
    
    input_dict = dict(
        a=np.ones(5),
        b=np.zeros(5),
    )
    
    var_list, out_array = conv_utils.convert_dict_to_2d_array(input_dict)

    assert 'a' in var_list
    assert 'b' in var_list

    assert out_array.shape[0] == 5
    assert out_array.shape[1] == 2


def test_convert_table_to_list_of_dicts():
    """Test function for test_convert_table_to_list_of_dicts"""

    input_dict = dict(        
        a=np.ones(5),
        b=np.zeros(5),
    )

    out_list = conv_utils.convert_table_to_list_of_dicts(input_dict)
    assert len(out_list) == 5

    assert out_list[0] == dict(a=1, b=0)

    
