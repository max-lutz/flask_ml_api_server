import pandas as pd
#custom function
def extract_first_letter(serie):
    return pd.DataFrame(serie.str[0])