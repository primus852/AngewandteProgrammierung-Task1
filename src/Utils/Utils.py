import numpy as np


class Utils:

    @staticmethod
    def trim_all_columns(df):
        """
        Trim whitespace from ends of each value across all series in dataframe
        """
        trim_strings = lambda x: x.strip() if isinstance(x, str) else x
        return df.applymap(trim_strings)

    @staticmethod
    def missing_row_count(df, search_string):
        """
        :param df: pd.DataFrame
        :param search_string: String
        :return: Integer
        """
        rows_with_missing = 0
        for index, row in df.iterrows():

            for col_index, col in row.iteritems():

                if col == search_string:
                    rows_with_missing += 1

        return rows_with_missing
