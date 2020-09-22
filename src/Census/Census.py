import pandas as pd
import pathlib
import seaborn as sns

from src.Utils.Utils import Utils


class Census:

    def __init__(self):
        # Config Pandas for less precision
        pd.set_option('display.precision', 2)

        self.data = self._load_data()
        self.clean_data = self._make_ready()

    @staticmethod
    def _load_data():
        """
        Load the data from the csv
        :return:
        """
        dataset = pathlib.Path.cwd() / 'data' / 'adult.data'

        colnames = ['age',
                    'workclass',
                    'fnlwgt',
                    'education',
                    'education-num',
                    'martial-status',
                    'occupation',
                    'relationship',
                    'race',
                    'sex',
                    'capital-gain',
                    'capital-loss',
                    'hours-per-week',
                    'native-country',
                    'income'
                    ]

        df = Utils.trim_all_columns(pd.read_csv(dataset, names=colnames, header=None, converters={"header": float}))

        return df

    def analyze_data(self):
        """
        Analyze the data for missing fields (rows with missing fields)
        :return:
        """
        rows_with_missing = Utils.missing_row_count(self.data, '?')
        rows_with_missing_pct = 100 * rows_with_missing / len(self.data)

        return rows_with_missing, rows_with_missing_pct

    def analyze_data_column(self, col_name, save_plot = True):
        """
        Analyze a single column for missing data
        :param save_plot:
        :param col_name:
        :return:
        """
        val_unique = len(self.data[col_name].unique())
        val_missing = 0
        if '?' in self.data[col_name].value_counts():
            val_missing = self.data[col_name].value_counts()['?']
        val_missing_pct = 100 * val_missing / len(self.data[col_name])

        most_frequent = self.data[col_name].value_counts().idxmax()

        # Save a plot
        if save_plot:

            p = pathlib.Path('plots'.format(col_name))
            p.mkdir(parents=True, exist_ok=True)
            print(col_name)
            # cplot = sns.violinplot(x=self.data.columns, y=col_name, data=self.data, hue='income', palette='Set1')
            bplot = sns.boxplot(y=col_name, x='income',
                                data=self.data,
                                width=0.5,
                                palette="colorblind")

            fig = bplot.get_figure()
            fig.savefig(p / '{}_box.png'.format(col_name))

        return val_unique, val_missing, val_missing_pct, most_frequent

    def _make_ready(self):
        return self.data.drop(['income'], axis=1)
