# Sagar Sapkota 2014-2018
# FP-Growth Data Mining Frequent Pattern Algorithm
# Author: Sagar Sapkota <spkt.sagar@gmail.com>
#
# License: MIT License
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TransactionEncoder(BaseEstimator, TransformerMixin):
    """Encoder class for transaction data in Python lists

    Parameters
    ------------

    Attributes
    ------------
    columns: list
      List of unique names in the `X` input list of lists

    """

    def __init__(self):
        pass

    def fit(self, X):
        """Learn unique column names from transaction DataFrame

        Parameters
        ------------
        X : list of lists
          A python list of lists, where the outer list stores the
          n transactions and the inner list stores the items in each
          transaction.

          For example,
          [['Apple', 'Beer', 'Rice', 'Chicken'],
           ['Apple', 'Beer', 'Rice'],
           ['Apple', 'Beer'],
           ['Apple', 'Bananas'],
           ['Milk', 'Beer', 'Rice', 'Chicken'],
           ['Milk', 'Beer', 'Rice'],
           ['Milk', 'Beer'],
           ['Apple', 'Bananas']]

        """
        unique_items = set()
        for transaction in X:
            for item in transaction:
                unique_items.add(item)
        self.columns = sorted(unique_items)
        columns_mapping = {}
        for col_idx, item in enumerate(self.columns):
            columns_mapping[item] = col_idx
        self.columns_mapping = columns_mapping
        return self

    def transform(self, X):
        """Transform transactions into a one-hot encoded NumPy array.

        Parameters
        ------------
        X : list of lists
          A python list of lists, where the outer list stores the
          n transactions and the inner list stores the items in each
          transaction.

          For example,
          [['Apple', 'Beer', 'Rice', 'Chicken'],
           ['Apple', 'Beer', 'Rice'],
           ['Apple', 'Beer'],
           ['Apple', 'Bananas'],
           ['Milk', 'Beer', 'Rice', 'Chicken'],
           ['Milk', 'Beer', 'Rice'],
           ['Milk', 'Beer'],
           ['Apple', 'Bananas']]

        Returns
        ------------
        array : NumPy array [n_transactions, n_unique_items]
           The one-hot encoded boolean array of the input transactions,
           where the columns represent the unique items found in the input
           array in alphabetic order.

           For example,
           array([[True , False, True , True , False, True ],
                  [True , False, True , False, False, True ],
                  [True , False, True , False, False, False],
                  [True , True , False, False, False, False],
                  [False, False, True , True , True , True ],
                  [False, False, True , False, True , True ],
                  [False, False, True , False, True , False],
                  [True , True , False, False, False, False]])
          The corresponding column labels are available as self.columns_, e.g.,
          ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']
        """
        array = np.zeros((len(X), len(self.columns)), dtype=bool)
        for row_idx, transaction in enumerate(X):
            for item in transaction:
                col_idx = self.columns_mapping[item]
                array[row_idx, col_idx] = True
        return array

    def inverse_transform(self, array):
        """Transforms an encoded NumPy array back into transactions.

        Parameters
        ------------
        array : NumPy array [n_transactions, n_unique_items]
            The NumPy one-hot encoded boolean array of the input transactions,
            where the columns represent the unique items found in the input
            array in alphabetic order

            For example,
            ```
            array([[True , False, True , True , False, True ],
                  [True , False, True , False, False, True ],
                  [True , False, True , False, False, False],
                  [True , True , False, False, False, False],
                  [False, False, True , True , True , True ],
                  [False, False, True , False, True , True ],
                  [False, False, True , False, True , False],
                  [True , True , False, False, False, False]])
            ```
            The corresponding column labels are available as self.columns_,
            e.g., ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

        Returns
        ------------
        X : list of lists
            A python list of lists, where the outer list stores the
            n transactions and the inner list stores the items in each
            transaction.

          For example,
          ```
          [['Apple', 'Beer', 'Rice', 'Chicken'],
           ['Apple', 'Beer', 'Rice'],
           ['Apple', 'Beer'],
           ['Apple', 'Bananas'],
           ['Milk', 'Beer', 'Rice', 'Chicken'],
           ['Milk', 'Beer', 'Rice'],
           ['Milk', 'Beer'],
           ['Apple', 'Bananas']]
          ```

        """
        return [[self.columns[idx]
                 for idx, cell in enumerate(row) if cell]
                for row in array]

    def fit_transform(self, X):
        """Fit a TransactionEncoder encoder and transform a dataset."""
        return self.fit(X).transform(X)
