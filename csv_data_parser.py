import pandas as pd


class DataParser:
    def __init__(self, filepath):
        try:
            self.data = pd.read_csv(filepath)
        except:
            print('Unable to read file')
            raise FileNotFoundError

    def get_verified_account_tweets(self):
        return self.data.loc[self.data['user_verified'] == True]

    def get_user_with_loc_tweets(self):
        return self.data.loc[self.data['user_location'].isnull() == False]

    def get_user_with_desc_tweets(self):
        return self.data.loc[self.data['user_description'].isnull == False]