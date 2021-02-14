import csv_data_parser as cdp

if __name__ == '__main__':
    parser = cdp.DataParser('./dataset/vaccination_tweets.csv')
    print(parser.get_user_with_loc_tweets().head())