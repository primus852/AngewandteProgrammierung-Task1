from src.Census.Census import Census

if __name__ == '__main__':
    census = Census()

    # Clean the Dataset
    rows_with_missing, rows_with_missing_pct = census.analyze_data()
    print('Dataset - Total: {} // Rows with missing: {} ({}%)'.format(len(census.data), rows_with_missing, round(rows_with_missing_pct, 2)))

    # Show the results of the single columns
    for name, data in census.data.iteritems():
        val_unique, val_missing, val_missing_pct, most_frequent = census.analyze_data_column(name)
        print('{} - Unique: {} / Missing: {} ({}%) / Most frequent: {}'.format(name, val_unique, val_missing, round(val_missing_pct, 2), most_frequent))

