def check_ol_skewn_iqr(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)] #outliers
        skewness = df[col].skew()                                        #skewness
        
        print(f"Column: {col}")
        print(f"Outliers:\n{len(outliers)}")
        print(f"Skewness: {skewness:.2f}")
        print(f"Q1: {q1:.2f}, Q2 (Median): {df[col].median():.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
        print("===================================")

def data_health(dataframe):
    # Corrected the selection of numeric columns
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    results_data = {
        'Column Name': [],
        'Outliers': [],
        'Skewness': [],
        'Q1': [],
        'Median': [],
        'Q3': [],
        'IQR': []
    }

    for col in numeric_columns:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)]
        skewness = dataframe[col].skew()

        # Append the results to the dictionary
        results_data['Column Name'].append(col)
        results_data['Outliers'].append(len(outliers))
        results_data['Skewness'].append(skewness)
        results_data['Q1'].append(q1)
        results_data['Median'].append(dataframe[col].median())  # Corrected reference to dataframe variable
        results_data['Q3'].append(q3)
        results_data['IQR'].append(iqr)

    results_df = pd.DataFrame(results_data)
    return results_df


#defining function to plot histgram and boxplot for numeric columns
def plot_histogram_boxplot(column_data, figsize = 'default',tick_params = 'default'):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    # histogram 
    sns.histplot(column_data, kde=True, color='skyblue', ax=axes[0])
    axes[0].ticklabel_format(style='plain', axis='x')
    axes[0].set_title(f' {column_data.name}')
    axes[0].set_xlabel(column_data.name)
    axes[0].set_ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    if True:
       axes[0].tick_params(axis = input(),labelrotation = int(input()))
    # table

    stats = column_data.describe()
    cell_text = [[f"{index}", f"{value:.2f}"] for index, value in stats.items()]
    axes[1].axis('tight')
    axes[1].axis('off')
    axes[1].table(cellText=cell_text, cellLoc='left', edges='vertical', loc='center')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    
    
    # Box Plot
    axes[2].boxplot(column_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgray'))
    axes[2].ticklabel_format(style='plain', axis='x')
    axes[2].set_title(f' {column_data.name}')
    axes[2].set_xlabel(column_data.name)
    plt.grid(True)
    plt.tight_layout()    
    if True:
        axes[2].tick_params(axis = input(),labelrotation = int(input()))

def outlier_removed(data,column):
    q1 = np.quantile(data[column],0.25)
    q3 = np.quantile(data[column],0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    #iqr = q3 - q1
    #outliers = df[(df['property_area'] < lower_bound) | (df['property_area'] > upper_bound)]
    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    no_outliers = data[~outliers_mask]
                   
    return no_outliers

def test_of_independence(df, target_column, columns_to_exclude):
    results = {}
    for column in df.columns:
        if column == target_column or column in columns_to_exclude:
            continue
        
        contingency_table = pd.crosstab(df[column], df[target_column])
        
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        if (expected < 5).any():
            print(f"Column '{column}' has expected frequency less than 5.")
            results[column] = 'Frequency check failed'
            continue
        if p < 0.05:
            print(f"Column '{column}' is associated with the target variable (p-value={p}).")
            results[column] = 'Associated'
        else:
            print(f"Column '{column}' is NOT associated with the target variable (p-value={p}).")
            results[column] = 'Not associated'
    
    return results


#chi2_results = test_of_independence(cat, 'loan_status', ['loan_status'])
#chi2_results


def perform_normality_and_levene_tests(df, target_variable):
    groups = df[target_variable].unique()

    for column in df.select_dtypes(include=[np.number]).columns:
        if column == target_variable:
            continue

        group1 = df[df[target_variable] == groups[0]][column]
        group2 = df[df[target_variable] == groups[1]][column]

        shapiro_group1 = stats.shapiro(group1)[1]
        shapiro_group2 = stats.shapiro(group2)[1]
        levene_test = stats.levene(group1, group2)[1]

        print(f"Results for {column}:")
        print(f"  Shapiro Group 1 (p-value = {shapiro_group1}): {'Normal' if shapiro_group1 > 0.05 else 'Not normal'}")
        print(f"  Shapiro Group 2 (p-value = {shapiro_group2}): {'Normal' if shapiro_group2 > 0.05 else 'Not normal'}")
        print(f"  Levene Test (p-value = {levene_test}): {'Equal variances' if levene_test > 0.05 else 'Unequal variances'}\n")


#perform_normality_and_levene_tests(df, 'loan_status')


def subset_function(df,predictor,target):
    non_defaulter = df[df[target] == 'Fully Paid'][predictor]
    defaulter = df[df[target] == 'Default'][predictor]
    return

def subset_function(df, predictor, target):
    non_defaulter = pd.DataFrame(df[df[target] == 'Fully Paid'][predictor])
    defaulter = pd.DataFrame(df[df[target] == 'Default'][predictor])
    return non_defaulter, defaulter


#defining the code to categorise zip_codes into subzones,

def map_to_sub_zone(row):
    region_name = None
    for region, states in zo_states_dict.items():
        if row['addr_state'] in states:
            region_name = region
            break
    
    if region_name:
        zone_range = 250  
        sub_zone = (int(row['zip_code']) // zone_range) + 1  #creating 4 sub zones for each value predsnt in region.
        return f"{region_name} - Sub-Zone {sub_zone}"
    
    return None

#df1['sub-regions'] = df1.apply(map_to_sub_zone, axis=1)

def clean_title(title):
    p = inflect.engine()
    title = title.lower()
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    title = re.sub(r'\bstore\b', 'store manager', title)
    title = re.sub(r'\b1st\b', 'first', title)
    title = re.sub(r'\b2nd\b', 'second', title)
    title = re.sub(r'\b3rd\b', 'third', title)
    
    num_patterns = {
        r'\b4\b': 'four',
        r'\b5\b': 'five',
        r'\b6\b': 'six',
        r'\b7\b': 'seven',
        r'\b8\b': 'eight',
        r'\b9\b': 'nine',
    }
    for pattern, word in num_patterns.items():
        title = re.sub(pattern, word, title)
    
    # Replace other numbers with their words using inflect library
    title = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), title)
    
    # Handle specific cases
    title = re.sub(r'(\b[A-Z]+\b)', lambda x: x.group().replace(' ', ''), title)
    
    # Handle cases where numbers are connected to words with no spaces
    title = re.sub(r'(\b[A-Za-z]+)(\d+)', r'\1 \2', title)
    
    return title.strip()

# Function to categorize the titles
def categorize_title(title):
    title = title.lower()
    if 'school' in title:
        return 'Education'
    elif 'bank' in title:
        return 'Banking'
    elif 'services' in title:
        return 'Services'
    elif 'manager' in title:
        return 'managers'
    elif 'analyst' in title:
        return 'analyst'
    elif 'engineer' in title:
        return 'engineers'
    elif 'executive' in title:
        return 'executives'
    elif 'inc' in title:
        return 'inc companies'
    elif 'driver' in title:
        return 'drivers'
    elif 'service' in title:
        return 'Services'
    elif 'accounting' in title:
        return 'accounting companies'
    elif 'transportaion' in title:
        return 'transport companies'
    elif 'restaurant' in title:
        return 'restaurants'
    elif 'assistant' in title:
        return 'assistants'
    elif 'medical' in title:
        return 'health industry'
    elif 'pharmaceuticals' in title:
        return 'health industry'
    elif 'teacher' in title:
        return 'teachers'
    elif 'supervisor' in title:
        return 'supervisors'
    elif 'specialist' in title:
        return 'specialist'
    elif 'coordinator' in title:
        return 'coordinator'
    elif 'company' in title:
        return 'companies'
    elif 'Education' in title:
        return 'Education'
    elif 'broker' in title:
        return 'broker'
    elif 'university' in title:
        return 'education'
    elif 'hospital' in title:
        return 'health industry'
    elif 'officer' in title:
        return 'officers'
    elif 'army' in title:
        return 'us govt'
    elif 'government' in title:
        return 'us govt'
    
    else:
        return title

def cat1(title):
    title = title.lower()

    if title in ['others']:
        return title
    elif title in ['inc comanies', 'companies', 'services', 'ibm', 'jp morgan chase', 'walmart', 'walgreens', 'banking', 'education', 'accounting companies', 'wells fargo']:
        return 'corporates'
    elif title in ['managers', 'vice president', 'ceo', 'president', 'senior consultant', 'director', 'director of operations', 'financial advisor']:
        return 'Elite grade individuals'
    elif title in ['department of defense', 'department of homeland security', 'us navy', 'us air force', 'us govt']:
        return 'govt entities'
    else:
        return 'average grade individuals'

#defining function to plot histgram and boxplot for numeric columns
def plot_histogram_boxplot(column_data):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    # histogram 
    sns.histplot(column_data, bins=20, kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title(f' {column_data.name}')
    axes[0].set_xlabel(column_data.name)
    axes[0].set_ylabel('Frequency')
    plt.grid(True)
    
    # Box Plot
    axes[1].boxplot(column_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgray'))
    axes[1].set_title(f' {column_data.name}')
    axes[1].set_xlabel(column_data.name)
    plt.grid(True)
    plt.tight_layout()    

def count_plot(data,x,y):
    sns.countplot(data  = data, x = x, hue = y)
    plt.title(f'Distributon of {x} with {y}  ')
    

def count_pl(data,x,y):
    sns.countplot(data = data,x = x ,hue = y)
   # plt.text(x = 6, y = 4, s = 6)
    #plt.text(i, y[i], y[i], ha = 'center')
    plt.tight_layout()



def bivar_gp(df, region_column, target_column):

    
    gp = df.groupby(by=[region_column])[target_column].value_counts().unstack(fill_value=0)

    # total count of target column to other independent categorical column
    gp['Total'] = gp.sum(axis=1)

    # Calculate the percentage of 0's in each category
    gp['Percentage_0'] = gp[0] / gp['Total'] * 100

    # renaming columns for better visualization
    gp = gp.reset_index()
    gp = gp.rename(columns={0: 'Count_0', 1: 'Count_1'})

    return gp




if __name__ == "__main__":
    print("running localy")