import pandas as pd
data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

filtered_data = data[data['year'].between(2015, 2018)]

def top_players(df, year, metric):
    return df[df['year'] == year].nlargest(10, metric)[['batter_name', metric]]

# 1. Print the top 10 players in 'H', 'avg', 'HR' and 'OBP' for each year from 2015 to 2018
print("Answer1)")
for year in range(2015, 2019):
    for metric in ['H', 'avg', 'HR', 'OBP']:
        print(f"Year {year}, Metric {metric}:")
        print(top_players(filtered_data, year, metric))
        print()

# 2. Print the player with the highest war by position(cp) in 2018
players_2018 = filtered_data[filtered_data['year'] == 2018]
players_2018 = players_2018[players_2018['cp'] != '지명타자']
top_war_players_list = []

for position in players_2018['cp'].unique():
    max_war_player = players_2018[players_2018['cp'] == position].nlargest(1, 'war')
    top_war_players_list.append(max_war_player)

top_war_players = pd.concat(top_war_players_list).reset_index(drop=True)
print("Answer2) Player with the highest war by position in 2018:")
print(top_war_players[['batter_name', 'war', 'tp']])
print()

# 3. find the highest correlation with salary
correlation_cols = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']
correlation_data = data[correlation_cols].corr()
correlation_with_salary = correlation_data['salary'].sort_values(ascending=False)

highest_correlation_stat = correlation_with_salary.drop('salary').idxmax()
highest_correlation_value = correlation_with_salary.drop('salary').max()

print(f"Answer3) The highest correlation with salary: '{highest_correlation_stat}', Correlation coefficient: {highest_correlation_value:.3f}")