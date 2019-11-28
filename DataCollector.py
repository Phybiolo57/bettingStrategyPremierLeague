import pandas as pd
import numpy as np
import pandasql as ps

#2017-2018 stats
data = pd.read_csv('/home/soumava/Documents/python projects linux/football stats/data/2017-18/players_raw.csv',',')
club_data = pd.read_csv('/home/soumava/Documents/python projects linux/football stats/football clubs/data/2017/premier_league.csv',',')

parent_df = pd.DataFrame()
club_db = pd.DataFrame()
player_stat_df = pd.DataFrame()

#Collected previous year player stats
data['Name']= data.apply(lambda x:'%s_%s' % (x['first_name'],x['second_name']),axis=1)

parent_df = data[['first_name', 'second_name', 'total_points', 'influence']]
parent_df['Name']= parent_df.apply(lambda x:'%s %s' % (x['first_name'],x['second_name']),axis=1)
parent_df.drop(columns=['first_name', 'second_name'])

club_df = club_data[['player_name', 'club_name', 'year']]

club_df.rename(columns={"player_name": "Name"}, inplace=True)
merged_df = pd.merge(parent_df, club_df, on='Name', how='inner')

q1 = """select sum(total_points) as TotalPts,club_name as Team,year from merged_df group by club_name,year"""
player_stat_df = ps.sqldf(q1,locals())
print(player_stat_df.head())

#Collecting previous season match details
match_data = pd.read_csv('/home/soumava/PycharmProjects/fifa football predictor/data/book.csv', ',')
query_previous_res = """select HomeTeam,AwayTeam,FTR from match_data"""
match_data_prev = ps.sqldf(query_previous_res,locals())

print(match_data_prev.shape)

final_analysis = pd.DataFrame(columns=['Team', 'HTW', 'HTL', 'ATW', 'ATL'])
print('--------Prev Match Data--------------')
print(match_data_prev.head())

query_temp_home_home_win = "select count(1) from match_data_prev where HomeTeam = '%s' and FTR = 'H'"
query_temp_home_home_loss = "select count(1) from match_data_prev where HomeTeam = '%s' and FTR = 'A'"
query_temp_away_away_loss = "select count(1) from match_data_prev where AwayTeam = '%s' and FTR = 'H'"
query_temp_away_away_win = "select count(1) from match_data_prev where AwayTeam = '%s' and FTR = 'A'"

teamSets = set()
for i in range(len(match_data_prev)) :
    str = match_data_prev.loc[i,'HomeTeam']
    if str not in teamSets:
        teamSets.add(str)
        query_prev_home_team_win = (query_temp_home_home_win % str)
        home_team_win_df = ps.sqldf(query_prev_home_team_win,locals())
        for k in range(len(home_team_win_df)):
            home_home_win = home_team_win_df.loc[k,'count(1)']

        query_prev_home_team_loss = (query_temp_home_home_loss %str)
        home_team_loss_df = ps.sqldf(query_prev_home_team_loss, locals())
        for k in range(len(home_team_loss_df)):
            home_home_loss = home_team_loss_df.loc[k,'count(1)']

        query_prev_away_team_win = (query_temp_away_away_win % str)
        away_team_win_df = ps.sqldf(query_prev_away_team_win, locals())
        for k in range(len(away_team_win_df)):
            away_away_win = away_team_win_df.loc[k,'count(1)']

        query_prev_away_team_loss = (query_temp_away_away_loss % str)
        away_team_loss_df = ps.sqldf(query_prev_away_team_loss, locals())
        for k in range(len(away_team_loss_df)):
            away_away_loss = away_team_loss_df.loc[k,'count(1)']

        final_analysis = final_analysis.append({'TeamShort':str,
                             'HTW':home_home_win,
                             'HTL':home_home_loss,
                             'ATW':away_away_win,
                             'ATL':away_away_loss},ignore_index=True)

print('------Final Data--------')

print(final_analysis.head())

final_analysis.drop(columns=['Team'])
output = pd.DataFrame()
manu_output = pd.DataFrame()
mancity_output = pd.DataFrame()

sql_join_1 = """select * From final_analysis, player_stat_df where player_stat_df.Team Like ('%' || Trim(final_analysis.TeamShort) || '%') """
# sql_join = """select * from final_analysis f inner join player_stat_df pdf on LEFT(f.Team,5) = LEFT(pdf.Team,5)"""
output = ps.sqldf(sql_join_1,locals())
print(output.head())

d_query = "select fa1.TeamShort as HT,fa2.TeamShort as AT, fa1.HTW as HHTW, fa2.HTW as AHTW, fa1.HTL as HHTL, fa2.HTL as AHTL, fa1.ATW as HATW, fa2.ATW as AATW, fa1.ATL as HATL, fa2.ATL as AATL, fa1.TotalPts as HTPts, fa2.TotalPts as ATPts from output fa1 inner join output fa2 where fa1.TeamShort <> fa2.TeamShort"
final_output = ps.sqldf(d_query,locals())
final_output.to_csv(r'/home/soumava/PycharmProjects/fifa football predictor/data/final_output.csv')
# print(ps.sqldf(d_query,locals()).head())



# output.to_csv(r'/home/soumava/PycharmProjects/fifa football predictor/data/stored.csv')

#Team HTW HTL ATW ATL
#Columns to be added
# print(ps.sqldf(q1,locals()).head())
# print(merged_df.head())

# names.append(data['first_name']+" "+data['second_name'], ignore_index=True)

# q1 = """SELECT concat(first_name,' ',second_name) as 'Name',total_points from data"""
# names = ps.sqldf(q1,locals())

# q2 = """SELECT player_name,club_name from club_data"""

eight_match_stats = pd.DataFrame()
eight_match_stats = pd.read_csv('data/new_match.csv',',')


q3 = """select
  HomeTeam,
  AwayTeam,
  case WHEN FTR = 'H' THEN "1" WHEN FTR = 'A' THEN "2" ELSE "3" END as HT
from
  eight_match_stats"""
stats = ps.sqldf(q3,locals())

q4 = """select
  op.HT
from
  stats op
  inner join final_output dt on ( op.HomeTeam = dt.HT and op.AwayTeam = dt.AT)
"""
result_data = ps.sqldf(q4,locals())

print('Hello')
# ['Chelsea','Man United','Arsenal','Man City','Tottenham','Liverpool','Everton','West Ham','Leicester','Southampton','Crystal Palace'
# ,'Stoke','Swansea','Watford','Newcastle','West Brom','Bournemouth','Brighton','Burnley','Huddersfield']