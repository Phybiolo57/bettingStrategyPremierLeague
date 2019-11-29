import pandas as pd
import numpy as np
import pandasql as ps
import tensorflow as tf


def map_results(results):
    features = {}

    for result in results:
        for key in result.keys():
            if key not in features:
                features[key] = []

            features[key].append(result[key])

    for key in features.keys():
        features[key] = np.array(features[key])


    return features, features['FT']

def train(features, labels, batchSize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    return dataset.shuffle(1000).repeat().batch(batchSize)

def formatFeatures(features,featureCategories):
    featureFinal = dict()
    numColumns = features.shape[1]

    for i in range(0, numColumns):
        featureFinal[featureCategories[i]] = features[:, i]

    return featureFinal


def separateFeaturesAndCategories(trainingData, train_data_Y, testData, test_data_Y):
    trainingFeatures = formatFeatures(trainingData.as_matrix(),trainingData.columns)
    trainingCategories = train_data_Y
    testFeatures = formatFeatures(testData.as_matrix(), testData.columns)
    testCategories = test_data_Y

    return trainingFeatures, trainingCategories, testFeatures, testCategories

#2017-2018 stats

example_data = pd.DataFrame()
example_data = pd.read_csv('training-data.csv',',').as_matrix()
print('example-data')

# testing_eg_data = formatFeatures(example_data[:, -1:])

data = pd.read_csv('players_raw.csv',',')
club_data = pd.read_csv('premier_league.csv',',')

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
match_data = pd.read_csv('book.csv', ',')
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
final_output.to_csv(r'final_output.csv')
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
eight_match_stats = pd.read_csv('new_match.csv',',')


q3 = """select
  HomeTeam,
  AwayTeam,
  case WHEN FTR = 'H' THEN "1" WHEN FTR = 'A' THEN "2" ELSE "3" END as FT
from
  eight_match_stats"""
stats = ps.sqldf(q3,locals())
stats.to_csv('ikr1.csv', sep='\t')

q4 = """select
  op.FT
from
  stats op
  inner join final_output dt on ( op.HomeTeam = dt.HT and op.AwayTeam = dt.AT)
"""
result_data = ps.sqldf(q4,locals())
result_data.to_csv('ikr.csv', sep='\t')
print('Hello')

q5 = """select
op.FT, fp.*
from
    stats op
    inner join final_output fp on ( op.HomeTeam = fp.HT and op.AwayTeam = fp.AT)

"""
file_data = ps.sqldf(q5,locals())
file_data.to_csv('ikr2.csv', sep='\t')

# ['Chelsea','Man United','Arsenal','Man City','Tottenham','Liverpool','Everton','West Ham','Leicester','Southampton','Crystal Palace'
# ,'Stoke','Swansea','Watford','Newcastle','West Brom','Bournemouth','Brighton','Burnley','Huddersfield']


train_data = file_data[['HHTW',	'AHTW',	'HHTL'	,'AHTL'	,'HATW',	'AATW',	'HATL'	,'AATL'	,'HTPts',	'ATPts']]
print(train_data)

train_data_X = train_data[0:30]
train_data_o = file_data['FT']
train_data_Y =train_data_o[0:30]


test_data_X = train_data[31:60]
test_data_Y = train_data_o[31:60]


trainingFeatures, trainingCategories, testFeatures, testCategories = \
separateFeaturesAndCategories(train_data_X, train_data_Y, test_data_X, test_data_Y)


# train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#         x=train_data_X,
#         y=train_data_Y,
#         batch_size=500,
#         num_epochs=None,
#         shuffle=True
#     )

# train_input_fn = lambda::

 # train_features, train_labels = map_results(train_results)
 # test_features, test_labels = map_results(test_results)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=trainingFeatures,
        y=trainingCategories,
        batch_size=5,
        num_epochs=None,
        shuffle=True
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=testFeatures,
        y=testCategories,
        num_epochs=1,
        shuffle=False
)

feature_columns = []

feature_columns = feature_columns + [
            tf.feature_column.numeric_column(key='HHTW'),
            tf.feature_column.numeric_column(key='AHTW'),
            tf.feature_column.numeric_column(key='HHTL'),
            tf.feature_column.numeric_column(key='AHTL'),
            tf.feature_column.numeric_column(key='HATW'),
            tf.feature_column.numeric_column(key='AATW'),
            tf.feature_column.numeric_column(key='HATL'),
            tf.feature_column.numeric_column(key='AATL'),
            tf.feature_column.numeric_column(key='HTPts'),
            tf.feature_column.numeric_column(key='ATPts'),
]

print(feature_columns)

model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[10],
        feature_columns=feature_columns,
        n_classes=3,
        label_vocabulary=['3','2','1'],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
))

for i in range(0, 200):
        model.train(input_fn= train_input_fn, steps=100)
        evaluation_result = model.evaluate(input_fn=test_input_fn)

        predictions = list(model.predict(input_fn=test_input_fn))
