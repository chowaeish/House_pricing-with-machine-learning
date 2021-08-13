import pandas as pd
import numpy as np
from eda import *
from data_prep import *
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#df = pd.read_csv("train.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = train.append(test).reset_index(drop=True)
df.head()
check_df(df)
# LabelEncoder #
df["LotShape"]
df.loc[(df["LotShape"] == "Reg"),"new_LotShape"] = 0
df.loc[(df["LotShape"] == "IR1"), "new_LotShape"] = 1
df.loc[(df["LotShape"] == "IR2"), "new_LotShape"] = 2
df.loc[(df["LotShape"] == "IR3"),"new_LotShape"] = 3
df["LotShape"]
df["new_LotShape"]

df["Utilities"]
df.loc[(df["Utilities"]=="AllPub"),"new_Utilities"] = 0
df.loc[(df["Utilities"]=="NoSewr"),"new_Utilities"] = 1
df.loc[(df["Utilities"]=="NoSeWa"), "new_Utilities"] = 2
df.loc[(df["Utilities"]=="ELO"), "new_Utilities"] = 3
df["Utilities"]
df["new_Utilities"].value_counts()

df["ExterQual"]
df.loc[df["ExterQual"] == "Ex","new_ExterQual"]= 0
df.loc[df["ExterQual"] == "Gd","new_ExterQual"]= 1
df.loc[df["ExterQual"] == "TA","new_ExterQual"]= 2
df.loc[df["ExterQual"] == "Fa","new_ExterQual"]= 3
df.loc[df["ExterQual"] == "Po","new_ExterQual"]= 4
df["ExterQual"]
df["new_ExterQual"].value_counts()

df["ExterCond"]
df.loc[df["ExterCond"] == "Ex","new_ExterCond"]= 0
df.loc[df["ExterCond"] == "Gd","new_ExterCond"]= 1
df.loc[df["ExterCond"] == "TA","new_ExterCond"]= 2
df.loc[df["ExterCond"] == "Fa","new_ExterCond"]= 3
df.loc[df["ExterCond"] == "Po","new_ExterCond"]= 4
df["ExterCond"]
df["new_ExterCond"].value_counts()

df["BsmtQual"]
df.loc[df["BsmtQual"]=="Ex","new_BsmtQual"] = 0
df.loc[df["BsmtQual"]=="Gd","new_BsmtQual"] = 1
df.loc[df["BsmtQual"]=="TA","new_BsmtQual"] = 2
df.loc[df["BsmtQual"]=="Fa","new_BsmtQual"] = 3
df.loc[df["BsmtQual"]=="Po","new_BsmtQual"] = 4
df.loc[df["BsmtQual"]=="NA","new_BsmtQual"] = 5
df["new_BsmtQual"].value_counts()

df["BsmtCond"]
df.loc[df["BsmtCond"]=="Ex","new_BsmtCond"] = 0
df.loc[df["BsmtCond"]=="Gd","new_BsmtCond"] = 1
df.loc[df["BsmtCond"]=="TA","new_BsmtCond"] = 2
df.loc[df["BsmtCond"]=="Fa","new_BsmtCond"] = 3
df.loc[df["BsmtCond"]=="Po","new_BsmtCond"] = 4
df.loc[df["BsmtCond"]=="NA","new_BsmtCond"] = 5
df["new_BsmtCond"].value_counts()

df["BsmtExposure"]
df.loc[df["BsmtExposure"]=="Gd","new_BsmtExposure"]=0
df.loc[df["BsmtExposure"]=="Av","new_BsmtExposure"]=1
df.loc[df["BsmtExposure"]=="Mn","new_BsmtExposure"]=2
df.loc[df["BsmtExposure"]=="No","new_BsmtExposure"]=3
df.loc[df["BsmtExposure"]=="NA","new_BsmtExposure"]=4
df["new_BsmtExposure"].value_counts()

df["BsmtFinType1"]
df.loc[df["BsmtFinType1"]=="GLQ","new_BsmtFinType1"]=0
df.loc[df["BsmtFinType1"]=="ALQ","new_BsmtFinType1"]=1
df.loc[df["BsmtFinType1"]=="BLQ", "new_BsmtFinType1"]=2
df.loc[df["BsmtFinType1"]=="Rec", "new_BsmtFinType1"]=3
df.loc[df["BsmtFinType1"]=="LwQ", "new_BsmtFinType1"]=4
df.loc[df["BsmtFinType1"]=="Unf", "new_BsmtFinType1"]=5
df.loc[df["BsmtFinType1"]=="NA","new_BsmtFinType1"]=6
df["new_BsmtFinType1"].value_counts()

df["BsmtFinType2"]
df.loc[df["BsmtFinType2"]=="GLQ","new_BsmtFinType2"]=0
df.loc[df["BsmtFinType2"]=="ALQ","new_BsmtFinType2"]=1
df.loc[df["BsmtFinType2"]=="BLQ", "new_BsmtFinType2"]=2
df.loc[df["BsmtFinType2"]=="Rec", "new_BsmtFinType2"]=3
df.loc[df["BsmtFinType2"]=="LwQ", "new_BsmtFinType2"]=4
df.loc[df["BsmtFinType2"]=="Unf", "new_BsmtFinType2"]=5
df.loc[df["BsmtFinType2"]=="NA","new_BsmtFinType2"]=6
df["new_BsmtFinType2"].value_counts()

df["HeatingQC"]
df.loc[df["HeatingQC"]=="Ex","new_HeatingQC"]=0
df.loc[df["HeatingQC"]=="Gd","new_HeatingQC"]=1
df.loc[df["HeatingQC"]=="TA","new_HeatingQC"]=2
df.loc[df["HeatingQC"]=="Fa","new_HeatingQC"]=3
df.loc[df["HeatingQC"]=="Po","new_HeatingQC"]=4
df["new_HeatingQC"].value_counts()

df["CentralAir"]
df.loc[df["CentralAir"]=="Y","new_CentralAir"]=0
df.loc[df["CentralAir"]=="N","new_CentralAir"]=1
df["new_CentralAir"].value_counts()

df["KitchenQual"]
df.loc[df["KitchenQual"]=="Ex","new_KitchenQual"]=0
df.loc[df["KitchenQual"]=="Gd","new_KitchenQual"]=1
df.loc[df["KitchenQual"]=="TA","new_KitchenQual"]=2
df.loc[df["KitchenQual"]=="Fa","new_KitchenQual"]=3
df.loc[df["KitchenQual"]=="Po","new_KitchenQual"]=4
df["new_KitchenQual"].value_counts()

df["Functional"]
df.loc[df["Functional"]=="Typ","new_Functional"] =0
df.loc[df["Functional"]=="Min1","new_Functional"] =1
df.loc[df["Functional"]=="Min2","new_Functional"] =2
df.loc[df["Functional"]=="Mod","new_Functional"] =3
df.loc[df["Functional"]=="Maj1","new_Functional"] =4
df.loc[df["Functional"]=="Maj2","new_Functional"] =5
df.loc[df["Functional"]=="Sev","new_Functional"] =6
df.loc[df["Functional"]=="Sal","new_Functional"] =7
df["new_Functional"].value_counts()

df["FireplaceQu"]
df.loc[df["FireplaceQu"]=="Ex","new_FireplaceQu"] = 0
df.loc[df["FireplaceQu"]=="Gd","new_FireplaceQu"] = 1
df.loc[df["FireplaceQu"]=="TA","new_FireplaceQu"] = 2
df.loc[df["FireplaceQu"]=="Fa","new_FireplaceQu"] = 3
df.loc[df["FireplaceQu"]=="Po","new_FireplaceQu"] = 4
df.loc[df["FireplaceQu"]=="NA","new_FireplaceQu"] = 5
df["new_FireplaceQu"].value_counts()

df["GarageFinish"]
df.loc[df["GarageFinish"]=="Fin","new_GarageFinish"]=0
df.loc[df["GarageFinish"]=="RFn","new_GarageFinish"]=1
df.loc[df["GarageFinish"]=="Unf","new_GarageFinish"]=2
df.loc[df["GarageFinish"]=="NA","new_GarageFinish"]=3
df["new_GarageFinish"].value_counts()

df["GarageQual"]
df.loc[df["GarageQual"]=="Ex","new_GarageQual"] = 0
df.loc[df["GarageQual"]=="Gd","new_GarageQual"] = 1
df.loc[df["GarageQual"]=="TA","new_GarageQual"] = 2
df.loc[df["GarageQual"]=="Fa","new_GarageQual"] = 3
df.loc[df["GarageQual"]=="Po","new_GarageQual"] = 4
df.loc[df["GarageQual"]=="NA","new_GarageQual"] = 5
df["new_GarageQual"].value_counts()

df["GarageCond"]
df.loc[df["GarageCond"]=="Ex","new_GarageCond"] = 0
df.loc[df["GarageCond"]=="Gd","new_GarageCond"] = 1
df.loc[df["GarageCond"]=="TA","new_GarageCond"] = 2
df.loc[df["GarageCond"]=="Fa","new_GarageCond"] = 3
df.loc[df["GarageCond"]=="Po","new_GarageCond"] = 4
df.loc[df["GarageCond"]=="NA","new_GarageCond"] = 5
df["new_GarageCond"].value_counts()


df["PoolQC"]
df.loc[df["PoolQC"]=="Ex","new_PoolQC"] = 0
df.loc[df["PoolQC"]=="Gd","new_PoolQC"] = 1
df.loc[df["PoolQC"]=="TA","new_PoolQC"] = 2
df.loc[df["PoolQC"]=="Fa","new_PoolQC"] = 3
df.loc[df["PoolQC"]=="NA","new_PoolQC"] = 4
df["new_PoolQC"].value_counts()

df["Fence"]
df.loc[df["Fence"]=="GdPrv","new_Fence"]=0
df.loc[df["Fence"]=="MnPrv","new_Fence"]=1
df.loc[df["Fence"]=="GdWo","new_Fence"]=2
df.loc[df["Fence"]=="MnWw","new_Fence"]=3
df.loc[df["Fence"]=="NA","new_Fence"]=4
df["new_Fence"].value_counts()

le=["LotShape","Utilities","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","CentralAir","KitchenQual","Functional","FireplaceQu","GarageFinish","GarageQual","GarageCond","PoolQC","Fence"]
for col in le:
    df.drop(col, axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=1)
cat_cols = cat_cols + cat_but_car


new_cat_cols = []
for i in cat_cols:
    if i not in le:
        new_cat_cols.append(i)

new_cat_cols




rare_analyser(df, "SalePrice", new_cat_cols)


df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", new_cat_cols)


useless_cols = [col for col in new_cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]

new_cat_cols = [col for col in new_cat_cols if col not in useless_cols]


for col in useless_cols:
    df.drop(col, axis=1, inplace=True)






df = one_hot_encoder(df, new_cat_cols, drop_first=False)

df.head(5)


df["new_PoolQC"]
missing_values_table(df)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 ]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)


df.isnull().sum().sum()


df.head(5)


#x=df.drop(["SalePrice", "Id"],axis=1)
#y=df["SalePrice"]

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

y = train_df["SalePrice"]
#y = np.log1p(train_df['SalePrice'])
x = train_df.drop(["Id", "SalePrice"], axis=1)


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc.fit_transform(x)


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

##################
# Hyperparameter Optimization
##################


gbm_model = GradientBoostingRegressor(random_state=17)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_model,
                                        x, y, cv=10, scoring="neg_mean_squared_error")))

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [2, 9],
              "n_estimators": [500, 1500],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=True).fit(x, y)


gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(x, y)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, x, y, cv=10, scoring="neg_mean_squared_error")))
#######################################
# Feature Selection
#######################################

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(gbm_final, x)


plot_importance(gbm_final, x, 20)


x.shape

feature_imp = pd.DataFrame({'Value': gbm_final.feature_importances_, 'Feature': x.columns})


num_summary(feature_imp, "Value", True)


feature_imp[feature_imp["Value"] > 0].shape

feature_imp[feature_imp["Value"] < 1].shape


zero_imp_cols = feature_imp[feature_imp["Value"] < 1]["Feature"].values


selected_cols = [col for col in x.columns if col not in zero_imp_cols]
len(selected_cols)

##################
# Hyperparameter Optimization
##################


gbm_model = GradientBoostingRegressor(random_state=17)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_model,
                                        x[selected_cols], y, cv=10, scoring="neg_mean_squared_error")))

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [2, 9],
              "n_estimators": [500, 1500],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=True).fit(x[selected_cols], y)


gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(x[selected_cols], y)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, x[selected_cols], y, cv=10, scoring="neg_mean_squared_error")))
##23401.0469




#######################################
# Sonuçların Yüklenmesi
#######################################
#test_df=pd.read_csv("test.csv")


submission_df = pd.DataFrame()

submission_df['Id'] = test_df["Id"]

y_pred_sub = gbm_final.predict(test_df[selected_cols])


y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.to_csv('submission.csv', index=False)