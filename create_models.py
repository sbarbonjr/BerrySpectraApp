import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

PATH = "/home/barbon/PycharmProjects/BerrySpectraApp/datasets"
FILES = [PATH+'/Data_SSC_V3.xlsx', PATH+'/Data_firmness_V3.xlsx', PATH+'/Data_TA_V3.xlsx']

def get_dataset(target, preprocessing):
    if target == "SSC (°Brix)":
        file_xlsx = FILES[0]
    elif target == "Firmness (gf)":
        file_xlsx = FILES[1]
    elif target == "Titratable acidity (%)":
        file_xlsx = FILES[2]
    else:
        print("File not found!")
        return None
    dataset = pd.read_excel(file_xlsx, header=1, sheet_name=preprocessing)
    return dataset

def outlier_removal(data, perc, seed):
  if perc==0:
    return(data)
  iso = IsolationForest(contamination=perc, random_state=seed)
  yhat = iso.fit_predict(data.iloc[:,3:231])
  mask = yhat != -1
  return data.iloc[mask, :]

def get_pls_cv (X_train, y_train, X_test, y_test, qtd_LV):
  pls = PLSRegression(n_components=qtd_LV)
  pls.fit(X_train, y_train)

  X_train_transformed = pls.transform(X_train)
  X_test_transformed = pls.transform(X_test)

  return(X_train_transformed, X_test_transformed, y_train, y_test, pls)

def create_model(target, preprocessing, best_LV, perc_isoF, seed_isoF, save_mean_spectra=False):
    #Target: SSC (°Brix)
    #PLS: SIM
    #Training Size: 453
    #perc_isoF: 0.10
    #seed_isoF: 42
    #R-squared (Cal): 0.9580
    #RMSE (Cal): 0.4900
    #MAE (Cal): 0.3780
    #R-squared (Pred): 0.7610
    #RMSE (Pred): 1.1203
    #MAE (Pred): 0.8546
    #LV: 11
    #Algorithm: RF Regression
    #Preprocessing: Norm + SNV + 1st
    
    dataset = get_dataset(target, preprocessing)
    
    if perc_isoF>0:
        dataset = outlier_removal(dataset, perc_isoF, seed_isoF)

    if save_mean_spectra:

        mean_spectra = pd.DataFrame(dataset.iloc[:,3:231].mean().round(4))
        mean_spectra.index = mean_spectra.index.astype(int)
        mean_spectra.columns = ["Absorbance"]
        mean_spectra.to_csv(PATH+"/MeanSpectra_"+target+".csv", header=True, index=True, index_label=["Wavelength"])

    
    y, X =  shuffle(pd.DataFrame(dataset[target]),
                   pd.DataFrame(dataset.iloc[:,3:231]), random_state=42)
        
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if best_LV>0:
        x_train, x_test, y_train, y_test, pls_model  = get_pls_cv(x_train, y_train, x_test, y_test, best_LV)
        joblib.dump(pls_model, PATH+"/"+target+'_PLS.pkl')

    print('x_train Shape:', x_train.shape)

    model = RandomForestRegressor(random_state=1, n_estimators=150)
    model.fit(x_train, y_train[target].values.ravel())
    y_pred = model.predict(x_test)
    mean_r2 = r2_score(y_test[target].values.ravel(), y_pred)
    mean_rmse = mean_squared_error(y_test[target].values.ravel(), y_pred, squared=False)
    mean_mae = mean_absolute_error(y_test[target].values.ravel(), y_pred)

    print('CAL - R-squared (R^2): %.3f' % mean_r2)
    print('CAL - Root Mean Squared Error (RMSE): %.3f' % mean_rmse)
    print('CAL - MAE: %.3f' % mean_mae)

    joblib.dump(model, PATH+"/"+target+'_random_forest_model.pkl')


create_model("SSC (°Brix)", "Norm + SNV + 1st", 11, 0.1, 42, True)
create_model("Firmness (gf)", "Norm + 2nd", 11, 0.05, 42, True)
create_model("Titratable acidity (%)", "Norm + 2nd", 0, 0, 0, True)

    