from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import os

def index(request):

     return render(request,'index.html')

def calculate(request):
    year = request.POST.get('year')
    present_price = request.POST.get('present_price')
    kms_driven = request.POST.get('kms_driven')
    fuel = request.POST.get('fuel_type')
    transmission = request.POST.get('transmission')

    if (int(year)>2022 or int(year)<2008):
        revalue = {"revalue": "ENTER year in between range:"}
        return render(request, 'result.html', revalue)
    elif (int(kms_driven)>500000):
        revalue = {"revalue": "This is overdriven car, don't go for it."}
        return render(request, 'result.html', revalue)
    else:
        Dataset = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\car data.csv')

        Dataset = Dataset[
            ['Selling_Price', 'Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Car_Name', 'Seller_Type',
             'Owner']]

        indep = Dataset.iloc[:, 1:6].values
        dep = Dataset.iloc[:, 0].values

        impu = SimpleImputer(missing_values=np.nan, strategy='mean')
        impu.fit(indep[:, 0:3])
        indep[:, 0:3] = impu.transform(indep[:, 0:3])

        le = LabelEncoder()
        indep[:, 4] = le.fit_transform(indep[:, 4])
        indep[:, 3] = le.fit_transform(indep[:, 3])

        x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size=0.2, random_state=0)

        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        print('Prediction: ', model.score(x_test, pred))
        print('Test Score: ', model.score(x_test, y_test), '\n')

        mse = mean_absolute_error(y_test, pred)
        # print(mse)

        i = 0
        for some in pred:
            if pred[i] > y_test[i]:
                pred[i] = pred[i] - mse
                i = i + 1
            else:
                i = i + 1

        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        user_input = [[int(year), int(present_price), int(kms_driven), int(fuel), int(transmission)]]
        price = model.predict(user_input)

        price = str(price)[1:-1]
        revalue = {"revalue": price}
        return render(request, 'result.html', revalue)