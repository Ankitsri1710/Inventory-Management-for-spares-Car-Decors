# Importing Libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# loading pickle files 
Perfume=pickle.load(open('sarima_model_perfumes.pkl','rb'))
AndriodHeadunits=pickle.load(open('sarima_model_ahu.pkl','rb'))
Speakers=pickle.load(open('sarima_model_speakers.pkl','rb'))
Cameras=pickle.load(open('holtwinter_model_cameras.pkl','rb'))
GLSC=pickle.load(open('sarima_model_GLSC.pkl','rb'))
AlloyWheels=pickle.load(open('sarima_model_aw.pkl','rb'))
Matts_3d=pickle.load(open('holtwinter_model_3DMatts.pkl','rb'))
matts_7d=pickle.load(open('hwe_model_7DMatts.pkl','rb'))
Armrests=pickle.load(open('sarima_Armrests.pkl','rb'))
BumperProtectors=pickle.load(open('Holtwinter_BP.pkl','rb'))
ChromeAccessories=pickle.load(open('sarima_ca.pkl','rb'))
Denting=pickle.load(open('sarima_denting.pkl','rb'))
DoorGuards=pickle.load(open('sarima_doorguards.pkl','rb'))
DoorVisors=pickle.load(open('sarima_doorvisors.pkl','rb'))
FancyGrills=pickle.load(open('sarima_fancygrills.pkl' ,'rb'))
Horns=pickle.load(open('Holtwinter_horns.pkl','rb'))
LedFogLamps=pickle.load(open('sarima_ledFL.pkl','rb'))
LedHeadlights=pickle.load(open('sarima_ledHL.pkl','rb'))
Painting=pickle.load(open('sarima_painting.pkl','rb'))
Sunfilm=pickle.load(open('sarima_sunfilm.pkl','rb'))
RearGuards=pickle.load(open('holtwinter_RearGuards.pkl','rb'))
SideBeading=pickle.load(open('holtwinter_SideBeading.pkl','rb'))
LeatherSeatCover=pickle.load(open('sarima_LeatherSeatCover.pkl','rb'))
Spoilers=pickle.load(open('sarima_model_spoilers.pkl','rb'))
Feature_list=[]
def Sarima_features(Feature):
    result=Feature.fit()
    pred = result.get_prediction(start=75,end=86,dynamic=True,full_results=True)
    prediction=round(pred.predicted_mean)
    prediction=prediction.to_frame().reset_index()
    prediction.columns=['Date','Predicted Sales']
    prediction['Predicted Sales']=prediction['Predicted Sales'].astype("int32")
    Feature_list.append(prediction)

def HoltWinter_features(Feature):
    pred = Feature.predict(start=75,end=86)
    prediction=round(pred)
    prediction=prediction.to_frame().reset_index()
    prediction.columns=['Date','Predicted Sales']
    prediction['Predicted Sales']=prediction['Predicted Sales'].astype("int32")
    Feature_list.append(prediction)
###### Perfume ###########
Sarima_features(Perfume)
perfume=Feature_list[0]
########## AndriodHeadUnits ###########
Sarima_features(AndriodHeadunits)
andriodHeadunits=Feature_list[1]
####### Speakers ###########
Sarima_features(Speakers)
speakers=Feature_list[2]
###### Cameras ###########
HoltWinter_features(Cameras)
cameras=Feature_list[3]
########GL Seat Covers ########
Sarima_features(GLSC)
glsc=Feature_list[4]
########### Alloywheels #########
Sarima_features(AlloyWheels)
alloywheels=Feature_list[5]
############ 3D Matts ##############
HoltWinter_features(Matts_3d)
matts_3d=Feature_list[6]
############ 7D Matts ##############
HoltWinter_features(matts_7d)
Matts_7d=Feature_list[7]
############# Armsrests ############
Sarima_features(Armrests)
armrests=Feature_list[8]
############ BumperProtector ########
HoltWinter_features(BumperProtectors)
bumperprotectors=Feature_list[9]
######## Chrome Accessories ##########
Sarima_features(ChromeAccessories)
ca=Feature_list[10]
######## Denting ############
Sarima_features(Denting)
denting=Feature_list[11]
########### DoorGuards ########
Sarima_features(DoorGuards)
doorguards=Feature_list[12]
########### DoorVisors ##########
Sarima_features(DoorVisors)
doorvisors=Feature_list[13]
########## FancyGrills ##########
Sarima_features(FancyGrills)
fancygrills=Feature_list[14]
########## Horns ############
HoltWinter_features(Horns)
horns=Feature_list[15]
########## Ledfoglamp ########
Sarima_features(LedFogLamps)
ledfl=Feature_list[16]
############# Ledheadlights #######
Sarima_features(LedHeadlights)
ledhl=Feature_list[17]
########## Painting ###############
Sarima_features(Painting)
painting=Feature_list[18]
########## Sunfilm ##############
Sarima_features(Sunfilm)
sunfilm=Feature_list[19]
########### RearGuards ############
HoltWinter_features(RearGuards)
rearguard=Feature_list[20]
########### SideBeading ##########
HoltWinter_features(SideBeading)
sidebeading=Feature_list[21]
########### Leather Seatcover #######
Sarima_features(LeatherSeatCover)
leatherseatcover=Feature_list[22]
######### Spoilers ############
Sarima_features(Spoilers)
spoiler=Feature_list[23]
####### DATA ############
Sales_data=pd.read_csv('C:\\Users\\HP\\Desktop\\Inventory Management Car decors\\Monthly sales data car decors.csv')
app_data=Sales_data
app_data.Date=pd.to_datetime(app_data.Date)
app_data=app_data.set_index('Date')

########## PLOTS ###############
import altair as alt
############ AndriodHeadunits ###########
actual_ah=p=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('AndriodHeadunits:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('AndriodHeadunits:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_ah=alt.Chart(andriodHeadunits).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
ah_plot=actual_ah+predicted_ah
################### Perfumes ###############
actual_perfume=p=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Perfumes:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Perfumes:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_perfume=alt.Chart(perfume).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
perfumes_plot=actual_perfume+predicted_perfume

########### Speakers ###########
actual_speakers=p=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Speakers:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Speakers:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_speakers=alt.Chart(speakers).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
speakers_plot=actual_speakers+predicted_speakers

############## Cameras ###############
actual_cameras=p=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Cameras:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Cameras:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_cameras=alt.Chart(cameras).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
cameras_plot=actual_cameras+predicted_cameras

############## GL seat covers ##########
actual_glsc=p=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('GLSteeringCovers:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('GLSteeringCovers:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_glsc=alt.Chart(glsc).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
glsc_plot=actual_glsc+predicted_glsc

################ Alloy Wheels #############
actual_aw=p=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('AlloyWheels:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('AlloyWheels:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_aw=alt.Chart(alloywheels).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
aw_plot=actual_aw+predicted_aw

################ 3D Matts ################
actual_3dm=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('3DMatts:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('3DMatts:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_3dm=alt.Chart(matts_3d).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_3dm=actual_3dm+predicted_3dm

################ 7D Matts ################
actual_7dm=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('7DMatts:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('7DMatts:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_7dm=alt.Chart(Matts_7d).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_7dm=actual_7dm+predicted_7dm

############ Armrests ###################
actual_armrests=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Armrests:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Armrests:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_armrests=alt.Chart(armrests).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_armrests=actual_armrests+predicted_armrests

########### Bumper Protectors ##############
actual_bp=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('BumperProtectors:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('BumperProtectors:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_bp=alt.Chart(bumperprotectors).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_bp=actual_bp+predicted_bp

############ ChromeAccessories ################
actual_ca=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('ChromeAccessories:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('ChromeAccessories:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_ca=alt.Chart(ca).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_ca=actual_ca+predicted_ca

############## denting ###############
actual_denting=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Denting:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Denting:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_denting=alt.Chart(denting).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_denting=actual_denting+predicted_denting
############ Doorguards #############
actual_doorguards=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('DoorGuards:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('DoorGuards:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_doorguards=alt.Chart(doorguards).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_doorguards=actual_doorguards+predicted_doorguards

################ DoorVisors ##########
actual_doorvisors=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('DoorVisors:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('DoorVisors:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_doorvisors=alt.Chart(doorvisors).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_doorvisors=actual_doorvisors+predicted_doorvisors

########## Fancygrills ######
actual_fancygrills=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('FancyGrills:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('FancyGrills:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_fancygrills=alt.Chart(fancygrills).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_fancygrills=actual_fancygrills+predicted_fancygrills

########### Horns ##########
actual_horns=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Horns:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Horns:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_horns=alt.Chart(horns).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_horns=actual_horns+predicted_horns

############# LedFoglamps ##############
actual_ledfl=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('LedFogLamps:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('LedFogLamps:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_ledfl=alt.Chart(ledfl).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_ledfl=actual_ledfl+predicted_ledfl

############ LedHeadlights #############
actual_ledhl=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('LedHeadlights:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('LedHeadlights:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_ledhl=alt.Chart(ledhl).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_ledhl=actual_ledhl+predicted_ledhl

################ painting ########
actual_painting=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Painting:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Painting:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_painting=alt.Chart(painting).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_painting=actual_painting+predicted_painting

################## Sunfilm ##############
actual_sunfilm=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Sunfilm:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Sunfilm:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_sunfilm=alt.Chart(sunfilm).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_sunfilm=actual_sunfilm+predicted_sunfilm

############## RearGuards #############
actual_rearguard=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('RearGuards:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('RearGuards:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_rearguard=alt.Chart(rearguard).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_rearguard=actual_rearguard+predicted_rearguard

############# Side Beading ##############
actual_sidebeading=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('SideBeading:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('SideBeading:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_sidebeading=alt.Chart(sidebeading).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_sidebeading=actual_sidebeading+predicted_sidebeading

########### Leather Seat Covers ##############
actual_leatherseatcover=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('LeatherSeatCovers:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('LeatherSeatCovers:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_leatherseatcover=alt.Chart(leatherseatcover).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_leatherseatcover=actual_leatherseatcover+predicted_leatherseatcover

############### Spoilers ###################
actual_spoiler=alt.Chart(Sales_data).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                             y=alt.Y('Spoilers:Q',axis=alt.Axis(title='Sales')),
                                                    tooltip=[alt.Tooltip('Spoilers:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
predicted_spoiler=alt.Chart(spoiler).mark_line(point=True).encode(x=alt.X('Date:T',axis=alt.Axis(title='Date')),
                                                             y=alt.Y('Predicted Sales:Q',axis=alt.Axis(title='Sales')),
                                                               color=alt.value('Red'),
                                                                       tooltip=[alt.Tooltip('Predicted Sales:Q',title='Sales per month'),
                                                                             alt.Tooltip('Date:T')]).properties(width=800,height=300)
plot_spoiler=actual_spoiler+predicted_spoiler

# Sales icon 
image_url="https://icon-library.com/images/sales-icon/sales-icon-3.jpg"

## Downloadble link ##
import base64
import time
timestr=time.strftime("%Y%m%d-%H%M%S")
def get_downloadable__link(df):
    csv =df.to_csv(index=True)
    b64=base64.b64encode(csv.encode()).decode()
    new_filename="new_csv_file_{}_.csv".format(timestr)
    st.markdown("DOWNLOAD FILE")
    href =f'<a href="data:file/csv;base64,{b64}" download="{new_filename}"> click here!</a>'
    st.markdown(href,unsafe_allow_html=True)

## Main App ##    
def main():
    st.set_page_config(
        page_title="Sales-App",page_icon=image_url,layout="wide")
    img='''
    <div>
    <center><img src="https://innodatatics.com/images/excelr-logo-small.png"alt="Company logo" style="width:200px;height:50px;">
    </center>
    </div>
    '''
    st.markdown(img,unsafe_allow_html=True)
    st.title('Inventory Management for Spare: Car Decors')
    html_temp = '''
    <div style ="background-color-:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Sales Prediction App </h2>
    </div>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader("Here is our dataset.")
    st.sidebar.title("Dataset")
    data_opt=['Select data','Monthly Sales data']
    data_opt=st.sidebar.selectbox('Select Your data.',data_opt)
    if data_opt=='Monthly Sales data':
        Sales_data['Date']=Sales_data['Date'].astype('str')
        st.dataframe(Sales_data)
    else:
        st.write("No data Selected")
    if st.button('Download'):
        if data_opt=='Monthly Sales data':
            get_downloadable__link(Sales_data)
        else:
           st.write("No data to download")
    
    Decor_items=['Select Decors','3DMatts','7DMatts','AlloyWheels','AndriodHeadunits','Armrests','BumperProtectors',
                 'Cameras','ChromeAccessories','Denting','DoorGuards','DoorVisors','FancyGrills',
                 'GLSteeringCovers','Horns','LeatherSeatCovers','LedFogLamps','LedHeadlights',
                 'Painting','Perfumes','RearGuards','SideBeading','Speakers','Spoilers','Sunfilm']
    st.sidebar.title('Decor Items')
    decor_items=st.sidebar.selectbox('Select Car Decors whose sales you want to predict',Decor_items)
    st.subheader(decor_items)
    st.spinner("Hello")
    
    if st.sidebar.button("Predict"):
        if decor_items=='Perfumes':
            st.subheader(decor_items)
            st.write('Next 12 month prediction')
            perfume['Date']=perfume['Date'].astype("str")
            st.dataframe(perfume)
            st.write('Actual and Predicted Sales Plot')
            perfumes_plot
        elif decor_items=='AndriodHeadunits':
            st.write('Next 12 month prediction')
            andriodHeadunits['Date']=andriodHeadunits['Date'].astype("str")
            st.dataframe(andriodHeadunits)
            st.write('Actual and Predicted Sales Plot')
            ah_plot
        elif decor_items=='Speakers':
            st.write('Next 12 month prediction')
            speakers['Date']=speakers['Date'].astype("str")
            st.dataframe(speakers)
            st.write('Actual and Predicted Sales Plot')
            speakers_plot
        elif decor_items=='Cameras':
            st.write('Next 12 month prediction')
            cameras['Date']=cameras['Date'].astype("str")
            st.dataframe(cameras)
            st.write('Actual and Predicted Sales Plot')
            cameras_plot
        elif decor_items=='GLSteeringCovers':
            st.write('Next 12 month prediction')
            glsc['Date']=glsc['Date'].astype("str")
            st.dataframe(glsc)
            st.write('Actual and Predicted Sales Plot')
            glsc_plot
        elif decor_items=='AlloyWheels':
            st.write('Next 12 month prediction')
            alloywheels['Date']=alloywheels['Date'].astype("str")
            st.dataframe(alloywheels)
            st.write('Actual and Predicted Sales Plot')
            aw_plot
        elif decor_items=='3DMatts':
            st.write('Next 12 month prediction')
            matts_3d['Date']=matts_3d['Date'].astype("str")
            st.dataframe(matts_3d)
            st.write('Actual and Predicted Sales Plot')
            plot_3dm
        elif decor_items=='7DMatts':
            st.write('Next 12 month prediction')
            Matts_7d['Date']=Matts_7d['Date'].astype("str")
            st.dataframe(Matts_7d)
            st.write('Actual and Predicted Sales Plot')
            plot_7dm
        elif decor_items=='Armrests':
            st.write('Next 12 month prediction')
            armrests['Date']=armrests['Date'].astype("str")
            st.dataframe(armrests)
            st.write('Actual and Predicted Sales Plot')
            plot_armrests
        elif decor_items=='BumperProtectors':
            st.write('Next 12 month prediction')
            bumperprotectors['Date']=bumperprotectors['Date'].astype("str")
            st.dataframe(bumperprotectors)
            st.write('Actual and Predicted Sales Plot')
            plot_bp
        elif decor_items=='ChromeAccessories':
            st.write('Next 12 month prediction')
            ca['Date']=ca['Date'].astype("str")
            st.dataframe(ca)
            st.write('Actual and Predicted Sales Plot')
            plot_ca
        elif decor_items=='Denting':
            st.write('Next 12 month prediction')
            denting['Date']=denting['Date'].astype("str")
            st.dataframe(denting)
            st.write('Actual and Predicted Sales Plot')
            plot_denting
        elif decor_items=='DoorGuards':
            st.write('Next 12 month prediction')
            doorguards['Date']=doorguards['Date'].astype("str")
            st.dataframe(doorguards)
            st.write('Actual and Predicted Sales Plot')
            plot_doorguards
        elif decor_items=='DoorVisors':
            st.write('Next 12 month prediction')
            doorvisors['Date']=doorvisors['Date'].astype("str")
            st.dataframe(doorvisors)
            st.write('Actual and Predicted Sales Plot')
            plot_doorvisors
        elif decor_items=='FancyGrills':
            st.write('Next 12 month prediction')
            fancygrills['Date']=fancygrills['Date'].astype("str")
            st.dataframe(fancygrills)
            st.write('Actual and Predicted Sales Plot')
            plot_fancygrills
        elif decor_items=='Horns':
            st.write('Next 12 month prediction')
            horns['Date']=horns['Date'].astype("str")
            st.dataframe(horns)
            st.write('Actual and Predicted Sales Plot')
            plot_horns
        elif decor_items=='LedFogLamps':
            st.write('Next 12 month prediction')
            ledfl['Date']=ledfl['Date'].astype("str")
            st.dataframe(ledfl)
            st.write('Actual and Predicted Sales Plot')
            plot_ledfl
        elif decor_items=='LedHeadlights':
            st.write('Next 12 month prediction')
            ledhl['Date']=ledhl['Date'].astype("str")
            st.dataframe(ledhl)
            st.write('Actual and Predicted Sales Plot')
            plot_ledhl
        elif decor_items=='Painting':
            st.write('Next 12 month prediction')
            painting['Date']=painting['Date'].astype("str")
            st.dataframe(painting)
            st.write('Actual and Predicted Sales Plot')
            plot_painting
        elif decor_items=='Sunfilm':
            st.write('Next 12 month prediction')
            sunfilm['Date']=sunfilm['Date'].astype("str")
            st.dataframe(sunfilm)
            st.write('Actual and Predicted Sales Plot')
            plot_sunfilm
        elif decor_items=='RearGuards':
            st.write('Next 12 month prediction')
            rearguard['Date']=rearguard['Date'].astype("str")
            st.dataframe(rearguard)
            st.write('Actual and Predicted Sales Plot')
            plot_rearguard
        elif decor_items=='SideBeading':
            st.write('Next 12 month prediction')
            sidebeading['Date']=sidebeading['Date'].astype("str")
            st.dataframe(sidebeading)
            st.write('Actual and Predicted Sales Plot')
            plot_sidebeading
        elif decor_items=='LeatherSeatCovers':
            st.write('Next 12 month prediction')
            leatherseatcover['Date']=leatherseatcover['Date'].astype("str")
            st.dataframe(leatherseatcover)
            st.write('Actual and Predicted Sales Plot')
            plot_leatherseatcover
        elif decor_items=='Spoilers':
            st.write('Next 12 month prediction')
            spoiler['Date']=spoiler['Date'].astype("str")
            st.dataframe(spoiler)
            st.write('Actual and Predicted Sales Plot')
            plot_spoiler
        else:
            st.write('No decor item selected.')
           
            
if __name__=='__main__':
    main()