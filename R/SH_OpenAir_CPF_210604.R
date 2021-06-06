
library(openair)

# For minutes wind data -> df

data<- read.csv("D:\\Data_backup\\Dropbox\\PMF_paper\\data_YSLEE\\PMF results_raw_YSLEE.csv", header = TRUE) 
winddata<- read.csv("D:\\OneDrive - SNU\\R\\Openair_20_1103\\wind_1486.csv", header = TRUE)

data$date <- as.POSIXct(strptime(data$date, format = "%Y-%m-%d", tz = "GMT"))

winddata$date <- as.POSIXct(strptime(winddata$date, format = "%Y-%m-%d", tz = "GMT"))
winddata$date2 <- as.POSIXct(strptime(winddata$date2, format = "%Y-%m-%d %H:%M", tz = "GMT"))

df<-merge(x = data, y = winddata, by = 'date', all.x = TRUE)
df = df[c(order(df$date2)),]
rownames(df) <- NULL

head(df)

# For hourly wind data -> data

data<- read.csv("D:\\Data_backup\\Dropbox\\PMF_paper\\SH_PMF_meteo_hourly.csv", header = TRUE) 


data$date <- as.POSIXct(strptime(data$date, format = "%Y-%m-%d %H:%M", tz = "GMT"))


#mydata$date <- as.POSIXct(strptime(mydata$date, format = "%d/%m/%Y %H:%M", tz = "GMT"))


## Haze splitting

df_haze = df[(df$PM25_observed>35.0), ]


# Drawing

polarPlot(df_haze, pollutant = "Industry.smelting", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL)

polarPlot(df_haze, pollutant = "Industry.Oil.", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL)

polarPlot(df, pollutant = "Mobile", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL)

polarPlot(df_haze, pollutant = "SS", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL, statistic= 'cpf', percentile = 75)

polarPlot(df_haze, pollutant = "Soil", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL)

polarPlot(data, pollutant = "Combustion.for.heating", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL)

polarPlot(data, pollutant = "Coal", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL)


windRose(data, ws='ws', wd='wd')
windRose(data, type='month')


# Examples

#polarPlot(mydata, pollutant = "f6", col = "jet", key.position = "bottom",
#          key.header = "mean PM25(ug/m3)", key.footer = NULL, type="weekday" )


#polarPlot(mydata, pollutant = "난방연소", col = "jet", key.position = "bottom",
#          key.header = "mean PM25(ug/m3)", key.footer = NULL, ,type="season", statistic = "cpf", percentile = 75)
#
#
#polarPlot(mydata, pollutant = "���.��Ŀ����.", col = "jet", key.position = "bottom",
#          key.header = "mean PM25 (ug/m3)", key.footer = NULL, ,type="season")
#
#polarPlot(mydata, pollutant = "���.�Ұ���.", col = "jet", key.position = "bottom",
#          key.header = "mean PM25 (ug/m3)", key.footer = NULL, type="weekend")
#
