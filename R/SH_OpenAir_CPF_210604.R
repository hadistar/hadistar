
library(openair)

# For minutes wind data -> df

data<- read.csv("D:\\Dropbox\\PMF_paper\\SH_PMF_meteo_hourly_v8.csv", header = TRUE) 
#winddata<- read.csv("D:\\OneDrive - SNU\\R\\Openair_20_1103\\wind_1486.csv", header = TRUE)

data$date <- as.POSIXct(strptime(data$date, format = "%Y-%m-%d", tz = "GMT"))

winddata$date <- as.POSIXct(strptime(winddata$date, format = "%Y-%m-%d", tz = "GMT"))
winddata$date2 <- as.POSIXct(strptime(winddata$date2, format = "%Y-%m-%d %H:%M", tz = "GMT"))

df<-merge(x = data, y = winddata, by = 'date', all.x = TRUE)
df = df[c(order(df$date2)),]
rownames(df) <- NULL

head(df)

# For hourly wind data -> data

data<- read.csv("D:\\Dropbox\\PMF_paper\\SH_PMF_meteo_hourly_v8.csv", header = TRUE) 


data$date <- as.POSIXct(strptime(data$date, format = "%Y-%m-%d %H:%M", tz = "GMT"))


#mydata$date <- as.POSIXct(strptime(mydata$date, format = "%d/%m/%Y %H:%M", tz = "GMT"))


## Haze splitting

df_haze = df[(df$PM25_observed>35.0), ]

data_haze = data[(data$PM25_observed>=35.0), ]
data_nonhaze = data[(data$PM25_observed<35.0), ]


# Drawing

# for percentile

for(i in colnames(data)[45:53]) {

  if (i != 'date' & i != 'wd' & i != 'ws'){
    print(i)
    
    for (per in seq(0,90,10)){
      
      jpeg(paste(i,', ',per,' to ',per+10,'.jpeg'))
      polarPlot(data, pollutant = i, col = "jet", key.position = "bottom",
                stati = 'cpf', percentile= c(per,per+10),
                #key.header = "mean PM25(ug/m3)", 
                key.footer = NULL, main = i)
      dev.off()
    }
  }
    
}


# for original
for(i in colnames(data)){
  
  if (i != 'date' & i != 'wd' & i != 'ws'){
    print(i)
    jpeg(paste(i,'.jpeg'))
    polarPlot(data_nonhaze, pollutant = i, col = "jet", key.position = "bottom",
              key.header = "mean PM25(ug/m3)", key.footer = NULL, main = i)
    dev.off()
  }
}

# 참고자료


polarPlot(data, pollutant = 'coal', col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL, main = i,
          statistic= 'cpf', percentile = 75,type="weekday")

polarPlot(data, pollutant = "PM25_observed", col = "jet", key.position = "bottom",
          key.header = "mean PM25(ug/m3)", key.footer = NULL, main = 'A')

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
