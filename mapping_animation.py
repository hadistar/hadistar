import folium
import pandas as pd
import branca.colormap as cm
from selenium import webdriver
import time

df = pd.read_csv('Anew_source1.csv')

m = folium.Map(location=[35.5502, 126.982], zoom_start=7,
                tiles='cartodbpositron')

colormap = cm.LinearColormap(colors = ['white','yellow','orange','red'],
                             index = [0,15,30,45],
#                             tick_labels=[0,15,30,45],
#                             scale_width=400, scale_heigh=50,
                             vmin=0, vmax=45,
                             caption='concentration (ug/m3)')

for i, day in enumerate(df.columns[2:]):
    print(i, day)
    for pt in range(int(len(df)/4)):
        pt = pt *4
        color = colormap(df.iloc[pt][i+2])
        folium.CircleMarker(location = [df.iloc[pt][0],df.iloc[pt][1]],
                            radius=0.01,
                            fill=True,
                            color=color,
                            fill_color=color,
                            fill_opacity=0.2,
                            line_opacity=0.2).add_to(m)
    m.add_child(colormap)
    m.save('./test/test_'+str(day).replace('/','_')+'.html')


    browser = webdriver.Chrome('D:/chromedriver.exe')
    browser.get('D:/hadistar/test/test_'+str(day).replace('/','_')+'.html')

    #Give the map tiles some time to load
    time.sleep(10)
    browser.save_screenshot('./test/test_'+str(day).replace('/','_')+'.png')
    browser.quit()
